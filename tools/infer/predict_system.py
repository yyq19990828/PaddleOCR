# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import copy
import numpy as np
import json
import time
import logging
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import (
    draw_ocr_box_txt,
    get_rotate_crop_image,
    get_minarea_rect_crop,
    slice_generator,
    merge_fragmented,
    preprocess_infer,
)

logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, f"mg_crop_{bno+self.crop_image_res_index}.jpg"
                ),
                img_crop_list[bno],
            )
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True, slice={}):
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        if slice:
            slice_gen = slice_generator(
                img,
                horizontal_stride=slice["horizontal_stride"],
                vertical_stride=slice["vertical_stride"],
            )
            elapsed = []
            dt_slice_boxes = []
            for slice_crop, v_start, h_start in slice_gen:
                dt_boxes, elapse = self.text_detector(slice_crop, use_slice=True)
                if dt_boxes.size:
                    dt_boxes[:, :, 0] += h_start
                    dt_boxes[:, :, 1] += v_start
                    dt_slice_boxes.append(dt_boxes)
                    elapsed.append(elapse)
            dt_boxes = np.concatenate(dt_slice_boxes)

            dt_boxes = merge_fragmented(
                boxes=dt_boxes,
                x_threshold=slice["merge_x_thres"],
                y_threshold=slice["merge_y_thres"],
            )
            elapse = sum(elapsed)
        else:
            dt_boxes, elapse = self.text_detector(img)

        time_dict["det"] = elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict["all"] = end - start
            return None, None, time_dict
        else:
            logger.debug(
                "dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse)
            )
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)
            time_dict["cls"] = elapse
            logger.debug(
                "cls num  : {}, elapsed : {}".format(len(img_crop_list), elapse)
            )
        if len(img_crop_list) > 1000:
            logger.debug(
                f"rec crops num: {len(img_crop_list)}, time and memory cost may be large."
            )

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict["rec"] = elapse
        logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict["all"] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def main():
    config, logger = preprocess_infer()
    
    # 创建一个args对象，用于兼容现有的TextSystem接口
    class Args:
        pass
    
    args = Args()
    global_config = config["Global"]
    det_config = config.get("Det", {})
    rec_config = config.get("Rec", {})
    cls_config = config.get("Cls", {})
    
    # 全局参数
    args.image_dir = global_config.get("image_dir", "./")
    args.use_gpu = global_config.get("use_gpu", True)
    args.use_xpu = global_config.get("use_xpu", False)
    args.use_npu = global_config.get("use_npu", False)
    args.use_mlu = global_config.get("use_mlu", False)
    args.use_gcu = global_config.get("use_gcu", False)
    args.use_onnx = global_config.get("use_onnx", False)
    args.ir_optim = global_config.get("ir_optim", True)
    args.use_tensorrt = global_config.get("use_tensorrt", False)
    args.min_subgraph_size = global_config.get("min_subgraph_size", 15)
    args.precision = global_config.get("precision", "fp32")
    args.gpu_mem = global_config.get("gpu_mem", 500)
    args.gpu_id = global_config.get("gpu_id", 0)
    args.max_batch_size = global_config.get("max_batch_size", 10)
    args.benchmark = global_config.get("benchmark", False)
    args.warmup = global_config.get("warmup", False)
    args.drop_score = global_config.get("drop_score", 0.5)
    args.vis_font_path = global_config.get("vis_font_path", "./doc/fonts/simfang.ttf")
    args.draw_img_save_dir = global_config.get("draw_img_save_dir", "./inference_results/")
    args.save_crop_res = global_config.get("save_crop_res", False)
    args.crop_res_save_dir = global_config.get("crop_res_save_dir", "./output/")
    args.process_id = global_config.get("process_id", 0)
    args.total_process_num = global_config.get("total_process_num", 1)
    args.page_num = global_config.get("page_num", 0)
    args.show_log = global_config.get("show_log", True)
    
    # 检测模型参数
    args.det_model_dir = det_config.get("det_model_dir", "")
    args.det_algorithm = det_config.get("det_algorithm", "DB")
    args.det_limit_side_len = det_config.get("det_limit_side_len", 960)
    args.det_limit_type = det_config.get("det_limit_type", "max")
    args.det_box_type = det_config.get("det_box_type", "quad")
    args.det_db_thresh = det_config.get("det_db_thresh", 0.3)
    args.det_db_box_thresh = det_config.get("det_db_box_thresh", 0.6)
    args.det_db_unclip_ratio = det_config.get("det_db_unclip_ratio", 1.5)
    args.use_dilation = det_config.get("use_dilation", False)
    args.det_db_score_mode = det_config.get("det_db_score_mode", "fast")
    # EAST算法参数
    args.det_east_score_thresh = det_config.get("det_east_score_thresh", 0.8)
    args.det_east_cover_thresh = det_config.get("det_east_cover_thresh", 0.1)
    args.det_east_nms_thresh = det_config.get("det_east_nms_thresh", 0.2)
    # SAST算法参数
    args.det_sast_score_thresh = det_config.get("det_sast_score_thresh", 0.5)
    args.det_sast_nms_thresh = det_config.get("det_sast_nms_thresh", 0.2)
    # PSE算法参数
    args.det_pse_thresh = det_config.get("det_pse_thresh", 0)
    args.det_pse_box_thresh = det_config.get("det_pse_box_thresh", 0.85)
    args.det_pse_min_area = det_config.get("det_pse_min_area", 16)
    args.det_pse_scale = det_config.get("det_pse_scale", 1)
    # FCE算法参数
    args.scales = det_config.get("scales", [8, 16, 32])
    args.alpha = det_config.get("alpha", 1.0)
    args.beta = det_config.get("beta", 1.0)
    args.fourier_degree = det_config.get("fourier_degree", 5)
    
    # 识别模型参数
    args.rec_model_dir = rec_config.get("rec_model_dir", "")
    args.rec_algorithm = rec_config.get("rec_algorithm", "SVTR_LCNet")
    args.rec_image_shape = rec_config.get("rec_image_shape", "3, 48, 320")
    args.rec_batch_num = rec_config.get("rec_batch_num", 6)
    args.rec_char_dict_path = rec_config.get("rec_char_dict_path", "./ppocr/utils/ppocr_keys_v1.txt")
    args.use_space_char = rec_config.get("use_space_char", True)
    args.rec_image_inverse = rec_config.get("rec_image_inverse", True)
    args.max_text_length = rec_config.get("max_text_length", 25)
    
    # 角度分类器参数
    args.use_angle_cls = cls_config.get("use_angle_cls", False)
    args.cls_model_dir = cls_config.get("cls_model_dir", "")
    args.cls_image_shape = cls_config.get("cls_image_shape", "3, 48, 192")
    args.cls_batch_num = cls_config.get("cls_batch_num", 6)
    args.cls_thresh = cls_config.get("cls_thresh", 0.9)
    
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id :: args.total_process_num]
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    logger.info(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
    )

    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0
    for idx, image_file in enumerate(image_file_list):
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                logger.debug("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        for index, img in enumerate(imgs):
            starttime = time.time()
            dt_boxes, rec_res, time_dict = text_sys(img)
            elapse = time.time() - starttime
            total_time += elapse
            if len(imgs) > 1:
                logger.debug(
                    str(idx)
                    + "_"
                    + str(index)
                    + "  Predict time of %s: %.3fs" % (image_file, elapse)
                )
            else:
                logger.debug(
                    str(idx) + "  Predict time of %s: %.3fs" % (image_file, elapse)
                )
            for text, score in rec_res:
                logger.debug("{}, {:.3f}".format(text, score))

            res = [
                {
                    "transcription": rec_res[i][0],
                    "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                }
                for i in range(len(dt_boxes))
            ]
            if len(imgs) > 1:
                save_pred = (
                    os.path.basename(image_file)
                    + "_"
                    + str(index)
                    + "\t"
                    + json.dumps(res, ensure_ascii=False)
                    + "\n"
                )
            else:
                save_pred = (
                    os.path.basename(image_file)
                    + "\t"
                    + json.dumps(res, ensure_ascii=False)
                    + "\n"
                )
            save_results.append(save_pred)

            if is_visualize:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=drop_score,
                    font_path=font_path,
                )
                if flag_gif:
                    save_file = image_file[:-3] + "png"
                elif flag_pdf:
                    save_file = image_file.replace(".pdf", "_" + str(index) + ".png")
                else:
                    save_file = image_file
                cv2.imwrite(
                    os.path.join(draw_img_save_dir, os.path.basename(save_file)),
                    draw_img[:, :, ::-1],
                )
                logger.debug(
                    "The visualized image saved in {}".format(
                        os.path.join(draw_img_save_dir, os.path.basename(save_file))
                    )
                )

    logger.info("The predict total time is {}".format(time.time() - _st))
    if args.benchmark:
        text_sys.text_detector.autolog.report()
        text_sys.text_recognizer.autolog.report()

    with open(
        os.path.join(draw_img_save_dir, "system_results.txt"), "w", encoding="utf-8"
    ) as f:
        f.writelines(save_results)


if __name__ == "__main__":
    main()
