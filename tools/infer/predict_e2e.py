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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import numpy as np
import time
import sys

import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from tools.infer.utility import preprocess_infer

logger = get_logger()


class TextE2E(object):
    def __init__(self, args):
        if os.path.exists(f"{args.e2e_model_dir}/inference.yml"):
            model_config = utility.load_config(f"{args.e2e_model_dir}/inference.yml")
            model_name = model_config.get("Global", {}).get("model_name", "")
            if model_name:
                raise ValueError(
                    f"{model_name} is not supported. Please check if the model is supported by the PaddleOCR wheel."
                )

        self.args = args
        self.e2e_algorithm = args.e2e_algorithm
        self.use_onnx = args.use_onnx
        pre_process_list = [
            {"E2EResizeForTest": {}},
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        postprocess_params = {}
        if self.e2e_algorithm == "PGNet":
            pre_process_list[0] = {
                "E2EResizeForTest": {
                    "max_side_len": args.e2e_limit_side_len,
                    "valid_set": "totaltext",
                }
            }
            postprocess_params["name"] = "PGPostProcess"
            postprocess_params["score_thresh"] = args.e2e_pgnet_score_thresh
            postprocess_params["character_dict_path"] = args.e2e_char_dict_path
            postprocess_params["valid_set"] = args.e2e_pgnet_valid_set
            postprocess_params["mode"] = args.e2e_pgnet_mode
        else:
            logger.info("unknown e2e_algorithm:{}".format(self.e2e_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            _,
        ) = utility.create_predictor(
            args, "e2e", logger
        )  # paddle.jit.load(args.det_model_dir)
        # self.predictor.eval()

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {"image": img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()

        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)
            preds = {}
            preds["f_border"] = outputs[0]
            preds["f_char"] = outputs[1]
            preds["f_direction"] = outputs[2]
            preds["f_score"] = outputs[3]
        else:
            self.input_tensor.copy_from_cpu(img)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)

            preds = {}
            if self.e2e_algorithm == "PGNet":
                preds["f_border"] = outputs[0]
                preds["f_char"] = outputs[1]
                preds["f_direction"] = outputs[2]
                preds["f_score"] = outputs[3]
            else:
                raise NotImplementedError
        post_result = self.postprocess_op(preds, shape_list)
        points, strs = post_result["points"], post_result["texts"]
        dt_boxes = self.filter_tag_det_res_only_clip(points, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, strs, elapse


def main():
    config, logger = preprocess_infer()
    
    # 创建一个args对象，用于兼容现有的TextE2E接口
    class Args:
        pass
    
    args = Args()
    global_config = config["Global"]
    e2e_config = config.get("E2E", {})
    
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
    args.benchmark = global_config.get("benchmark", False)
    
    # E2E参数 - 从E2E配置段读取
    args.e2e_model_dir = e2e_config.get("e2e_model_dir", "")
    args.e2e_algorithm = e2e_config.get("e2e_algorithm", "PGNet")
    args.e2e_limit_side_len = e2e_config.get("e2e_limit_side_len", 768)
    args.e2e_limit_type = e2e_config.get("e2e_limit_type", "max")
    args.e2e_char_dict_path = e2e_config.get("e2e_char_dict_path", "./ppocr/utils/ic15_dict.txt")
    args.use_space_char = global_config.get("use_space_char", True)  # 通常在Global中设置
    
    # PGNet特定参数
    args.e2e_pgnet_score_thresh = e2e_config.get("e2e_pgnet_score_thresh", 0.5)
    args.e2e_pgnet_mode = e2e_config.get("e2e_pgnet_mode", "fast")
    args.e2e_pgnet_polygon = e2e_config.get("e2e_pgnet_polygon", True)
    args.e2e_pgnet_valid_set = e2e_config.get("e2e_pgnet_valid_set", "totaltext")
    
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextE2E(args)
    count = 0
    total_time = 0
    draw_img_save = global_config.get("draw_img_save_dir", "./inference_results")
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        points, strs, elapse = text_detector(img)
        if count > 0:
            total_time += elapse
        count += 1
        logger.info("Predict time of {}: {}".format(image_file, elapse))
        src_im = utility.draw_e2e_res(points, strs, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save, "e2e_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
        logger.info("The visualized image saved in {}".format(img_path))
    if count > 1:
        logger.info("Avg Time: {}".format(total_time / (count - 1)))


if __name__ == "__main__":
    main()
