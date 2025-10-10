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

import logging
import os
import cv2
import random
import numpy as np
import paddle
import importlib.util
import sys
import subprocess


def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))


def get_check_global_params(mode):
    check_params = [
        "use_gpu",
        "max_text_length",
        "image_shape",
        "image_shape",
        "character_type",
        "loss_type",
    ]
    if mode == "train_eval":
        check_params = check_params + [
            "train_batch_size_per_card",
            "test_batch_size_per_card",
        ]
    elif mode == "test":
        check_params = check_params + ["test_batch_size_per_card"]
    return check_params


def _check_image_file(path):
    img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file, infer_list=None):
    imgs_lists = []
    if infer_list and not os.path.exists(infer_list):
        raise Exception("not found infer list {}".format(infer_list))
    if infer_list:
        with open(infer_list, "r") as f:
            lines = f.readlines()
        for line in lines:
            image_path = line.strip().split("\t")[0]
            image_path = os.path.join(img_file, image_path)
            imgs_lists.append(image_path)
    else:
        if img_file is None or not os.path.exists(img_file):
            raise Exception("not found any img file in {}".format(img_file))

        img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
        if os.path.isfile(img_file) and _check_image_file(img_file):
            imgs_lists.append(img_file)
        elif os.path.isdir(img_file):
            for single_file in os.listdir(img_file):
                file_path = os.path.join(img_file, single_file)
                if os.path.isfile(file_path) and _check_image_file(file_path):
                    imgs_lists.append(file_path)

    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def get_image_file_list_from_configs(data_dir_list, label_file_list):
    all_imgs_lists = []
    
    # Handle the case where infer_list has only one element but infer_img has multiple
    if len(label_file_list) == 1 and len(data_dir_list) > 1:
        # Use the single label file with all data directories
        label_file = label_file_list[0]
        if not os.path.exists(label_file):
            raise Exception("not found label file {}".format(label_file))

        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            image_paths_str = line.strip().split("\t")[0]
            
            # Handle cases where image_paths_str is a list-like string
            if image_paths_str.startswith("[") and image_paths_str.endswith("]"):
                try:
                    # Safely evaluate the string as a Python list
                    image_relative_paths = eval(image_paths_str)
                    if isinstance(image_relative_paths, list) and len(image_relative_paths) > 0:
                        image_relative_path = image_relative_paths[0] # Take the first one as per user's instruction
                    else:
                        print(f"Warning: Could not parse image path list from: {image_paths_str}")
                        continue
                except Exception as e:
                    print(f"Warning: Error evaluating image path string '{image_paths_str}': {e}")
                    image_relative_path = image_paths_str # Fallback to original string if parsing fails
            else:
                image_relative_path = image_paths_str

            # Try each data directory to find the image
            found = False
            for base_dir in data_dir_list:
                image_path = os.path.join(base_dir, image_relative_path)
                if os.path.isfile(image_path) and _check_image_file(image_path):
                    all_imgs_lists.append(image_path)
                    found = True
                    break
            
            if not found:
                # If not found with any base_dir, try with image_relative_path as absolute path
                if os.path.isfile(image_relative_path) and _check_image_file(image_relative_path):
                    all_imgs_lists.append(image_relative_path)
                else:
                    # For debugging, print the components before joining
                    print(f"Debug: data_dir_list='{data_dir_list}', image_relative_path='{image_relative_path}'")
                    print(f"Warning: Image file not found in any data directory or as absolute path: {image_relative_path}")
    else:
        # Original logic for when lengths match or other cases
        for i, label_file in enumerate(label_file_list):
            if not os.path.exists(label_file):
                raise Exception("not found label file {}".format(label_file))

            with open(label_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                image_paths_str = line.strip().split("\t")[0]
                
                # Handle cases where image_paths_str is a list-like string
                if image_paths_str.startswith("[") and image_paths_str.endswith("]"):
                    try:
                        # Safely evaluate the string as a Python list
                        image_relative_paths = eval(image_paths_str)
                        if isinstance(image_relative_paths, list) and len(image_relative_paths) > 0:
                            image_relative_path = image_relative_paths[0] # Take the first one as per user's instruction
                        else:
                            print(f"Warning: Could not parse image path list from: {image_paths_str}")
                            continue
                    except Exception as e:
                        print(f"Warning: Error evaluating image path string '{image_paths_str}': {e}")
                        image_relative_path = image_paths_str # Fallback to original string if parsing fails
                else:
                    image_relative_path = image_paths_str

                # Determine the base directory for the image
                base_dir = ""
                if len(data_dir_list) == 1:
                    base_dir = data_dir_list[0]
                elif len(data_dir_list) > i:
                    base_dir = data_dir_list[i]
                else:
                    raise Exception(
                        f"data_dir_list and label_file_list length mismatch or data_dir_list is empty. "
                        f"data_dir_list length: {len(data_dir_list)}, current label_file index: {i}"
                    )

                # The image_relative_path should be relative to the base_dir
                # No need to remove 't/' prefix if base_dir is correctly determined
                
                image_path = os.path.join(base_dir, image_relative_path)
                if os.path.isfile(image_path) and _check_image_file(image_path):
                    all_imgs_lists.append(image_path)
                else:
                    # If not found with base_dir, try with image_relative_path as absolute path
                    # This handles cases where image_relative_path might already be an absolute path
                    if os.path.isfile(image_relative_path) and _check_image_file(image_relative_path):
                        all_imgs_lists.append(image_relative_path)
                    else:
                        # For debugging, print the components before joining
                        print(f"Debug: base_dir='{base_dir}', image_relative_path='{image_relative_path}'")
                        print(f"Warning: Image file not found: {image_path} (from base_dir) or {image_relative_path} (as absolute path)")

    if len(all_imgs_lists) == 0:
        raise Exception("not found any img file from the provided configs")
    all_imgs_lists = sorted(list(set(all_imgs_lists))) # Use set to remove duplicates, then sort
    return all_imgs_lists


def binarize_img(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # conversion to grayscale image
        # use cv2 threshold binarization
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img


def alpha_to_color(img, alpha_color=(255, 255, 255)):
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img


def check_and_read(img_path):
    if os.path.basename(img_path)[-3:].lower() == "gif":
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            logger = logging.getLogger("ppocr")
            logger.info("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    elif os.path.basename(img_path)[-3:].lower() == "pdf":
        from paddle.utils import try_import

        fitz = try_import("fitz")
        from PIL import Image

        imgs = []
        with fitz.open(img_path) as pdf:
            for pg in range(0, pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
            return imgs, False, True
    return None, False, False


def load_vqa_bio_label_maps(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    old_lines = [line.strip() for line in lines]
    lines = ["O"]
    for line in old_lines:
        # "O" has already been in lines
        if line.upper() in ["OTHER", "OTHERS", "IGNORE"]:
            continue
        lines.append(line)
    labels = ["O"]
    for line in lines[1:]:
        labels.append("B-" + line)
        labels.append("I-" + line)
    label2id_map = {label.upper(): idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label.upper() for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


def set_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def check_install(module_name, install_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"Warning! The {module_name} module is NOT installed")
        print(
            f"Try install {module_name} module automatically. You can also try to install manually by pip install {install_name}."
        )
        python = sys.executable
        try:
            subprocess.check_call(
                [python, "-m", "pip", "install", install_name],
                stdout=subprocess.DEVNULL,
            )
            print(f"The {module_name} module is now installed")
        except subprocess.CalledProcessError as exc:
            raise Exception(f"Install {module_name} failed, please install manually")
    else:
        print(f"{module_name} has been installed.")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """reset"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
