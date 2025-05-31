import random
import torch
import torch.utils.data as data
from PIL import Image
import cv2
import numpy as np
import os
import json
from typing import Tuple
from pycocotools import mask as maskUtils

from lib.utils.imgproc import center_crop, mask_smooth, synthesize_data, synthesize_data_multi
from lib.dataset.COCO import COCO

class SA1B(COCO):
    def __init__(self, root_dirs, template_size=128, template_scale=2, search_size=256, search_scale=4, search_scale_std=2, max_object_num=5, dual_template=False, template_mask=False, argumentation=True):
        """
        Args:
            root_dirs (list[string]): Directorys with all the images and annotations. The list should be like:
                root_dirs = [root_dir_1, root_dir_2, ...]
                The structure of each directory should be like:
                    root_dir_1
                    ├── ...
                    ├── sa_1946.jpg
                    ├── sa_1946.json
                    ├── sa_1947.jpg
                    ├── sa_1947.json
                    └── ...
                The json structure should be like:
                    {
                        "image": {
                            "file_name": "sa_1946.jpg",
                            "image_id": 1946
                            "width": 1920,
                            "height": 1080
                        },
                        "annotations": [
                            {
                                "bbox": [x, y, w, h],
                                "id": 1,
                                "area": 123.0,
                                "segmentation": {
                                    "counts": "counts",
                                    "size": [h, w]
                                }
                            },
                            {
                                "bbox": [x, y, w, h],
                                "id": 2,
                                "area": 456.0,
                                "segmentation": {
                                    "counts": "counts",
                                    "size": [h, w]
                                }
                            }
                        ]
                    }
            template_size (int): Size of the template image.
            template_scale (int): Scale of the template image.
            search_size (int): Size of the search image.
            search_scale (int): Scale of the search image.
            search_scale_std (int): Standard deviation of the search scale.
            max_object_num (int): Maximum number of objects.
            dual_template (bool): Whether to use dual template.
            template_mask (bool): Whether to return the mask of the template image.
            argumentation (bool): Whether to augment the data.
        """
        super(COCO, self).__init__(template_size, template_scale, search_size, search_scale, search_scale_std, max_object_num, dual_template, template_mask, argumentation)

        self.root_dirs = root_dirs

        self.build_dataset_idx()

    def build_dataset_idx(self):
        self.images = {}
        self.objects = {}

        for root_dir in self.root_dirs:
            files = os.listdir(root_dir)
            images = [file for file in files if file.endswith('.jpg')]
            annos = [file for file in files if file.endswith('.json')]
            for anno in annos:
                with open(os.path.join(root_dir, anno), "r") as f:
                    anno_data = json.load(f)
                image_id = anno_data['image']['image_id']
                self.images[image_id] = {
                    'file_name': anno_data['image']['file_name'],
                    'width': anno_data['image']['width'],
                    'height': anno_data['image']['height'],
                    'folder': root_dir,
                    'objects': []
                }
                for object in anno_data['annotations']:
                    if self.filter(object, image_id):
                        self.images[image_id]['objects'].append(object['id'])
                        self.objects[object['id']] = {
                            'bbox': object['bbox'],
                            'segmentation': object['segmentation'],
                            'image_id': image_id
                        }

    def filter(self, object, image_id):
        # Filter the object by conditions
        h, w = object['bbox'][3], object['bbox'][2]
        H, W = self.images[image_id]['height'], self.images[image_id]['width']
        if w > W // 2 or h > H // 2 or w / h < 0.25 or w / h > 4 or w < 20 or h < 20:
            return False
        if object['area'] / (w * h) < 0.5:
            return False
        return True

    def read_image(self, image_id):
        # Read image
        img_path = os.path.join(self.images[image_id]['folder'], self.images[image_id]['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        return img

    def rle_decode(self, mask_annos, h, w):
        # Decode the rle mask, no need to use h and w
        mask_annos['counts'] = mask_annos['counts'].encode('utf-8')
        mask = maskUtils.decode(mask_annos)
        return mask
        