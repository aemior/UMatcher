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
from lib.dataset.StandardDataset import StandardDataset

class COCO(StandardDataset):
    def __init__(self, root_dir, anno_file, template_size=128, template_scale=2, search_size=256, search_scale=4, search_scale_std=2, max_object_num=5, dual_template=False, template_mask=False, argumentation=True):
        """
        Args:
            root_dir (string): Directory with all the images. The directory structure should be like:
                root_dir
                ├── ...
                ├── 000000079188.jpg
                ├── 000000079189.jpg
                ├── 000000079190.jpg
                └── ...
            anno_file (string): Json annotation file in COCO format. The structure should be like:
                {
                    "images": [
                        {
                            "file_name": "000000079188.jpg",
                            "id": 79188
                        },
                        {
                            "file_name": "000000079189.jpg",
                            "id": 79189
                        },
                        ...
                    ],
                    "annotations": [
                        {
                            "image_id": 79188,
                            "bbox": [x, y, w, h],
                            "id": 1,
                            "area": 123.0,
                            "segmentation": {
                                "counts": "counts",
                                "size": [h, w]
                            }
                        },
                        {
                            "image_id": 79188,
                            "bbox": [x, y, w, h],
                            "id": 2,
                            "area": 456.0,
                            "segmentation": [[
                                p1_x,
                                p1_y,
                                ...
                            ]]
                        },
                        ...
                    ]
                } 
            template_size (int): Size of the template image.
            template_scale (int): Scale of the template image.
            search_size (int): Size of the search image.
            search_scale (int): Scale of the search image.
            search_scale_std (int): Standard deviation of the search scale.
            max_object_num (int): Maximum number of objects.
            dual_template (bool): Whether to use dual template.
            template_mask (bool): Whether to use template mask.
            argumentation (bool): Whether to augment the data.
        """
        super(COCO, self).__init__(template_size, template_scale, search_size, search_scale, search_scale_std, max_object_num, dual_template, template_mask, argumentation)

        self.root_dir = root_dir
        self.anno_file = anno_file

        self.build_dataset_idx()

    def filter(self, object):
        # Filter the object by conditions
        h, w = object['bbox'][3], object['bbox'][2]
        if h==0 or w==0:
            return False
        H, W = self.images[object['image_id']]['height'], self.images[object['image_id']]['width']
        if w > W // 2 or h > H // 2 or w / h < 0.25 or w / h > 4 or w < 20 or h < 20:
            return False
        if object['area'] / (w * h) < 0.5:
            return False
        return True

    def build_dataset_idx(self):
        self.images = {}
        self.objects = {}
        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        for img in anno['images']:
            self.images[img['id']] = {
                'file_name': img['file_name'],
                'height': img['height'],
                'width': img['width'],
                'objects': []
            }

        for object in anno['annotations']:
            if not self.filter(object):
                continue
            self.images[object['image_id']]['objects'].append(object['id'])
            self.objects[object['id']] = {
                'bbox': object['bbox'],
                'segmentation': object['segmentation'],
                'image_id': object['image_id']
            }

    def rle_decode(self, contours, h, w):
        # Decode the rle mask
        rle = maskUtils.frPyObjects(contours, h, w)
        mask = maskUtils.decode(rle)
        return mask

    def contours_to_mask(self, object_id):
        # Convert segmentation contours to mask
        h, w = self.images[self.objects[object_id]['image_id']]['height'], self.images[self.objects[object_id]['image_id']]['width']
        contours = self.objects[object_id]['segmentation']
        mask = np.zeros((h, w), dtype=np.uint8)
        # Convert contours to opencv format
        if "counts" in contours:
            return self.rle_decode(contours, h, w)
        contours = [np.array(c).reshape(-1, 1, 2).astype(np.int32) for c in contours]
        mask = cv2.fillPoly(mask, contours, 1) # type: ignore
        return mask

    def read_image(self, image_id):
        # Read image
        img_path = os.path.join(self.root_dir, self.images[image_id]['file_name'])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        return img

    def merge_mask(self, rgb_img, mask):
        # convert the rgb image to rgba using the mask
        rgba_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=np.uint8)
        rgba_img[:, :, :3] = rgb_img
        rgba_img[:, :, 3] = mask * 255
        return rgba_img


    def __len__(self):
        return len(self.objects)

    def get_template(self, foreground_crop, bg_id):
        img_template, trans_mask, bbox_template = synthesize_data(
            foreground_crop,
            self.read_image(bg_id),
        )
        template_img = cv2.resize(
            center_crop(
                img_template,
                self.tlbr2chw(bbox_template),
                self.template_scale
            ), # type: ignore
            (self.template_size, self.template_size)
        ) # type: ignore
        mask_template = cv2.resize(
            center_crop(
                trans_mask,
                self.tlbr2chw(bbox_template),
                self.template_scale
            ),
            (self.template_size, self.template_size)
        )
        return template_img, mask_template

    def convert2tensor(self, img:np.ndarray) -> torch.tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32))/255.0

    def __getitem__(self, idx) -> dict:
        """
        Args:
            idx (int): Index of the sequence, which stands for the idx-th object in the dataset.
        Returns:
            A dictionary containing the following items:
                - template_img (torch.Tensor): the template image. shape: (3, template_size, template_size)
                - search_img (torch.Tensor): the search image. shape: (3, search_size, search_size)
                - ground_bbox (torch.Tensor): the ground truth bounding box. shape: (max_object_num, 4), format: [cx, cy, w, h] normalized by search_size
                - box_num (torch.Tensor): the number of boxes. shape: (1)
                - template_mask (torch.Tensor): the mask of the template image. shape: (1, search_size, search_size)
        """
        object_id = list(self.objects.keys())[idx]
        image_id = self.objects[object_id]['image_id']

        mask = self.contours_to_mask(object_id)
        foreground = self.merge_mask(self.read_image(image_id), mask)
        fg_bbox = self.objects[object_id]['bbox']
        x, y, w, h = map(int, fg_bbox)
        foreground_crop = foreground[y:y+h, x:x+w]
        foreground_crop[:, :, 3] = mask_smooth(foreground_crop[:, :, 3])
        sample_list = [img_id for img_id in self.images.keys() if img_id != image_id]
        template_bg_id, search_bg_id = random.sample(sample_list, 2)

        img_search, bboxs_search = synthesize_data_multi(
            self.read_image(search_bg_id),
            foreground_crop,
            self.max_object_num, self.search_size, self.search_scale, self.search_scale_std
        )

        template_img_a, mask_template_a = self.get_template(foreground_crop, template_bg_id)
        template_img_a = self.convert2tensor(template_img_a)
        if self.template_mask:
            template_mask_a = torch.tensor(mask_template_a).unsqueeze(0).float()

        if self.dual_template:
            template_img_b, mask_template_b = self.get_template(foreground_crop, template_bg_id)
            template_img_b = self.convert2tensor(template_img_b)
            if self.template_mask:
                template_mask_b = torch.tensor(mask_template_b).unsqueeze(0).float()

        search_img = cv2.resize(img_search, (self.search_size, self.search_size))
        search_img = self.convert2tensor(search_img)

        ground_bbox = np.zeros((self.max_object_num, 4), dtype=np.float32)
        for i, bbox in enumerate(bboxs_search):
            if i >= self.max_object_num:
                break
            ground_bbox[i] = self.tlbr2chw(bbox)
        ground_bbox /= self.search_size
        ground_bbox = torch.from_numpy(np.array(ground_bbox).astype(np.float32))
        box_num = torch.tensor([len(bboxs_search)], dtype=torch.int64)


        if self.dual_template:
            res =  {
                'template_img_a': template_img_a,
                'template_img_b': template_img_b,
                'search_img': search_img,
                'ground_bbox': ground_bbox,
                'box_num': box_num
            }
            if self.template_mask:
                res['template_mask_a'] = template_mask_a
                res['template_mask_b'] = template_mask_b
        else:
            res =  {
                'template_img': template_img_a,
                'search_img': search_img,
                'ground_bbox': ground_bbox,
                'box_num': box_num
            }
            if self.template_mask:
                res['template_mask'] = template_mask_a
        return res
            