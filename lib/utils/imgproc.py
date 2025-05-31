import math
import torchvision
import torch
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes, make_grid 
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import random

import cv2
import numpy as np

from typing import Union, Tuple, List

def center_crop(img, bbox, scale, return_roi=False) -> np.ndarray:
    """
    Crop the image around the center of the bounding box.
    Args:
        img: numpy array of shape (H, W, C)
        bbox: list of integers [cx, cy, w, h]
        scale: float
        return_roi: bool
    Returns:
        crop_img: numpy array of shape (size, size, C)
    """
    cx, cy, w, h = bbox
    size = int(np.sqrt(w * h) * scale)
    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = x1 + size
    y2 = y1 + size

    x1_pad = max(0, -x1)
    y1_pad = max(0, -y1)
    x2_pad = max(x2 - img.shape[1], 0)
    y2_pad = max(y2 - img.shape[0], 0)

    roi_x1 = max(x1, 0)
    roi_y1 = max(y1, 0)
    roi_x2 = min(x2, img.shape[1])
    roi_y2 = min(y2, img.shape[0])

    roi = img[int(roi_y1):int(roi_y2), int(roi_x1):int(roi_x2)]

    if img.ndim == 2:  # Single-channel image
        crop_img = cv2.copyMakeBorder(
            #roi, int(y1_pad), int(y2_pad), int(x1_pad), int(x2_pad), cv2.BORDER_CONSTANT, value=[0]
            roi, int(y1_pad), int(y2_pad), int(x1_pad), int(x2_pad), cv2.BORDER_REPLICATE
        )
    else:  # Multi-channel image
        crop_img = cv2.copyMakeBorder(
            roi, int(y1_pad), int(y2_pad), int(x1_pad), int(x2_pad), cv2.BORDER_REPLICATE
        )

    if return_roi:
        return crop_img, [roi_x1, roi_y1, roi_x2, roi_y2]
    return crop_img



def noisy_crop(img, bbox, scale_mean, scale_std, shift_limit):
    """
    Crop the image around the bounding box with random scale and shift.
    Args:
        img: numpy array of shape (H, W, C)
        bbox: bbox in src image, list of integers [cx, cy, w, h]
        scale_mean: float
        scale_std: float
        shift_limit: float
    Returns:
        crop_img: numpy array of shape (size, size, C)
        normalized_bbox: bbox in crop image, list of floats [cx, cy, w, h]
    """
    cx, cy, w, h = bbox

    # Step 1: Determine the crop size with a normal distribution
    crop_sz = int(np.random.normal(scale_mean, scale_std) * math.sqrt(w * h))
    crop_sz = max(3, crop_sz)  # Ensure crop size is at least 3 (The target is 1 pix at least, and a bbox around it) to avoid errors

    # Step 2: Determine the shift with a uniform distribution
    shift_x = np.random.uniform(-shift_limit, shift_limit) * crop_sz
    shift_y = np.random.uniform(-shift_limit, shift_limit) * crop_sz

    # Step 3: Calculate the crop region
    x1 = int(cx - crop_sz / 2 + shift_x)
    y1 = int(cy - crop_sz / 2 + shift_y)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    # Step 4: Ensure the crop region is within image bounds
    x1_pad = max(0, -x1)
    y1_pad = max(0, -y1)
    x2_pad = max(x2 - img.shape[1], 0)
    y2_pad = max(y2 - img.shape[0], 0)

    roi_x1 = max(x1, 0)
    roi_y1 = max(y1, 0)
    roi_x2 = min(x2, img.shape[1])
    roi_y2 = min(y2, img.shape[0])

    roi = img[int(roi_y1):int(roi_y2), int(roi_x1):int(roi_x2)]

    # Step 5: Perform the cropping and padding
    crop_img = cv2.copyMakeBorder(
        roi, int(y1_pad), int(y2_pad), int(x1_pad), int(x2_pad), cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Step 6: Calculate and normalize the bounding box
    new_cx = (cx - roi_x1 + x1_pad) / crop_sz
    new_cy = (cy - roi_y1 + y1_pad) / crop_sz
    new_w = min(w / crop_sz, 1.0)
    new_h = min(h / crop_sz, 1.0)
    normalized_bbox = [new_cx, new_cy, new_w, new_h]

    return crop_img, normalized_bbox

def noisy_crop(img, bbox, scale_mean, scale_std):
    """
    Crop the image around the bounding box with random scale and shift.
    Args:
        img: numpy array of shape (H, W, C)
        bbox: bbox in src image, list of integers [cx, cy, w, h]
        scale_mean: float
        scale_std: float
    Returns:
        crop_img: numpy array of shape (size, size, C)
        normalized_bbox: bbox in crop image, list of floats [cx, cy, w, h]
    """
    cx, cy, w, h = bbox

    # Step 1: Determine the crop size with a normal distribution
    crop_sz = int(np.random.normal(scale_mean, scale_std) * math.sqrt(w * h))
    crop_sz = max(min(w,h), crop_sz)  # Ensure crop size is at least min(w,h) to avoid errors
    crop_sz = min(int(max(img.shape[1],img.shape[0]) * scale_mean), crop_sz)  # Ensure crop size not too large to avoid errors

    # Step 2: Calculate the maximum allowable shift so that the target center stays within the crop bounds
    max_shift_x = int((crop_sz / 2) * 0.9)
    max_shift_y = int((crop_sz / 2) * 0.9)

    # Step 3: Randomly shift the bounding box within allowable limits
    shift_x = np.random.uniform(-max_shift_x, max_shift_x)
    shift_y = np.random.uniform(-max_shift_y, max_shift_y)


    # Step 4: Calculate the crop region
    x1 = int(cx - crop_sz / 2 + shift_x)
    y1 = int(cy - crop_sz / 2 + shift_y)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    # Step 5: Ensure the crop region is within image bounds
    x1_pad = max(0, -x1)
    y1_pad = max(0, -y1)
    x2_pad = max(x2 - img.shape[1], 0)
    y2_pad = max(y2 - img.shape[0], 0)

    roi_x1 = max(x1, 0)
    roi_y1 = max(y1, 0)
    roi_x2 = min(x2, img.shape[1])
    roi_y2 = min(y2, img.shape[0])

    roi = img[int(roi_y1):int(roi_y2), int(roi_x1):int(roi_x2)]

    # Step 6: Perform the cropping and padding
    crop_img = cv2.copyMakeBorder(
        roi, int(y1_pad), int(y2_pad), int(x1_pad), int(x2_pad), cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Step 7: Adjust the bounding box to the crop coordinates
    # Original bounding box in terms of corner points
    original_x1 = cx - w / 2
    original_y1 = cy - h / 2
    original_x2 = cx + w / 2
    original_y2 = cy + h / 2

    # Adjust bounding box corners to fit within the crop
    new_x1 = max(original_x1 - roi_x1 + x1_pad, 0)
    new_y1 = max(original_y1 - roi_y1 + y1_pad, 0)
    new_x2 = min(original_x2 - roi_x1 + x1_pad, crop_sz)
    new_y2 = min(original_y2 - roi_y1 + y1_pad, crop_sz)

    # Recalculate new bounding box in terms of cx, cy, w, h
    new_cx = (new_x1 + new_x2) / 2 / crop_sz
    new_cy = (new_y1 + new_y2) / 2 / crop_sz
    new_w = (new_x2 - new_x1) / crop_sz
    new_h = (new_y2 - new_y1) / crop_sz

    # Clamp to ensure values stay within [0, 1]
    new_cx = np.clip(new_cx, 0, 1)
    new_cy = np.clip(new_cy, 0, 1)
    new_w = np.clip(new_w, 0, 1)
    new_h = np.clip(new_h, 0, 1)

    normalized_bbox = [new_cx, new_cy, new_w, new_h]

    return crop_img, normalized_bbox



def is_normal_bbox(bbox):
    """
    Check if the bounding box is normal.
    Args:
        bbox: list of integers [cx, cy, w, h]
    Returns:
        bool
    """
    cx, cy, w, h = bbox
    return cx - w/2 >= 0 and cy - h/2 >= 0 and cx + w/2 <= 1 and cy + h/2 <= 1

def crop_no_absence(img, bbox, scale):
    """
    Crop the image without the bounding box with fixed scale.
    Args:
        img: numpy array of shape (H, W, C)
        bbox: bbox in src image, list of integers [cx, cy, w, h]
        scale: float
    Returns:
        crop_img: numpy array of shape (size, size, C)
    """
    cx, cy, w, h = bbox

    w = max(w, 3)
    h = max(h, 3)
    # Step 1: Calculate the crop size
    crop_sz = int(math.sqrt(w * h) * scale)
    crop_sz = max(min(w,h), crop_sz)  # Ensure crop size is at least min(w,h) to avoid errors
    crop_sz = min(int(max(img.shape[1],img.shape[0]) * scale), crop_sz)  # Ensure crop size not too large to avoid errors

    # Step 2: Calculate the shift to remove the bounding box
    min_shift = int((crop_sz / 2))
    shift_x = np.random.choice([-min_shift, min_shift, 0])
    if shift_x == 0:
        shift_y = np.random.choice([-min_shift, min_shift])
    else:
        shift_y = np.random.choice([-min_shift, min_shift, 0])

    # Step 3: Calculate the crop region
    x1 = int(cx - crop_sz / 2 + shift_x)
    y1 = int(cy - crop_sz / 2 + shift_y)
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    # Step 4: Ensure the crop region is within image bounds
    x1_pad = max(0, -x1)
    y1_pad = max(0, -y1)
    x2_pad = max(x2 - img.shape[1], 0)
    y2_pad = max(y2 - img.shape[0], 0)

    roi_x1 = max(x1, 0)
    roi_y1 = max(y1, 0)
    roi_x2 = min(x2, img.shape[1])
    roi_y2 = min(y2, img.shape[0])

    if (roi_x2 - roi_x1) <= 1 or (roi_y2 - roi_y1) <= 1:
        return np.zeros((crop_sz, crop_sz, img.shape[2]), dtype=img.dtype)

    roi = img[int(roi_y1):int(roi_y2), int(roi_x1):int(roi_x2)]

    # Step 5: Perform the cropping and padding
    crop_img = cv2.copyMakeBorder(
        roi, int(y1_pad), int(y2_pad), int(x1_pad), int(x2_pad), cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return crop_img



"""
def crop_no_absence(img, bbox, crop_size):
    img_h, img_w = img.shape[:2]
    cx, cy, w, h = bbox

    # Step 1: Calculate the bounding box coordinates
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Step 2: Check for feasibility
    if (crop_size > img_w or crop_size > img_h or
        (x1 <= crop_size and x2 >= img_w - crop_size and
         y1 <= crop_size and y2 >= img_h - crop_size)):
        # The crop size is too large or bbox is too large to find a valid region
        return np.zeros((crop_size, crop_size, img.shape[2]), dtype=img.dtype)

    # Step 3: Find a valid crop region
    valid_regions = []
    for _ in range(1000):  # Try 1000 times to find a valid region
        x = np.random.randint(0, img_w - crop_size + 1)
        y = np.random.randint(0, img_h - crop_size + 1)
        
        # Check if the random region overlaps with the bbox
        if (x + crop_size <= x1 or x >= x2 or
            y + crop_size <= y1 or y >= y2):
            valid_regions.append((x, y))
            break

    if not valid_regions:
        return np.zeros((crop_size, crop_size, img.shape[2]), dtype=img.dtype)

    # Choose one of the valid regions randomly
    x, y = valid_regions[np.random.randint(len(valid_regions))]
    cropped_img = img[y:y+crop_size, x:x+crop_size]

    return cropped_img
"""

def draw_bboxes_on_batch(images, bboxes, scores, ground_bbox=None, is_absence=None, score_map=None, save_path=None):
    """
    Draw bounding boxes on a batch of images and save as a single grid image.

    :param images: A torch tensor of shape (n, c, h, w)
    :param bboxes: A torch tensor of shape (n, 4)
    :param scores: A torch tensor of shape (n)
    :param gt_bbox: A torch tensor of shape (n, 4)
    :param is_absence: A torch tensor of shape (n)
    :param score_map: A torch tensor of shape (n, 1, h, w)
    :param save_path: Path to save the output grid image
    """
    n, c, h, w = images.shape
    images_with_bboxes = []

    for i in range(n):
        image = images[i]
        bbox = bboxes[i].unsqueeze(0)  # (1, 4)
        score = scores[i].item()
        
        # Convert the normalized bbox coordinates to absolute pixel coordinates
        bbox[0, 0] *= w  # x-center
        bbox[0, 1] *= h  # y-center
        bbox[0, 2] *= w  # width
        bbox[0, 3] *= h  # height
        
        # Convert center coordinates to top-left coordinates
        bbox[0, 0] -= bbox[0, 2] / 2  # x1
        bbox[0, 1] -= bbox[0, 3] / 2  # y1
        bbox[0, 2] = bbox[0, 0] + bbox[0, 2]  # x2
        bbox[0, 3] = bbox[0, 1] + bbox[0, 3]  # y2
        
        # Draw bounding box on the image
        label = f'{score:.2f}'

        # If score_map is provided, overlay it onto the image
        if score_map is not None:
            # Resize score_map to match the image size (h, w) with mosaic effect
            score_map_resized = F.interpolate(score_map[i].unsqueeze(0), size=(h // 10, w // 10), mode='bilinear', align_corners=False)
            resized_score_map = F.interpolate(score_map_resized, size=(h, w), mode='nearest')
            resized_score_map = resized_score_map.squeeze(0)  # shape (h, w)
            
            # Convert the score map to a heatmap using matplotlib's viridis colormap
            score_map_np = resized_score_map.squeeze().cpu().numpy()
            heatmap_np = plt.get_cmap('viridis')(score_map_np)[:, :, :3]  # Get RGB channels only
            heatmap = torch.from_numpy(heatmap_np).permute(2, 0, 1)  # Convert to tensor and permute to (C, H, W)
            heatmap = heatmap.squeeze().to(image.device)  # shape (1, 3, h, w) to match the image batch dimension

            
            # Blend the heatmap with the original image (simple overlay with 50% opacity)
            alpha = 0.5
            image = (image.float() * (1 - alpha) + heatmap.float() * alpha)

        # Convert the image into 0-255 range and uint8 type
        image = (image * 255).byte()

        color = "green"        
        if score < 0.5:
            color = "cyan"

        drawn_image = draw_bounding_boxes(image, bbox, labels=[label], colors=[color], width=2)
        if ground_bbox is not None:

            gt_bbox = ground_bbox[i].unsqueeze(0)
            color = "red"

            if is_absence is not None:
                if is_absence[i] == 0.0:
                    gt_bbox[0, 0] = 0.5
                    gt_bbox[0, 1] = 0.5
                    gt_bbox[0, 2] = 0.5
                    gt_bbox[0, 3] = 0.5
                    color = "blue"

            gt_bbox[0, 0] *= w  # x-center 
            gt_bbox[0, 1] *= h  # y-center
            gt_bbox[0, 2] *= w  # width
            gt_bbox[0, 3] *= h  # height

            gt_bbox[0, 0] -= gt_bbox[0, 2] / 2  # x1
            gt_bbox[0, 1] -= gt_bbox[0, 3] / 2
            gt_bbox[0, 2] = gt_bbox[0, 0] + gt_bbox[0, 2]
            gt_bbox[0, 3] = gt_bbox[0, 1] + gt_bbox[0, 3]

            drawn_image = draw_bounding_boxes(drawn_image, gt_bbox, colors=[color], width=2)


        images_with_bboxes.append(drawn_image)

    # Make a grid of images
    grid_image = make_grid(images_with_bboxes, nrow=4)
    
    if save_path is not None:
        # Convert the grid to a PIL image and save
        pil_image = to_pil_image(grid_image)
        pil_image.save(save_path)

    return grid_image

def draw_bboxes_on_batch_multi(images, bboxes, scores, batch_indices, ground_bbox=None, box_nums=None, is_absence=None, score_map=None, save_path=None):
    """
    Draw bounding boxes on a batch of images and save as a single grid image.

    :param images: A torch tensor of shape (n, c, h, w)
    :param bboxes: A torch tensor of shape (n_peaks, 4) representing the bounding boxes (cx, cy, w, h) for the whole batch.
    :param scores: A torch tensor of shape (n_peaks) representing the scores for each bounding box.
    :param batch_indices: A torch tensor of shape (n_peaks), where each element indicates the image in the batch the bbox belongs to.
    :param ground_bbox: A torch tensor of shape (n, max_box_num, 4) containing ground truth bounding boxes.
    :param box_nums: A torch tensor of shape (n), indicating the actual number of ground truth boxes for each image.
    :param is_absence: A torch tensor of shape (n), indicating absence status for each image.
    :param score_map: A torch tensor of shape (n, 1, h, w) for optional score maps overlayed on each image.
    :param save_path: Path to save the output grid image.
    """
    n, c, h, w = images.shape
    images_with_bboxes = []

    for i in range(n):
        image = images[i]
        
        # Overlay score map if provided
        if score_map is not None:
            score_map_resized = F.interpolate(score_map[i].unsqueeze(0), size=(h // 10, w // 10), mode='bilinear', align_corners=False)
            resized_score_map = F.interpolate(score_map_resized, size=(h, w), mode='nearest').squeeze()
            score_map_np = resized_score_map.cpu().numpy()
            heatmap_np = plt.get_cmap('viridis')(score_map_np)[:, :, :3]  # Extract RGB channels
            heatmap = torch.from_numpy(heatmap_np).permute(2, 0, 1).to(image.device)
            alpha = 0.5
            image = (image.float() * (1 - alpha) + heatmap.float() * alpha)
        
        image = (image * 255).byte()  # Convert to 0-255 range and uint8

        # Draw predicted bounding boxes that belong to the current image
        for j in range(len(batch_indices)):
            if batch_indices[j] == i:
                bbox = bboxes[j].unsqueeze(0)  # (1, 4)
                score = scores[j].item()
                
                # Convert normalized bbox to pixel coordinates
                bbox[0, 0] *= w  # x-center
                bbox[0, 1] *= h  # y-center
                bbox[0, 2] *= w  # width
                bbox[0, 3] *= h  # height
                
                # Convert center coordinates to top-left coordinates
                bbox[0, 0] -= bbox[0, 2] / 2  # x1
                bbox[0, 1] -= bbox[0, 3] / 2  # y1
                bbox[0, 2] = bbox[0, 0] + bbox[0, 2]  # x2
                bbox[0, 3] = bbox[0, 1] + bbox[0, 3]  # y2
                
                # Draw bounding box with color based on score
                label = f'{score:.2f}'
                color = "green" if score >= 0.5 else "cyan"
                image = draw_bounding_boxes(image, bbox, labels=[label], colors=[color], width=2)
        
        # Draw ground truth bounding boxes if provided
        if ground_bbox is not None and box_nums is not None:
            for j in range(box_nums[i]):
                gt_bbox = ground_bbox[i, j].unsqueeze(0)
                color = "red"
                
                if is_absence is not None and is_absence[i] == 0.0:
                    gt_bbox[0, :] = torch.tensor([0.5, 0.5, 0.25, 0.25])
                    color = "blue"
                
                gt_bbox[0, 0] *= w  # x-center
                gt_bbox[0, 1] *= h  # y-center
                gt_bbox[0, 2] *= w  # width
                gt_bbox[0, 3] *= h  # height
                
                gt_bbox[0, 0] -= gt_bbox[0, 2] / 2  # x1
                gt_bbox[0, 1] -= gt_bbox[0, 3] / 2
                gt_bbox[0, 2] = gt_bbox[0, 0] + gt_bbox[0, 2]
                gt_bbox[0, 3] = gt_bbox[0, 1] + gt_bbox[0, 3]
                
                image = draw_bounding_boxes(image, gt_bbox, colors=[color], width=2)

        images_with_bboxes.append(image)


    if save_path is not None:
        # Create grid of images
        grid_image = make_grid(images_with_bboxes, nrow=8)
        pil_image = to_pil_image(grid_image)
        pil_image.save(save_path)

    return images_with_bboxes

def draw_match_result(template, search, n_pair=2, save_path=None):
    """
    Draw the matching result between the template and search images.
    Args:
        template: tensor of shape (N, C, H, W)
        search: list of search image tensor (with bbox draw) [(C, H, W)]
        n_pair: number of template-search pairs to draw each row
        save_path: str
    Returns:
        grid_image: tensor of the grid image
    """

    bs = template.shape[0]
    grid_images = []
    pair_size = 2
    if len(search) == bs: # one template to one search
        for i in range(bs):
            template_img = template[i]
            # resize the template image to the same size as the search image
            template_img = F.interpolate(template_img.unsqueeze(0), size=(search[i].shape[1], search[i].shape[2]), mode='bilinear', align_corners=False).squeeze()
            template_img = (template_img.cpu() * 255).byte()
            search_img = search[i]
            grid_images.append(template_img)
            grid_images.append(search_img)
    elif len(search) == 2*bs: # one template to two search
        pair_size = 3
        for i in range(bs):
            template_img = template[i]
            # resize the template image to the same size as the search image
            template_img = F.interpolate(template_img.unsqueeze(0), size=(search[2*i].shape[1], search[2*i].shape[2]), mode='bilinear', align_corners=False).squeeze()
            template_img = (template_img.cpu() * 255).byte()
            search_img1 = search[i]
            search_img2 = search[bs+i]
            grid_images.append(template_img)
            grid_images.append(search_img1)
            grid_images.append(search_img2)

    # Create grid of images
    grid_image = make_grid(grid_images, nrow=pair_size*n_pair)
    if save_path is not None:
        pil_image = to_pil_image(grid_image)
        pil_image.save(save_path)
    
    return grid_image




#def random_perspective_transform(image, mask):
    """
    对前景图片和其mask做随机透视变换
    """
    """
    height, width = image.shape[:2]
    
    # 定义随机透视变换矩阵参数
    margin = 0.3  # 允许的随机偏移量百分比
    pts1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    pts2 = np.float32([
        [random.uniform(0, margin) * width, random.uniform(0, margin) * height],
        [width - random.uniform(0, margin) * width, random.uniform(0, margin) * height],
        [width - random.uniform(0, margin) * width, height - random.uniform(0, margin) * height],
        [random.uniform(0, margin) * width, height - random.uniform(0, margin) * height]
    ])

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # 应用透视变换
    transformed_image = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    transformed_mask = cv2.warpPerspective(mask, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    return transformed_image, transformed_mask, matrix
    """

def random_perspective_transform(image, mask, min_scale=0.5, max_scale=2.0, max_attempts=10):
    """
    对前景图片和其mask做随机透视变换，同时保证变换后的区域大小在合理范围内。
    
    参数:
    - min_scale: 最小缩放比例
    - max_scale: 最大缩放比例
    - max_attempts: 最大尝试次数，用于防止在极端情况下陷入死循环
    """
    height, width = image.shape[:2]
    margin = 0.1  # 允许的随机偏移量百分比
    pts1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    for _ in range(max_attempts):
        # 生成随机变换后的角点
        pts2 = np.float32([
            [random.uniform(0, margin) * width, random.uniform(0, margin) * height],
            [width - random.uniform(0, margin) * width, random.uniform(0, margin) * height],
            [width - random.uniform(0, margin) * width, height - random.uniform(0, margin) * height],
            [random.uniform(0, margin) * width, height - random.uniform(0, margin) * height]
        ])

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # 计算原始面积和变换后区域的面积
        orig_area = width * height
        new_area = cv2.contourArea(pts2)

        # 检查缩放是否在合理范围内
        if min_scale * orig_area <= new_area <= max_scale * orig_area:
            # 面积符合要求，进行透视变换
            transformed_image = cv2.warpPerspective(image, matrix, (width, height), 
                                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            transformed_mask = cv2.warpPerspective(mask, matrix, (width, height), 
                                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
            return transformed_image, transformed_mask, matrix
    
    return image, mask, np.eye(3)

def random_perspective_transform_new(image, mask, min_scale=0.5, max_scale=2.0):
    """
    对前景图片和其mask做随机透视变换，保证变换后的外接矩形区域大小在合理范围内。
    
    参数:
    - min_scale: 最小缩放比例（基于外接矩形面积）
    - max_scale: 最大缩放比例（基于外接矩形面积）
    """
    height, width = image.shape[:2]
    orig_area = width * height
    pts1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    # 生成随机缩放比例
    scale = random.uniform(min_scale, max_scale)
    target_area = scale * orig_area
    
    # 生成宽高缩放因子（允许宽高比变化，但限制在合理范围内）
    # 这里采用对数空间采样以获得更均匀的宽高比分布
    """
    while True:
        ratio = np.exp(random.uniform(-1, 1))  # 宽高比变化范围: e^-1 ~ e^1 ≈ 0.37~2.72
        s_w = np.sqrt(target_area / orig_area * ratio)
        s_h = np.sqrt(target_area / orig_area / ratio)
        
        # 限制最大缩放倍数避免极端变形
        if 0.33 < s_w < 3.0 and 0.33 < s_h < 3.0:
            break
    """
    s_w = np.random.normal(1.0, 1)
    s_h = np.random.normal(1.0, 1)
    # 限制再 0.33 ~ 3.0 之间
    s_w = np.clip(s_w, 0.33, 3.0)
    s_h = np.clip(s_h, 0.33, 3.0)
    
    # 计算外接矩形尺寸
    w_new = s_w * width
    h_new = s_h * height
    
    # 生成随机中心偏移（限制在图像尺寸的20%范围内）
    max_offset_x = width * 0.2
    max_offset_y = height * 0.2
    cx = width/2 + random.uniform(-max_offset_x, max_offset_x)
    cy = height/2 + random.uniform(-max_offset_y, max_offset_y)
    
    # 计算外接矩形起始坐标
    x0 = cx - w_new/2
    y0 = cy - h_new/2
    
    # 生成变换后的角点（保持凸四边形特性）
    margin = 0.15  # 控制四边形边缘的弯曲程度
    pts2 = np.float32([
        [x0 + random.uniform(0, margin*w_new),        y0 + random.uniform(0, margin*h_new)],       # 左上
        [x0 + w_new - random.uniform(0, margin*w_new), y0 + random.uniform(0, margin*h_new)],       # 右上
        [x0 + w_new - random.uniform(0, margin*w_new), y0 + h_new - random.uniform(0, margin*h_new)], # 右下
        [x0 + random.uniform(0, margin*w_new),        y0 + h_new - random.uniform(0, margin*h_new)]  # 左下
    ])
    
    # 确保四边形凸性
    try:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    except:
        return image, mask, np.eye(3)
    
    # 执行变换
    transformed_image = cv2.warpPerspective(image, matrix, (width, height), 
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    transformed_mask = cv2.warpPerspective(mask, matrix, (width, height),
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return transformed_image, transformed_mask, matrix

def apply_foreground_to_background(background, foreground, mask, position):
    """
    将前景图像按照mask和位置融合到背景图像中
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    # 计算位置和大小
    x, y = position
    x_end = min(x + fg_w, bg_w)
    y_end = min(y + fg_h, bg_h)
    
    fg_resized = foreground[:y_end - y, :x_end - x]
    mask_resized = mask[:y_end - y, :x_end - x]

    # 提取背景区域
    background_region = background[y:y_end, x:x_end]

    # 将前景图像和背景图像融合（使用mask的alpha值）
    alpha = mask_resized / 255.0
    for c in range(3):  # 处理每个通道
        background_region[:, :, c] = alpha * fg_resized[:, :, c] + (1 - alpha) * background_region[:, :, c]

    background[y:y_end, x:x_end] = background_region

    # 创建新的mask
    new_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    new_mask[y:y_end, x:x_end] = mask_resized

    return background, new_mask

def calculate_bounding_box(transformed_mask, position):
    """
    根据透视变换矩阵和位置重新计算前景图像的边界框
    """
    # 找到mask中非零像素的坐标
    points = cv2.findNonZero(transformed_mask)

    # 如果没有找到有效点，返回空的边界框
    if points is None:
        return [0, 0, 0, 0]

    # 计算非零像素点的边界
    x_min, y_min = np.min(points[:, 0, :], axis=0)
    x_max, y_max = np.max(points[:, 0, :], axis=0)

    # 考虑放置位置的偏移量
    x_min += position[0]
    x_max += position[0]
    y_min += position[1]
    y_max += position[1]

    return [x_min, y_min, x_max, y_max]

# Function to smooth the mask
def mask_smooth(mask):
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

import cv2
import numpy as np
import random

def resize_background_to_fit_foreground(background, fg_width, fg_height):
    bg_h, bg_w = background.shape[:2]
    if bg_h < fg_height or bg_w < fg_width:
        new_size = (int(fg_width * 2), int(fg_height * 2))
        background = cv2.resize(background, new_size)
    return background

def generate_random_position(bg_width, bg_height, fg_width, fg_height):
    x = random.randint(0, bg_width - fg_width)
    y = random.randint(0, bg_height - fg_height)
    return x, y

def synthesize_data(foreground, background):
    # 分离前景图像和mask
    fg_rgb = foreground[:, :, :3]  # 前景图像的RGB通道
    fg_alpha = foreground[:, :, 3]  # 前景图像的Alpha通道

    # 对前景图像和mask做随机透视变换
    transformed_fg, transformed_mask, matrix = random_perspective_transform(fg_rgb, fg_alpha)
    check_img(transformed_fg)


    # 如果背景图像小于前景图像，则缩放背景图像
    background = resize_background_to_fit_foreground(background, transformed_fg.shape[1], transformed_fg.shape[0])

    # 获取背景图像的大小
    bg_h, bg_w = background.shape[:2]

    # 生成随机位置，确保前景图像不会超出背景图像的范围
    try:
        position = generate_random_position(bg_w, bg_h, transformed_fg.shape[1], transformed_fg.shape[0])
    except Exception as e:
        import pdb
        pdb.set_trace()

    # 将前景图像应用到背景图像上
    output_image, output_mask = apply_foreground_to_background(background, transformed_fg, transformed_mask, position)

    # 计算新的边界框
    bounding_box = calculate_bounding_box(transformed_mask, position)

    return output_image, output_mask, bounding_box


def synthesize_data_by_path(background_image_path, foreground_image_path):
    # 读取背景图像
    background = cv2.imread(background_image_path)
    check_img(background)

    # 读取前景图像（包括alpha通道）
    foreground = cv2.imread(foreground_image_path, cv2.IMREAD_UNCHANGED)
    check_img(foreground)

    output_image, mask, bbox = synthesize_data(foreground, background)

    return output_image, bbox

    

def synthesize_data_multi_by_path(background_image_path, foreground_image_path, max_instance_num, search_size, scale_mean, scale_std):
    # 读取背景图像
    background = cv2.imread(background_image_path)
    check_img(background)
    # 读取前景图像（包括alpha通道）
    foreground = cv2.imread(foreground_image_path, cv2.IMREAD_UNCHANGED)
    check_img(foreground)
    return synthesize_data_multi(background, foreground, max_instance_num, search_size, scale_mean, scale_std)

def synthesize_data_multi(background, foreground, max_instance_num, search_size, scale_mean, scale_std) -> Tuple[np.ndarray, List[List[int]]]:

    # 随机截取背景图像的一块方形区域，并缩放到 (search_size x search_size)
    bg_h, bg_w = background.shape[:2]
    min_size = 20  # 最小裁剪区域
    crop_size = random.randint(min_size, min(bg_h, bg_w))  # 随机生成裁剪区域大小
    crop_x = random.randint(0, bg_w - crop_size)
    crop_y = random.randint(0, bg_h - crop_size)
    cropped_bg = background[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    # 将裁剪后的背景缩放到 (search_size x search_size)
    background = cv2.resize(cropped_bg, (search_size, search_size))


    # 分离前景图像和mask
    fg_rgb = foreground[:, :, :3]  # 前景图像的RGB通道
    fg_alpha = foreground[:, :, 3]  # 前景图像的Alpha通道

    # 随机生成实例的数量
    n_instances = random.randint(1, max_instance_num)
    
    output_image = background.copy()
    bounding_boxes = []

    for _ in range(n_instances):
        # 对前景图像和mask做随机透视变换
        transformed_fg, transformed_mask, matrix = random_perspective_transform(fg_rgb, fg_alpha)
        check_img(transformed_fg)

        # 获取前景图像的宽和高
        fg_h, fg_w = transformed_fg.shape[:2]

        # 根据缩放规则计算缩放比例 r
        scale = np.random.normal(loc=scale_mean, scale=scale_std)  # 从正态分布生成随机缩放比例
        scale = max(scale_mean/2, scale) # 缩放比例不小于1
        scale = min(scale, 2*scale_mean) # 缩放比例不大于2倍的平均值
        r = search_size / (scale * np.sqrt(fg_w * fg_h))
        max_r = search_size / (max(fg_w, fg_h)+1)
        r = min(r, max_r)
        new_w = int(fg_w * r)
        new_h = int(fg_h * r)

        # 缩放前景图像
        scaled_fg = cv2.resize(transformed_fg, (new_w, new_h))
        scaled_mask = cv2.resize(transformed_mask, (new_w, new_h))

        # 随机生成放置位置，确保前景图像不会超出背景图像的范围
        position = generate_random_position(search_size, search_size, new_w, new_h)

        # 将前景图像应用到背景图像上
        output_image, output_mask = apply_foreground_to_background(output_image, scaled_fg, scaled_mask, position)

        # 计算当前前景目标的边界框
        bounding_box = calculate_bounding_box(scaled_mask, position)
        bounding_boxes.append(bounding_box)

    return output_image, bounding_boxes

def check_img(img):
    if img is None:
        raise ValueError("The image is None.")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("The image has zero height or width.")
    if img.shape[0] < 10 or img.shape[1] < 10:
        raise ValueError("The image is too small.")
    if img.shape[0] > 3000 or img.shape[1] > 3000:
        raise ValueError("The image is too large.")
    return True
