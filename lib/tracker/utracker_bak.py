import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import l1_loss
import torchvision.ops as ops  # For GIoU computation
import numpy as np
import os
import cv2
import requests
import base64

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.focal_loss import FocalLoss
from lib.utils.imgproc import center_crop
from lib.utils.heatmap_util import generate_heatmap
from lib.model.search_branch import build_search_branch
from lib.model.template_branch import build_template_branch
from lib.model.mobileone import reparameterize_model
from lib.utils.imgproc import draw_bboxes_on_batch_multi
#from lib.tracker.bytetrack.byte_tracker import BYTETracker

def get_device_of_module(module):
    return next(module.parameters()).device

class UTracker(nn.Module):
    def __init__(self, cfg):
        super(UTracker, self).__init__()
        self.template_branch = None
        self.search_branch = None
        self.last_pos = None
        self.last_result = None
        self.last_max_result = None
        self.KF = init_kalman_filter()
        self.KF_prediction = None
        self.embedding_list = []
        self.search_size = cfg.DATA.SEARCH.SIZE
        self.template_size = cfg.DATA.TEMPLATE.SIZE
        self.template_embedding = None
        self.template_image = None
        self.mean_embedding = None
        self.init_template_embedding = None
        self.save_path = None
        self.remote_sam = RemoteSAM('127.0.0.1', 5000)
        self.train_data = {"search_img":[], "gt_bbox":[]}  # 用于存储训练数据
        self.frame_cnt = 0
        #self.bytetracker = BYTETracker(cfg)

    def init(self, template, bbox):
        self.template_branch = reparameterize_model(self.template_branch)
        self.search_branch = reparameterize_model(self.search_branch)
        self.embedding_list = []
        self.update_template(template, bbox)
        self.init_template_embedding = self.template_embedding.clone().detach()
        self.last_pos = bbox
        self.last_score = 1.0
        self.loss_cnt = 0
        self.frame_cnt = 0
        # 初始状态（位置和尺度已知，速度和加速度为0）
        # 一阶模型
        initial_state = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]], [0], [0], [0], [0]], np.float32)
        # 二阶模型
        #initial_state = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]], [0], [0], [0], [0], [0], [0], [0], [0]], np.float32)
        self.KF.statePost = initial_state


    def update_template(self, template, bbox):
        bbox = [int(x) for x in bbox] 
        template_img = center_crop(template, bbox, 2)
        template_img = cv2.resize(template_img, (self.template_size, self.template_size))
        #template_img = self.remote_sam.build_template(template, bbox)
        self.template_image = template_img.copy()
        template_img = torch.from_numpy(template_img.astype(np.float32)).permute(2, 0, 1)
        c,h,w = template_img.shape
        template_img = template_img.view(1, c, h, w)
        with torch.no_grad():
            self.template_embedding = self.template_branch(template_img.to(get_device_of_module(self))/255.0)
        self.embedding_list.append(self.template_embedding)

    def mean_template(self):
        with torch.no_grad():
            if self.mean_embedding is None:
                mean_embedding = torch.stack(self.embedding_list).mean(dim=0)
                # normalize the mean embedding
                #self.template_embedding = F.normalize(mean_embedding, p=2, dim=1) 
                self.mean_embedding = F.normalize(mean_embedding, p=2, dim=1) 
            else:
                #self.mean_embedding = F.normalize(self.mean_embedding*0.5+self.embedding_list[-1]*0.5, p=2, dim=1) 
                self.mean_embedding = F.normalize(self.mean_embedding+self.embedding_list[-1], p=2, dim=1) 

    def trans_pos(self, pred_bbox, w_i, h_i):
        cx, cy, w, h = pred_bbox
        offset_x, offset_y, w, h = (cx-0.5) * w_i, (cy-0.5) * h_i, w * w_i, h * h_i
        return [self.last_pos[0] + offset_x, self.last_pos[1] + offset_y, w, h]

    def match_last_pos(self):
        # return the pos from last_result that is closest to last_pos
        if self.last_result is None:
            return None
        else:
            # match the closest one by IOU
            kf_pos = self.KF_prediction
            last_pos = self.last_pos
            max_iou = 0
            max_idx = 0
            for idx, pos in enumerate(self.last_result):
                iou = self.compute_iou(last_pos, pos)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = idx
            current_pos = self.last_result[max_idx]
            if max_iou < 0.5:
                max_iou = 0
                max_idx = 0
                for idx, pos in enumerate(self.last_result):
                    iou = self.compute_iou(kf_pos, pos)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = idx

                if max_iou < 0.5:
                    if max_iou > 0.3:
                        current_pos = kf_pos
                    else:
                        max_score = 0
                        for idx,score in enumerate(self.last_result_score):
                            if score > max_score:
                                max_score = score
                                max_idx = idx
                        if max_score > 0.5:
                            current_pos = self.last_result[max_idx]
                        else:
                            current_pos = self.last_pos
                else:
                    current_pos = self.last_result[max_idx]

            """
            for idx, pos in enumerate(self.last_result):
                iou = self.compute_iou(kf_pos, pos)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = idx
            current_pos = self.last_result[max_idx]
            if max_iou < 0.3:
                max_iou = 0
                max_idx = 0
                last_pos = self.last_pos
                for idx, pos in enumerate(self.last_result):
                    iou = self.compute_iou(last_pos, pos)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = idx
                if max_iou < 0.3:
                    current_pos = self.last_pos
                else:
                    current_pos = self.last_result[max_idx]
            """
            # update KF
            measurement = np.array([[current_pos[0]], [current_pos[1]], [current_pos[2]], [current_pos[3]]], np.float32)
            self.KF.correct(measurement)

            return current_pos

    def fuse_last_pos(self):
        if self.last_result is None:
            return None
        else:
            #import pdb; pdb.set_trace()
            last_pos = self.last_pos
            max_iou = 0
            max_idx = 0
            gate_score = 0
            gate_idx = 0
            for idx, pos in enumerate(self.last_result):
                iou = self.compute_iou(self.KF_prediction, pos)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = idx
                if iou > 0.2 and self.last_result_score[idx] > gate_score:
                    gate_score = self.last_result_score[idx]
                    gate_idx = idx
            if max_iou > 0.5 or (max_iou > 0.2 and self.last_result_score[max_idx] > 0.2):
                current_pos = self.last_result[max_idx]
                return current_pos, self.last_result_score[max_idx]
            elif gate_score > 0.1:
                current_pos = self.last_result[gate_idx]
                return current_pos, gate_score
            else:
                #print("last_max_score", self.last_max_score)
                if self.last_max_score > 0.4: 
                    return self.last_max_result, self.last_max_score
                else:
                    return self.last_pos, 0

    
    def compute_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        inter_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        inter_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        inter_area = inter_x * inter_y
        union_area = w1*h1 + w2*h2 - inter_area
        return inter_area / union_area

    def match_max_score(self, peak_scores):
        if self.last_result is None:
            return None
        else:
            self.last_max_score = self.last_result_score[np.argmax(peak_scores)]
            return self.last_result[np.argmax(peak_scores)]

    def match_image(self, image, base_pos, embedding):
        base_pos = [base_pos[0], base_pos[1], int(base_pos[2]*0.9), int(base_pos[3]*0.9)]
        search_img, ori_roi = center_crop(image, base_pos, 4, return_roi=True)
        w_i, h_i = search_img.shape[1], search_img.shape[0]
        search_img = cv2.resize(search_img, (self.search_size, self.search_size))
        self.search_img = search_img.copy()
        search_img = torch.from_numpy(search_img.astype(np.float32)).permute(2, 0, 1)
        c,h,w = search_img.shape
        search_img = search_img.view(1, c, h, w)
        with torch.no_grad():
            result = self.search_branch(search_img.to(get_device_of_module(self))/255.0, embedding)
        pred_bbox = result[1].detach().cpu().numpy()
        for pos in pred_bbox:
            self.last_result.append(self.trans_pos(pos, w_i, h_i))
        for score in result[-1].detach().cpu().numpy():
            self.last_result_score.append(score)
        return result[0].detach().cpu().numpy()

    def track(self, search):
        self.frame_cnt += 1
        kf_res = self.KF.predict()
        self.KF_prediction = [kf_res[0][0], kf_res[1][0], kf_res[2][0], kf_res[3][0]]
        self.last_result = []
        self.last_result_score = []
        crop_region = self.last_pos
        if crop_region[-1] < 5:
            crop_region[-1] = 5
        if crop_region[-2] < 5:
            crop_region[-2] = 5
        crop_region[0] = np.clip(crop_region[0], 0, search.shape[1])
        crop_region[1] = np.clip(crop_region[1], 0, search.shape[0])
        # if self.mean_embedding is not None:
        #     score_map_ = self.match_image(search, crop_region, self.mean_embedding)
        # else:
        score_map_ = self.match_image(search, crop_region, self.template_embedding)
        #Draw score map on self.search_img
        # Resize score map from (1,1,16,16) to (256,256)
        score_map = F.interpolate(torch.from_numpy(score_map_), 
                                 size=(self.search_size, self.search_size),
                                 mode='bicubic').squeeze(0).squeeze(0).numpy()

        # Normalize score map to 0-255 range
        score_map = ((score_map - score_map.min()) * 255 / 
                     (score_map.max() - score_map.min())).astype(np.uint8)

        # Convert to heatmap
        score_map_color = cv2.applyColorMap(score_map, cv2.COLORMAP_JET)

        # Blend with search image
        alpha = 0.5
        self.search_img = cv2.addWeighted(self.search_img, alpha, 
                                         score_map_color, 1-alpha, 0)
         
        #crop_region = [self.last_pos[0], self.last_pos[1], int(self.last_pos[2]*0.9), int(self.last_pos[3]*0.9)]

        """
        if self.mean_embedding is not None:
            import pdb; pdb.set_trace()
            if self.KF_prediction[-1] > 5 and self.KF_prediction[-2] > 5:
                self.match_image(search, self.KF_prediction, self.mean_embedding)
            else:
                self.match_image(search, crop_region, self.mean_embedding)
        """


        """
        if self.init_template_embedding is not None:
            self.match_image(search, self.last_pos, self.init_template_embedding)
        """
            


        #self.last_pos = self.last_result[0]
        if len(self.last_result) >= 1:
            self.loss_cnt = 0
            """
            self.last_max_result = self.match_max_score(self.last_result_score)
            #self.last_pos = self.match_last_pos()
            new_pose, new_score = self.fuse_last_pos()
            # update KF
            IOU_MAX = self.compute_iou(new_pose, self.last_max_result)
            IOU_KF = self.compute_iou(new_pose, self.KF_prediction)
            # update the template embedding
            #if IOU_MAX > 0.75 and IOU_KF > 0.75:
            #if new_score > 0 and new_score < 0.5 and IOU_KF > 0.8:
            if new_score > 0 and IOU_KF > 0.7:
                self.update_template(search, self.KF_prediction)
                self.mean_template()
                #import pdb; pdb.set_trace()
                print("Update template X")
            self.last_pos = new_pose
            """
            new_pose = self.match_max_score(self.last_result_score)
            if np.max(self.last_result_score) < 0.2:
                new_pose[2:] = self.KF_prediction[2:]
            measurement = np.array([[new_pose[0]], [new_pose[1]], [new_pose[2]], [new_pose[3]]], np.float32)
            self.KF.correct(measurement)
            # if (np.max(self.last_result_score) > 0.5):
            #     self.update_template(search, new_pose)
            #     self.mean_template()
            self.last_pos = new_pose 
        else:
            if self.loss_cnt < 1:
                self.loss_cnt += 1
                self.last_pos = self.KF_prediction
        
        # cx, cy, w, h = self.last_pos
        # clip position and size
        self.last_pos[0] = np.clip(self.last_pos[0], 0, search.shape[1])
        self.last_pos[1] = np.clip(self.last_pos[1], 0, search.shape[0])
        self.last_pos[2] = np.clip(self.last_pos[2], 5, search.shape[1])
        self.last_pos[3] = np.clip(self.last_pos[3], 5, search.shape[0])

        return self.last_pos

    
    def collect_data(self, search, bbox_last, bbox_gt):
        self.update_template(search, bbox_gt)
        # collect data for training
        search_img, ori_roi = center_crop(search, bbox_last, 4, return_roi=True)
        w_i, h_i = search_img.shape[1], search_img.shape[0]
        search_img = cv2.resize(search_img, (self.search_size, self.search_size))

        # 添加以下代码以计算bbox_gt在search_img中的坐标
        cx_gt, cy_gt, w_gt, h_gt = bbox_gt
        cx_last, cy_last, w_last, h_last = bbox_last
        
        # 计算裁剪参数
        size = int(np.sqrt(w_last * h_last) * 4)
        x1 = cx_last - size // 2
        y1 = cy_last - size // 2
        
        # 转换到裁剪图像中的坐标（考虑填充后的整体图像）
        cx_crop = cx_gt - x1
        cy_crop = cy_gt - y1
        
        # 计算缩放比例
        scale_factor = self.search_size / size
        
        # 应用缩放
        cx_new = cx_crop * scale_factor
        cy_new = cy_crop * scale_factor
        w_new = w_gt * scale_factor
        h_new = h_gt * scale_factor
        
        # 最终的坐标（可根据需求处理为整数或浮点）
        bbox_gt_in_search = (cx_new, cy_new, w_new, h_new)
        search_img = torch.from_numpy(search_img.astype(np.float32)).permute(2, 0, 1) / 255
        bbox_gt_in_search = torch.tensor(bbox_gt_in_search).float()

        self.train_data["search_img"].append(search_img)
        self.train_data["gt_bbox"].append(bbox_gt_in_search)

    def save_data(self, path):
        torch.save(self.train_data, path)
    
    def load_data(self, path):
        self.train_data = torch.load(path)

    def optim_template_embedding(self, lr=1e-4, step=1000, idxs=None):
        if len(self.train_data["search_img"]) == 0:
            print("No data to train")
            return

        if idxs is None:
            search_img = torch.stack(self.train_data["search_img"])
            gt_bbox = torch.stack(self.train_data["gt_bbox"])
        else:
            search_img = torch.stack(self.train_data["search_img"])[idxs]
            gt_bbox = torch.stack(self.train_data["gt_bbox"])[idxs]
        if self.template_embedding is None:
            init_template_embedding = torch.randn(1, self.template_branch.embed_dim, 1, 1)
        else:
            init_template_embedding = self.template_embedding

        search_img = search_img.to(get_device_of_module(self))
        gt_bbox = gt_bbox.to(get_device_of_module(self))
        init_template_embedding = init_template_embedding.clone().detach().requires_grad_(True)
        optimizer = torch.optim.AdamW([init_template_embedding], lr=lr)
        ground_absence = torch.tensor([1.0]*search_img.shape[0]).to(get_device_of_module(self))
        # 关闭 self.search_branch 中所有参数的梯度计算
        for param in self.search_branch.parameters():
            param.requires_grad_(False)
        for i in range(step):
            optimizer.zero_grad()
            # 广播 init_template_embedding 到 [n, d, 1, 1]
            norm_template_embedding = F.normalize(init_template_embedding, p=2, dim=1)
            broadcasted_template_embedding = norm_template_embedding.expand(search_img.size(0), -1, -1, -1).to(get_device_of_module(self))
            out = self.search_branch(search_img, broadcasted_template_embedding)
            loss = self.get_loss(out, gt_bbox, ground_absence)
            loss.backward()
            optimizer.step()
            print("Loss:", loss.item(), "step:", i)
            if i % 10 == 0:
                #self.visualize_embedding_result(init_template_embedding, f"experiments/debug_tracker_11/optim_{i}.jpg")
                self.visualize_embedding_result(init_template_embedding, os.path.join(self.save_path,f"optim_{i}.jpg"))

        return F.normalize(init_template_embedding, p=2, dim=1).detach().cpu()

    def visualize_embedding_result(self, template_embedding, output_path):
        if len(self.train_data["search_img"]) == 0:
            print("No data to show")
            return

        search_img = torch.stack(self.train_data["search_img"])
        gt_bbox = torch.stack(self.train_data["gt_bbox"])
        init_template_embedding = F.normalize(template_embedding.clone().detach(), p=2, dim=1)
        search_img = search_img.to(get_device_of_module(self))
        gt_bbox = gt_bbox.to(get_device_of_module(self))
        template_embedding = init_template_embedding.to(get_device_of_module(self))

        with torch.no_grad():
            broadcasted_template_embedding = template_embedding.expand(search_img.size(0), -1, -1, -1)
            resault = self.search_branch(search_img, broadcasted_template_embedding)

        images = search_img.detach().cpu()
        ground_bboxes = torch.zeros(gt_bbox.shape[0], 5, 4)
        for i in range(gt_bbox.shape[0]):
            ground_bboxes[i,0] = gt_bbox[i] / self.search_size
        scores = resault[-1].detach().cpu()
        bboxes = resault[1].detach().cpu()
        batch_indices = resault[2].detach().cpu()
        is_absence = torch.tensor([1.0]*search_img.shape[0]).detach().cpu()
        box_num = [1]*search_img.shape[0]

        draw_bboxes_on_batch_multi(images, bboxes, scores, batch_indices, ground_bboxes, box_num, is_absence, score_map=resault[0].detach().cpu(), save_path=output_path)

    def draw_results_on_image(self, search_img):
        for bbox in self.last_result:
            cx, cy, w, h = bbox
            cv2.rectangle(search_img, (int(cx-w//2), int(cy-h//2)), (int(cx+w//2), int(cy+h//2)), (255, 255, 0), 5)
        cx, cy, w, h = self.last_max_result
        cv2.rectangle(search_img, (int(cx-w//2), int(cy-h//2)), (int(cx+w//2), int(cy+h//2)), (0, 255, 255), 5)
        cx, cy, w, h = self.KF_prediction
        cv2.rectangle(search_img, (int(cx-w//2), int(cy-h//2)), (int(cx+w//2), int(cy+h//2)), (255, 0, 255), 5)
        return search_img



        
        
        

        






    def stastic_result(self, iou, score_map, is_absence, iou_threshold=0.5, score_threshold=0.5):
        batch_size = iou.shape[0]
        true_positives = []
        false_positives = []
        scores = []
        num_gt_objects = 0
        for i in range(batch_size):
            max_score = score_map[i].max()
            scores.append(max_score)
            if is_absence[i] == 1: # Target present
                num_gt_objects += 1
                if max_score > score_threshold and iou[i] > score_threshold:
                    true_positives.append(1)
                    false_positives.append(0)
                else:
                    true_positives.append(0)
                    false_positives.append(1)
            else: # Target absent
                if max_score < score_threshold:
                    true_positives.append(1)
                    false_positives.append(0)
                else:
                    true_positives.append(0)
                    false_positives.append(1)

        return true_positives, false_positives, scores, num_gt_objects
    
    def calculate_ap_from_aggregated_data(self, true_positives, false_positives, scores, num_gt_objects):
        # Sort by score
        indices = np.argsort(scores)[::-1]
        true_positives = np.array(true_positives)[indices]
        false_positives = np.array(false_positives)[indices]
        
        # Calculate cumulative sum
        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)
        
        # Calculate precision and recall
        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall = cumulative_tp / num_gt_objects  # Use the number of ground truth objects
        
        # Compute AP
        ap = np.trapz(precision, recall)
        
        return ap
            

    def get_loss(self, result, ground_bbox, is_absence, val=False):
        # gt gaussian map
        gt_gaussian_maps = generate_heatmap(ground_bbox, self.search_size, self.stride)
        gt_gaussian_maps = gt_gaussian_maps.to(result[0].device)

        pred_boxes_vec = box_cxcywh_to_xyxy(result[1]).view(-1, 4)  # (N,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_cxcywh_to_xyxy(ground_bbox)[:, None, :].view(-1, 4).clamp(min=0.0,max=1.0)

        giou_loss, l1_loss = self.compute_box_loss(pred_boxes_vec, result[2], gt_boxes_vec)

        gt_gaussian_maps *= is_absence.view(-1, 1, 1, 1)
        location_loss = self.focal_loss(result[0], gt_gaussian_maps)

        return location_loss + giou_loss + l1_loss

    def compute_box_loss(self, pred_bboxes, batch_indices, gt_bboxes):
        """
        Compute the matching and loss between predicted bboxes and ground truth bboxes.
        
        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes [n_pred, 4 (cx, cy, w, h)]
            batch_indices (torch.Tensor): Indices indicating the batch to which each pred bbox belongs [n_pred]
            gt_bboxes (torch.Tensor): Ground truth bounding boxes [bs, 4 (cx, cy, w, h)]
        
        Returns:
            giou_loss (torch.Tensor): GIoU loss
            l1_loss (torch.Tensor): L1 loss
        """
        bs = gt_bboxes.shape[0]  # batch size
        total_giou_loss = 0
        total_l1_loss = 0
        n_matched = 0  # Count matched boxes for averaging loss

        for i in range(bs):
            # Get the predicted boxes and ground truth boxes for the current batch element
            pred_mask = batch_indices == i
            pred_boxes_i = pred_bboxes[pred_mask]  # [n_pred_i, 4]
            gt_boxes_i = gt_bboxes[i].unsqueeze(0)  # [1, 4] since each image has exactly one GT box
            
            n_pred = pred_boxes_i.shape[0]

            if n_pred == 0:
                continue

            # Convert to [x_min, y_min, x_max, y_max] format for GIoU calculation
            pred_boxes_i_xyxy = ops.box_convert(pred_boxes_i, in_fmt="cxcywh", out_fmt="xyxy")
            gt_boxes_i_xyxy = ops.box_convert(gt_boxes_i, in_fmt="cxcywh", out_fmt="xyxy")
            
            # Compute pairwise GIoU matrix
            giou_matrix = ops.generalized_box_iou(pred_boxes_i_xyxy, gt_boxes_i_xyxy)  # [n_pred_i, 1]
            
            # Convert GIoU to a cost matrix (1 - GIoU, because we want to minimize the cost)
            cost_matrix = 1 - giou_matrix  # [n_pred_i, 1]

            # Perform Hungarian matching (minimizing 1 - GIoU)
            # Since there is only one GT box, we can directly select the best match
            best_match_idx = torch.argmin(cost_matrix, dim=0)  # [1]

            # Matched pairs of indices for predicted and ground truth boxes
            matched_pred_boxes = pred_boxes_i[best_match_idx]  # Matched predicted boxes [1, 4]
            matched_gt_boxes = gt_boxes_i  # Matched GT boxes [1, 4]

            # Compute GIoU loss for matched pairs
            matched_pred_boxes_xyxy = ops.box_convert(matched_pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            matched_gt_boxes_xyxy = ops.box_convert(matched_gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            
            giou = ops.generalized_box_iou(matched_pred_boxes_xyxy, matched_gt_boxes_xyxy)  # [1]
            giou_loss = 1 - giou  # GIoU loss

            # Compute L1 loss for matched pairs
            l1_loss = torch.abs(matched_pred_boxes - matched_gt_boxes).sum(dim=-1)  # [1]

            # Accumulate loss
            total_giou_loss += giou_loss.sum()
            total_l1_loss += l1_loss.sum()
            n_matched += 1  # Number of matched boxes
        
        # Compute the final average losses
        if n_matched > 0:
            avg_giou_loss = total_giou_loss / n_matched
            avg_l1_loss = total_l1_loss / n_matched
        else:
            avg_giou_loss = torch.tensor(0.0)
            avg_l1_loss = torch.tensor(0.0)

        return avg_giou_loss, avg_l1_loss



    def set_loss(self, cfg):
        self.focal_loss = FocalLoss()
        self.giou_loss = giou_loss
        self.l1_loss = l1_loss
        self.search_size = cfg.DATA.SEARCH.SIZE
        self.stride = cfg.MODEL.BACKBONE.STRIDE
        self.loss_weight = {
            'cls': cfg.TRAIN.LOSS.CLS_WEIGHT,
            'l1': cfg.TRAIN.LOSS.BBOX_WEIGHT,
            'iou': cfg.TRAIN.LOSS.IOU_WEIGHT
        }
        

# 初始化卡尔曼滤波器
# 一阶动态模型，状态变量包括位置和速度
# 初始化卡尔曼滤波器
def init_kalman_filter():
    kf = cv2.KalmanFilter(8, 4, 0)  # 8个状态变量，4个观测变量
    
    # 状态转移矩阵
    dt = 1.0  # 时间步长
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, dt, 0, 0, 0],
        [0, 1, 0, 0, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0, dt, 0],
        [0, 0, 0, 1, 0, 0, 0, dt],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ], np.float32)
    
    # 观测矩阵
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
    ], np.float32)
    
    # 过程噪声协方差矩阵
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
    
    # 观测噪声协方差矩阵
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    
    # 后验误差协方差矩阵
    kf.errorCovPost = np.eye(8, dtype=np.float32)
    
    return kf

# 二阶动态模型，状态变量包括位置、速度和加速度
"""
def init_kalman_filter():
    kf = cv2.KalmanFilter(12, 4, 0)  # 12个状态变量，4个观测变量
    
    # 状态转移矩阵
    dt = 1.0  # 时间步长
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, dt, 0, 0, 0, 0.5*dt**2, 0, 0, 0],
        [0, 1, 0, 0, 0, dt, 0, 0, 0, 0.5*dt**2, 0, 0],
        [0, 0, 1, 0, 0, 0, dt, 0, 0, 0, 0.5*dt**2, 0],
        [0, 0, 0, 1, 0, 0, 0, dt, 0, 0, 0, 0.5*dt**2],
        [0, 0, 0, 0, 1, 0, 0, 0, dt, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, dt, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, dt, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, dt],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], np.float32)
    
    # 观测矩阵
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ], np.float32)
    
    # 过程噪声协方差矩阵
    kf.processNoiseCov = np.eye(12, dtype=np.float32) * 1e-2
    
    # 观测噪声协方差矩阵
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    
    # 后验误差协方差矩阵
    kf.errorCovPost = np.eye(12, dtype=np.float32)
    
    return kf
    
def constraint_loss(score_map, is_absence):
    # 将 is_absence 调整为 (n, 1, 1, 1) 形状
    is_absence = is_absence.view(-1, 1, 1, 1)
    # 生成反转掩码，当 is_absence 为 0 时，该值为 1，否则为 0
    mask = 1 - is_absence
    # 计算 score_map 和全零 map 的 L1 损失
    l1_loss = nn.L1Loss(reduction='none')
    zero_map = torch.zeros_like(score_map)
    loss = l1_loss(score_map, zero_map)
    # 只保留需要惩罚的部分
    masked_loss = loss * mask
    # 对损失进行平均
    return masked_loss.sum() / score_map.shape[0]
"""

def build_utracker(cfg):
    model = UTracker(cfg)
    model.template_branch = build_template_branch(cfg)
    model.search_branch = build_search_branch(cfg)
    model.set_loss(cfg)
    return model

class RemoteSAM:
    def __init__(self, ip, port):
        self.url = f'http://{ip}:{port}/process'

    def build_template(self, image, bbox):
        try:
            # 转换OpenCV BGR图像为Base64
            _, buffer = cv2.imencode('.jpg', image)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                'image': image_b64,
                'bbox': [float(x) for x in bbox]  # 确保数值类型可序列化
            }
            
            response = requests.post(self.url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if result['status'] != 'success':
                raise Exception(result.get('message', 'Unknown error'))
            
            return self._decode_template(result['template'])
        except Exception as e:
            raise RuntimeError(f"Remote processing failed: {str(e)}")

    def _decode_template(self, base64_str):
        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)