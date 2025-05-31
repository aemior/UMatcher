import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import l1_loss
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import torchvision.ops as ops  # For GIoU computation

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.focal_loss import FocalLoss
from lib.utils.imgproc import center_crop
from lib.utils.heatmap_util import generate_heatmap, generate_multi_heatmap
from lib.model.mobileone import reparameterize_model
from lib.model.search_branch import build_search_branch, SearchBranch
from lib.model.template_branch import build_template_branch, TemplateBranch
from lib.model.autoencoder import build_auto_encoder, AutoEncoder

def get_device_of_module(module):
    return next(module.parameters()).device

class UMatcher(nn.Module):
    def __init__(self, cfg):
        super(UMatcher, self).__init__()
        self.template_branch: Optional[TemplateBranch] = None
        self.search_branch: Optional[SearchBranch] = None
        self.auto_encoder: Optional[AutoEncoder] = None
        self.dual_template = cfg.DATA.TEMPLATE.DUAL
        if self.dual_template:
            self.temperature = cfg.TRAIN.LOSS.TEMPTURE
        self.last_pos = None
        self.search_size = cfg.DATA.SEARCH.SIZE
        self.template_size = cfg.DATA.TEMPLATE.SIZE

    def forward(self, data, val=False):
        return self.forward_cross(data, val)

    def forward_standard(self, data, val=False):
        template_embedding = self.template_branch(data['template_img'])
        result = self.search_branch(data['search_img'], template_embedding)
        loss = self.get_loss(result, data['ground_bbox'], data['is_absence'])
        if val:
            return loss, result
        else:
            return loss

    def forward_cross(self, data, val=False):
        """
        :param data: dict, contains the following keys:
            - template_img: torch.Tensor [bs, 3, H, W]
            - search_img: torch.Tensor [bs, 3, H, W]
            - ground_bbox: torch.Tensor [bs, 4 (cx, cy, w, h)]
            - box_num: torch.Tensor [bs]
            - template_mask: torch.Tensor [bs, 1, H, W] (ignored here)
        :param val: bool
        
        :return: If val=False, a dict with 'total_loss', 'iou_loss', 'l1_loss', and 'cls_loss'.
                 If val=True, the raw model outputs.
        """
        search_img = data['search_img']
        ground_bbox = data['ground_bbox']
        box_num = data['box_num']

        # Shift search_img, ground_bbox, and box_num
        search_img_shifted = torch.roll(search_img, 1, dims=0)
        ground_bbox_shifted = torch.roll(ground_bbox, 1, dims=0)
        box_num_shifted = torch.roll(box_num, 1, dims=0)

        # Embed template
        if self.dual_template:
            template_img = data['template_img_a']
            template_img_ = data['template_img_b']
            template_embedding_a = self.template_branch(template_img)
            template_embedding_b = self.template_branch(template_img_)
            template_embedding = torch.cat([template_embedding_a, template_embedding_b], dim=0)
        else:
            template_img = data['template_img']
            template_embedding_a = self.template_branch(template_img)
            template_embedding = torch.cat([template_embedding_a, template_embedding_a], dim=0)

        # Duplicate and concatenate
        search_img = torch.cat([search_img, search_img_shifted], dim=0)
        ground_bbox = torch.cat([ground_bbox, ground_bbox_shifted], dim=0)
        box_num = torch.cat([box_num, box_num_shifted], dim=0)

        # Construct ground_absence
        bs = template_img.shape[0]
        ground_absence = torch.tensor([1.0]*bs + [0.0]*bs, device=template_img.device)

        # Forward
        result = self.search_branch(search_img, template_embedding)
        if val:
            return result

        # Calculate losses
        l1_loss, giou_loss = self.compute_box_loss(result[1], result[2], ground_bbox, ground_absence)
        location_loss = self.compute_score_loss(result[0], ground_bbox, box_num, ground_absence)
        # if dual template cal contrastive loss using template_embedding and template_embedding_
        # the shape of the embedding is [bs, embedding_dim, 1, 1] and normed
        # Cal CLIP like contrastive loss
        if self.dual_template:
            contrastive_loss = self.clip_contrastive_loss(template_embedding_a, template_embedding_b)
        else:
            contrastive_loss = None

        # Weighted sum
        loss = (self.loss_weight['iou'] * giou_loss
                + self.loss_weight['l1'] * l1_loss
                + self.loss_weight['cls'] * location_loss)

        if self.dual_template:
            loss += self.loss_weight['ctr'] * contrastive_loss

        return {
            'total_loss': loss,
            'iou_loss': giou_loss,
            'l1_loss': l1_loss,
            'cls_loss': location_loss,
            'ctr_loss': contrastive_loss
        }

    def clip_contrastive_loss(self, template_embedding, template_embedding_):
        # flatten the embedding (the embedding is alraedy normed)
        template_embedding_a = template_embedding.view(template_embedding.shape[0], -1)
        template_embedding_b = template_embedding_.view(template_embedding_.shape[0], -1)

        # cosine similarity
        logits_per_img = torch.matmul(template_embedding_a, template_embedding_b.T) / self.temperature
        logits_per_img_t = logits_per_img.T

        # labels
        batch_size = template_embedding_a.shape[0]
        labels = torch.arange(batch_size, device=template_embedding_a.device)

        # cross entropy loss
        loss = F.cross_entropy(logits_per_img, labels)
        loss_t = F.cross_entropy(logits_per_img_t, labels)

        return (loss + loss_t) / 2


    def compute_score_loss(self, pred_score_map, gt_bboxes, box_nums, is_absence):
        """
        Compute the loss for the score map.

        Args:
            pred_score_map (torch.Tensor): Predicted score map [bs, 1, H, W]
            gt_bboxes (torch.Tensor): Ground truth bounding boxes [bs, max_bbox, 4 (cx, cy, w, h)]
            box_nums (torch.Tensor): Number of valid GT boxes for each image in the batch [bs]
            is_absence (torch.Tensor): Indicator of target presence [bs]
        
        Returns:
            score_loss (torch.Tensor): Score loss
        """
        gt_gaussian_maps = generate_multi_heatmap(gt_bboxes, box_nums.flatten(), self.search_size, self.stride).to(is_absence.device)
        gt_gaussian_maps *= is_absence.view(-1, 1, 1, 1)
        location_loss = self.focal_loss(pred_score_map, gt_gaussian_maps) + constraint_loss(pred_score_map, is_absence)
        return location_loss
        

    def compute_box_loss(self, pred_bboxes, batch_indices, gt_bboxes, box_num):
        """
        Compute the matching and loss between predicted bboxes and ground truth bboxes.
        
        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes [n_pred, 4 (cx, cy, w, h)]
            batch_indices (torch.Tensor): Indices indicating the batch to which each pred bbox belongs [n_pred]
            gt_bboxes (torch.Tensor): Ground truth bounding boxes [bs, max_bbox, 4 (cx, cy, w, h)]
            box_num (torch.Tensor): Number of valid GT boxes for each image in the batch [bs]
        
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
            gt_boxes_i = gt_bboxes[i, :int(box_num[i]), :]  # [n_gt_i, 4]
            
            n_pred = pred_boxes_i.shape[0]
            n_gt = gt_boxes_i.shape[0]

            if n_pred == 0 or n_gt == 0:
                continue

            # Convert to [x_min, y_min, x_max, y_max] format for GIoU calculation
            pred_boxes_i_xyxy = ops.box_convert(pred_boxes_i, in_fmt="cxcywh", out_fmt="xyxy")
            gt_boxes_i_xyxy = ops.box_convert(gt_boxes_i, in_fmt="cxcywh", out_fmt="xyxy")
            
            # Compute pairwise GIoU matrix
            giou_matrix = ops.generalized_box_iou(pred_boxes_i_xyxy, gt_boxes_i_xyxy)  # [n_pred_i, n_gt_i]
            
            # Convert GIoU to a cost matrix (1 - GIoU, because we want to minimize the cost)
            cost_matrix = 1 - giou_matrix  # [n_pred_i, n_gt_i]

            # Perform Hungarian matching (minimizing 1 - GIoU)
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())  # Hungarian matching

            # Matched pairs of indices for predicted and ground truth boxes
            matched_pred_boxes = pred_boxes_i[row_ind]  # Matched predicted boxes [n_matched, 4]
            matched_gt_boxes = gt_boxes_i[col_ind]  # Matched GT boxes [n_matched, 4]

            # Compute GIoU loss for matched pairs
            matched_pred_boxes_xyxy = ops.box_convert(matched_pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            matched_gt_boxes_xyxy = ops.box_convert(matched_gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            
            giou = ops.generalized_box_iou(matched_pred_boxes_xyxy, matched_gt_boxes_xyxy)  # [n_matched]
            giou_loss = 1 - giou  # GIoU loss

            # Compute L1 loss for matched pairs
            l1_loss = torch.abs(matched_pred_boxes - matched_gt_boxes).sum(dim=-1)  # [n_matched]

            # Accumulate loss
            total_giou_loss += giou_loss.sum()
            total_l1_loss += l1_loss.sum()
            n_matched += len(row_ind)  # Number of matched boxes
        
        # Compute the final average losses
        if n_matched > 0:
            avg_giou_loss = total_giou_loss / n_matched
            avg_l1_loss = total_l1_loss / n_matched
        else:
            avg_giou_loss = torch.tensor(0.0)
            avg_l1_loss = torch.tensor(0.0)

        return avg_giou_loss, avg_l1_loss




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
        gt_gaussian_maps = generate_heatmap(ground_bbox.view(1,-1,4), self.search_size, self.stride)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes_vec = box_cxcywh_to_xyxy(result[1]).view(-1, 4)  # (N,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_cxcywh_to_xyxy(ground_bbox)[:, None, :].view(-1, 4).clamp(min=0.0,max=1.0)

        # compute giou and iou
        try:
            giou_loss, iou = self.giou_loss(pred_boxes_vec, gt_boxes_vec, is_absence)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # compute l1 loss
        if is_absence is not None:
            l1_loss = self.l1_loss(pred_boxes_vec * is_absence.view(-1, 1), gt_boxes_vec * is_absence.view(-1, 1))
        else:
            l1_loss = self.l1_loss(pred_boxes_vec, gt_boxes_vec)

        # compute location loss
        gt_gaussian_maps *= is_absence.view(-1, 1, 1, 1)
        location_loss = self.focal_loss(result[0], gt_gaussian_maps) + constraint_loss(result[0], is_absence)

        # weighted sum
        loss = self.loss_weight['iou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['cls'] * location_loss

        if val:
            true_positives, false_positives, scores, num_gt_objects = self.stastic_result(iou, result[0], is_absence)
            return {'total_loss':loss, 'results':result, 'true_positives':true_positives, 'false_positives':false_positives, 'scores':scores, 'num_gt_objects':num_gt_objects}
        else:
            return {'total_loss':loss, 'iou_loss':giou_loss, 'l1_loss':l1_loss, 'cls_loss':location_loss}

    def get_loss_multi(self, result, ground_bbox, box_nums):
        # gt gaussian map
        gt_gaussian_maps = generate_heatmap(ground_bbox.view(1,-1,4), self.search_size, self.stride)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes_vec = box_cxcywh_to_xyxy(result[1]).view(-1, 4)

    def set_loss(self, cfg):
        self.focal_loss = FocalLoss()
        self.giou_loss = giou_loss
        self.l1_loss = l1_loss
        self.search_size = cfg.DATA.SEARCH.SIZE
        self.stride = cfg.MODEL.BACKBONE.STRIDE
        self.loss_weight = {
            'cls': cfg.TRAIN.LOSS.CLS_WEIGHT,
            'l1': cfg.TRAIN.LOSS.BBOX_WEIGHT,
            'iou': cfg.TRAIN.LOSS.IOU_WEIGHT,
            'ctr': cfg.TRAIN.LOSS.CONTRA_WEIGHT
        }

    def export_onnx(self, folder_name, template_size, search_size, embedding_dim, opset_version=11, half_precision=False):
        template_branch = reparameterize_model(self.template_branch) 
        search_branch = reparameterize_model(self.search_branch)
        single_model = CoreModel(template_branch, search_branch)

        # Convert to half precision if requested
        dtype = torch.float16 if half_precision else torch.float32
        if half_precision:
            template_branch = template_branch.to('cuda').half()
            search_branch = search_branch.to('cuda').half()
            single_model = single_model.to('cuda').half()
            print('Using half precision (float16) for export')

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Create inputs with the appropriate precision
        device = 'cuda' if half_precision else 'cpu'
        dummy_input = (torch.randn(1, 3, template_size, template_size, dtype=dtype).to(device), 
                  torch.randn(1, 3, search_size, search_size, dtype=dtype).to(device))
        search_input = (torch.randn(1, 3, search_size, search_size, dtype=dtype).to(device), 
                   torch.randn(1, embedding_dim, 1, 1, dtype=dtype).to(device))
        template_input = torch.randn(1, 3, template_size, template_size, dtype=dtype).to(device)

        # Define input and output names
        input_names = ['template_img', 'search_img']
        output_names = ['score_map', 'scale_map', 'offset_map']
        input_names_template = ['template_img']
        output_names_template = ['template_embedding']
        input_names_search = ['search_img', 'template_embedding']
        
        print('Exporting fused model to ONNX...')
        torch.onnx.export(single_model, 
                          dummy_input, 
                          os.path.join(folder_name, 'umatcher.onnx'), 
                          opset_version=opset_version, 
                          input_names=input_names, 
                          output_names=output_names,
                          do_constant_folding=True
                          )

        print('Exporting template branch to ONNX...')
        torch.onnx.export(template_branch, 
                          template_input, 
                          os.path.join(folder_name, 'template_branch.onnx'), 
                          opset_version=opset_version, 
                          input_names=input_names_template, 
                          output_names=output_names_template,
                          do_constant_folding=True
                          )

        print('Exporting search branch to ONNX...')
        torch.onnx.export(search_branch, 
                          search_input, 
                          os.path.join(folder_name, 'search_branch.onnx'), 
                          opset_version=opset_version,
                          input_names=input_names_search,
                          output_names=output_names,
                          do_constant_folding=True
                          )
        print('Exporting complete!')

class CoreModel(nn.Module):
    def __init__(self, template_branch, search_branch):
        super(CoreModel, self).__init__()
        self.template_branch = template_branch
        self.search_branch = search_branch
    def forward(self, template_img, search_img):
        template_embedding = self.template_branch(template_img)
        result = self.search_branch(search_img, template_embedding)
        return result
        
    
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

def build_umatcher(cfg):
    model = UMatcher(cfg)
    model.template_branch = build_template_branch(cfg)
    model.search_branch = build_search_branch(cfg)
    model.set_loss(cfg)
    return model