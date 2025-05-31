import numpy as np
import torch


def generate_heatmap(bboxes, patch_size=320, stride=16):
    """
    Generate ground truth heatmap same as CenterNet
    Args:
        bboxes (torch.Tensor): shape of [num_search, bs, 4]

    Returns:
        gaussian_maps: list of generated heatmap

    """
    gaussian_maps = []
    heatmap_size = patch_size // stride
    for i in range(bboxes.shape[0]):
        gt_scoremap = torch.zeros(heatmap_size, heatmap_size)
        bboxe = bboxes[i] / patch_size * heatmap_size - 0.5
        gt_wh = bboxe[2:]
        radius = CenterNetHeatMap.get_gaussian_radius(gt_wh, 0.3)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        centers_int = torch.round(bboxe[:2])
        CenterNetHeatMap.draw_gaussian(gt_scoremap, centers_int, radius)
        gaussian_maps.append(gt_scoremap.unsqueeze(0))
    
    return torch.stack(gaussian_maps, dim=0)


def generate_multi_heatmap(bboxes, box_nums, patch_size=256, stride=16):
    """
    Generate ground truth heatmap same as CenterNet
    Args:
        bboxes (torch.Tensor): shape of [batch_size, num_search, 4]
        box_nums (torch.Tensor): shape of [batch_size]
        patch_size (int): size of patch image
        stride (int): stride of heatmap
    Returns:
        gt_scoremaps: list of generated heatmap [batch_size, 1, heatmap_size, heatmap_size]
    """
    heatmap_size = patch_size // stride
    bs = bboxes.shape[0]
    gt_scoremaps = []
    for single_patch_bboxes, box_num in zip(bboxes, box_nums):
        gt_scoremap = torch.zeros(heatmap_size, heatmap_size)
        bboxes = single_patch_bboxes * heatmap_size - 0.5
        gt_wh = bboxes[:, 2:]
        radius = CenterNetHeatMap.get_gaussian_radius(gt_wh, 0.3)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        centers_int = torch.round(bboxes[:, :2])
        for i in range(box_num):
            CenterNetHeatMap.draw_gaussian(gt_scoremap, centers_int[i], radius[i])
        gt_scoremaps.append(gt_scoremap.unsqueeze(0))

    # mearge all scoremaps into bs x 1 x heatmap_size x heatmap_size and return
    return torch.stack(gt_scoremaps, dim=0)


    



class CenterNetHeatMap(object):
    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = CenterNetHeatMap.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            CenterNetHeatMap.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        # box_tensor = torch.Tensor(box_size)
        box_tensor = box_size
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m: m + 1, -n: n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CenterNetHeatMap.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top: y + bottom, x - left: x + right]
        masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top: y + bottom, x - left: x + right] = masked_fmap
        # return fmap


def compute_grids(features, strides):
    """
    grids regret to the input image size
    """
    grids = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        shifts_x = torch.arange(
            0, w * strides[level],
            step=strides[level],
            dtype=torch.float32, device=feature.device)
        shifts_y = torch.arange(
            0, h * strides[level],
            step=strides[level],
            dtype=torch.float32, device=feature.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                          strides[level] // 2
        grids.append(grids_per_level)
    return grids


def get_center3x3(locations, centers, strides, range=3):
    '''
    Inputs:
        locations: M x 2
        centers: N x 2
        strides: M
    '''
    range = (range - 1) / 2
    M, N = locations.shape[0], centers.shape[0]
    locations_expanded = locations.view(M, 1, 2).expand(M, N, 2)  # M x N x 2
    centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # M x N x 2
    strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)  # M x N
    centers_discret = ((centers_expanded / strides_expanded).int() * strides_expanded).float() + \
                      strides_expanded / 2  # M x N x 2
    dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
    dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
    return (dist_x <= strides_expanded[:, :, 0] * range) & \
           (dist_y <= strides_expanded[:, :, 0] * range)


def get_pred(score_map_ctr, size_map, offset_map, feat_size):
    max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)

    idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
    size = size_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
    offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

    return size * feat_size, offset