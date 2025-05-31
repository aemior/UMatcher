import torch.utils.data as data
import torch

class StandardDataset(data.Dataset):
    def __init__(self, template_size=128, template_scale=2, search_size=256, search_scale=4, search_scale_std=2, max_object_num=5, dual_template=True, template_mask=False, argumentation=True):
        self.template_size = template_size
        self.template_scale = template_scale
        self.search_size = search_size
        self.search_scale = search_scale
        self.search_scale_std = search_scale_std
        self.max_object_num = max_object_num
        self.dual_template = dual_template
        self.template_mask = template_mask
        self.argumentation = argumentation

    def tlbr2chw(self, tlbr):
        # Conver the bounding box from top-left bottom-right ([x_min, y_min, x_max, y_max]) format to center height width format
        cx = (tlbr[0] + tlbr[2]) / 2
        cy = (tlbr[1] + tlbr[3]) / 2
        return [cx, cy, tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]]

    def tlwh2cwh(self, tlwh):
        # Conver the bounding box from top-left width height ([x_min, y_min, width, height]) format to center height width format
        cx = tlwh[0] + tlwh[2] / 2
        cy = tlwh[1] + tlwh[3] / 2
        return [cx, cy, tlwh[2], tlwh[3]]

    def __getitem__(self, idx) -> dict:
        """
        Args:
            idx (int): Index of the sequence, which stands for the idx-th object in the dataset.
        Returns:
            A dictionary containing the following items:
                - template_img_a (torch.Tensor): the template image. shape: (3, template_size, template_size)
                - template_img_b (torch.Tensor): the template image. shape: (3, template_size, template_size)
                - search_img (torch.Tensor): the search image. shape: (3, search_size, search_size)
                - ground_bbox (torch.Tensor): the ground truth bounding box. shape: (max_object_num, 4), format: [cx, cy, w, h] normalized by search_size
                - box_num (torch.Tensor): the number of boxes. shape: (1)
                - template_mask_a (torch.Tensor): the mask of the template image. shape: (1, search_size, search_size)
                - template_mask_b (torch.Tensor): the mask of the template image. shape: (1, search_size, search_size)
        """
        template_img_a = torch.zeros(3, self.template_size, self.template_size)
        template_img_b = torch.zeros(3, self.template_size, self.template_size)
        search_img = torch.zeros(3, self.search_size, self.search_size)
        ground_bbox = torch.zeros(self.max_object_num, 4)
        box_num = torch.zeros(1)
        template_mask_a = torch.zeros(1, self.search_size, self.search_size)
        template_mask_b = torch.zeros(1, self.search_size, self.search_size)

        return {
            'template_img_a': template_img_a,
            'template_mask_a': template_mask_a,
            'template_img_b': template_img_b,
            'template_mask_b': template_mask_b,
            'search_img': search_img,
            'ground_bbox': ground_bbox,
            'box_num': box_num
        }