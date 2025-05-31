#include "center_crop.h"

cv::Mat CenterCrop(const cv::Mat& image, cv::Rect target_rect, float crop_scale) {
    int cx = target_rect.x + target_rect.width / 2;
    int cy = target_rect.y + target_rect.height / 2;
    int w = target_rect.width;
    int h = target_rect.height;
    
    int size = static_cast<int>(std::sqrt(w * h) * crop_scale);
    int x1 = cx - size / 2;
    int y1 = cy - size / 2;
    int x2 = x1 + size;
    int y2 = y1 + size;
    
    int x1_pad = std::max(0, -x1);
    int y1_pad = std::max(0, -y1);
    int x2_pad = std::max(x2 - image.cols, 0);
    int y2_pad = std::max(y2 - image.rows, 0);
    
    int roi_x1 = std::max(x1, 0);
    int roi_y1 = std::max(y1, 0);
    int roi_x2 = std::min(x2, image.cols);
    int roi_y2 = std::min(y2, image.rows);
    
    cv::Mat roi = image(cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1));
    
    cv::Mat crop_img;
    cv::copyMakeBorder(roi, crop_img, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_REPLICATE);
    
    return crop_img;
}