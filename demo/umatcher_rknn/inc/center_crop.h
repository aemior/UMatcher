#ifndef CENTER_CROP_H
#define CENTER_CROP_H
#include <opencv2/opencv.hpp>

cv::Mat CenterCrop(const cv::Mat& image, cv::Rect target_rect, float crop_scale);

#endif // CENTER_CROP_H