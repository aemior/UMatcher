#ifndef UDETECTOR_H
#define UDETECTOR_H

#include "umatcher.h"
#include "center_crop.h"

class UDetector{
public:
    UDetector(const char* param_template, const char* bin_template,
        const char* param_search, const char* bin_search,
        int template_size, float template_scale,
        int search_size, float search_scale,
        int stride, int embdding_dim,
        bool useGPU);

    ~UDetector();

    void SetTemplate(cv::Mat &image, cv::Rect &bbox);
    float scale_factor;
    std::vector<MATCH_RESULT> Detect(cv::Mat &image, float threshold, std::vector<float> pyramid, float overlap);
    cv::Mat DrawDetections(cv::Mat &image, std::vector<MATCH_RESULT> &detections);

private:
    int template_size, search_size, stride, embdding_dim;
    float template_scale, search_scale;
    float* template_embedding;
    std::vector<MATCH_RESULT> NMS(std::vector<MATCH_RESULT> &input, float nms_thr);
    UMatcher* umatcher;
};

#endif //UDETECTOR_H