#ifndef UTRACKER_H
#define UTRACKER_H

#include "umatcher.h"
#include "center_crop.h"

class UTracker {
public:
    UTracker(const char* template_model_path, const char* search_model_path,
             int template_size, float template_scale,
             int search_size, float search_scale,
             int stride, int embdding_dim,
             bool useGPU);

    ~UTracker();

    void Init(cv::Mat &image, cv::Rect &bbox);
    MATCH_RESULT Track(cv::Mat &image);
    cv::Mat DrawResult(cv::Mat &image);
    float CalculateIoU(const MATCH_RESULT& boxA, const MATCH_RESULT& boxB);
private:
    int template_size, search_size, stride, embdding_dim;
    float template_scale, search_scale;
    float* template_embedding;
    MATCH_RESULT last_pos;
    void UpdateTemplate(cv::Mat &image, cv::Rect& bbox);
    MATCH_RESULT TransPos(MATCH_RESULT input_box, int w_i, int h_i);
    MATCH_RESULT MatchPos(std::vector<MATCH_RESULT>& candinate);
    UMatcher* umatcher;
    
    // KalmanFilter
    float alpha_kf = 0.5f; // weight for KF-IoU
    int tau_kf = 10;       // threshold frames to use KF
    int success_count = 0; // consecutive successful updates
    
    cv::KalmanFilter kf;
    void InitKalmanFilter();
    cv::Mat PredictKalman();
    void UpdateKalman(const cv::Rect& bbox);
};

#endif //UTRACKER_H