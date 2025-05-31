#ifndef UMATCHER_H
#define UMATCHER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <net.h>

typedef struct MATCH_RESULT {
    float cx;
    float cy;
    float w;
    float h;
    float score;
} MATCH_RESULT;

class UMatcher {
public:
    UMatcher(const char* param_template, const char* bin_template,
        const char* param_search, const char* bin_search,
        int template_size, float template_scale,
        int search_size, float search_scale,
        int stride, int embdding_dim,
        bool useGPU);

    ~UMatcher();

    static UMatcher* matcher;
    ncnn::Net* TemplateBranch;
    ncnn::Net* SearchBranch;
    static bool hasGPU;

    int EmbeddingTemplate(cv::Mat &image, float* embedding);
    void SetTemplateEmbedding(float* embedding);
    std::vector<MATCH_RESULT> DecodeBBox(ncnn::Mat &score_map, ncnn::Mat &size_map, ncnn::Mat &offset_map, float score_threshold);
    std::vector<MATCH_RESULT> Match(cv::Mat &image, float* embedding, float score_threshold);
    std::vector<MATCH_RESULT> Match(cv::Mat &image, float score_threshold);

private:
    int template_size, search_size, stride, embdding_dim;
    float template_scale, search_scale;
    float* template_embedding;

};

#endif // UMATCHER_H


