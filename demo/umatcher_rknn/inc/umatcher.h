#ifndef UMATCHER_H
#define UMATCHER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "rknn_api.h"

typedef struct MATCH_RESULT {
    float cx;
    float cy;
    float w;
    float h;
    float score;
} MATCH_RESULT;

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    rknn_tensor_mem* input_mems[2];
    rknn_tensor_mem* output_mems[3];
    rknn_output outputs[3];

    int model_channel;
    int model_width;
    int model_height;
    int emb_dim;
    bool debug_flag;
    bool is_quant;
} rknn_app_context_t;

int init_umatcher_model(const char* model_path, rknn_app_context_t* app_ctx, bool debug_flag);

int release_umatcher_model(rknn_app_context_t* app_ctx);

int inference_umatcher_template(rknn_app_context_t* app_ctx, uchar* img, float* embedding);

int inference_umatcher_search(rknn_app_context_t* app_ctx, uchar* img, __fp16* embedding);

class UMatcher {
public:
    UMatcher(const char* template_model_path, const char* search_model_path,
        int template_size, float template_scale,
        int search_size, float search_scale,
        int stride, int embdding_dim,
        bool quant);

    ~UMatcher();

    static UMatcher* matcher;
    rknn_app_context_t TemplateBranch;
    rknn_app_context_t SearchBranch;

    int EmbeddingTemplate(cv::Mat &image, float* embedding);
    void SetTemplateEmbedding(float* embedding);
    std::vector<MATCH_RESULT> DecodeBBox(rknn_app_context_t* search_branch, float score_threshold);
    std::vector<MATCH_RESULT> Match(cv::Mat &image, float* embedding, float score_threshold);
    std::vector<MATCH_RESULT> Match(cv::Mat &image, __fp16* embedding, float score_threshold);
    std::vector<MATCH_RESULT> Match(cv::Mat &image, float score_threshold);

private:
    int template_size, search_size, stride, embdding_dim;
    float template_scale, search_scale;
    __fp16* template_embedding;

};

#endif // UMATCHER_H


