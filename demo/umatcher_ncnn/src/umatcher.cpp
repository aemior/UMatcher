#include <iostream>
#include "umatcher.h"

UMatcher* UMatcher::matcher = nullptr;
bool UMatcher::hasGPU = false;

UMatcher::UMatcher(const char* param_template, const char* bin_template,
                         const char* param_search, const char* bin_search,
                         int template_size, float template_scale,
                         int search_size, float search_scale,
                         int stride, int embdding_dim, bool useGPU)
{
    TemplateBranch = new ncnn::Net();
    SearchBranch = new ncnn::Net();
    TemplateBranch->opt.use_vulkan_compute = useGPU;
    SearchBranch->opt.use_vulkan_compute = useGPU;

#if NCNN_VULKAN
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    TemplateBranch->opt.use_vulkan_compute = hasGPU && useGPU;
    SearchBranch->opt.use_vulkan_compute = hasGPU && useGPU;
    TemplateBranch->opt.use_fp16_arithmetic = true;
    SearchBranch->opt.use_fp16_arithmetic = true;
    TemplateBranch->load_param(param_template);
    TemplateBranch->load_model(bin_template);
    SearchBranch->load_param(param_search);
    SearchBranch->load_model(bin_search);

    this->template_size = template_size;
    this->search_size = search_size;
    this->template_scale = template_scale;
    this->search_scale = search_scale;
    this->stride = stride;
    this->embdding_dim = embdding_dim;
    this->template_embedding = new float[embdding_dim];
    for (int i = 0; i < embdding_dim; i++)
    {
        this->template_embedding[i] = 0.0f; // Initialize template embedding to zero
    }
}

UMatcher::~UMatcher()
{
    delete TemplateBranch;
    delete SearchBranch;
    delete[] template_embedding;
}

int UMatcher::EmbeddingTemplate(cv::Mat &image, float* embedding)
{
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, template_size, template_size);
    const float mean_vals[3] = { 0.0f, 0.0f, 0.0f }; // Assuming mean is zero for simplicity
    const float norm_vals[3] = { 1/255.0f, 1/255.0f, 1/255.0f };
    input.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = TemplateBranch->create_extractor();
    ex.input("in0", input);

    ncnn::Mat out;
    ex.extract("out0", out);

    if (out.w != 1 || out.h != 1 || out.c != embdding_dim) {
        std::cout << "Error: Invalid output dimensions from template branch." << std::endl;
        std::cout << "Expected: 1x1x" << embdding_dim << ", got: " << out.w << "x" << out.h << "x" << out.c << std::endl;
        return -1; // Invalid output dimensions
    }

    // Copy embedding values
    // Normalize to unit vector
    float norm = 0.0f;
    for (int i = 0; i < embdding_dim; i++) {
        embedding[i] = out.channel(i)[0];
        norm += embedding[i] * embedding[i];
    }
    norm = sqrt(norm)+1e-12f;
    for (int i = 0; i < embdding_dim; i++) {
        embedding[i] /= norm;
    }

    return 0; // Success
}

void UMatcher::SetTemplateEmbedding(float* embedding)
{
    for (int i = 0; i < embdding_dim; i++) {
        template_embedding[i] = embedding[i];
    }
}

void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

std::vector<MATCH_RESULT> UMatcher::Match(cv::Mat &image, float* embedding, float score_threshold) {
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, search_size, search_size);
    const float mean_vals[3] = { 0.0f, 0.0f, 0.0f }; // Assuming mean is zero for simplicity
    const float norm_vals[3] = { 1/255.0f, 1/255.0f, 1/255.0f };
    input.substract_mean_normalize(mean_vals, norm_vals);
    
    ncnn::Mat embed = ncnn::Mat(embdding_dim, (void*)embedding).reshape(1, 1, embdding_dim);

    ncnn::Extractor ex = SearchBranch->create_extractor();
    ex.input("in0", input);
    ex.input("in1", embed);

    ncnn::Mat score, scale, offset;
    ex.extract("out0", score);
    ex.extract("out1", scale);
    ex.extract("out2", offset);

    // std::cout << score.w << 'x' << score.h << 'x' << score.c << std::endl;
    // std::cout << offset.w << 'x' << offset.h << 'x' << offset.c << std::endl;
    // std::cout << scale.w << 'x' << scale.h << 'x' << scale.c << std::endl;

    // save a score heatmap for debugging
    // cv::Mat score_heatmap(score.h, score.w, CV_32FC1, score.data);
    // cv::Mat score_heatmap_normalized;
    // cv::normalize(score_heatmap, score_heatmap_normalized, 0, 255, cv::NORM_MINMAX);
    // cv::Mat score_heatmap_8u;
    // score_heatmap_normalized.convertTo(score_heatmap_8u, CV_8UC1);
    // cv::applyColorMap(score_heatmap_8u, score_heatmap_8u, cv::COLORMAP_JET);
    // cv::imwrite("score_heatmap.png", score_heatmap_8u);

    // pretty_print(offset);
    std::vector<MATCH_RESULT> results = DecodeBBox(score, scale, offset, score_threshold);
    return results;
}

/*
std::vector<MATCH_RESULT> UMatcher::DecodeBBox(ncnn::Mat &score_map, ncnn::Mat &size_map, ncnn::Mat &offset_map, float score_threshold) {
    // The shape of the score_map, size_map, and offset_map should be (w,h,c):
    // score_map: 16x16x1
    // size_map: 16x16x2
    // offset_map: 16x16x2
    std::vector<MATCH_RESULT> results;



}
*/

std::vector<MATCH_RESULT> UMatcher::DecodeBBox(ncnn::Mat &score_map, ncnn::Mat &size_map, ncnn::Mat &offset_map, float score_threshold) {
    std::vector<MATCH_RESULT> results;
    
    const int stride = this->stride;  // 从类成员获取stride
    const int feat_sz = score_map.w;  // 特征图尺寸（假设宽高相等）

    // 遍历特征图每个位置
    for (int y = 0; y < feat_sz; ++y) {
        const float* score_row = score_map.row(y);
        for (int x = 0; x < feat_sz; ++x) {
            float score = score_row[x];
            if (score <= score_threshold) continue;

            // 获取offset值（注意ncnn::Mat的通道顺序）
            float offset_x = offset_map.channel(0).row(y)[x];
            float offset_y = offset_map.channel(1).row(y)[x];

            // 计算中心点坐标（保持浮点精度）
            float cx = (x + offset_x) / stride;
            float cy = (y + offset_y) / stride;

            // 获取尺寸值
            float w = size_map.channel(0).row(y)[x];
            float h = size_map.channel(1).row(y)[x];

            // 填充结果（注意类型转换可能导致精度丢失）
            MATCH_RESULT res;
            res.cx = cx * search_size;
            res.cy = cy * search_size;
            res.w = w * search_size;
            res.h = h * search_size;
            res.score = score;
            
            results.push_back(res);
        }
    }

    return results;
}