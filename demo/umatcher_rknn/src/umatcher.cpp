#include <iostream>
#include "umatcher.h"

UMatcher::UMatcher(const char* template_model_path, const char* search_model_path,
                         int template_size, float template_scale,
                         int search_size, float search_scale,
                         int stride, int embdding_dim, bool debug_flag)
{
    memset(&TemplateBranch, 0, sizeof(rknn_app_context_t));
    memset(&SearchBranch, 0, sizeof(rknn_app_context_t));
    init_umatcher_model(template_model_path, &TemplateBranch, debug_flag);
    init_umatcher_model(search_model_path, &SearchBranch, debug_flag);
    SearchBranch.emb_dim = embdding_dim;

    this->template_size = template_size;
    this->search_size = search_size;
    this->template_scale = template_scale;
    this->search_scale = search_scale;
    this->stride = stride;
    this->embdding_dim = embdding_dim;
    this->template_embedding = new __fp16[embdding_dim];
    for (int i = 0; i < embdding_dim; i++)
    {
        this->template_embedding[i] = 0.0f; // Initialize template embedding to zero
    }
}

UMatcher::~UMatcher()
{
    release_umatcher_model(&TemplateBranch);
    release_umatcher_model(&SearchBranch);
    delete[] template_embedding;
}

int UMatcher::EmbeddingTemplate(cv::Mat &image, float* embedding)
{
    inference_umatcher_template(&TemplateBranch, image.data, embedding);

    // Copy embedding values
    // Normalize to unit vector
    float norm = 0.0f;
    for (int i = 0; i < embdding_dim; i++) {
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

std::vector<MATCH_RESULT> UMatcher::Match(cv::Mat &image, float* embedding, float score_threshold) {
    __fp16 f16_embedding[128];
    for (int i=0; i<128; ++i) {
        f16_embedding[i] = embedding[i];
    }
    std::vector<MATCH_RESULT> results;
    inference_umatcher_search(&SearchBranch, image.data, f16_embedding);
    results = DecodeBBox(&SearchBranch, score_threshold);
    return results;
}

std::vector<MATCH_RESULT> UMatcher::Match(cv::Mat &image, float score_threshold) {
    std::vector<MATCH_RESULT> results;
    inference_umatcher_search(&SearchBranch, image.data, template_embedding);
    results = DecodeBBox(&SearchBranch, score_threshold);
    return results;
}

std::vector<MATCH_RESULT> UMatcher::DecodeBBox(rknn_app_context_t* search_branch, float score_threshold) {
    std::vector<MATCH_RESULT> results;

    float* score_map = (float*)(search_branch->outputs[0].buf); // 16x16 hw
    float* size_map = (float*)(search_branch->outputs[1].buf); // 2x16x16 chw
    float* offset_map = (float*)(search_branch->outputs[2].buf); // 2x16x16 chw
    
    const int stride = this->stride;  // 从类成员获取stride
    const int feat_sz = 16;  // 特征图尺寸（假设宽高相等）

    // 遍历特征图每个位置
    for (int y = 0; y < feat_sz; ++y) {
        const float* score_row = score_map+(y*feat_sz);
        for (int x = 0; x < feat_sz; ++x) {
            float score = score_row[x];
            if (score <= score_threshold) continue;

            float offset_x = (offset_map+y*feat_sz)[x];
            float offset_y = (offset_map+feat_sz*feat_sz+y*feat_sz)[x];;

            // 计算中心点坐标（保持浮点精度）
            float cx = (x + offset_x) / stride;
            float cy = (y + offset_y) / stride;

            // 获取尺寸值
            float w = (size_map+y*feat_sz)[x];
            float h = (size_map+feat_sz*feat_sz+y*feat_sz)[x];;

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

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_umatcher_model(const char* model_path, rknn_app_context_t* app_ctx, bool debug_flag) {
    int ret;
    rknn_context ctx = 0;
    app_ctx->debug_flag = debug_flag;

    FILE* fp = fopen(model_path, "rb");
    if(fp == NULL) {
        printf("\033[1m\033[91m Failed \033[0m to open file %s\n", model_path);
    }

    fseek(fp, 0, SEEK_END);
    long model_size = ftell(fp);
    rewind(fp);

    uint8_t* model = (uint8_t*)malloc(model_size);
    if(model == NULL) {
        printf("Failed to malloc memory for model\n");
        fclose(fp);
    }

    size_t read_count = fread(model, 1, model_size, fp);
    if(read_count != model_size) {
        printf("Failed to read model\n");
        free(model);
        fclose(fp);
    }

    fclose(fp);
    if(debug_flag)
        ret = rknn_init(&ctx, model, model_size, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
    else
        ret = rknn_init(&ctx, model, model_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    if (debug_flag) {
        printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
        printf("input tensors:\n");
    }

    // Get Model Input Info
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        //        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        if(debug_flag) dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    if (debug_flag)
        printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        if(debug_flag) dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        if (debug_flag)
            printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        if (debug_flag)
            printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    if (debug_flag)
        printf("model input height=%d, width=%d, channel=%d\n",
            app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
    rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;

    // Create input tensor memory
    // default input type is int8 (normalize and quantize need compute in outside)
    // if set uint8, will fuse normalize and quantize to npu
    app_ctx->input_attrs[0].type = input_type;
    // default fmt is NHWC, npu only support NHWC in zero copy mode
    app_ctx->input_attrs[0].fmt = input_layout;

    app_ctx->input_mems[0] = rknn_create_mem(app_ctx->rknn_ctx, app_ctx->input_attrs[0].size_with_stride);

    if (io_num.n_input > 1) {
        app_ctx->input_attrs[1].type = RKNN_TENSOR_FLOAT16;
        // app_ctx->input_attrs[1].fmt = RKNN_TENSOR_NCHW;
        app_ctx->input_attrs[1].pass_through = 1;

        app_ctx->input_mems[1] = rknn_create_mem(app_ctx->rknn_ctx, app_ctx->input_attrs[1].size_with_stride);
    }

    // Create output tensor memory
    for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
        // default output type is depend on model, this require float32 to compute top5
        // allocate float32 output tensor
        int output_size;
        if(app_ctx->is_quant)
            output_size = app_ctx->output_attrs[i].n_elems * sizeof(char);
        else
            output_size = app_ctx->output_attrs[i].n_elems * sizeof(float);
        app_ctx->output_mems[i]  = rknn_create_mem(app_ctx->rknn_ctx, output_size);
    }

    // Set input tensor memory
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->input_mems[0], &app_ctx->input_attrs[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
    }
    if (io_num.n_input > 1) {
        ret = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->input_mems[1], &app_ctx->input_attrs[1]);
        if (ret < 0) {
            printf("rknn_set_io_mem 1 fail! ret=%d\n", ret);
            return -1;
        }
    }

    // Set output tensor memory
    for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
        // default output type is depend on model, this require float32 to compute top5
        if(app_ctx->is_quant)
            app_ctx->output_attrs[i].type = RKNN_TENSOR_INT8;
        else
            app_ctx->output_attrs[i].type = RKNN_TENSOR_FLOAT32;
        // set output memory and attribute
        ret = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->output_mems[i], &app_ctx->output_attrs[i]);
        if (ret < 0) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    return 0;
}

int release_umatcher_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_umatcher_template(rknn_app_context_t* app_ctx, uchar* img, float* embedding) {
    int ret;

    if ((!app_ctx) || !(img))
    {
        return -1;
    }

    // Copy input data to input tensor memory
    int width  = app_ctx->input_attrs[0].dims[2];
    int stride = app_ctx->input_attrs[0].w_stride;

    if (width == stride) {
        memcpy(app_ctx->input_mems[0]->virt_addr, img, width * app_ctx->input_attrs[0].dims[1] * app_ctx->input_attrs[0].dims[3]);
    } else {
        int height  = app_ctx->input_attrs[0].dims[1];
        int channel = app_ctx->input_attrs[0].dims[3];
        // copy from src to dst with stride
        uint8_t* src_ptr = img;
        uint8_t* dst_ptr = (uint8_t*)app_ctx->input_mems[0]->virt_addr;
        // width-channel elements
        int src_wc_elems = width * channel;
        int dst_wc_elems = stride * channel;
        for (int h = 0; h < height; ++h) {
            memcpy(dst_ptr, src_ptr, src_wc_elems);
            src_ptr += src_wc_elems;
            dst_ptr += dst_wc_elems;
        }
    }

    // Run
    ret               = rknn_run(app_ctx->rknn_ctx, NULL);
    if(app_ctx->debug_flag)
    {
        rknn_perf_detail perf_detail;
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
        printf("PERF DETAIL\n%s\n", perf_detail.perf_data);
    }

    // Get Output
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        app_ctx->outputs[i].index = i;
        app_ctx->outputs[i].want_float = (!app_ctx->is_quant);
        app_ctx->outputs[i].buf = app_ctx->output_mems[i]->virt_addr;
    }

    for (int i=0; i<128; ++i) {
        embedding[i] = *((float*)(app_ctx->outputs[0].buf)+i);
    }

    // Post Process
    // post_process(app_ctx, app_ctx->outputs, app_ctx->scale_w, app_ctx->scale_h, box_conf_threshold, nms_threshold, od_results);

    return ret;
}

int inference_umatcher_search(rknn_app_context_t* app_ctx, uchar* img, __fp16* embedding) {
    int ret;

    if ((!app_ctx) || !(img))
    {
        return -1;
    }

    // Copy input data to input tensor memory
    int width  = app_ctx->input_attrs[0].dims[2];
    int stride = app_ctx->input_attrs[0].w_stride;

    if (width == stride) {
        memcpy(app_ctx->input_mems[0]->virt_addr, img, width * app_ctx->input_attrs[0].dims[1] * app_ctx->input_attrs[0].dims[3]);
    } else {
        int height  = app_ctx->input_attrs[0].dims[1];
        int channel = app_ctx->input_attrs[0].dims[3];
        // copy from src to dst with stride
        uint8_t* src_ptr = img;
        uint8_t* dst_ptr = (uint8_t*)app_ctx->input_mems[0]->virt_addr;
        // width-channel elements
        int src_wc_elems = width * channel;
        int dst_wc_elems = stride * channel;
        for (int h = 0; h < height; ++h) {
            memcpy(dst_ptr, src_ptr, src_wc_elems);
            src_ptr += src_wc_elems;
            dst_ptr += dst_wc_elems;
        }
    }
    int width_e  = app_ctx->input_attrs[1].dims[1];
    int stride_e = app_ctx->input_attrs[1].dims[4];
    memcpy(app_ctx->input_mems[1]->virt_addr, embedding, sizeof(__fp16)*app_ctx->emb_dim);

    // Run
    ret               = rknn_run(app_ctx->rknn_ctx, NULL);
    if(app_ctx->debug_flag)
    {
        rknn_perf_detail perf_detail;
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
        printf("PERF DETAIL\n%s\n", perf_detail.perf_data);
    }

    // Get Output
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        app_ctx->outputs[i].index = i;
        app_ctx->outputs[i].want_float = (!app_ctx->is_quant);
        app_ctx->outputs[i].buf = app_ctx->output_mems[i]->virt_addr;
    }

    return ret;

}