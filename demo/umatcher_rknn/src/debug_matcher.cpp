#include <iostream>
#include "umatcher.h"

// 绘制目标框并保存图像
void draw_and_save_boxes(cv::Mat& image, const std::vector<MATCH_RESULT>& results) {
    // 遍历所有检测结果
    for (const auto& res : results) {
        // 计算矩形框的左上角和右下角坐标[6,8](@ref)
        cv::Point pt1(static_cast<int>(res.cx - res.w/2), 
                     static_cast<int>(res.cy - res.h/2));
        cv::Point pt2(static_cast<int>(res.cx + res.w/2), 
                     static_cast<int>(res.cy + res.h/2));
        
        // 根据得分设置颜色（得分越高越绿）[7](@ref)
        cv::Scalar color(0, static_cast<int>(res.score * 255), 0); // BGR格式
        
        // 在图像上绘制矩形框[8](@ref)
        cv::rectangle(image, pt1, pt2, color, 2, cv::LINE_AA);  // 线宽2，抗锯齿
    }
    
    // 保存结果图像[5,11](@ref)
    if (!cv::imwrite("result.png", image)) {
        std::cerr << "保存图像失败！" << std::endl;
    }
}

int main() {
    UMatcher matcher(
        "../data/template_branch.rknn",
        "../data/search_branch.rknn",
        128,2,
        256,4,
        16,128,false
    );
    cv::Mat img = cv::imread("../data/debug_template.png");

    float gt_embedding[128] = {0.0, 0.1934814453125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10174560546875, 0.0, 0.0732421875, 0.0, 0.04327392578125, 0.0, 0.0, 0.045654296875, 0.0, 0.0, 0.07366943359375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.224853515625, 0.0, 0.09075927734375, 0.0, 0.0679931640625, 0.0, 0.13525390625, 0.260498046875, 0.0, 0.0, 0.0, 0.2264404296875, 0.0, 0.0, 0.0270538330078125, 0.0128021240234375, 0.0, 0.0, 0.016571044921875, 0.04302978515625, 0.0, 0.0, 0.0, 0.0, 0.041015625, 0.007640838623046875, 0.0, 0.0, 0.0, 0.026763916015625, 0.0, 0.0, 0.1849365234375, 0.0, 0.059051513671875, 0.110107421875, 0.0, 0.0, 0.46240234375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15185546875, 0.0, 0.0, 0.0797119140625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25341796875, 0.0, 0.0, 0.0, 0.1328125, 0.146728515625, 0.066650390625, 0.0, 0.11541748046875, 0.0, 0.03668212890625, 0.0, 0.0, 0.0, 0.0, 0.00852203369140625, 0.0, 0.51611328125, 0.0231781005859375, 0.01433563232421875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08013916015625, 0.150390625, 0.0, 0.0, 0.0126190185546875, 0.0, 0.0, 0.1368408203125, 0.0, 0.0, 0.0, 0.0, 0.0};

    float embedding[128];
    for (int i=0; i<128; ++i) {
        embedding[i]=0;
    }

    std::cout << "GT:\n";
    for (int i=0; i<128; ++i) {
        std::cout << gt_embedding[i] << ", ";
    }
    std::cout << std::endl;


    matcher.EmbeddingTemplate(img, embedding);

    std::cout << "PD:\n";
    for (int i=0; i<128; ++i) {
        std::cout << embedding[i] << ", ";
    }
    std::cout << std::endl;

    // 初始化计算变量
    float dot_product = 0.0f;
    float norm_gt = 0.0f;
    float norm_emb = 0.0f;

    // 计算点积和模长平方
    for (int i = 0; i < 128; ++i) {
        dot_product += gt_embedding[i] * embedding[i];
        norm_gt += gt_embedding[i] * gt_embedding[i];
        norm_emb += embedding[i] * embedding[i];
    }

    // 计算模长
    float denom = std::sqrt(norm_gt) * std::sqrt(norm_emb);
    
    // 避免除零错误
    float epsilon = std::numeric_limits<float>::epsilon();
    float similarity = (denom < epsilon) ? 0.0f : dot_product / denom;

    // 打印结果（保留4位小数）
    std::cout.precision(4);
    std::cout << "余弦相似度: " << std::fixed << similarity << std::endl;
    cv::Mat s_img = cv::imread("../data/debug_search.jpg");

    auto result = matcher.Match(s_img, embedding, 0.5);
    std::cout << "RES SIZE" << result.size() << std::endl;
    draw_and_save_boxes(s_img, result);


    return 0;
}