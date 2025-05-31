#include "udetector.h"


UDetector::UDetector(const char* param_template, const char* bin_template,
    const char* param_search, const char* bin_search,
    int template_size, float template_scale,
    int search_size, float search_scale,
    int stride, int embdding_dim,
    bool useGPU) {
    umatcher = new UMatcher(param_template, bin_template,
        param_search, bin_search,
        template_size, template_scale,
        search_size, search_scale,
        stride, embdding_dim, useGPU);
    
    this->template_size = template_size;
    this->search_size = search_size;
    this->template_scale = template_scale;
    this->search_scale = search_scale;
    this->stride = stride;
    this->embdding_dim = embdding_dim;
    this->template_embedding = new float[embdding_dim];

    for (int i = 0; i < embdding_dim; i++) {
        this->template_embedding[i] = 0.0f; // Initialize template embedding to zero
    }

}

UDetector::~UDetector() {
    delete umatcher;
    delete[] template_embedding;
}

void UDetector::SetTemplate(cv::Mat &image, cv::Rect &bbox) {
    // Crop the template image based on the bounding box
    cv::Mat cropped_template = CenterCrop(image, bbox, template_scale);
    float original_w = cropped_template.cols;
    scale_factor = static_cast<float>(template_size) / original_w;
    cv::resize(cropped_template, cropped_template, cv::Size(template_size, template_size));
    umatcher->EmbeddingTemplate(cropped_template, template_embedding);
}

std::vector<MATCH_RESULT> UDetector::NMS(std::vector<MATCH_RESULT> &input, float nms_thr) {
    if (input.empty()) {
        return input;
    }
    
    // Sort by score in descending order
    std::sort(input.begin(), input.end(), [](const MATCH_RESULT &a, const MATCH_RESULT &b) {
        return a.score > b.score;
    });
    
    std::vector<MATCH_RESULT> result;
    std::vector<bool> suppressed(input.size(), false);
    
    for (size_t i = 0; i < input.size(); i++) {
        if (suppressed[i]) {
            continue;
        }
        
        result.push_back(input[i]);
        
        // Calculate IoU with remaining boxes
        for (size_t j = i + 1; j < input.size(); j++) {
            if (suppressed[j]) {
                continue;
            }
            
            // Convert center format to corner format for IoU calculation
            float x1_i = input[i].cx - input[i].w / 2.0f;
            float y1_i = input[i].cy - input[i].h / 2.0f;
            float x2_i = input[i].cx + input[i].w / 2.0f;
            float y2_i = input[i].cy + input[i].h / 2.0f;
            
            float x1_j = input[j].cx - input[j].w / 2.0f;
            float y1_j = input[j].cy - input[j].h / 2.0f;
            float x2_j = input[j].cx + input[j].w / 2.0f;
            float y2_j = input[j].cy + input[j].h / 2.0f;
            
            // Calculate intersection
            float inter_x1 = std::max(x1_i, x1_j);
            float inter_y1 = std::max(y1_i, y1_j);
            float inter_x2 = std::min(x2_i, x2_j);
            float inter_y2 = std::min(y2_i, y2_j);
            
            float inter_area = 0.0f;
            if (inter_x2 > inter_x1 && inter_y2 > inter_y1) {
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
            }
            
            // Calculate union
            float area_i = input[i].w * input[i].h;
            float area_j = input[j].w * input[j].h;
            float union_area = area_i + area_j - inter_area;
            
            // Calculate IoU
            float iou = 0.0f;
            if (union_area > 0) {
                iou = inter_area / union_area;
            }
            
            // Suppress if IoU is above threshold
            if (iou > nms_thr) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

std::vector<MATCH_RESULT> UDetector::Detect(cv::Mat &image, float threshold, std::vector<float> pyramid, float overlap) {
    std::vector<MATCH_RESULT> all_boxes;
    int original_h = image.rows;
    int original_w = image.cols;
    
    for (size_t scale_idx = 0; scale_idx < pyramid.size(); scale_idx++) {
        float scale = scale_factor * pyramid[scale_idx];
        
        int scaled_w = static_cast<int>(original_w * scale);
        int scaled_h = static_cast<int>(original_h * scale);
        
        // Skip if both dimensions are smaller than search size
        if (scaled_w < search_size && scaled_h < search_size) {
            continue;
        }
        
        cv::Mat scaled_img;
        cv::resize(image, scaled_img, cv::Size(scaled_w, scaled_h));
        
        // Pad the image to ensure full coverage
        int pad_w = std::max(0, search_size - scaled_w);
        int pad_h = std::max(0, search_size - scaled_h);
        
        if (pad_w > 0 || pad_h > 0) {
            cv::Mat padded_img = cv::Mat::zeros(scaled_h + pad_h, scaled_w + pad_w, scaled_img.type());
            scaled_img.copyTo(padded_img(cv::Rect(0, 0, scaled_w, scaled_h)));
            scaled_img = padded_img;
            scaled_w += pad_w;
            scaled_h += pad_h;
        }
        
        int step = static_cast<int>(search_size * (1 - overlap));
        step = std::max(1, step);
        
        std::vector<int> x_starts;
        int current_x = 0;
        while (current_x <= scaled_w - search_size) {
            x_starts.push_back(current_x);
            current_x += step;
        }
        if (!x_starts.empty() && x_starts.back() + search_size < scaled_w) {
            x_starts.push_back(scaled_w - search_size);
        } else if (x_starts.empty() && scaled_w >= search_size) {
            x_starts.push_back(0);
        }
        
        std::vector<int> y_starts;
        int current_y = 0;
        while (current_y <= scaled_h - search_size) {
            y_starts.push_back(current_y);
            current_y += step;
        }
        if (!y_starts.empty() && y_starts.back() + search_size < scaled_h) {
            y_starts.push_back(scaled_h - search_size);
        } else if (y_starts.empty() && scaled_h >= search_size) {
            y_starts.push_back(0);
        }
        
        for (int x_start : x_starts) {
            for (int y_start : y_starts) {
                cv::Mat window;
                scaled_img(cv::Rect(x_start, y_start, search_size, search_size)).copyTo(window);
                std::vector<MATCH_RESULT> pred_bboxes = umatcher->Match(window, template_embedding, threshold);
                if (pred_bboxes.empty()) {
                    continue;
                }
                
                for (size_t i = 0; i < pred_bboxes.size(); i++) {
                    MATCH_RESULT bbox = pred_bboxes[i];
                    
                    bbox.cx += x_start;
                    bbox.cy += y_start;

                    bbox.cx /= scale;
                    bbox.cy /= scale;
                    bbox.w /= scale;
                    bbox.h /= scale;

                    all_boxes.push_back(bbox);
                }
            }
        }
    }
    
    if (all_boxes.empty()) {
        return all_boxes;
    }
    
    // Apply NMS
    all_boxes = NMS(all_boxes, overlap);
    
    return all_boxes;
}

cv::Mat UDetector::DrawDetections(cv::Mat &image, std::vector<MATCH_RESULT> &detections) {
    cv::Mat output_image = image.clone();
    
    for (const auto &det : detections) {
        int center_x = static_cast<int>(det.cx);
        int center_y = static_cast<int>(det.cy);
        int width = static_cast<int>(det.w);
        int height = static_cast<int>(det.h);
        
        int x = center_x - width / 2;
        int y = center_y - height / 2;
        cv::Rect bbox(x, y, width, height);
        
        // Draw rectangle
        cv::rectangle(output_image, bbox, cv::Scalar(0, 255, 0), 2);
        
        // Draw score
        std::string score_text = std::to_string(det.score);
        cv::putText(output_image, score_text, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }
    
    return output_image;
}