#include "utracker.h"

UTracker::UTracker(const char* template_model_path, const char* search_model_path,
                   int template_size, float template_scale,
                   int search_size, float search_scale,
                   int stride, int embdding_dim,
                   bool useGPU) {
    umatcher = new UMatcher(template_model_path, search_model_path,
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

UTracker::~UTracker() {
    delete umatcher;
    delete[] template_embedding;
}

void UTracker::InitKalmanFilter() {
    // Initialize Kalman Filter with 8 state variables (x, y, w, h, vx, vy, vw, vh)
    kf.init(8, 4);
    kf.transitionMatrix = (cv::Mat_<float>(8, 8) << 1, 0, 0, 0, 1, 0, 0, 0,
                                                    0, 1, 0, 0, 0, 1, 0, 0,
                                                    0, 0, 1, 0, 0, 0, 1, 0,
                                                    0, 0, 0, 1, 0, 0, 0, 1,
                                                    0, 0, 0, 0, 1, 0, 0, 0,
                                                    0, 0, 0, 0, 0, 1, 0, 0,
                                                    0, 0, 0, 0, 0, 0, 1, 0,
                                                    0, 0, 0, 0, 0, 0, 0, 1);
    kf.measurementMatrix = (cv::Mat_<float>(4, 8) << 1, 0, 0, 0, 0, 0, 0, 0,
                                                     0, 1, 0, 0, 0, 0, 0, 0,
                                                     0, 0, 1, 0, 0, 0, 0, 0,
                                                     0, 0, 0, 1, 0, 0, 0, 0);
    kf.processNoiseCov = (cv::Mat_<float>(8, 8) << 1e-2, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 1e-2, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 1e-2, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 1e-2, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 1e-3, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 1e-3, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 1e-3, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 1e-3);
    kf.measurementNoiseCov = (cv::Mat_<float>(4, 4) << 1e-1, 0, 0, 0,
                                                       0, 1e-1, 0, 0,
                                                       0, 0, 1e-1, 0,
                                                       0, 0, 0, 1e-1);
}

void UTracker::Init(cv::Mat &image, cv::Rect &bbox) {
    last_pos.cx = bbox.x + bbox.width / 2;
    last_pos.cy = bbox.y + bbox.height / 2;
    last_pos.w = bbox.width;
    last_pos.h = bbox.height;
    last_pos.score = 1.0f; // Initial score can be set to 1.0 or any other value
    success_count = 0; // Reset success count
    // Initialize Kalman Filter
    InitKalmanFilter();
    // Set the initial state of the Kalman Filter
    kf.statePre = (cv::Mat_<float>(8, 1) << last_pos.cx, last_pos.cy, last_pos.w, last_pos.h,
                                             0, 0, 0, 0); // Initial velocity is set to ze
    kf.statePost = kf.statePre.clone();
    // Update the template embedding
    UpdateTemplate(image, bbox);
}

void UTracker::UpdateTemplate(cv::Mat &image, cv::Rect &bbox) {
    // Crop the template image based on the bounding box
    cv::Mat cropped_template = CenterCrop(image, bbox, template_scale);
    // Resize the cropped template to the template size
    cv::resize(cropped_template, cropped_template, cv::Size(template_size, template_size));
    // Update the template embedding
    umatcher->EmbeddingTemplate(cropped_template, template_embedding);
    umatcher->SetTemplateEmbedding(template_embedding);
}

MATCH_RESULT UTracker::TransPos(MATCH_RESULT input_box, int w_i, int h_i) {
    float w_f = w_i / static_cast<float>(search_size);
    float h_f = h_i / static_cast<float>(search_size);
    MATCH_RESULT output_box;
    output_box.cx = (last_pos.cx - w_i*0.5) + input_box.cx * w_f;
    output_box.cy = (last_pos.cy - h_i*0.5) + input_box.cy * h_f;
    output_box.w = input_box.w * w_f;
    output_box.h = input_box.h * h_f;
    output_box.score = input_box.score; // Keep the score unchanged
    return output_box;
}

MATCH_RESULT UTracker::Track(cv::Mat &image) {
    cv::Rect last_rect(last_pos.cx - last_pos.w / 2, last_pos.cy - last_pos.h / 2,
                             last_pos.w, last_pos.h);
    // Crop the search region around the last position
    cv::Mat search_region = CenterCrop(image, last_rect, search_scale);
    int w_i = search_region.cols;
    int h_i = search_region.rows;
    // Resize the search region to the search size
    cv::resize(search_region, search_region, cv::Size(search_size, search_size));
    // Get the candidate positions from the matcher
    std::vector<MATCH_RESULT> candidates_raw = umatcher->Match(search_region, 0.1);
    std::vector<MATCH_RESULT> candidates;
    for (const auto &cand : candidates_raw) {
        candidates.push_back(TransPos(cand, w_i, h_i));
    }
    return MatchPos(candidates);
}

// Helper function to calculate IoU
float UTracker::CalculateIoU(const MATCH_RESULT& boxA, const MATCH_RESULT& boxB) {
    float xA1 = boxA.cx - boxA.w / 2, yA1 = boxA.cy - boxA.h / 2;
    float xA2 = boxA.cx + boxA.w / 2, yA2 = boxA.cy + boxA.h / 2;
    float xB1 = boxB.cx - boxB.w / 2, yB1 = boxB.cy - boxB.h / 2;
    float xB2 = boxB.cx + boxB.w / 2, yB2 = boxB.cy + boxB.h / 2;
    
    float ix1 = std::max(xA1, xB1), iy1 = std::max(yA1, yB1);
    float ix2 = std::min(xA2, xB2), iy2 = std::min(yA2, yB2);
    float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float union_area = boxA.w * boxA.h + boxB.w * boxB.h - inter;
    return union_area > 0 ? inter / union_area : 0.0f;
}

MATCH_RESULT UTracker::MatchPos(std::vector<MATCH_RESULT>& candidates) {
    if (candidates.empty()) {
        success_count = 0;
        MATCH_RESULT result = last_pos;
        result.score = 0.0f;
        return result;
    }

    bool use_kf = (success_count >= tau_kf);
    MATCH_RESULT kf_pred;
    
    if (use_kf) {
        // Predict next state
        cv::Mat pred = kf.predict();
        kf_pred.cx = pred.at<float>(0, 0);
        kf_pred.cy = pred.at<float>(1, 0);
        kf_pred.w = pred.at<float>(2, 0);
        kf_pred.h = pred.at<float>(3, 0);
    }

    float best_score = -1.0f;
    MATCH_RESULT best_det;
    
    for (const auto& cand : candidates) {
        float combined_score;
        if (use_kf) {
            float kf_iou = CalculateIoU(kf_pred, cand);
            combined_score = alpha_kf * kf_iou + (1 - alpha_kf) * cand.score;
        } else {
            combined_score = cand.score;
        }
        
        if (combined_score > best_score) {
            best_score = combined_score;
            best_det = cand;
        }
    }

    // Update success counter and Kalman filter
    if (best_score > 0.2f) {
        success_count++;
        // Correct with measurement
        cv::Mat measurement = (cv::Mat_<float>(4, 1) << best_det.cx, best_det.cy, best_det.w, best_det.h);
        kf.correct(measurement);
        last_pos = best_det;
        last_pos.score = best_score;
        return last_pos;
    } else {
        success_count = 0;
        last_pos.score = 0.0f;
        MATCH_RESULT result = last_pos;
        return result;
    }
}

cv::Mat UTracker::DrawResult(cv::Mat &image) {
    cv::Mat result = image.clone();
    
    // Get bounding box coordinates
    int x1 = static_cast<int>(last_pos.cx - last_pos.w / 2);
    int y1 = static_cast<int>(last_pos.cy - last_pos.h / 2);
    int x2 = static_cast<int>(last_pos.cx + last_pos.w / 2);
    int y2 = static_cast<int>(last_pos.cy + last_pos.h / 2);
    
    // Choose colors based on tracking status
    cv::Scalar box_color, bg_color;
    if (last_pos.score > 0) {
        box_color = cv::Scalar(76, 175, 80);  // Material Green 500 (BGR)
        bg_color = cv::Scalar(76, 175, 80);   // Green background
    } else {
        box_color = cv::Scalar(67, 56, 202);  // Material Red 500 (BGR)
        bg_color = cv::Scalar(67, 56, 202);   // Red background
    }
    
    cv::Scalar shadow_color(255, 150, 150);  // Shadow color
    cv::Scalar text_color(255, 255, 255);    // White text
    
    // Draw corner lines with shadows
    int corner_length = static_cast<int>(std::min({30, static_cast<int>(last_pos.w) / 6, static_cast<int>(last_pos.h) / 6}));
    int thickness = 3;
    int shadow_offset = 2;
    
    // Shadow for corners (offset to bottom-right)
    // Top-left corner shadow
    cv::line(result, cv::Point(x1 + shadow_offset, y1 + shadow_offset), 
             cv::Point(x1 + corner_length + shadow_offset, y1 + shadow_offset), shadow_color, thickness);
    cv::line(result, cv::Point(x1 + shadow_offset, y1 + shadow_offset), 
             cv::Point(x1 + shadow_offset, y1 + corner_length + shadow_offset), shadow_color, thickness);
    
    // Top-right corner shadow
    cv::line(result, cv::Point(x2 + shadow_offset, y1 + shadow_offset), 
             cv::Point(x2 - corner_length + shadow_offset, y1 + shadow_offset), shadow_color, thickness);
    cv::line(result, cv::Point(x2 + shadow_offset, y1 + shadow_offset), 
             cv::Point(x2 + shadow_offset, y1 + corner_length + shadow_offset), shadow_color, thickness);
    
    // Bottom-left corner shadow
    cv::line(result, cv::Point(x1 + shadow_offset, y2 + shadow_offset), 
             cv::Point(x1 + corner_length + shadow_offset, y2 + shadow_offset), shadow_color, thickness);
    cv::line(result, cv::Point(x1 + shadow_offset, y2 + shadow_offset), 
             cv::Point(x1 + shadow_offset, y2 - corner_length + shadow_offset), shadow_color, thickness);
    
    // Bottom-right corner shadow
    cv::line(result, cv::Point(x2 + shadow_offset, y2 + shadow_offset), 
             cv::Point(x2 - corner_length + shadow_offset, y2 + shadow_offset), shadow_color, thickness);
    cv::line(result, cv::Point(x2 + shadow_offset, y2 + shadow_offset), 
             cv::Point(x2 + shadow_offset, y2 - corner_length + shadow_offset), shadow_color, thickness);
    
    // Main corner lines
    // Top-left corner
    cv::line(result, cv::Point(x1, y1), cv::Point(x1 + corner_length, y1), box_color, thickness);
    cv::line(result, cv::Point(x1, y1), cv::Point(x1, y1 + corner_length), box_color, thickness);
    
    // Top-right corner
    cv::line(result, cv::Point(x2, y1), cv::Point(x2 - corner_length, y1), box_color, thickness);
    cv::line(result, cv::Point(x2, y1), cv::Point(x2, y1 + corner_length), box_color, thickness);
    
    // Bottom-left corner
    cv::line(result, cv::Point(x1, y2), cv::Point(x1 + corner_length, y2), box_color, thickness);
    cv::line(result, cv::Point(x1, y2), cv::Point(x1, y2 - corner_length), box_color, thickness);
    
    // Bottom-right corner
    cv::line(result, cv::Point(x2, y2), cv::Point(x2 - corner_length, y2), box_color, thickness);
    cv::line(result, cv::Point(x2, y2), cv::Point(x2, y2 - corner_length), box_color, thickness);
    
    // Display score or MISS with background
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.6;
    int text_thickness = 2;
    
    std::string text;
    if (last_pos.score > 0) {
        char score_text[50];
        sprintf(score_text, "Score: %.3f", last_pos.score);
        text = std::string(score_text);
    } else {
        text = "MISS";
    }
    
    // Get text size for background rectangle
    int baseline;
    cv::Size text_size = cv::getTextSize(text, font, font_scale, text_thickness, &baseline);
    
    // Background rectangle coordinates
    int bg_x1 = 10, bg_y1 = 10;
    int bg_x2 = bg_x1 + text_size.width + 10;
    int bg_y2 = bg_y1 + text_size.height + 10;
    
    // Draw background shadow (offset to bottom-right)
    int bg_shadow_offset = 3;
    cv::rectangle(result, cv::Point(bg_x1 + bg_shadow_offset, bg_y1 + bg_shadow_offset), 
                  cv::Point(bg_x2 + bg_shadow_offset, bg_y2 + bg_shadow_offset), shadow_color, -1);
    
    // Draw background rectangle
    cv::rectangle(result, cv::Point(bg_x1, bg_y1), cv::Point(bg_x2, bg_y2), bg_color, -1);
    
    // Draw white text on colored background
    int text_x = bg_x1 + 5;
    int text_y = bg_y1 + text_size.height + 5;
    cv::putText(result, text, cv::Point(text_x, text_y), font, font_scale, text_color, text_thickness);
    
    return result;
}