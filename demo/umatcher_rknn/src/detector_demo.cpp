#include "udetector.h"
#include <iostream>
#include <string>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  --template_branch <path>   Path to template branch model (default: ../data/umatcher_rknn/template_branch)\n"
              << "  --search_branch <path>     Path to search branch model (default: ../data/umatcher_rknn/search_branch)\n"
              << "  --template_img <path>      Path to template image (default: ../data/test_1.png)\n"
              << "  --search_img <path>        Path to search image (default: ../data/test_1.png)\n"
              << "  --template_roi <cx cy w h> ROI in cxcywh format (default: 110 233 52 99)\n"
              << "  --output_path <path>       Output path (default: data/detection_result.png)\n"
              << "  --live_mode               Enable live mode\n"
              << "  --help                    Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Default values
    std::string template_branch = "../data/umatcher_rknn/template_branch.rknn";
    std::string search_branch = "../data/umatcher_rknn/search_branch.rknn";
    std::string template_img = "../data/test_1.png";
    std::string search_img = "../data/test_1.png";
    std::string output_path = "../data/detection_result.png";
    bool live_mode = false;
    int template_roi[4] = {110, 233, 52, 99}; // cx, cy, w, h

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--template_branch" && i + 1 < argc) {
            template_branch = argv[++i];
        } else if (arg == "--search_branch" && i + 1 < argc) {
            search_branch = argv[++i];
        } else if (arg == "--template_img" && i + 1 < argc) {
            template_img = argv[++i];
        } else if (arg == "--search_img" && i + 1 < argc) {
            search_img = argv[++i];
        } else if (arg == "--output_path" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--live_mode") {
            live_mode = true;
        } else if (arg == "--template_roi" && i + 4 < argc) {
            template_roi[0] = std::stoi(argv[++i]);
            template_roi[1] = std::stoi(argv[++i]);
            template_roi[2] = std::stoi(argv[++i]);
            template_roi[3] = std::stoi(argv[++i]);
        }
    }

    // Load images
    cv::Mat template_image = cv::imread(template_img);
    if (template_image.empty()) {
        std::cerr << "Failed to load template image: " << template_img << std::endl;
        return -1;
    }

    cv::Mat search_image = cv::imread(search_img);
    if (search_image.empty()) {
        std::cerr << "Failed to load search image: " << search_img << std::endl;
        return -1;
    }

    // Initialize detector with .ncnn.param and .ncnn.bin extensions
    UDetector detector(
        (template_branch).c_str(),
        (search_branch).c_str(),
        128, 2.0,
        256, 4.0,
        16, 128,
        false
    );

    cv::Rect template_rect;
    
    if (live_mode) {
        // Interactive mode: use selectROI
        template_rect = cv::selectROI("Select ROI", template_image);
        cv::destroyWindow("Select ROI");
    } else {
        // Convert cxcywh to xywh format
        int cx = template_roi[0];
        int cy = template_roi[1];
        int w = template_roi[2];
        int h = template_roi[3];
        template_rect = cv::Rect(cx - w/2, cy - h/2, w, h);
    }

    detector.SetTemplate(template_image, template_rect);
    std::cout << "Template has set." << std::endl;

    std::vector<MATCH_RESULT> res = detector.Detect(search_image, 0.3, {0.7, 1.0, 1.3}, 0.1);
    std::cout << "Detections found: " << res.size() << std::endl;

    cv::Mat result_img = detector.DrawDetections(search_image, res);

    if (live_mode) {
        // Show results interactively
        cv::imshow("Detection Result", result_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    } else {
        // Save results to file
        cv::imwrite(output_path, result_img);
        std::cout << "Results saved to: " << output_path << std::endl;
    }

    return 0;
}
