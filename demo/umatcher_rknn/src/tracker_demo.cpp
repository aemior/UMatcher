#include "utracker.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --template_branch PATH   Path to template branch (default: ../data/umatcher_rknn/template_branch)\n"
              << "  --search_branch PATH     Path to search branch (default: ../data/umatcher_rknn/search_branch)\n"
              << "  --input_path PATH        Path to input video (default: ../data/girl_dance.mp4)\n"
              << "  --init_roi cx cy w h     ROI in cxcywh format (default: 547 188 43 57)\n"
              << "  --output_path PATH       Output video path (default: ../data/tracking_result.mp4)\n"
              << "  --live_mode             Enable live mode with ROI selection\n"
              << "  --help                  Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Default parameters
    std::string template_branch = "../data/umatcher_rknn/template_branch.rknn";
    std::string search_branch = "../data/umatcher_rknn/search_branch.rknn";
    std::string input_path = "../data/girl_dance.mp4";
    std::vector<int> init_roi = {547, 188, 43, 57};
    std::string output_path = "../data/tracking_result.mp4";
    bool live_mode = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--template_branch" && i + 1 < argc) {
            template_branch = argv[++i];
        } else if (arg == "--search_branch" && i + 1 < argc) {
            search_branch = argv[++i];
        } else if (arg == "--input_path" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--init_roi" && i + 4 < argc) {
            init_roi[0] = std::stoi(argv[++i]);
            init_roi[1] = std::stoi(argv[++i]);
            init_roi[2] = std::stoi(argv[++i]);
            init_roi[3] = std::stoi(argv[++i]);
        } else if (arg == "--output_path" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--live_mode") {
            live_mode = true;
        }
    }

    // Initialize UTracker
    UTracker tracker(template_branch.c_str(), search_branch.c_str(),
                    128, 2.0f, 256, 4.0f, 16, 128, false);

    // Open video
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << input_path << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Failed to read first frame" << std::endl;
        return -1;
    }

    cv::Rect bbox;
    
    if (live_mode) {
        // Live mode: use OpenCV ROI selection
        bbox = cv::selectROI("Select ROI", frame, false);
        cv::destroyWindow("Select ROI");
        std::cout << "Selected ROI: " << bbox << std::endl;
    } else {
        // Convert cxcywh to xyxy format
        int cx = init_roi[0], cy = init_roi[1], w = init_roi[2], h = init_roi[3];
        bbox = cv::Rect(cx - w/2, cy - h/2, w, h);
    }

    // Initialize tracker
    tracker.Init(frame, bbox);

    cv::VideoWriter writer;
    if (!live_mode) {
        // Setup video writer for file output
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size frame_size(frame.cols, frame.rows);
        writer.open(output_path, fourcc, fps, frame_size);
        
        if (!writer.isOpened()) {
            std::cerr << "Failed to open output video: " << output_path << std::endl;
            return -1;
        }
    }

    // Tracking loop
    std::cout << "Starting tracking..." << std::endl;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        MATCH_RESULT result = tracker.Track(frame);
        std::cout << "\rTracked position: (" << result.cx << ", " << result.cy 
                  << ") with size (" << result.w << ", " << result.h 
                  << ") and score " << result.score;

        cv::Mat output_frame = tracker.DrawResult(frame);

        if (live_mode) {
            cv::imshow("Tracking Result", output_frame);
            if (cv::waitKey(1) == 27) break; // ESC to exit
        } else {
            writer.write(output_frame);
        }
    }
    std::cout << std::endl;

    if (!live_mode) {
        writer.release();
        std::cout << "Output saved to: " << output_path << std::endl;
    }

    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}