#include <iostream>
#include <filesystem>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "YOLOv8_LT.h"

using torch::indexing::Slice;
using torch::indexing::None;

namespace fs = std::filesystem;

int main() {
    std::string model_path = "/home/jh/YOLOv8-LibTorch-CPP-Inference/weights/yolov8m_seg_1280_b4_best.torchscript";
    float confThreshold = 0.2;
    float iouThreshold = 0.3;
    int input_width = 1280;
    int input_height = 1280;
    bool show_bbox = true;
    bool show_label = true;
    bool show_mask = true;
    std::string font_path = "/home/jh/YOLOv8-LibTorch-CPP-Inference/resource/uming.ttc";
    std::vector<std::string> class_ch = {"鱷魚", "人手孔", "裂縫", "排水孔", "伸縮縫", "補綻", "坑洞"};

    // load model
    YOLOv8_LT detector(
        model_path,
        font_path,
        class_ch,
        confThreshold,
        iouThreshold,
        input_width,
        input_height,
        show_bbox,
        show_label,
        show_mask
    );

    std::string image_dir = "/home/jh/Pictures/test/";
    std::string output_dir = "/home/jh/Pictures/test_/";
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }
    
    try {
        // Record start time 
        auto start = std::chrono::high_resolution_clock::now();
        
        // Go through all images in the directory
        for (const auto& entry : fs::directory_iterator(image_dir)){
            // Load image
            cv::Mat image = cv::imread(entry.path().string());
            if (image.empty()) {
                std::cerr << "Could not read the image: " << entry.path() << std::endl;
                continue;
            }
            std::cout << "Read image: " << entry.path() << std::endl;

            // Inference
            auto [result_image, detResults] = detector.infer(image);

            // Save result image
            cv::imwrite(output_dir + entry.path().filename().string(), result_image);

            // Show the results
            for (const auto& result : detResults) {
                int x1 = result.bbox.x;
                int y1 = result.bbox.y;
                int x2 = result.bbox.x + result.bbox.width;
                int y2 = result.bbox.y + result.bbox.height;
                float conf = result.conf;
                int cls = result.classID;
                std::cout << "Rect: [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]  Conf: " << conf << "  Class: " << class_ch[cls] << std::endl;
            }

            // record end time
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Use " << elapsed.count() << " seconds." << std::endl;
        }
    } catch (const c10::Error& e) {
        std::cout << e.msg() << std::endl;
    }
    return 0;
} // End of main