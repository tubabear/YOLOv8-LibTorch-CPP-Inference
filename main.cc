#include <iostream>
#include <filesystem>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "YOLOv8_LT.h"

using torch::indexing::Slice;
using torch::indexing::None;

namespace fs = std::filesystem;

int main() {
    // Device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA :torch::kCPU);

    std::string model_path = "/home/jh/YOLOv8-LibTorch-CPP-Inference/weights/yolov8m_seg_1280_b4_best.torchscript";
    std::string image_dir = "/home/jh/Pictures/test/";
    
    // Class labels in Chinese
    std::vector<std::string> clsCh {"鱷魚", "人手孔", "裂縫", "排水孔", "伸縮縫", "補綻", "坑洞"};

    torch::jit::script::Module model;
    try {
        // Load the model
        model = torch::jit::load(model_path);
        model.eval();
        model.to(device, torch::kFloat32);

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
            auto [result_image, detResults] = YOLOv8_LT(image, model);

            // Save result image
            cv::imwrite("./test/" + entry.path().filename().string(), result_image);


            // Show the results
            for (const auto& result : detResults) {
                int x1 = result.bbox.x;
                int y1 = result.bbox.y;
                int x2 = result.bbox.x + result.bbox.width;
                int y2 = result.bbox.y + result.bbox.height;
                float conf = result.conf;
                int cls = result.classID;
                std::cout << "Rect: [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]  Conf: " << conf << "  Class: " << clsCh[cls] << std::endl;
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