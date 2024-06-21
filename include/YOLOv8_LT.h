#ifndef YOLOV8_LT_H
#define YOLOV8_LT_H

#include <opencv2/opencv.hpp>
#include <torch/script.h> // LibTorch
#include <opencv2/freetype.hpp>
#include <vector>
#include <string>

/**
 * @struct detectionResult
 * @brief Struct to hold the detection results from YOLOv8.
 * 
 * This struct contains the bounding box, class ID, confidence score, and contours 
 * of the detected objects.
 */
struct detectionResult 
{
    cv::Rect bbox; ///< Bounding box of the detected object.
    int classID; ///< Class ID of the detected object.
    float conf; ///< Confidence score of the detected object.
    std::vector<cv::Point> contours; ///< Contours of the detected object.
};

class YOLOv8_LT{
    public:
        YOLOv8_LT(
            const std::string& model_path,
            const std::string font_path,
            std::vector<std::string> class_Ch,
            float confThreshold = 0.2,
            float iouThreshold = 0.3,
            int input_width = 1280,
            int input_height = 1280,
            bool show_bbox = true,
            bool show_label = true,
            bool show_mask = true
        );

        std::pair<cv::Mat, std::vector<detectionResult>> infer(const cv::Mat& image, bool show_bbox = true, bool show_label = true, bool show_mask = true);
    
    private:
        torch::jit::script::Module model;
        std::string font_path;
        std::vector<std::string> class_Ch;
        float confThreshold;
        float iouThreshold;
        int input_width;
        int input_height;
        bool show_bbox;
        bool show_label;
        bool show_mask;

        torch::Device device;
        cv::Ptr<cv::freetype::FreeType2> ft2;

        torch::Tensor preprocess(const cv::Mat& image);

        std::pair<cv::Mat, std::vector<detectionResult>> postprocess(
            torch::jit::IValue& output,
            const cv::Mat& image
        );

        std::pair<cv::Mat, std::vector<std::vector<cv::Point>>> drawDetected(
            const cv::Mat& image,
            const torch::Tensor& keep,
            const torch::Tensor& mask_tensor
        );

    };

#endif // YOLOV8_LT_H