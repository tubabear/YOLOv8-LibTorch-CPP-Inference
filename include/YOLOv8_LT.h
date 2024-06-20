#ifndef YOLOV8_LT_H
#define YOLOV8_LT_H

#include <opencv2/opencv.hpp>
#include <torch/script.h> // LibTorch
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

/**
 * @brief Perform object detection using the YOLOv8 model.
 * 
 * This function takes an input image and a YOLOv8 model to perform object detection.
 * It returns a pair consisting of the processed image and a vector of detection results.
 * 
 * @param image The input image on which object detection is performed.
 * @param model The YOLOv8 model loaded using LibTorch.
 * @param confThreshold Confidence threshold for detections. Default is 0.2.
 * @param iouThreshold Intersection over Union (IoU) threshold for non-maximum suppression. Default is 0.3.
 * @param show_bbox Boolean flag to indicate whether to display bounding boxes. Default is true.
 * @param show_label Boolean flag to indicate whether to display labels. Default is true.
 * @param show_mask Boolean flag to indicate whether to display masks. Default is true.
 * @return A pair containing the processed image and a vector of detection results.
 */

std::pair<cv::Mat, std::vector<detectionResult>> YOLOv8_LT(cv::Mat& image, torch::jit::script::Module& model, float confThreshold=0.2, float iouThreshold=0.3, bool show_bbox=true, bool show_label=true, bool show_mask=true);

#endif // YOLOV8_LT_H