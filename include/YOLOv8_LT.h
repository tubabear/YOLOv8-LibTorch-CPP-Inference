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

/**
 * @class YOLOv8_LT
 * @brief A class for performing object detection using the YOLOv8 LibTorch model.
 * 
 * This class provides functionalities to load a YOLOv8 model, preprocess input images,
 * perform inference, and postprocess the output to draw detections.
 */
class YOLOv8_LT {
public:
    /**
     * @brief Constructor for YOLOv8_LT.
     * @param model_path Path to the YOLOv8 model.
     * @param font_path Path to the font file for displaying labels in chinese.
     * @param class_Ch Vector of class names in Chinese.
     * @param confThreshold Confidence threshold for detected objects.
     * @param iouThreshold Intersection over Union (IoU) threshold for detected bbox.
     * @param input_width Width of the input image.
     * @param input_height Height of the input image.
     * @param show_bbox Flag to indicate whether to show bounding boxes.
     * @param show_label Flag to indicate whether to show labels.
     * @param show_mask Flag to indicate whether to show masks.
     */
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

    /**
     * @brief Performs inference on an input image.
     * @param image Input image for inference.
     * @param show_bbox Flag to indicate whether to show bounding boxes.
     * @param show_label Flag to indicate whether to show labels.
     * @param show_mask Flag to indicate whether to show masks.
     * @return A pair containing the processed image and a vector of detection results.
     */
    std::pair<cv::Mat, std::vector<detectionResult>> infer(const cv::Mat& image, bool show_bbox = true, bool show_label = true, bool show_mask = true);

private:
    torch::jit::script::Module model; ///< The loaded YOLOv8 model.
    std::string font_path; ///< Path to the chinese font file.
    std::vector<std::string> class_Ch; ///< Vector of class names in Chinese.
    float confThreshold; ///< Confidence threshold for detected objects.
    float iouThreshold; ///< Intersection over Union (IoU) threshold for detection.
    int input_width; ///< Width of the input image.
    int input_height; ///< Height of the input image.
    bool show_bbox; ///< Flag to indicate whether to show bounding boxes.
    bool show_label; ///< Flag to indicate whether to show labels.
    bool show_mask; ///< Flag to indicate whether to show masks.

    torch::Device device; ///< Device to run the model on (CPU/GPU).
    cv::Ptr<cv::freetype::FreeType2> ft2; ///< FreeType2 instance for text rendering.

    /**
     * @brief Preprocesses the given image and converts it into a tensor.
     * 
     * This function takes an image in the form of a cv::Mat object and preprocesses it. 
     * The preprocessed image is then stored in the provided torch::Tensor reference.
     * 
     * @param image The input image to be preprocessed, given as a cv::Mat.
     * @param image_tensor The output tensor where the preprocessed image will be stored.
     */
    void preprocess(const cv::Mat& image, torch::Tensor& image_tensor);

    /**
     * @brief Postprocesses the model output and generates detection results.
     * 
     * This function takes the output from the model, the original input image, 
     * and processes them to produce a result image and detection results. The results 
     * include annotated images and detected objects stored in the results vector.
     * 
     * @param output The model's output to be postprocessed, given as a torch::jit::IValue.
     * @param image The original input image, given as a cv::Mat.
     * @param result_image The output image with annotations, given as a cv::Mat.
     * @param results A vector to store the detected objects and their attributes.
     */
    void postprocess(
        torch::jit::IValue& output,
        const cv::Mat& image,
        cv::Mat& result_image,
        std::vector<detectionResult>& results
    );

    /**
     * @brief Draws detected objects on the image and extracts their contours.
     * 
     * This function takes the original image, detection tensors, and processes them 
     * to draw the detected objects on the result image. It also extracts the contours 
     * of the detected objects and stores them in the provided vector.
     * 
     * @param image The original input image, given as a cv::Mat.
     * @param keep A tensor indicating which objects to keep after detection.
     * @param mask_tensor A tensor containing the mask of detected objects.
     * @param result_image The output image with drawn detections, given as a cv::Mat.
     * @param contours A vector to store the contours of the detected objects.
     */
    void drawDetected(
        const cv::Mat& image,
        const torch::Tensor& keep,
        const torch::Tensor& mask_tensor,
        cv::Mat& result_image,
        std::vector<std::vector<cv::Point>>& contours
    );
}; // End of YOLOv8_LT class

#endif // YOLOV8_LT_H