#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/freetype.hpp>
#include <random>

using torch::indexing::Slice;
using torch::indexing::None;

float generate_scale(cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size[0];
    int target_w = target_size[1];

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
} // End of generate_scale

float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size) {
    // Check if input image already matches target size
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        } else {
            output_image = input_image.clone();
            return 1.;
        }
    }

    // Calculate resize scale to maintain aspect ratio
    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    // Calculate padding sizes
    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    // Resize the image with interpolation
    cv::resize(input_image, output_image,
               cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_AREA);
    
    // Add padding to the resized image
    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114.));
    return resize_scale;
}// End of letterbox

torch::Tensor xyxy2xywh(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    y.index_put_({"...", 0}, (x.index({"...", 0}) + x.index({"...", 2})).div(2));
    y.index_put_({"...", 1}, (x.index({"...", 1}) + x.index({"...", 3})).div(2));
    y.index_put_({"...", 2}, x.index({"...", 2}) - x.index({"...", 0}));
    y.index_put_({"...", 3}, x.index({"...", 3}) - x.index({"...", 1}));
    return y;
} // End of xyxy2xywh

torch::Tensor xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto dw = x.index({"...", 2}).div(2);
    auto dh = x.index({"...", 3}).div(2);
    y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
    y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
    y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
    y.index_put_({"...", 3}, x.index({"...", 1}) + dh);
    return y;
} // End of xywh2xyxy

torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold) {
    // Return an empty tensor if there are no bounding boxes
    if (bboxes.numel() == 0)
        return torch::empty({0}, bboxes.options().dtype(torch::kLong));

    // Extract the coordinates of the bounding boxes and move them to CPU
    auto x1_t = bboxes.select(1, 0).contiguous().to(torch::kCPU);
    auto y1_t = bboxes.select(1, 1).contiguous().to(torch::kCPU);
    auto x2_t = bboxes.select(1, 2).contiguous().to(torch::kCPU);
    auto y2_t = bboxes.select(1, 3).contiguous().to(torch::kCPU);

    // Calculate the areas of the bounding boxes
    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    // Sort the scores in descending order and get the indices
    auto order_t = std::get<1>(
        scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true)).to(torch::kCPU);

    auto ndets = bboxes.size(0);
    torch::Tensor suppressed_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kByte)).to(torch::kCPU);
    torch::Tensor keep_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kLong)).to(torch::kCPU);

    auto suppressed = suppressed_t.data_ptr<uint8_t>();
    auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>();
    auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>();
    auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();

    int64_t num_to_keep = 0;

    // Iterate through the bounding boxes
    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1)
            continue;
        keep[num_to_keep++] = i;
        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];

        // Compare the current bounding box with the rest
        for (int64_t _j = _i + 1; _j < ndets; _j++) {
            auto j = order[_j];
            if (suppressed[j] == 1)
                continue;
            auto xx1 = std::max(ix1, x1[j]);
            auto yy1 = std::max(iy1, y1[j]);
            auto xx2 = std::min(ix2, x2[j]);
            auto yy2 = std::min(iy2, y2[j]);

            auto w = std::max(static_cast<float>(0), xx2 - xx1);
            auto h = std::max(static_cast<float>(0), yy2 - yy1);
            auto inter = w * h;
            auto ovr = inter / (iarea + areas[j] - inter);
            if (ovr > iou_threshold)
                suppressed[j] = 1;
        }
    }
    return keep_t.narrow(0, 0, num_to_keep);
} // End of nms

torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300) {
    auto bs = prediction.size(0); // batch size
    auto nc = prediction.size(1) - 4 - 32; // number of class
    auto nm = prediction.size(1) - nc - 4; // number of mask weight
    auto mi = 4 + nc; // mask weight inital 
    auto xc = prediction.index({Slice(), Slice(4, mi)}).amax(1) > conf_thres;

    prediction = prediction.transpose(-1, -2);
    prediction.index_put_({"...", Slice({None, 4})}, xywh2xyxy(prediction.index({"...", Slice(None, 4)})));

    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++) {
        // box(4)+conf+clssID+number of mask weight
        output.push_back(torch::zeros({0, 6 + nm}, prediction.device()));
    }
    
    for (int xi = 0; xi < prediction.size(0); xi++) {
        auto x = prediction[xi];
        x = x.index({xc[xi]});
        auto x_split = x.split({4, nc, nm}, 1);
        auto box = x_split[0], cls = x_split[1], mask = x_split[2];
        auto [conf, j] = cls.max(1, true);
        x = torch::cat({box, conf, j.toType(torch::kFloat), mask}, 1);
        x = x.index({conf.view(-1) > conf_thres});
        int n = x.size(0);
        if (!n) { continue; }

        // NMS
        auto c = x.index({Slice(), Slice{5, 6}}) * 7680;
        auto boxes = x.index({Slice(), Slice(None, 4)}) + c;
        auto scores = x.index({Slice(), 4});
        auto i = nms(boxes, scores, iou_thres);
        i = i.index({Slice(None, max_det)});
        output[xi] = x.index({i});
    }

    return torch::stack(output);
} // End of non_max_suppression

torch::Tensor clip_boxes(torch::Tensor& boxes, const std::vector<int>& shape) {
    // Clip x1 and x2 to be within the width boundary
    boxes.index_put_({"...", 0}, boxes.index({"...", 0}).clamp(0, shape[1]));
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}).clamp(0, shape[1]));

    // Clip y1 and y2 to be within the height boundary
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}).clamp(0, shape[0]));
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}).clamp(0, shape[0]));

    return boxes;
} // End of clip_boxes

torch::Tensor scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape) {
    // Calculate the scaling factor and padding
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

    // Adjust bounding box coordinates
    boxes.index_put_({"...", 0}, boxes.index({"...", 0}) - pad0);
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}) - pad0);
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}) - pad1);
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}) - pad1);
    boxes.index_put_({"...", Slice(None, 4)}, boxes.index({"...", Slice(None, 4)}).div(gain));

    return boxes;
} // End of scale_boxes

cv::Scalar getRandomColor() {
    static std::random_device rd; // Random device to seed the generator
    static std::mt19937 gen(rd()); // Mersenne Twister random number generator
    static std::uniform_int_distribution<> dis(0, 255); // Uniform distribution for values between 0 and 255

    // Generate and return a random color
    return cv::Scalar(dis(gen), dis(gen), dis(gen));
} // End of getRandomColor

cv::Mat drawDetected(cv::Mat image, torch::Tensor& keep, torch::Tensor& mask_tensor, float threshold = 0.5) {
    // Number of detected results
    int num_results = keep.size(0);

    // Class labels in Chinese
    std::vector<std::string> clsCh {"鱷魚", "人手孔", "裂縫", "排水孔", "伸縮縫", "補綻", "坑洞"};

    // Initialize FreeType2 font
    cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData("./resource/uming.ttc", 0); 
    
    // Iterate over each detected results
    for (int i = 0; i < num_results; ++i) {
        // Extract bbox and mask weights
        float x1 = keep[i][0].item<float>();
        float y1 = keep[i][1].item<float>();
        float x2 = keep[i][2].item<float>();
        float y2 = keep[i][3].item<float>();
        float confidence = keep[i][4].item<float>();
        int class_id = keep[i][5].item<int>();

        // Extract mask weights
        auto mask_weights = keep[i].slice(0, 6, 38);

        // Extract corresponding mask tensor
        auto mask = mask_tensor.squeeze();
        auto weighted_mask = torch::einsum("i,ijk->jk", {mask_weights, mask}).to(torch::kCPU);
        auto sigmoid_mask = torch::sigmoid(weighted_mask);

        auto sizes = sigmoid_mask.sizes();
        int height = sizes[0];
        int width = sizes[1];

        cv::Mat mask_mat(cv::Size(height, width), CV_32F, sigmoid_mask.data_ptr<float>());
        
        // Resize mask to padded image size
        int long_side = std::max(image.cols, image.rows);
        cv::Mat resized_mask;
        cv::resize(mask_mat, resized_mask, cv::Size(long_side, long_side), 0, 0, cv::INTER_LINEAR);
        cv::normalize(resized_mask, resized_mask, 0, 1, cv::NORM_MINMAX);

        // cut off padding area
        int x = (resized_mask.cols - image.cols) / 2 ;
        int y = (resized_mask.rows - image.rows) / 2 ;
        resized_mask = resized_mask(cv::Rect(x, y, image.cols, image.rows));

        // Apply binary threshold
        cv::threshold(resized_mask, resized_mask, threshold, 1, cv::THRESH_BINARY);
        
        // Crop mask to bbox size
        cv::Rect bbox(cv::Point(static_cast<int>(x1), static_cast<int>(y1)), cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        cv::Mat cropped_mask = resized_mask(bbox);

        // Apply cropped mask to image
        cv::Scalar random_color = getRandomColor();
        cv::Mat roi = image(bbox);
        std::vector<cv::Mat> channels(3, cropped_mask);
        cv::Mat merged_mask;
        cv::merge(channels, merged_mask);

        cv::Mat colored_mask;
        cv::Mat color_mask = cv::Mat(merged_mask.size(), merged_mask.type(), cv::Scalar(random_color[0], random_color[1], random_color[2]));

        cv::multiply(merged_mask, color_mask, colored_mask);

        // Convert to the same type as roi
        colored_mask.convertTo(colored_mask, roi.type());

        // Apply the mask with transparency
        cv::addWeighted(roi, 1.0, colored_mask, 0.5, 0, roi);

        // Draw bounding box
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), random_color, 2);

        // Draw text
        std::string label = clsCh[class_id];
        int baseLine = 0;
        int offsetY = -5;
        cv::Size labelSize = ft2->getTextSize(label, 20, -1, &baseLine);
        y1 = std::max(y1, static_cast<float>(labelSize.height));
        cv::rectangle(image, cv::Point(x1, y1 - labelSize.height + offsetY),cv::Point(x1 + labelSize.width, y1 + baseLine + offsetY),random_color, cv::FILLED);
        ft2->putText(image, label, cv::Point(x1, y1 + offsetY), 20, cv::Scalar(255, 255, 255), -1, cv::LINE_AA, true);
    }

    return image;
} // End of drawDetected

int main() {
    // Device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA :torch::kCPU);

    // classes names
    std::vector<std::string> classes {"alligator", "cover", "crack", "draincover", "expansion", "patch", "pothole"};
    try {
        // Load the model (e.g. yolov8s.torchscript)
        std::string model_path = "/home/jh/YOLOv8-LibTorch-CPP-Inference/weights/yolov8m_seg_1280_b4_best.torchscript";
        torch::jit::script::Module yolo_model;
        yolo_model = torch::jit::load(model_path);
        yolo_model.eval();
        yolo_model.to(device, torch::kFloat32);

        // Load image and preprocess
        cv::Mat image = cv::imread("/home/jh/Pictures/test/RCX-8097_20240108_095833882_pave.jpg");
        cv::Mat input_image;
        letterbox(image, input_image, {1280, 1280});

        torch::Tensor image_tensor = torch::from_blob(input_image.data, {input_image.rows, input_image.cols, 3}, torch::kByte).to(device);
        image_tensor = image_tensor.toType(torch::kFloat32).div(255);
        image_tensor = image_tensor.permute({2, 0, 1});
        image_tensor = image_tensor.unsqueeze(0);
        std::vector<torch::jit::IValue> inputs {image_tensor};
        
        //Inference
        torch::IValue output = yolo_model.forward(inputs);
        torch::Tensor result_tensor;
        torch::Tensor mask_tensor;
        
        if (output.isTuple()) {
            auto outputs = output.toTuple()->elements();
            if (!outputs.empty() && outputs[0].isTensor()) {
                result_tensor = outputs[0].toTensor();
                mask_tensor = outputs[1].toTensor();
            } else {
                std::cerr << "First element is not a tensor." << std::endl;
                return -1;
            }
        } else {
            std::cerr << "Output is not a tuple." << std::endl;
            return -1;
        }

        // NMS
        auto keep = non_max_suppression(result_tensor)[0];
        auto boxes = keep.index({Slice(), Slice(None, 4)});
        keep.index_put_({Slice(), Slice(None, 4)}, scale_boxes({input_image.rows, input_image.cols}, boxes, {image.rows, image.cols}));

        // Draw masks and bbox
        cv::Mat result_image = drawDetected(image, keep, mask_tensor);
        cv::imwrite("./results.jpg", result_image);

        // Show the results
        for (int i = 0; i < keep.size(0); i++) {
            int x1 = keep[i][0].item().toFloat();
            int y1 = keep[i][1].item().toFloat();
            int x2 = keep[i][2].item().toFloat();
            int y2 = keep[i][3].item().toFloat();
            float conf = keep[i][4].item().toFloat();
            int cls = keep[i][5].item().toInt();
            std::cout << "Rect: [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]  Conf: " << conf << "  Class: " << classes[cls] << std::endl;
        }
    } catch (const c10::Error& e) {
        std::cout << e.msg() << std::endl;
    }
    return 0;
} // End of main