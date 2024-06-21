#include <gtest/gtest.h>
#include "YOLOv8_LT.h"

using namespace yolov8;

class YOLOv8_LT_Test : public ::testing::Test {
protected:
    std::string model_path;
    std::string font_path;
    std::vector<std::string> class_ch;
    float confThreshold;
    float iouThreshold;
    int input_width;
    int input_height;
    bool show_bbox;
    bool show_label;
    bool show_mask;

    void SetUp() override {
        model_path = "../../weights/yolov8m_seg_1280_b4_best.torchscript";
        font_path = "../../resource/uming.ttc";
        class_ch = {"鱷魚", "人手孔", "裂縫", "排水孔", "伸縮縫", "補綻", "坑洞"};
        confThreshold = 0.2;
        iouThreshold = 0.3;
        input_width = 1280;
        input_height = 1280;
        show_bbox = true;
        show_label = true;
        show_mask = true;
    }
};

TEST_F(YOLOv8_LT_Test, Initialization) {
    YOLOv8_LT yolo(
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

    EXPECT_EQ(yolo.getConfThreshold(), confThreshold);
    EXPECT_EQ(yolo.getIouThreshold(), iouThreshold);
    EXPECT_EQ(yolo.getInputWidth(), input_width);
    EXPECT_EQ(yolo.getInputHeight(), input_height);
}

TEST_F(YOLOv8_LT_Test, Infer) {
    YOLOv8_LT yolo(model_path, font_path, class_ch);
    cv::Mat image = cv::imread("../../tests/test_image.jpg");
    ASSERT_FALSE(image.empty()) << "Could not read the image: test_image.jpg";

    auto [result_image, results] = yolo.infer(image);
    EXPECT_GT(results.size(), 0);
    EXPECT_EQ(result_image.size(), image.size());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
