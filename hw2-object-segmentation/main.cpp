/*
 * Author: Abdulhalim Kiraz
 * Course: EE576
 * Project No: 2
 * File: main.cpp
 * Description: Loads three images from the Washington RGB-D Dataset,
 * applies color-map based segmentation with connected components,
 * filters segments by minimum size, and runs YOLOv8-seg for
 * deep-learning based segmentation. Results are displayed in a 2x2 grid.
 */

#include "Functions.h"
#include "Project2_Params.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// Runs the full segmentation pipeline on a single image and saves the result.
void processImage(const std::string& imagePath, const std::string& label) {
    std::cout << "\n==============================" << std::endl;
    std::cout << "Image: " << label << std::endl;
    std::cout << "==============================" << std::endl;

    // Load and resize the image.
    cv::Mat src = cv::imread(imagePath);
    if (src.empty()) {
        std::cerr << "Error: Could not load image: " << imagePath << std::endl;
        return;
    }
    cv::resize(src, src, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));

    // Step 1: Enhance contrast with CLAHE.
    cv::Mat enhanced = enhanceImage(src);

    // Step 2 (a): Build color map and quantize the image.
    std::vector<cv::Vec3b> cmap = buildColorMap(NUM_COLORS);
    cv::Mat labelImg;
    cv::Mat quantized = quantizeImage(enhanced, cmap, NUM_COLORS, labelImg);
    std::cout << "Color map: M=" << NUM_COLORS << " colors." << std::endl;

    // Step 3 (a): Find all connected segments.
    std::vector<SegmentInfo> allSegments;
    cv::Mat segAll = runConnectedComponents(labelImg, cmap, NUM_COLORS, allSegments);
    std::cout << "Total segments found: " << allSegments.size() << std::endl;

    // Step 4 (a): Filter segments by minimum size.
    std::vector<SegmentInfo> keptSegments;
    cv::Mat segFiltered = filterSegments(allSegments,
                                         src.rows, src.cols,
                                         TAU_MIN,
                                         labelImg, cmap,
                                         keptSegments);

    // Step 5 (b): Deep-learning segmentation with YOLOv8-seg.
    cv::Mat segYolo = runYoloSeg(enhanced, YOLO_MODEL_PATH,
                                  YOLO_CONF_THR, YOLO_NMS_THR);

    // Compose the 2x2 result grid.
    cv::Mat p0 = enhanced.clone();
    cv::Mat p1 = segAll.clone();
    cv::Mat p2 = segFiltered.clone();
    cv::Mat p3 = segYolo.clone();

    addLabel(p0, "1. Original (enhanced)");
    addLabel(p1, "2. All segments [n=" + std::to_string(allSegments.size()) + "]");
    addLabel(p2, "3. Filtered [tau=" + std::to_string(TAU_MIN).substr(0, 5)
                 + ", n=" + std::to_string(keptSegments.size()) + "]");
    addLabel(p3, "4. YOLOv8-seg");

    std::vector<cv::Mat> panels;
    panels.push_back(p0);
    panels.push_back(p1);
    panels.push_back(p2);
    panels.push_back(p3);

    cv::Mat grid = makeGrid(panels);

    // Save and display the result.
    std::string outPath = label + "_result.png";
    cv::imwrite(outPath, grid);
    std::cout << "Saved: " << outPath << std::endl;

    cv::imshow("EE576 HW2 - " + label, grid);
    cv::waitKey(0);
}

int main() {
    std::cout << "EE576 Project 2 - Image Segmentation" << std::endl;

    // Process two object images and one scene image.
    processImage(IMG_BOWL,   "bowl");
    processImage(IMG_BANANA,  "banana");
    processImage(IMG_KITCHEN, "kitchen");

    std::cout << "\nDone." << std::endl;
    return 0;
}
