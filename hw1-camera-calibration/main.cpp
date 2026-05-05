/*
 * Author: Abdulhalim Kiraz
 * Course: EE576
 * Project No: 1
 * File: main.cpp
 * Description: Loads two consecutive images, collects corresponding points,
 * estimates a constrained affine-type transform manually and a full homography
 * with OpenCV, and displays the results on a 1x3 canvas.
 */

#include "Functions.h"
#include "Project1_Params.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

int main() {
    // Build file paths for the two consecutive images.
    std::string img1_path = DATA_DIR + std::to_string(START_INDEX) + IMG_EXT;
    std::string img2_path = DATA_DIR + std::to_string(START_INDEX + 1) + IMG_EXT;

    // Load images.
    cv::Mat img1 = cv::imread(img1_path);
    cv::Mat img2 = cv::imread(img2_path);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images." << std::endl;
        std::cerr << "Tried paths: " << img1_path << " and " << img2_path << std::endl;
        return -1;
    }

    // Resize both images to a common display size.
    cv::resize(img1, img1, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));
    cv::resize(img2, img2, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));

    // Create a 1x3 canvas: first image on the left, second image on the right.
    cv::Mat canvas = cv::Mat::zeros(TARGET_HEIGHT, TARGET_WIDTH * GRID_COLS, img1.type());
    img1.copyTo(canvas(cv::Rect(0, 0, TARGET_WIDTH, TARGET_HEIGHT)));
    img2.copyTo(canvas(cv::Rect(TARGET_WIDTH * 2, 0, TARGET_WIDTH, TARGET_HEIGHT)));

    // Collect raw clicks from the display canvas.
    std::vector<cv::Point2f> allClicks, pts1, pts2;
    cv::namedWindow("Pseudo-Inverse Method");
    cv::setMouseCallback("Pseudo-Inverse Method", onMouse, &allClicks);

    std::cout << "Select corresponding points in this order:" << std::endl;
    std::cout << "1) Click a point in the FIRST image." << std::endl;
    std::cout << "2) Click the matching point in the SECOND image." << std::endl;
    std::cout << "Repeat until " << MIN_CORRESPONDING_POINTS << " pairs are selected." << std::endl;

    // Wait until enough clicks are collected or ESC is pressed.
    while (allClicks.size() < static_cast<size_t>(MIN_CORRESPONDING_POINTS * 2)) {
        cv::imshow("Pseudo-Inverse Method", canvas);
        if (cv::waitKey(UI_DELAY_MS) == KEY_ESC) {
            std::cout << "Program terminated by user." << std::endl;
            return 0;
        }
    }

    // Convert raw alternating clicks into two point vectors.
    processClicks(allClicks, pts1, pts2);

    if (pts1.size() != pts2.size() || pts1.size() < MIN_CORRESPONDING_POINTS) {
        std::cerr << "Error: Invalid number of corresponding points." << std::endl;
        return -1;
    }

    // Manual method:
    // clicks were collected as pts1 = image 1 and pts2 = image 2.
    // Since the assignment asks to warp the second image into the middle cell,
    // the transform must map image 2 -> image 1.
    cv::Mat H_manual = solveConstrainedTransform(pts2, pts1);
    std::cout << "Manual H Matrix:\n" << H_manual << std::endl;

    cv::Mat result_manual;
    cv::warpPerspective(img2, result_manual, H_manual, img1.size());
    result_manual.copyTo(canvas(cv::Rect(TARGET_WIDTH, 0, TARGET_WIDTH, TARGET_HEIGHT)));

    cv::imshow("Pseudo-Inverse Method", canvas);

    // OpenCV method: also map image 2 -> image 1.
    cv::Mat H_opencv;
    findHomographyMap(pts1, pts2, img1, img2, H_opencv);
    std::cout << "\nOpenCV H Matrix:\n" << H_opencv << std::endl;

    cv::waitKey(0);
    return 0;
}