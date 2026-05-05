/*
 * Author: Abdulhalim Kiraz
 * Course: EE576
 * Project No: 1
 * File: Functions.cpp
 * Description: Implements mouse input, manual constrained transform estimation,
 * click processing, and OpenCV homography visualization.
 */

#include "Functions.h"
#include "Project1_Params.h"
#include <iostream>

void onMouse(int event, int x, int y, int /*flags*/, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        std::vector<cv::Point2f>* points =
            reinterpret_cast<std::vector<cv::Point2f>*>(userdata);

        points->push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
        std::cout << "Point registered at: (" << x << ", " << y << ")" << std::endl;
    }
}

cv::Mat solveConstrainedTransform(const std::vector<cv::Point2f>& srcPts,
                                  const std::vector<cv::Point2f>& dstPts) {
    const int N = static_cast<int>(srcPts.size());

    // Build Ah = a for the constrained model:
    // [x' y' 1]^T = H [x y 1]^T
    // with the last row fixed as [0 0 1].
    cv::Mat A = cv::Mat::zeros(2 * N, 6, CV_32F);
    cv::Mat a = cv::Mat::zeros(2 * N, 1, CV_32F);

    for (int i = 0; i < N; ++i) {
        // Equation for x'
        A.at<float>(2 * i, 0) = srcPts[i].x;
        A.at<float>(2 * i, 1) = srcPts[i].y;
        A.at<float>(2 * i, 2) = 1.0f;
        a.at<float>(2 * i, 0) = dstPts[i].x;

        // Equation for y'
        A.at<float>(2 * i + 1, 3) = srcPts[i].x;
        A.at<float>(2 * i + 1, 4) = srcPts[i].y;
        A.at<float>(2 * i + 1, 5) = 1.0f;
        a.at<float>(2 * i + 1, 0) = dstPts[i].y;
    }

    // Solve for h with SVD-based pseudo-inverse.
    cv::Mat h;
    cv::solve(A, a, h, cv::DECOMP_SVD);

    // Construct the final 3x3 matrix.
    cv::Mat H = cv::Mat::eye(3, 3, CV_32F);
    H.at<float>(0, 0) = h.at<float>(0);
    H.at<float>(0, 1) = h.at<float>(1);
    H.at<float>(0, 2) = h.at<float>(2);
    H.at<float>(1, 0) = h.at<float>(3);
    H.at<float>(1, 1) = h.at<float>(4);
    H.at<float>(1, 2) = h.at<float>(5);

    return H;
}

void processClicks(const std::vector<cv::Point2f>& allClicks,
                   std::vector<cv::Point2f>& pts1,
                   std::vector<cv::Point2f>& pts2) {
    for (size_t i = 0; i < allClicks.size(); ++i) {
        if (i % 2 == 0) {
            // Even-index clicks belong to the first image.
            pts1.push_back(allClicks[i]);
        } else {
            // Odd-index clicks belong to the second image, which is displayed
            // in the third panel of the 1x3 canvas. 
            cv::Point2f adjusted = allClicks[i];
            adjusted.x -= (2 * TARGET_WIDTH);
            pts2.push_back(adjusted);
        }
    }
}

void findHomographyMap(const std::vector<cv::Point2f>& pts1,
                       const std::vector<cv::Point2f>& pts2,
                       const cv::Mat& img1,
                       const cv::Mat& img2,
                       cv::Mat& H_out) {
    // The display layout compares warped image 2 against image 1.
    // Therefore the homography is estimated from image 2 -> image 1.
    H_out = cv::findHomography(pts2, pts1);

    if (H_out.empty()) {
        std::cerr << "Error: OpenCV homography could not be computed." << std::endl;
        return;
    }

    // Draw corresponding points for visualization.
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;

    for (size_t i = 0; i < pts1.size(); ++i) {
        keypoints1.emplace_back(pts1[i], 1.f);
        keypoints2.emplace_back(pts2[i], 1.f);
        matches.emplace_back(static_cast<int>(i), static_cast<int>(i), 0.0f);
    }

    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    cv::imshow("OpenCV Matches", img_matches);

    // Warp the second image into the coordinate system of the first image.
    cv::Mat result_opencv;
    cv::warpPerspective(img2, result_opencv, H_out, img1.size());

    // Build the 1x3 comparison canvas.
    cv::Mat canvas_opencv = cv::Mat::zeros(TARGET_HEIGHT, TARGET_WIDTH * GRID_COLS, img1.type());
    img1.copyTo(canvas_opencv(cv::Rect(0, 0, TARGET_WIDTH, TARGET_HEIGHT)));
    result_opencv.copyTo(canvas_opencv(cv::Rect(TARGET_WIDTH, 0, TARGET_WIDTH, TARGET_HEIGHT)));
    img2.copyTo(canvas_opencv(cv::Rect(TARGET_WIDTH * 2, 0, TARGET_WIDTH, TARGET_HEIGHT)));

    cv::imshow("OpenCV Result", canvas_opencv);
}