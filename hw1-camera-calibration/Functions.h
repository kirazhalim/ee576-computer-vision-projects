/*
 * Author: Abdulhalim Kiraz
 * Course: EE576
 * Project No: 1
 * File: Functions.h
 * Description: Function declarations for mouse interaction, manual transform
 * estimation, click processing, and OpenCV homography mapping.
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Mouse callback used to store clicked points on the display canvas.
 *
 * @param event OpenCV mouse event type.
 * @param x X coordinate of the click.
 * @param y Y coordinate of the click.
 * @param flags Extra event flags (unused).
 * @param userdata Pointer to std::vector<cv::Point2f> storing the clicks.
 */
void onMouse(int event, int x, int y, int flags, void* userdata);

/**
 * @brief Estimates the constrained transform
 * [ h11 h12 h13
 *   h21 h22 h23
 *   0   0   1 ]
 * from source points to destination points using pseudo-inverse.
 *
 * @param srcPts Source points.
 * @param dstPts Destination points.
 * @return cv::Mat Estimated 3x3 constrained transformation matrix.
 */
cv::Mat solveConstrainedTransform(const std::vector<cv::Point2f>& srcPts,
                                  const std::vector<cv::Point2f>& dstPts);

/**
 * @brief Splits alternating canvas clicks into first-image and second-image points.
 *
 * The click order is assumed to be:
 * first image point, second image point, first image point, second image point, ...
 *
 * @param allClicks Raw click list collected from the display canvas.
 * @param pts1 Output points from the first image.
 * @param pts2 Output points from the second image.
 */
void processClicks(const std::vector<cv::Point2f>& allClicks,
                   std::vector<cv::Point2f>& pts1,
                   std::vector<cv::Point2f>& pts2);

/**
 * @brief Computes OpenCV homography, shows matches, and displays the warped result.
 *
 * The function maps the second image onto the first image for comparison in the
 * 1x3 output layout.
 *
 * @param pts1 Points selected from the first image.
 * @param pts2 Points selected from the second image.
 * @param img1 First input image.
 * @param img2 Second input image.
 * @param H_out Output homography matrix estimated by OpenCV.
 */
void findHomographyMap(const std::vector<cv::Point2f>& pts1,
                       const std::vector<cv::Point2f>& pts2,
                       const cv::Mat& img1,
                       const cv::Mat& img2,
                       cv::Mat& H_out);

#endif