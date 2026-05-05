/*
 * Author: Abdulhalim Kiraz
 * Course: EE576
 * Project No: 2
 * File: Functions.h
 * Description: Function declarations for image enhancement, color map
 * construction, quantization, connected component segmentation,
 * segment filtering, YOLOv8-seg inference, and display utilities.
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Holds information about a single connected segment.
struct SegmentInfo {
    int          label;       // color bin index (0 to M-1)
    int          pixelCount;  // number of pixels in the segment
    cv::Vec3b    color;       // BGR color of this segment
};

/**
 * @brief Applies CLAHE to the L channel in LAB color space.
 *
 * Enhances local contrast without amplifying noise, which helps
 * the quantization step produce more meaningful regions.
 *
 * @param src Input BGR image.
 * @return Enhanced BGR image.
 */
cv::Mat enhanceImage(const cv::Mat& src);

/**
 * @brief Generates M visually distinct colors using HSV color space.
 *
 * Hue is spread evenly across [0, 179]. Saturation and value are
 * fixed so that all colors are vivid and easy to distinguish.
 *
 * @param M Number of colors to generate.
 * @return Vector of M BGR colors.
 */
std::vector<cv::Vec3b> buildColorMap(int M);

/**
 * @brief Maps each pixel to its nearest color in the color map.
 *
 * The image is first converted to grayscale. The intensity range
 * [0, 255] is divided into M equal bins, and each bin is assigned
 * a color from the color map.
 *
 * @param src       Input BGR image.
 * @param cmap      Color map from buildColorMap().
 * @param M         Number of colors.
 * @param labelImg  Output: single-channel Mat with bin index per pixel.
 * @return Quantized BGR image.
 */
cv::Mat quantizeImage(const cv::Mat& src,
                      const std::vector<cv::Vec3b>& cmap,
                      int M,
                      cv::Mat& labelImg);

/**
 * @brief Runs connected component labeling for each color bin.
 *
 * For each of the M color bins, a binary mask is extracted and
 * connectedComponentsWithStats is applied. Each resulting connected
 * region is stored as a SegmentInfo entry.
 *
 * @param labelImg  Bin-index image from quantizeImage().
 * @param cmap      Color map.
 * @param M         Number of colors.
 * @param segments  Output: all detected segments.
 * @return BGR image with every segment painted in its color.
 */
cv::Mat runConnectedComponents(const cv::Mat& labelImg,
                               const std::vector<cv::Vec3b>& cmap,
                               int M,
                               std::vector<SegmentInfo>& segments);

/**
 * @brief Keeps only segments larger than tau_min * N1 * N2 pixels.
 *
 * Prints a table of kept segments and computes size mean and variance.
 *
 * @param segments     All segments from runConnectedComponents().
 * @param rows         Image height.
 * @param cols         Image width.
 * @param tauMin       Minimum size threshold (fraction of image area).
 * @param labelImg     Bin-index image used for repainting.
 * @param cmap         Color map.
 * @param keptSegments Output: segments that passed the filter.
 * @return BGR image showing only the kept segments.
 */
cv::Mat filterSegments(const std::vector<SegmentInfo>& segments,
                       int rows, int cols,
                       double tauMin,
                       const cv::Mat& labelImg,
                       const std::vector<cv::Vec3b>& cmap,
                       std::vector<SegmentInfo>& keptSegments);

/**
 * @brief Runs YOLOv8-seg segmentation using the OpenCV DNN module.
 *
 * Loads the ONNX model, runs inference, and overlays colored instance
 * masks on the image. Returns a placeholder if the model is not found.
 *
 * @param src       Input BGR image.
 * @param modelPath Path to yolov8n-seg.onnx.
 * @param confThr   Confidence threshold.
 * @param nmsThr    NMS IoU threshold.
 * @return BGR image with segmentation masks overlaid.
 */
cv::Mat runYoloSeg(const cv::Mat& src,
                   const std::string& modelPath,
                   float confThr,
                   float nmsThr);

/**
 * @brief Arranges four panels into a 2x2 composite image.
 *
 * @param panels Vector of exactly 4 BGR images (same size).
 * @return 2x2 composite BGR image.
 */
cv::Mat makeGrid(const std::vector<cv::Mat>& panels);

/**
 * @brief Draws a dark title bar with white text at the top of an image.
 *
 * @param img   Image to annotate (modified in place).
 * @param text  Title text.
 */
void addLabel(cv::Mat& img, const std::string& text);

#endif
