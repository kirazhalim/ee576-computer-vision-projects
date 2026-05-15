/*
 * Author: Abdulhalim Kiraz
 * Student ID: 2021401219
 * Course: EE576
 * Project No: 4
 * File: Functions.h
 * Description: Function declarations for optical flow, sparse tracking,
 * region comparison, and 2x2 display.
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct RegionFlow {
    int label;
    int area;
    cv::Point2f centroid;
    cv::Point2f averageFlow;
};

struct SparseTrack {
    cv::Point2f start;
    cv::Point2f end;
    float error;
};

struct SparseTrackingResult {
    int siftCountFrame1;
    int siftCountFrame2;
    std::vector<SparseTrack> tracks;
};

struct OrientationStats {
    int comparedCount;
    double averageCosineSimilarity;
    double averageAngleDifference;
};

struct RegionComparison {
    int label;
    int trackCount;
    cv::Point2f denseAverage;
    cv::Point2f sparseAverage;
    double differenceNorm;
};

struct DatasetConfig {
    std::string name;
    std::string dataDir;
    std::string maskDir;
    std::string imagePrefix;
    std::string imageExt;
    std::string excludedSuffix;
    bool useProvidedMask;
};

/**
 * @brief Loads the available frame indices from a numbered image sequence.
 *
 * @param dataDir Relative path to the image directory.
 * @param imagePrefix Prefix before the frame index.
 * @param imageExt Image extension, such as ".jpg".
 * @param excludedSuffix File-name suffix to skip before extension.
 * @return Sorted list of frame indices.
 */
std::vector<int> loadFrameIndices(const std::string& dataDir,
                                  const std::string& imagePrefix,
                                  const std::string& imageExt,
                                  const std::string& excludedSuffix);

/**
 * @brief Reads one numbered frame from the image sequence.
 *
 * @param dataDir Relative path to the image directory.
 * @param imagePrefix Prefix before the frame index.
 * @param index Frame index.
 * @param imageExt Image extension.
 * @return Loaded BGR image.
 */
cv::Mat loadFrame(const std::string& dataDir,
                  const std::string& imagePrefix,
                  int index,
                  const std::string& imageExt);

/**
 * @brief Reads and binarizes the mask belonging to one frame.
 *
 * @param maskDir Relative path to the mask directory.
 * @param index Frame index.
 * @param imageExt Mask image extension.
 * @param size Output mask size.
 * @return Binary mask.
 */
cv::Mat loadMask(const std::string& maskDir,
                 int index,
                 const std::string& imageExt,
                 const cv::Size& size);

/**
 * @brief Produces a simple moving-region mask from frame difference.
 *
 * @param frame1 First BGR frame.
 * @param frame2 Second BGR frame.
 * @return Binary motion mask.
 */
cv::Mat createAutoMotionMask(const cv::Mat& frame1,
                             const cv::Mat& frame2);

/**
 * @brief Computes dense optical flow between two consecutive frames.
 *
 * @param frame1 First BGR frame.
 * @param frame2 Second BGR frame.
 * @return Two-channel flow image containing dx and dy.
 */
cv::Mat computeDenseFlow(const cv::Mat& frame1, const cv::Mat& frame2);

/**
 * @brief Computes average dense flow vectors for connected mask regions.
 *
 * @param flow Dense optical flow.
 * @param mask Binary moving-region mask.
 * @return Average flow information for each region.
 */
std::vector<RegionFlow> computeRegionFlows(const cv::Mat& flow,
                                           const cv::Mat& mask);

/**
 * @brief Draws masked regions and their average dense flow vectors.
 *
 * @param frame Base BGR frame.
 * @param mask Binary moving-region mask.
 * @param regions Region average flow values.
 * @return Visualization image.
 */
cv::Mat drawDenseFlowPanel(const cv::Mat& frame,
                           const cv::Mat& mask,
                           const std::vector<RegionFlow>& regions);

/**
 * @brief Computes SIFT points and tracks first-frame points with Lucas-Kanade.
 *
 * @param frame1 First BGR frame.
 * @param frame2 Second BGR frame.
 * @return Sparse tracking result.
 */
SparseTrackingResult computeSparseTracks(const cv::Mat& frame1,
                                         const cv::Mat& frame2,
                                         const cv::Mat& mask = cv::Mat());

/**
 * @brief Computes sparse average flow separately for each mask region.
 *
 * @param frame1 First BGR frame.
 * @param frame2 Second BGR frame.
 * @param mask Binary moving-region mask.
 * @return Average sparse flow information for each region.
 */
std::vector<RegionFlow> computeSparseRegionFlows(const cv::Mat& frame1,
                                                 const cv::Mat& frame2,
                                                 const cv::Mat& mask);

/**
 * @brief Compares dense and sparse region average flow vectors.
 *
 * @param denseRegions Dense average flow values by mask region.
 * @param sparseRegions Sparse average flow values by mask region.
 * @return Region-by-region difference values.
 */
std::vector<RegionComparison> compareRegionFlows(
    const std::vector<RegionFlow>& denseRegions,
    const std::vector<RegionFlow>& sparseRegions);

/**
 * @brief Draws sparse optical flow tracks.
 *
 * @param frame Base BGR frame.
 * @param result Sparse tracking result.
 * @return Visualization image.
 */
cv::Mat drawSparseTrackingPanel(const cv::Mat& frame,
                                const SparseTrackingResult& result,
                                const std::vector<RegionFlow>& sparseRegions);

/**
 * @brief Compares dense and sparse flow orientations.
 *
 * @param denseFlow Dense optical flow field.
 * @param sparseResult Sparse tracking result.
 * @return Average cosine similarity and angle difference.
 */
OrientationStats compareFlowOrientations(const cv::Mat& denseFlow,
                                         const SparseTrackingResult& sparseResult);

/**
 * @brief Draws a dark title bar and a short label on an image.
 *
 * @param img Image to annotate.
 * @param text Label text.
 */
void addLabel(cv::Mat& img, const std::string& text);

/**
 * @brief Arranges four equal-size panels into a 2x2 grid.
 *
 * @param panels Vector of four BGR images.
 * @return Composite grid image.
 */
cv::Mat makeGrid(const std::vector<cv::Mat>& panels);

/**
 * @brief Runs the frame loop and shows dense/sparse flow results.
 */
void runGridViewer();

#endif
