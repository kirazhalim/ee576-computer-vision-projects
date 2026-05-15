/*
 * Author: Abdulhalim Kiraz
 * Student ID: 2021401219
 * Course: EE576
 * Project No: 4
 * File: Functions.cpp
 * Description: Implements optical flow, sparse tracking, region comparison,
 * and the 2x2 display loop.
 */

#include "Functions.h"
#include "Project4_Params.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <opencv2/features2d.hpp>

namespace {

std::string framePath(const std::string& dataDir,
                      const std::string& imagePrefix,
                      int index,
                      const std::string& imageExt) {
    return dataDir + "/" + imagePrefix + std::to_string(index) + imageExt;
}

bool startsWith(const std::string& s, const std::string& prefix) {
    if (s.size() < prefix.size()) {
        return false;
    }
    return s.compare(0, prefix.size(), prefix) == 0;
}

bool endsWith(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

cv::Mat preparePanel(const cv::Mat& src, const std::string& label) {
    cv::Mat panel;
    cv::resize(src, panel, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));
    addLabel(panel, label);
    return panel;
}

void printDenseFlowSummary(int frameIndex1,
                           int frameIndex2,
                           const std::string& datasetName,
                           const std::vector<RegionFlow>& regions) {
    std::cout << "\nDataset: " << datasetName << std::endl;
    std::cout << "Frames " << frameIndex1 << " -> " << frameIndex2 << std::endl;
    std::cout << "Dense motion regions: " << regions.size() << std::endl;

    for (size_t i = 0; i < regions.size(); ++i) {
        std::cout << "  Region " << regions[i].label
                  << " area=" << regions[i].area
                  << " avg=(" << std::fixed << std::setprecision(3)
                  << regions[i].averageFlow.x << ", "
                  << regions[i].averageFlow.y << ")" << std::endl;
    }
}

void printSparseSummary(const SparseTrackingResult& result) {
    std::cout << "Sparse SIFT keypoints: frame1=" << result.siftCountFrame1
              << ", frame2=" << result.siftCountFrame2 << std::endl;
    std::cout << "Sparse LK tracks: " << result.tracks.size() << std::endl;
}

void printOrientationSummary(const OrientationStats& stats) {
    std::cout << "Orientation comparison count: " << stats.comparedCount << std::endl;
    std::cout << "Average cosine similarity: " << std::fixed << std::setprecision(3)
              << stats.averageCosineSimilarity << std::endl;
    std::cout << "Average angle difference: " << std::fixed << std::setprecision(3)
              << stats.averageAngleDifference << " deg" << std::endl;
}

void printRegionComparisons(const std::vector<RegionComparison>& comparisons) {
    std::cout << "Masked region dense-sparse differences: "
              << comparisons.size() << std::endl;

    for (size_t i = 0; i < comparisons.size(); ++i) {
        std::cout << "  Region " << comparisons[i].label
                  << " tracks=" << comparisons[i].trackCount
                  << " dense=(" << std::fixed << std::setprecision(3)
                  << comparisons[i].denseAverage.x << ", "
                  << comparisons[i].denseAverage.y << ")"
                  << " sparse=("
                  << comparisons[i].sparseAverage.x << ", "
                  << comparisons[i].sparseAverage.y << ")"
                  << " norm=" << comparisons[i].differenceNorm << std::endl;
    }
}

std::vector<DatasetConfig> makeDatasets() {
    std::vector<DatasetConfig> datasets;

    // TUM has provided masks. Kitchen masks are made from frame difference.
    DatasetConfig tum;
    tum.name = "TUM";
    tum.dataDir = TUM_DATA_DIR;
    tum.maskDir = TUM_MASK_DIR;
    tum.imagePrefix = TUM_IMAGE_PREFIX;
    tum.imageExt = TUM_IMAGE_EXT;
    tum.excludedSuffix = TUM_EXCLUDED_SUFFIX;
    tum.useProvidedMask = true;
    datasets.push_back(tum);

    DatasetConfig kitchen;
    kitchen.name = "Kitchen";
    kitchen.dataDir = KITCHEN_DATA_DIR;
    kitchen.maskDir = "";
    kitchen.imagePrefix = KITCHEN_IMAGE_PREFIX;
    kitchen.imageExt = KITCHEN_IMAGE_EXT;
    kitchen.excludedSuffix = KITCHEN_EXCLUDED_SUFFIX;
    kitchen.useProvidedMask = false;
    datasets.push_back(kitchen);

    return datasets;
}

std::vector<int> findConsecutivePairs(const std::vector<int>& indices) {
    std::vector<int> pairStarts;
    for (size_t i = 0; i + 1 < indices.size(); ++i) {
        if (indices[i + 1] == indices[i] + 1) {
            pairStarts.push_back(indices[i]);
        }
    }
    return pairStarts;
}

} // namespace

std::vector<int> loadFrameIndices(const std::string& dataDir,
                                  const std::string& imagePrefix,
                                  const std::string& imageExt,
                                  const std::string& excludedSuffix) {
    std::vector<cv::String> files;
    cv::glob(dataDir + "/*" + imageExt, files, false);

    std::vector<int> indices;
    for (size_t i = 0; i < files.size(); ++i) {
        std::string path = files[i];
        size_t slash = path.find_last_of("/\\");
        size_t dot = path.find_last_of('.');
        if (slash == std::string::npos || dot == std::string::npos || dot <= slash) {
            continue;
        }

        std::string stem = path.substr(slash + 1, dot - slash - 1);
        // This removes kitchen depth images from the RGB frame list.
        if (!excludedSuffix.empty() && endsWith(stem, excludedSuffix)) {
            continue;
        }
        if (!imagePrefix.empty() && !startsWith(stem, imagePrefix)) {
            continue;
        }

        std::string indexText = stem.substr(imagePrefix.size());
        indices.push_back(std::stoi(indexText));
    }

    std::sort(indices.begin(), indices.end());
    return indices;
}

cv::Mat loadFrame(const std::string& dataDir,
                  const std::string& imagePrefix,
                  int index,
                  const std::string& imageExt) {
    return cv::imread(framePath(dataDir, imagePrefix, index, imageExt));
}

cv::Mat loadMask(const std::string& maskDir,
                 int index,
                 const std::string& imageExt,
                 const cv::Size& size) {
    cv::Mat mask = cv::imread(framePath(maskDir, "", index, imageExt), cv::IMREAD_GRAYSCALE);

    if (mask.empty()) {
        return cv::Mat(size, CV_8U, cv::Scalar(MASK_MAX_VALUE));
    }

    if (mask.size() != size) {
        cv::resize(mask, mask, size, 0, 0, cv::INTER_NEAREST);
    }

    cv::threshold(mask, mask, MASK_THRESHOLD_VALUE, MASK_MAX_VALUE, cv::THRESH_BINARY);
    return mask;
}

cv::Mat createAutoMotionMask(const cv::Mat& frame1,
                             const cv::Mat& frame2) {
    cv::Mat gray1, gray2, diff, mask;
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    // A simple motion mask is enough for the kitchen sequence.
    cv::absdiff(gray1, gray2, diff);
    cv::GaussianBlur(diff, diff, cv::Size(AUTO_MASK_BLUR_SIZE, AUTO_MASK_BLUR_SIZE), 0);
    cv::threshold(diff, mask, AUTO_MASK_THRESHOLD_VALUE, MASK_MAX_VALUE, cv::THRESH_BINARY);

    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(AUTO_MASK_MORPH_SIZE, AUTO_MASK_MORPH_SIZE));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
    cv::dilate(mask, mask, element, cv::Point(-1, -1), AUTO_MASK_DILATE_ITERATIONS);

    return mask;
}

cv::Mat computeDenseFlow(const cv::Mat& frame1, const cv::Mat& frame2) {
    cv::Mat gray1, gray2, flow;
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    cv::calcOpticalFlowFarneback(gray1, gray2, flow,
                                 FARNEBACK_PYR_SCALE,
                                 FARNEBACK_LEVELS,
                                 FARNEBACK_WINDOW_SIZE,
                                 FARNEBACK_ITERATIONS,
                                 FARNEBACK_POLY_N,
                                 FARNEBACK_POLY_SIGMA,
                                 FARNEBACK_FLAGS);
    return flow;
}

std::vector<RegionFlow> computeRegionFlows(const cv::Mat& flow,
                                           const cv::Mat& mask) {
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

    std::vector<RegionFlow> regions;
    for (int label = 1; label < count; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < MIN_REGION_AREA) {
            continue;
        }

        cv::Mat regionMask = labels == label;
        cv::Scalar avg = cv::mean(flow, regionMask);

        RegionFlow region;
        region.label = label;
        region.area = area;
        region.centroid = cv::Point2f(static_cast<float>(centroids.at<double>(label, 0)),
                                      static_cast<float>(centroids.at<double>(label, 1)));
        region.averageFlow = cv::Point2f(static_cast<float>(avg[0]),
                                         static_cast<float>(avg[1]));
        regions.push_back(region);
    }

    return regions;
}

cv::Mat drawDenseFlowPanel(const cv::Mat& frame,
                           const cv::Mat& mask,
                           const std::vector<RegionFlow>& regions) {
    cv::Mat panel = frame.clone();
    cv::Mat overlay = panel.clone();
    overlay.setTo(cv::Scalar(REGION_BORDER_BLUE, REGION_BORDER_GREEN, REGION_BORDER_RED), mask);
    cv::addWeighted(overlay, MASK_OVERLAY_ALPHA, panel, 1.0 - MASK_OVERLAY_ALPHA, 0.0, panel);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(panel, contours, -1,
                     cv::Scalar(REGION_BORDER_BLUE, REGION_BORDER_GREEN, REGION_BORDER_RED),
                     REGION_BORDER_THICKNESS);

    for (size_t i = 0; i < regions.size(); ++i) {
        cv::Point2f start = regions[i].centroid;
        cv::Point2f end = start + regions[i].averageFlow * static_cast<float>(DENSE_ARROW_SCALE);
        cv::arrowedLine(panel, start, end,
                        cv::Scalar(DENSE_ARROW_BLUE, DENSE_ARROW_GREEN, DENSE_ARROW_RED),
                        DENSE_ARROW_THICKNESS, cv::LINE_AA);
    }

    addLabel(panel, "Part 1: dense average flow");
    return panel;
}

SparseTrackingResult computeSparseTracks(const cv::Mat& frame1,
                                         const cv::Mat& frame2,
                                         const cv::Mat& mask) {
    cv::Mat gray1, gray2;
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(SIFT_FEATURE_COUNT);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    sift->detect(gray1, keypoints1, mask);
    sift->detect(gray2, keypoints2);

    SparseTrackingResult result;
    result.siftCountFrame1 = static_cast<int>(keypoints1.size());
    result.siftCountFrame2 = static_cast<int>(keypoints2.size());

    if (keypoints1.empty()) {
        return result;
    }

    std::vector<cv::Point2f> points1, points2;
    cv::KeyPoint::convert(keypoints1, points1);

    std::vector<uchar> status;
    std::vector<float> errors;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                              LK_TERM_COUNT, LK_TERM_EPS);

    cv::calcOpticalFlowPyrLK(gray1, gray2, points1, points2, status, errors,
                             cv::Size(LK_WINDOW_WIDTH, LK_WINDOW_HEIGHT),
                             LK_MAX_LEVEL, criteria);

    for (size_t i = 0; i < points1.size(); ++i) {
        if (!status[i]) {
            continue;
        }

        SparseTrack track;
        track.start = points1[i];
        track.end = points2[i];
        track.error = errors[i];
        result.tracks.push_back(track);
    }

    std::sort(result.tracks.begin(), result.tracks.end(),
              [](const SparseTrack& a, const SparseTrack& b) {
                  return a.error < b.error;
              });

    if (result.tracks.size() > static_cast<size_t>(MAX_SPARSE_TRACKS_TO_DRAW)) {
        result.tracks.resize(MAX_SPARSE_TRACKS_TO_DRAW);
    }

    return result;
}

std::vector<RegionFlow> computeSparseRegionFlows(const cv::Mat& frame1,
                                                 const cv::Mat& frame2,
                                                 const cv::Mat& mask) {
    cv::Mat labels, stats, centroids;
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

    std::vector<RegionFlow> regions;
    for (int label = 1; label < count; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < MIN_REGION_AREA) {
            continue;
        }

        cv::Mat regionMask = labels == label;
        SparseTrackingResult sparse = computeSparseTracks(frame1, frame2, regionMask);

        // Regions with very few tracks do not give a useful average vector.
        if (sparse.tracks.size() < static_cast<size_t>(MIN_REGION_TRACKS)) {
            continue;
        }

        cv::Point2f sum(0.0f, 0.0f);
        for (size_t i = 0; i < sparse.tracks.size(); ++i) {
            sum += sparse.tracks[i].end - sparse.tracks[i].start;
        }

        RegionFlow region;
        region.label = label;
        region.area = static_cast<int>(sparse.tracks.size());
        region.centroid = cv::Point2f(static_cast<float>(centroids.at<double>(label, 0)),
                                      static_cast<float>(centroids.at<double>(label, 1)));
        region.averageFlow = sum * (1.0f / static_cast<float>(sparse.tracks.size()));
        regions.push_back(region);
    }

    return regions;
}

std::vector<RegionComparison> compareRegionFlows(
    const std::vector<RegionFlow>& denseRegions,
    const std::vector<RegionFlow>& sparseRegions) {
    std::vector<RegionComparison> comparisons;

    for (size_t i = 0; i < denseRegions.size(); ++i) {
        for (size_t j = 0; j < sparseRegions.size(); ++j) {
            if (denseRegions[i].label != sparseRegions[j].label) {
                continue;
            }

            RegionComparison comparison;
            comparison.label = denseRegions[i].label;
            comparison.trackCount = sparseRegions[j].area;
            comparison.denseAverage = denseRegions[i].averageFlow;
            comparison.sparseAverage = sparseRegions[j].averageFlow;
            comparison.differenceNorm =
                cv::norm(comparison.denseAverage - comparison.sparseAverage);
            comparisons.push_back(comparison);
        }
    }

    return comparisons;
}

cv::Mat drawSparseTrackingPanel(const cv::Mat& frame,
                                const SparseTrackingResult& result,
                                const std::vector<RegionFlow>& sparseRegions) {
    cv::Mat panel = frame.clone();

    for (size_t i = 0; i < result.tracks.size(); ++i) {
        cv::Point2f start = result.tracks[i].start;
        cv::Point2f end = start + (result.tracks[i].end - start)
                                * static_cast<float>(SPARSE_ARROW_SCALE);

        cv::arrowedLine(panel, start, end,
                        cv::Scalar(SPARSE_ARROW_BLUE, SPARSE_ARROW_GREEN, SPARSE_ARROW_RED),
                        SPARSE_ARROW_THICKNESS, cv::LINE_AA);
        cv::circle(panel, start, SPARSE_POINT_RADIUS,
                   cv::Scalar(SPARSE_POINT_BLUE, SPARSE_POINT_GREEN, SPARSE_POINT_RED),
                   cv::FILLED, cv::LINE_AA);
    }

    for (size_t i = 0; i < sparseRegions.size(); ++i) {
        cv::Point2f start = sparseRegions[i].centroid;
        cv::Point2f end = start + sparseRegions[i].averageFlow
                                  * static_cast<float>(SPARSE_REGION_ARROW_SCALE);
        cv::arrowedLine(panel, start, end,
                        cv::Scalar(SPARSE_REGION_ARROW_BLUE,
                                   SPARSE_REGION_ARROW_GREEN,
                                   SPARSE_REGION_ARROW_RED),
                        SPARSE_REGION_ARROW_THICKNESS, cv::LINE_AA);
    }

    addLabel(panel, "Part 2: SIFT + LK sparse flow");
    return panel;
}

OrientationStats compareFlowOrientations(const cv::Mat& denseFlow,
                                         const SparseTrackingResult& sparseResult) {
    OrientationStats stats;
    stats.comparedCount = 0;
    stats.averageCosineSimilarity = 0.0;
    stats.averageAngleDifference = 0.0;

    double cosineSum = 0.0;
    double angleSum = 0.0;

    for (size_t i = 0; i < sparseResult.tracks.size(); ++i) {
        int x = static_cast<int>(std::round(sparseResult.tracks[i].start.x));
        int y = static_cast<int>(std::round(sparseResult.tracks[i].start.y));

        if (x < 0 || y < 0 || x >= denseFlow.cols || y >= denseFlow.rows) {
            continue;
        }

        cv::Point2f denseVec = denseFlow.at<cv::Point2f>(y, x);
        cv::Point2f sparseVec = sparseResult.tracks[i].end - sparseResult.tracks[i].start;

        double denseNorm = cv::norm(denseVec);
        double sparseNorm = cv::norm(sparseVec);
        if (denseNorm < MIN_VECTOR_NORM || sparseNorm < MIN_VECTOR_NORM) {
            continue;
        }

        double cosine = denseVec.dot(sparseVec) / (denseNorm * sparseNorm);
        cosine = std::max(-1.0, std::min(1.0, cosine));

        cosineSum += cosine;
        angleSum += std::acos(cosine) * RADIANS_TO_DEGREES;
        ++stats.comparedCount;
    }

    if (stats.comparedCount > 0) {
        stats.averageCosineSimilarity = cosineSum / stats.comparedCount;
        stats.averageAngleDifference = angleSum / stats.comparedCount;
    }

    return stats;
}

void addLabel(cv::Mat& img, const std::string& text) {
    cv::rectangle(img, cv::Point(0, 0), cv::Point(img.cols, LABEL_BAR_HEIGHT),
                  cv::Scalar(LABEL_BAR_COLOR, LABEL_BAR_COLOR, LABEL_BAR_COLOR),
                  cv::FILLED);
    cv::putText(img, text, cv::Point(LABEL_TEXT_X, LABEL_TEXT_Y),
                cv::FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                cv::Scalar(LABEL_TEXT_COLOR, LABEL_TEXT_COLOR, LABEL_TEXT_COLOR),
                LABEL_FONT_THICKNESS, cv::LINE_AA);
}

cv::Mat makeGrid(const std::vector<cv::Mat>& panels) {
    cv::Mat top, bottom, grid;
    cv::hconcat(panels[0], panels[1], top);
    cv::hconcat(panels[2], panels[3], bottom);
    cv::vconcat(top, bottom, grid);
    return grid;
}

void runGridViewer() {
    std::vector<DatasetConfig> datasets = makeDatasets();
    int datasetIndex = 0;
    int pos = 0;

    std::cout << "Controls: n/space/right = next, p/left = previous, "
              << "d = switch dataset, q = quit" << std::endl;

    while (true) {
        DatasetConfig dataset = datasets[datasetIndex];
        std::vector<int> indices = loadFrameIndices(dataset.dataDir,
                                                    dataset.imagePrefix,
                                                    dataset.imageExt,
                                                    dataset.excludedSuffix);
        std::vector<int> pairStarts = findConsecutivePairs(indices);

        if (indices.size() < 2 || pairStarts.empty()) {
            std::cerr << "Error: Not enough consecutive frames for "
                      << dataset.name << std::endl;
            return;
        }

        if (pos >= static_cast<int>(pairStarts.size())) {
            pos = 0;
        }

        int frameIndex1 = pairStarts[pos];
        int frameIndex2 = frameIndex1 + 1;

        cv::Mat frame1 = loadFrame(dataset.dataDir, dataset.imagePrefix,
                                   frameIndex1, dataset.imageExt);
        cv::Mat frame2 = loadFrame(dataset.dataDir, dataset.imagePrefix,
                                   frameIndex2, dataset.imageExt);

        if (frame1.empty() || frame2.empty()) {
            std::cerr << "Error: Could not load frame pair "
                      << frameIndex1 << " and " << frameIndex2 << std::endl;
            return;
        }

        cv::Mat denseFlow = computeDenseFlow(frame1, frame2);
        cv::Mat mask;
        if (dataset.useProvidedMask) {
            mask = loadMask(dataset.maskDir, frameIndex1, dataset.imageExt, frame1.size());
        } else {
            mask = createAutoMotionMask(frame1, frame2);
        }

        std::vector<RegionFlow> denseRegions = computeRegionFlows(denseFlow, mask);
        cv::Mat densePanel = drawDenseFlowPanel(frame1, mask, denseRegions);
        printDenseFlowSummary(frameIndex1, frameIndex2, dataset.name, denseRegions);

        SparseTrackingResult sparseResult = computeSparseTracks(frame1, frame2);
        std::vector<RegionFlow> sparseRegions =
            computeSparseRegionFlows(frame1, frame2, mask);
        std::vector<RegionComparison> regionComparisons =
            compareRegionFlows(denseRegions, sparseRegions);
        cv::Mat sparsePanel = drawSparseTrackingPanel(frame1, sparseResult, sparseRegions);
        printSparseSummary(sparseResult);

        OrientationStats orientationStats = compareFlowOrientations(denseFlow, sparseResult);
        printOrientationSummary(orientationStats);
        printRegionComparisons(regionComparisons);

        std::vector<cv::Mat> panels;
        panels.push_back(preparePanel(frame1, dataset.name + " frame n = "
                                      + std::to_string(frameIndex1)));
        panels.push_back(preparePanel(frame2, dataset.name + " frame n+1 = "
                                      + std::to_string(frameIndex2)));
        panels.push_back(densePanel);
        panels.push_back(sparsePanel);

        cv::Mat grid = makeGrid(panels);
        cv::imshow(WINDOW_NAME, grid);

        int key = cv::waitKey(UI_DELAY_MS);
        if (key == KEY_QUIT) {
            break;
        } else if (key == KEY_NEXT || key == KEY_SPACE || key == KEY_RIGHT_ARROW) {
            ++pos;
            if (pos >= static_cast<int>(pairStarts.size())) {
                pos = 0;
            }
        } else if ((key == KEY_PREV || key == KEY_LEFT_ARROW) && pos > 0) {
            --pos;
        } else if (key == KEY_SWITCH_DATASET) {
            datasetIndex = (datasetIndex + 1) % static_cast<int>(datasets.size());
            pos = 0;
        }
    }

    cv::destroyAllWindows();
}
