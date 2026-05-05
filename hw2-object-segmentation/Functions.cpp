/*
 * Author: Abdulhalim Kiraz
 * Course: EE576
 * Project No: 2
 * File: Functions.cpp
 * Description: Implements image enhancement, color map construction,
 * quantization, connected component segmentation, segment filtering,
 * YOLOv8-seg inference, and display utilities.
 */

#include "Functions.h"
#include "Project2_Params.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <numeric>

cv::Mat enhanceImage(const cv::Mat& src) {
    // Convert BGR to LAB so that only the lightness channel is modified.
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels(3);
    cv::split(lab, channels);

    // Apply CLAHE to the L channel with a clip limit of 2.0.
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);

    cv::merge(channels, lab);

    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    return result;
}

std::vector<cv::Vec3b> buildColorMap(int M) {
    std::vector<cv::Vec3b> cmap(M);

    for (int i = 0; i < M; i++) {
        // Spread hue evenly across [0, 179] (OpenCV HSV range).
        uchar h = static_cast<uchar>(179 * i / M);
        uchar s = 210;
        uchar v = 220;

        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, s, v));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cmap[i] = bgr.at<cv::Vec3b>(0, 0);
    }

    return cmap;
}

cv::Mat quantizeImage(const cv::Mat& src,
                      const std::vector<cv::Vec3b>& cmap,
                      int M,
                      cv::Mat& labelImg) {
    // Convert to grayscale for intensity-based bin assignment.
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    int rows = src.rows;
    int cols = src.cols;

    labelImg = cv::Mat(rows, cols, CV_32S);
    cv::Mat quantized(rows, cols, CV_8UC3);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // Map intensity [0, 255] to bin index [0, M-1].
            int bin = static_cast<int>(gray.at<uchar>(r, c)) * M / 256;
            if (bin >= M) bin = M - 1;

            labelImg.at<int>(r, c)    = bin;
            quantized.at<cv::Vec3b>(r, c) = cmap[bin];
        }
    }

    return quantized;
}

cv::Mat runConnectedComponents(const cv::Mat& labelImg,
                               const std::vector<cv::Vec3b>& cmap,
                               int M,
                               std::vector<SegmentInfo>& segments) {
    int rows = labelImg.rows;
    int cols = labelImg.cols;

    cv::Mat segAll = cv::Mat::zeros(rows, cols, CV_8UC3);
    segments.clear();

    for (int bin = 0; bin < M; bin++) {
        // Build a binary mask for pixels belonging to this bin.
        cv::Mat mask(rows, cols, CV_8U, cv::Scalar(0));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if (labelImg.at<int>(r, c) == bin)
                    mask.at<uchar>(r, c) = 255;

        if (cv::countNonZero(mask) == 0)
            continue;

        // Find connected components within this color bin.
        cv::Mat compLabels, stats, centroids;
        int nComps = cv::connectedComponentsWithStats(mask, compLabels, stats, centroids);

        // Component 0 is the background, so start from 1.
        for (int cid = 1; cid < nComps; cid++) {
            int pxCount = stats.at<int>(cid, cv::CC_STAT_AREA);

            SegmentInfo seg;
            seg.label      = bin;
            seg.pixelCount = pxCount;
            seg.color      = cmap[bin];
            segments.push_back(seg);

            // Paint this component on the output image.
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    if (compLabels.at<int>(r, c) == cid)
                        segAll.at<cv::Vec3b>(r, c) = cmap[bin];
        }
    }

    return segAll;
}

cv::Mat filterSegments(const std::vector<SegmentInfo>& segments,
                       int rows, int cols,
                       double tauMin,
                       const cv::Mat& labelImg,
                       const std::vector<cv::Vec3b>& cmap,
                       std::vector<SegmentInfo>& keptSegments) {
    // A segment must have at least this many pixels to be kept.
    int minPixels = static_cast<int>(tauMin * rows * cols);

    keptSegments.clear();
    for (size_t i = 0; i < segments.size(); i++) {
        if (segments[i].pixelCount >= minPixels)
            keptSegments.push_back(segments[i]);
    }

    // Print a summary table of the kept segments.
    std::cout << "\n--- Filtered Segments (tau_min=" << tauMin << ") ---" << std::endl;
    std::cout << std::left << std::setw(8)  << "Label"
                           << std::setw(12) << "Pixels"
                           << "Color (B,G,R)" << std::endl;
    std::cout << std::string(38, '-') << std::endl;

    for (size_t i = 0; i < keptSegments.size(); i++) {
        const SegmentInfo& s = keptSegments[i];
        std::cout << std::left << std::setw(8)  << s.label
                               << std::setw(12) << s.pixelCount
                               << "("  << (int)s.color[0]
                               << "," << (int)s.color[1]
                               << "," << (int)s.color[2] << ")" << std::endl;
    }

    // Compute size mean and variance.
    if (!keptSegments.empty()) {
        double sum = 0.0;
        for (size_t i = 0; i < keptSegments.size(); i++)
            sum += keptSegments[i].pixelCount;
        double mean = sum / keptSegments.size();

        double varSum = 0.0;
        for (size_t i = 0; i < keptSegments.size(); i++) {
            double diff = keptSegments[i].pixelCount - mean;
            varSum += diff * diff;
        }
        double variance = varSum / keptSegments.size();

        std::cout << "\nNumber of kept segments : " << keptSegments.size() << std::endl;
        std::cout << "Mean size (pixels)      : " << std::fixed << std::setprecision(1) << mean    << std::endl;
        std::cout << "Variance                : " << std::fixed << std::setprecision(1) << variance << std::endl;
    }

    // Mark which bins have at least one kept segment.
    std::vector<bool> keepBin(cmap.size(), false);
    for (size_t i = 0; i < keptSegments.size(); i++)
        keepBin[keptSegments[i].label] = true;

    // Repaint only the pixels belonging to kept bins.
    cv::Mat filtered = cv::Mat::zeros(rows, cols, CV_8UC3);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int bin = labelImg.at<int>(r, c);
            if (keepBin[bin])
                filtered.at<cv::Vec3b>(r, c) = cmap[bin];
        }

    return filtered;
}

cv::Mat runYoloSeg(const cv::Mat& src,
                   const std::string& modelPath,
                   float confThr,
                   float nmsThr) {
    // Return a placeholder if the model file is not found.
    std::ifstream f(modelPath);
    if (!f.good()) {
        cv::Mat placeholder = src.clone();
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
        cv::addWeighted(src, 0.5, gray, 0.5, 0, placeholder);
        cv::putText(placeholder, "Model not found: " + modelPath,
                    cv::Point(10, src.rows / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 0, 220), 2);
        return placeholder;
    }

    // Load the ONNX model via OpenCV DNN.
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // YOLOv8 expects a 640x640 blob normalized to [0, 1].
    cv::Mat blob;
    cv::dnn::blobFromImage(src, blob, 1.0 / 255.0,
                           cv::Size(YOLO_INPUT_SZ, YOLO_INPUT_SZ),
                           cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);

    // YOLOv8-seg produces two outputs:
    //   output0 : [1, 116, 8400]  boxes + scores + mask coefficients
    //   output1 : [1,  32, 160, 160]  prototype masks
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    // Reshape output0 from [1, 116, 8400] to [8400, 116].
    cv::Mat output0 = outs[0].reshape(1, outs[0].size[1]).t();
    cv::Mat output1 = outs[1];

    int protoH = output1.size[2];
    int protoW = output1.size[3];

    float xScale = static_cast<float>(src.cols) / YOLO_INPUT_SZ;
    float yScale = static_cast<float>(src.rows) / YOLO_INPUT_SZ;

    std::vector<cv::Rect>  boxes;
    std::vector<float>     confidences;
    std::vector<int>       classIds;
    std::vector<cv::Mat>   maskCoefs;

    // Parse each detection row.
    for (int i = 0; i < output0.rows; i++) {
        // Class scores are in columns 4 to 83.
        cv::Mat scores = output0.row(i).colRange(4, 84);
        cv::Point classIdPt;
        double maxScore;
        cv::minMaxLoc(scores, 0, &maxScore, 0, &classIdPt);

        if (maxScore < confThr)
            continue;

        float cx = output0.at<float>(i, 0);
        float cy = output0.at<float>(i, 1);
        float bw = output0.at<float>(i, 2);
        float bh = output0.at<float>(i, 3);

        int x1    = static_cast<int>((cx - bw / 2) * xScale);
        int y1    = static_cast<int>((cy - bh / 2) * yScale);
        int bwPx  = static_cast<int>(bw * xScale);
        int bhPx  = static_cast<int>(bh * yScale);

        boxes.push_back(cv::Rect(x1, y1, bwPx, bhPx));
        confidences.push_back(static_cast<float>(maxScore));
        classIds.push_back(classIdPt.x);
        maskCoefs.push_back(output0.row(i).colRange(84, 116).clone());
    }

    // Apply NMS to remove redundant boxes.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThr, nmsThr, indices);

    // Build a color map for drawing masks.
    std::vector<cv::Vec3b> maskColors = buildColorMap(80);

    // Reshape prototype masks to [32, protoH * protoW].
    cv::Mat protos = output1.reshape(1, 32);

    cv::Mat result = src.clone();

    for (size_t k = 0; k < indices.size(); k++) {
        int idx = indices[k];

        // Compute mask: maskCoefs * protos, then sigmoid.
        cv::Mat maskFlat = maskCoefs[idx] * protos;
        maskFlat = maskFlat.reshape(1, protoH);

        for (int r = 0; r < maskFlat.rows; r++)
            for (int c = 0; c < maskFlat.cols; c++) {
                float v = maskFlat.at<float>(r, c);
                maskFlat.at<float>(r, c) = 1.0f / (1.0f + std::exp(-v));
            }

        // Resize mask to original image dimensions.
        cv::Mat maskResized;
        cv::resize(maskFlat, maskResized, cv::Size(src.cols, src.rows));

        cv::Vec3b color = maskColors[classIds[idx] % 80];

        // Blend mask pixels: 60% original, 40% color.
        for (int r = 0; r < result.rows; r++)
            for (int c = 0; c < result.cols; c++)
                if (maskResized.at<float>(r, c) > 0.5f) {
                    cv::Vec3b& px = result.at<cv::Vec3b>(r, c);
                    px[0] = static_cast<uchar>(0.6 * px[0] + 0.4 * color[0]);
                    px[1] = static_cast<uchar>(0.6 * px[1] + 0.4 * color[1]);
                    px[2] = static_cast<uchar>(0.6 * px[2] + 0.4 * color[2]);
                }

        // Draw the bounding box.
        cv::Rect box = boxes[idx] & cv::Rect(0, 0, src.cols, src.rows);
        cv::rectangle(result, box, cv::Scalar(color[0], color[1], color[2]), 2);

        // Show class id and confidence.
        std::string lbl = "id:" + std::to_string(classIds[idx])
                        + " " + std::to_string(static_cast<int>(confidences[idx] * 100)) + "%";
        cv::putText(result, lbl, cv::Point(box.x, std::max(box.y - 5, 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
    }

    // Show total detection count.
    cv::putText(result, "Detections: " + std::to_string(indices.size()),
                cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(0, 220, 0), 2);

    return result;
}

cv::Mat makeGrid(const std::vector<cv::Mat>& panels) {
    // Combine four panels into a 2x2 grid.
    cv::Mat top, bottom, grid;
    cv::hconcat(panels[0], panels[1], top);
    cv::hconcat(panels[2], panels[3], bottom);
    cv::vconcat(top, bottom, grid);
    return grid;
}

void addLabel(cv::Mat& img, const std::string& text) {
    // Draw a dark bar at the top and write the title in white.
    cv::rectangle(img, cv::Point(0, 0), cv::Point(img.cols, 26),
                  cv::Scalar(30, 30, 30), cv::FILLED);
    cv::putText(img, text, cv::Point(6, 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
}
