/*
 * Author: Abdulhalim Kiraz
 * course: ee576
 * project no: 3
 * file: functions.h
 * description: function declarations for dataset loading, bow learning,
 * scene evaluation, and yolo baseline.
 */

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// stores one rgb-d object image and its related files.
struct ImageSample {
    std::string className;
    int classId;
    std::string imagePath;
    std::string maskPath;
    std::string depthPath;
    std::string locPath;
};

// stores the average bow histogram of one class.
struct BowModel {
    std::string className;
    int classId;
    cv::Mat averageHist;
    int trainCount;
};

// stores one predicted or ground-truth object box.
struct Detection {
    int classId;
    std::string className;
    cv::Rect box;
    double score;
};

// stores one precision-recall row for one class and threshold.
struct MetricRow {
    std::string partName;
    std::string className;
    double threshold;
    int tp;
    int fp;
    int fn;
    double precision;
    double recall;
};

// creates the outputs folder if it does not exist.
void makeOutputDir();

// reads the three selected object folders.
std::vector<ImageSample> loadObjectSamples();

// makes a deterministic train-test split.
void splitSamples(const std::vector<ImageSample>& samples,
                  std::vector<ImageSample>& trainSamples,
                  std::vector<ImageSample>& testSamples);

// saves the split counts to a text file.
void writeSplitSummary(const std::vector<ImageSample>& trainSamples,
                       const std::vector<ImageSample>& testSamples);

// builds the visual vocabulary from training descriptors.
cv::Mat buildVocabulary(const std::vector<ImageSample>& trainSamples);

// trains one average bow model for each class.
std::vector<BowModel> trainBowModels(const std::vector<ImageSample>& trainSamples,
                                     const cv::Mat& vocabulary);

// saves the bow settings and model counts.
void saveBowSummary(const cv::Mat& vocabulary,
                    const std::vector<BowModel>& models);

// evaluates bow recognition on object test images.
std::vector<MetricRow> evaluateBowObjects(const std::vector<ImageSample>& testSamples,
                                          const cv::Mat& vocabulary,
                                          const std::vector<BowModel>& models);

// loads kitchen scene ground truth boxes.
std::vector<std::vector<Detection> > loadSceneGroundTruth();

// evaluates automatic scene segmentation followed by bow.
std::vector<MetricRow> evaluateBowScenes(const cv::Mat& vocabulary,
                                         const std::vector<BowModel>& models,
                                         const std::vector<std::vector<Detection> >& gt);

// evaluates bow with gt boxes as diagnostic candidates.
std::vector<MetricRow> evaluateBowScenesGtCandidates(const cv::Mat& vocabulary,
                                                     const std::vector<BowModel>& models,
                                                     const std::vector<std::vector<Detection> >& gt);

// evaluates yolov8n on object test images.
std::vector<MetricRow> evaluateYoloObjects(const std::vector<ImageSample>& testSamples);

// evaluates yolov8n on kitchen scene images.
std::vector<MetricRow> evaluateYoloScenes(const std::vector<std::vector<Detection> >& gt);

// writes metric tables and pr csv files.
void writeMetrics(const std::string& txtPath,
                  const std::string& csvPath,
                  const std::vector<MetricRow>& rows,
                  const std::string& note);

// writes the final part 1.e summary.
void writeFinalSummary(const std::vector<MetricRow>& part1b,
                       const std::vector<MetricRow>& part1c,
                       const std::vector<MetricRow>& yoloObj,
                       const std::vector<MetricRow>& yoloScene);

// returns the best distance to the ideal pr point.
double bestDistanceToTopRight(const std::vector<MetricRow>& rows,
                              const std::string& className);

#endif
