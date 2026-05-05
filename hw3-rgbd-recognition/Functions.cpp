/*
 * Author: Abdulhalim Kiraz
 * course: ee576
 * project no: 3
 * file: functions.cpp
 * description: implements dataset loading, bow, scene evaluation,
 * and yolo baseline utilities.
 */

#include "Functions.h"
#include "MatParser.h"
#include "Project3_Params.h"

#include <opencv2/dnn.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sys/stat.h>

namespace {

// stores scores for one image-level sample.
struct ScoredSample {
    int gtClassId;
    std::vector<double> scores;
};

// checks a file name suffix.
bool endsWith(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// checks if a file can be opened.
bool fileExists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

// returns the file name part of a path.
std::string baseName(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return path;
    return path.substr(slash + 1);
}

// removes the file extension.
std::string removeExtension(const std::string& path) {
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return path;
    return path.substr(0, dot);
}

// maps a class name to its project id.
int classIndex(const std::string& className) {
    for (int i = 0; i < (int)CLASS_NAMES.size(); i++)
        if (CLASS_NAMES[i] == className)
            return i;
    return -1;
}

// maps a coco id to the selected class id.
int yoloClassToProjectClass(int cocoId) {
    for (int i = 0; i < (int)YOLO_COCO_IDS.size(); i++)
        if (YOLO_COCO_IDS[i] == cocoId)
            return i;
    return -1;
}

// keeps a rectangle inside image bounds.
cv::Rect clampRect(const cv::Rect& r, int cols, int rows) {
    int x1 = std::max(0, r.x);
    int y1 = std::max(0, r.y);
    int x2 = std::min(cols, r.x + r.width);
    int y2 = std::min(rows, r.y + r.height);
    if (x2 <= x1 || y2 <= y1) return cv::Rect();
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// computes intersection over union.
double rectIou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    double interArea = (double)inter.area();
    double unionArea = (double)a.area() + (double)b.area() - interArea;
    if (unionArea <= 0.0) return 0.0;
    return interArea / unionArea;
}

// loads an object mask or returns a full mask.
cv::Mat loadMask(const std::string& maskPath, const cv::Size& size) {
    cv::Mat mask;
    if (fileExists(maskPath))
        mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);

    if (mask.empty()) {
        mask = cv::Mat(size, CV_8U, cv::Scalar(255));
    } else {
        if (mask.size() != size)
            cv::resize(mask, mask, size, 0, 0, cv::INTER_NEAREST);
        cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
    }
    return mask;
}

// extracts sift descriptors inside the given mask.
cv::Mat computeSiftDescriptors(const cv::Mat& bgr, const cv::Mat& mask) {
    cv::Mat gray;
    if (bgr.channels() == 3)
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    else
        gray = bgr.clone();

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(SIFT_FEATURES);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat desc;
    sift->detectAndCompute(gray, mask, keypoints, desc);

    if (desc.empty() && !mask.empty())
        sift->detectAndCompute(gray, cv::Mat(), keypoints, desc);

    if (desc.empty())
        return desc;

    std::vector<int> order(desc.rows);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return keypoints[a].response > keypoints[b].response;
    });

    int keep = std::min((int)order.size(), MAX_DESCRIPTORS_PER_IMAGE);
    cv::Mat kept(keep, desc.cols, desc.type());
    for (int i = 0; i < keep; i++)
        desc.row(order[i]).copyTo(kept.row(i));

    return kept;
}

// extracts descriptors for one object sample.
cv::Mat computeSampleDescriptors(const ImageSample& sample) {
    cv::Mat img = cv::imread(sample.imagePath);
    if (img.empty()) return cv::Mat();

    cv::resize(img, img, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));
    cv::Mat mask = loadMask(sample.maskPath, img.size());
    cv::resize(mask, mask, img.size(), 0, 0, cv::INTER_NEAREST);
    return computeSiftDescriptors(img, mask);
}

// converts sift descriptors to a normalized bow histogram.
cv::Mat computeBowHist(const cv::Mat& descriptors, const cv::Mat& vocabulary) {
    cv::Mat hist = cv::Mat::zeros(1, vocabulary.rows, CV_32F);
    if (descriptors.empty() || vocabulary.empty())
        return hist;

    for (int i = 0; i < descriptors.rows; i++) {
        int bestIdx = 0;
        double bestDist = DBL_MAX;

        for (int k = 0; k < vocabulary.rows; k++) {
            double d = cv::norm(descriptors.row(i), vocabulary.row(k), cv::NORM_L2SQR);
            if (d < bestDist) {
                bestDist = d;
                bestIdx = k;
            }
        }
        hist.at<float>(0, bestIdx) += 1.0f;
    }

    double sumVal = cv::sum(hist)[0];
    if (sumVal > 0.0)
        hist /= sumVal;
    return hist;
}

// computes bow histogram for one image region.
cv::Mat computeImageBowHist(const std::string& imagePath,
                            const cv::Mat& roiMask,
                            const cv::Mat& vocabulary) {
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) return cv::Mat();

    cv::resize(img, img, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));
    cv::Mat mask = roiMask;
    if (!mask.empty() && mask.size() != img.size())
        cv::resize(mask, mask, img.size(), 0, 0, cv::INTER_NEAREST);

    cv::Mat desc = computeSiftDescriptors(img, mask);
    return computeBowHist(desc, vocabulary);
}

// compares two normalized histograms.
double chiSquareDistance(const cv::Mat& a, const cv::Mat& b) {
    double d = 0.0;
    for (int i = 0; i < a.cols; i++) {
        double av = a.at<float>(0, i);
        double bv = b.at<float>(0, i);
        double denom = av + bv + 1e-9;
        d += ((av - bv) * (av - bv)) / denom;
    }
    return 0.5 * d;
}

// scores one histogram against all class models.
std::vector<double> scoreAgainstModels(const cv::Mat& hist,
                                       const std::vector<BowModel>& models) {
    std::vector<double> scores(CLASS_NAMES.size(), 0.0);
    if (hist.empty()) return scores;

    for (size_t i = 0; i < models.size(); i++) {
        double dist = chiSquareDistance(hist, models[i].averageHist);
        scores[models[i].classId] = 1.0 / (1.0 + dist);
    }
    return scores;
}

// builds precision-recall rows from image-level scores.
std::vector<MetricRow> metricsFromScores(const std::string& partName,
                                         const std::vector<ScoredSample>& scored) {
    std::vector<MetricRow> rows;

    for (int c = 0; c < (int)CLASS_NAMES.size(); c++) {
        for (size_t t = 0; t < SCORE_THRESHOLDS.size(); t++) {
            double thr = SCORE_THRESHOLDS[t];
            int tp = 0, fp = 0, fn = 0;

            for (size_t i = 0; i < scored.size(); i++) {
                bool pred = scored[i].scores[c] >= thr;
                bool gt = scored[i].gtClassId == c;

                if (pred && gt) tp++;
                else if (pred && !gt) fp++;
                else if (!pred && gt) fn++;
            }

            MetricRow row;
            row.partName = partName;
            row.className = CLASS_NAMES[c];
            row.threshold = thr;
            row.tp = tp;
            row.fp = fp;
            row.fn = fn;
            row.precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
            row.recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
            rows.push_back(row);
        }
    }
    return rows;
}

// builds precision-recall rows from detection boxes.
std::vector<MetricRow> metricsFromDetections(
        const std::string& partName,
        const std::vector<std::vector<Detection> >& predictions,
        const std::vector<std::vector<Detection> >& gt) {
    std::vector<MetricRow> rows;

    for (int c = 0; c < (int)CLASS_NAMES.size(); c++) {
        for (size_t t = 0; t < SCORE_THRESHOLDS.size(); t++) {
            double thr = SCORE_THRESHOLDS[t];
            int tp = 0, fp = 0, fn = 0;

            for (size_t f = 0; f < gt.size(); f++) {
                std::vector<int> matched(gt[f].size(), 0);

                for (size_t p = 0; p < predictions[f].size(); p++) {
                    const Detection& pred = predictions[f][p];
                    if (pred.classId != c || pred.score < thr)
                        continue;

                    int bestGt = -1;
                    double bestIou = 0.0;
                    for (size_t g = 0; g < gt[f].size(); g++) {
                        if (matched[g] || gt[f][g].classId != c)
                            continue;
                        double iou = rectIou(pred.box, gt[f][g].box);
                        if (iou > bestIou) {
                            bestIou = iou;
                            bestGt = (int)g;
                        }
                    }

                    if (bestGt >= 0 && bestIou >= IOU_THRESHOLD) {
                        tp++;
                        matched[bestGt] = 1;
                    } else {
                        fp++;
                    }
                }

                for (size_t g = 0; g < gt[f].size(); g++)
                    if (!matched[g] && gt[f][g].classId == c)
                        fn++;
            }

            MetricRow row;
            row.partName = partName;
            row.className = CLASS_NAMES[c];
            row.threshold = thr;
            row.tp = tp;
            row.fp = fp;
            row.fn = fn;
            row.precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
            row.recall = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
            rows.push_back(row);
        }
    }
    return rows;
}

// extracts valid boxes from connected components.
std::vector<cv::Rect> componentsFromMask(const cv::Mat& mask) {
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);

    std::vector<cv::Rect> boxes;
    for (int i = 1; i < n; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < MIN_SCENE_AREA || area > MAX_SCENE_AREA)
            continue;

        cv::Rect r(stats.at<int>(i, cv::CC_STAT_LEFT),
                   stats.at<int>(i, cv::CC_STAT_TOP),
                   stats.at<int>(i, cv::CC_STAT_WIDTH),
                   stats.at<int>(i, cv::CC_STAT_HEIGHT));
        double aspect = (double)r.width / std::max(1, r.height);
        if (aspect < 0.15 || aspect > 5.0)
            continue;
        boxes.push_back(r);
    }
    return boxes;
}

// adds boxes while removing near duplicates.
void addBoxesUnique(std::vector<cv::Rect>& boxes,
                    const std::vector<cv::Rect>& candidates,
                    double duplicateIou) {
    for (size_t i = 0; i < candidates.size(); i++) {
        bool duplicate = false;
        for (size_t j = 0; j < boxes.size(); j++) {
            if (rectIou(candidates[i], boxes[j]) > duplicateIou) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
            boxes.push_back(candidates[i]);
    }
}

// extracts valid boxes from contours.
std::vector<cv::Rect> contoursFromMask(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> boxes;
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect r = cv::boundingRect(contours[i]);
        int area = r.area();
        if (area < MIN_SCENE_AREA || area > MAX_SCENE_AREA)
            continue;
        double aspect = (double)r.width / std::max(1, r.height);
        if (aspect < 0.15 || aspect > 5.0)
            continue;
        boxes.push_back(r);
    }
    return boxes;
}

// generates simple scene proposals from color, edge, and depth cues.
std::vector<cv::Rect> generateSceneCandidates(const cv::Mat& img,
                                              const cv::Mat& depth) {
    std::vector<cv::Rect> boxes;

    cv::Mat hsv, sat, val;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    sat = ch[1];
    val = ch[2];

    cv::Mat maskColor, maskBright;
    cv::threshold(sat, maskColor, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(val, maskBright, 35, 255, cv::THRESH_BINARY);
    cv::bitwise_and(maskColor, maskBright, maskColor);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(maskColor, maskColor, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(maskColor, maskColor, cv::MORPH_CLOSE, kernel);

    boxes = componentsFromMask(maskColor);

    cv::Mat gray, edges;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    cv::Canny(gray, edges, 60, 150);
    cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
    addBoxesUnique(boxes, contoursFromMask(edges), 0.70);

    if (!depth.empty()) {
        cv::Mat depth8, maskDepth;
        if (depth.type() != CV_8U)
            cv::normalize(depth, depth8, 0, 255, cv::NORM_MINMAX, CV_8U);
        else
            depth8 = depth.clone();

        cv::threshold(depth8, maskDepth, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::morphologyEx(maskDepth, maskDepth, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(maskDepth, maskDepth, cv::MORPH_CLOSE, kernel);
        addBoxesUnique(boxes, componentsFromMask(maskDepth), 0.70);

        // fixed depth levels give a few simple extra proposals.
        int depthThresholds[] = {50, 80, 110, 140, 170, 200};
        for (size_t k = 0; k < sizeof(depthThresholds) / sizeof(depthThresholds[0]); k++) {
            cv::threshold(depth8, maskDepth, depthThresholds[k], 255, cv::THRESH_BINARY);
            cv::morphologyEx(maskDepth, maskDepth, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(maskDepth, maskDepth, cv::MORPH_CLOSE, kernel);
            addBoxesUnique(boxes, componentsFromMask(maskDepth), 0.70);
        }
    }

    return boxes;
}

// classifies all candidate boxes in one scene image.
std::vector<Detection> classifySceneImage(const std::string& imagePath,
                                          const std::string& depthPath,
                                          const cv::Mat& vocabulary,
                                          const std::vector<BowModel>& models) {
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) return std::vector<Detection>();

    cv::resize(img, img, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));
    cv::Mat depth = cv::imread(depthPath, cv::IMREAD_UNCHANGED);
    if (!depth.empty())
        cv::resize(depth, depth, img.size(), 0, 0, cv::INTER_NEAREST);

    std::vector<cv::Rect> boxes = generateSceneCandidates(img, depth);
    std::vector<Detection> detections;

    for (size_t i = 0; i < boxes.size(); i++) {
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
        cv::rectangle(mask, boxes[i], cv::Scalar(255), cv::FILLED);

        cv::Mat hist = computeImageBowHist(imagePath, mask, vocabulary);
        std::vector<double> scores = scoreAgainstModels(hist, models);

        int bestClass = -1;
        double bestScore = 0.0;
        for (int c = 0; c < (int)scores.size(); c++) {
            if (scores[c] > bestScore) {
                bestScore = scores[c];
                bestClass = c;
            }
        }

        if (bestClass >= 0) {
            Detection d;
            d.classId = bestClass;
            d.className = CLASS_NAMES[bestClass];
            d.box = boxes[i];
            d.score = bestScore;
            detections.push_back(d);
        }
    }

    return detections;
}

// runs yolov8n on one image and keeps selected classes.
std::vector<Detection> runYoloOnImage(cv::dnn::Net& net, const cv::Mat& img) {
    std::vector<Detection> detections;
    if (img.empty()) return detections;

    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0 / 255.0,
                           cv::Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
                           cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());
    if (outs.empty()) return detections;

    cv::Mat data;
    cv::Mat out = outs[0];
    if (out.dims == 3) {
        int a = out.size[1];
        int b = out.size[2];
        if (a < b) {
            cv::Mat tmp(a, b, CV_32F, out.ptr<float>());
            data = tmp.t();
        } else {
            data = cv::Mat(a, b, CV_32F, out.ptr<float>()).clone();
        }
    } else {
        data = out.reshape(1, out.rows);
        if (data.cols < data.rows)
            data = data.t();
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> projectIds;

    float xScale = (float)img.cols / YOLO_INPUT_SIZE;
    float yScale = (float)img.rows / YOLO_INPUT_SIZE;

    for (int i = 0; i < data.rows; i++) {
        cv::Mat scores = data.row(i).colRange(4, data.cols);
        cv::Point classIdPt;
        double maxScore;
        cv::minMaxLoc(scores, 0, &maxScore, 0, &classIdPt);

        int cocoId = classIdPt.x;
        int projectId = yoloClassToProjectClass(cocoId);
        if (projectId < 0)
            continue;

        float cx = data.at<float>(i, 0);
        float cy = data.at<float>(i, 1);
        float w  = data.at<float>(i, 2);
        float h  = data.at<float>(i, 3);

        int left = (int)std::round((cx - 0.5f * w) * xScale);
        int top  = (int)std::round((cy - 0.5f * h) * yScale);
        int bw   = (int)std::round(w * xScale);
        int bh   = (int)std::round(h * yScale);

        cv::Rect box = clampRect(cv::Rect(left, top, bw, bh), img.cols, img.rows);
        if (box.area() <= 0)
            continue;

        boxes.push_back(box);
        confs.push_back((float)maxScore);
        projectIds.push_back(projectId);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confs, 0.0f, YOLO_NMS_THR, keep);

    for (size_t i = 0; i < keep.size(); i++) {
        int idx = keep[i];
        Detection d;
        d.classId = projectIds[idx];
        d.className = CLASS_NAMES[d.classId];
        d.box = boxes[idx];
        d.score = confs[idx];
        detections.push_back(d);
    }

    return detections;
}

// runs yolo on object test images.
std::vector<MetricRow> loadYoloAndEvaluateObjects(const std::vector<ImageSample>& testSamples) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(YOLO_MODEL_PATH);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::vector<ScoredSample> scored;
    for (size_t i = 0; i < testSamples.size(); i++) {
        cv::Mat img = cv::imread(testSamples[i].imagePath);
        if (img.empty()) continue;
        cv::resize(img, img, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));

        std::vector<Detection> dets = runYoloOnImage(net, img);
        ScoredSample s;
        s.gtClassId = testSamples[i].classId;
        s.scores.assign(CLASS_NAMES.size(), 0.0);
        for (size_t d = 0; d < dets.size(); d++)
            s.scores[dets[d].classId] = std::max(s.scores[dets[d].classId],
                                                 dets[d].score);
        scored.push_back(s);
    }

    return metricsFromScores("1.d_object_yolo", scored);
}

} // namespace

// creates the output folder.
void makeOutputDir() {
    mkdir(OUTPUT_DIR.c_str(), 0755);
}

// loads all selected object samples.
std::vector<ImageSample> loadObjectSamples() {
    std::vector<ImageSample> samples;

    for (int c = 0; c < (int)CLASS_NAMES.size(); c++) {
        std::string dir = DATASET_DIR + "/" + CLASS_DIRS[c];
        std::vector<cv::String> files;
        cv::glob(dir + "/*.png", files, false);
        std::sort(files.begin(), files.end());

        for (size_t i = 0; i < files.size(); i++) {
            std::string path = files[i];
            if (endsWith(path, "_mask.png") || endsWith(path, "_depth.png"))
                continue;

            std::string stem = removeExtension(path);
            ImageSample s;
            s.className = CLASS_NAMES[c];
            s.classId = c;
            s.imagePath = path;
            s.maskPath = stem + "_mask.png";
            s.depthPath = stem + "_depth.png";
            s.locPath = stem + "_loc.txt";
            samples.push_back(s);
        }
    }

    std::cout << "Loaded object samples: " << samples.size() << std::endl;
    return samples;
}

// splits samples class by class with a fixed seed.
void splitSamples(const std::vector<ImageSample>& samples,
                  std::vector<ImageSample>& trainSamples,
                  std::vector<ImageSample>& testSamples) {
    trainSamples.clear();
    testSamples.clear();

    for (int c = 0; c < (int)CLASS_NAMES.size(); c++) {
        std::vector<ImageSample> classSamples;
        for (size_t i = 0; i < samples.size(); i++)
            if (samples[i].classId == c)
                classSamples.push_back(samples[i]);

        std::mt19937 rng(SPLIT_SEED + c);
        std::shuffle(classSamples.begin(), classSamples.end(), rng);

        int nTrain = (int)std::round(TRAIN_RATIO * classSamples.size());
        for (int i = 0; i < (int)classSamples.size(); i++) {
            if (i < nTrain)
                trainSamples.push_back(classSamples[i]);
            else
                testSamples.push_back(classSamples[i]);
        }
    }

    std::cout << "Train samples: " << trainSamples.size() << std::endl;
    std::cout << "Test samples : " << testSamples.size() << std::endl;
}

// writes train and test counts.
void writeSplitSummary(const std::vector<ImageSample>& trainSamples,
                       const std::vector<ImageSample>& testSamples) {
    std::ofstream out((OUTPUT_DIR + "/split_summary.txt").c_str());
    out << "EE576 HW3 deterministic split\n";
    out << "Train ratio: " << TRAIN_RATIO << "\n";
    out << "Seed base: " << SPLIT_SEED << "\n\n";

    for (int c = 0; c < (int)CLASS_NAMES.size(); c++) {
        int tr = 0, te = 0;
        for (size_t i = 0; i < trainSamples.size(); i++)
            if (trainSamples[i].classId == c) tr++;
        for (size_t i = 0; i < testSamples.size(); i++)
            if (testSamples[i].classId == c) te++;
        out << CLASS_NAMES[c] << ": train=" << tr << ", test=" << te << "\n";
    }
}

// builds the bow vocabulary with k-means.
cv::Mat buildVocabulary(const std::vector<ImageSample>& trainSamples) {
    cv::TermCriteria term(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
                          50, 0.001);
    cv::BOWKMeansTrainer trainer(BOW_VOCAB_SIZE, term,
                                 BOW_KMEANS_ATTEMPTS,
                                 cv::KMEANS_PP_CENTERS);

    int descriptorRows = 0;
    for (size_t i = 0; i < trainSamples.size(); i++) {
        cv::Mat desc = computeSampleDescriptors(trainSamples[i]);
        if (!desc.empty()) {
            trainer.add(desc);
            descriptorRows += desc.rows;
        }
        if ((i + 1) % 100 == 0)
            std::cout << "  SIFT descriptors collected from " << (i + 1)
                      << " training images." << std::endl;
    }

    if (descriptorRows < BOW_VOCAB_SIZE)
        throw std::runtime_error("Not enough SIFT descriptors to build vocabulary.");

    std::cout << "Total descriptor rows for vocabulary: " << descriptorRows << std::endl;
    cv::Mat vocabulary = trainer.cluster();

    cv::FileStorage fs((OUTPUT_DIR + "/bow_vocabulary_1a.yml").c_str(),
                       cv::FileStorage::WRITE);
    fs << "vocabulary" << vocabulary;
    fs.release();

    return vocabulary;
}

// averages bow histograms for each class.
std::vector<BowModel> trainBowModels(const std::vector<ImageSample>& trainSamples,
                                     const cv::Mat& vocabulary) {
    std::vector<cv::Mat> sums(CLASS_NAMES.size());
    std::vector<int> counts(CLASS_NAMES.size(), 0);
    for (int c = 0; c < (int)CLASS_NAMES.size(); c++)
        sums[c] = cv::Mat::zeros(1, vocabulary.rows, CV_32F);

    for (size_t i = 0; i < trainSamples.size(); i++) {
        cv::Mat desc = computeSampleDescriptors(trainSamples[i]);
        cv::Mat hist = computeBowHist(desc, vocabulary);
        if (cv::sum(hist)[0] <= 0.0)
            continue;

        sums[trainSamples[i].classId] += hist;
        counts[trainSamples[i].classId]++;
    }

    std::vector<BowModel> models;
    cv::FileStorage fs((OUTPUT_DIR + "/bow_models_1a.yml").c_str(),
                       cv::FileStorage::WRITE);
    fs << "models" << "[";

    for (int c = 0; c < (int)CLASS_NAMES.size(); c++) {
        BowModel m;
        m.className = CLASS_NAMES[c];
        m.classId = c;
        m.trainCount = counts[c];
        if (counts[c] > 0)
            m.averageHist = sums[c] / counts[c];
        else
            m.averageHist = sums[c].clone();
        models.push_back(m);

        fs << "{";
        fs << "className" << m.className;
        fs << "trainCount" << m.trainCount;
        fs << "averageHist" << m.averageHist;
        fs << "}";
    }
    fs << "]";
    fs.release();

    return models;
}

// saves part 1.a summary.
void saveBowSummary(const cv::Mat& vocabulary,
                    const std::vector<BowModel>& models) {
    std::ofstream out((OUTPUT_DIR + "/bow_summary_1a.txt").c_str());
    out << "Part 1.a - BOW summary\n";
    out << "SIFT features per image: " << SIFT_FEATURES << "\n";
    out << "Max descriptors per image: " << MAX_DESCRIPTORS_PER_IMAGE << "\n";
    out << "Vocabulary size: " << vocabulary.rows << "\n";
    out << "Descriptor length: " << vocabulary.cols << "\n\n";
    out << "Rationale: SIFT keypoints are extracted inside the annotated object ";
    out << "masks. OpenCV BOWKMeansTrainer constructs the visual vocabulary. ";
    out << "Each class model is the average normalized BOW histogram of its ";
    out << "training samples.\n\n";

    for (size_t i = 0; i < models.size(); i++)
        out << models[i].className << ": train histograms="
            << models[i].trainCount << "\n";
}

// evaluates part 1.b.
std::vector<MetricRow> evaluateBowObjects(const std::vector<ImageSample>& testSamples,
                                          const cv::Mat& vocabulary,
                                          const std::vector<BowModel>& models) {
    std::vector<ScoredSample> scored;

    for (size_t i = 0; i < testSamples.size(); i++) {
        cv::Mat desc = computeSampleDescriptors(testSamples[i]);
        cv::Mat hist = computeBowHist(desc, vocabulary);

        ScoredSample s;
        s.gtClassId = testSamples[i].classId;
        s.scores = scoreAgainstModels(hist, models);
        scored.push_back(s);
    }

    return metricsFromScores("1.b_bow_objects", scored);
}

// loads scene boxes from the mat file.
std::vector<std::vector<Detection> > loadSceneGroundTruth() {
    std::string matPath = DATASET_DIR + "/kitchen_small/kitchen_small_1.mat";
    std::string csvPath = OUTPUT_DIR + "/scene_gt_kitchen_small_1.csv";
    return parseSceneGroundTruthMat(matPath, csvPath);
}

// evaluates part 1.c with automatic scene candidates.
std::vector<MetricRow> evaluateBowScenes(const cv::Mat& vocabulary,
                                         const std::vector<BowModel>& models,
                                         const std::vector<std::vector<Detection> >& gt) {
    std::vector<std::vector<Detection> > predictions(gt.size());

    for (size_t i = 0; i < gt.size(); i++) {
        std::string stem = DATASET_DIR + "/kitchen_small/kitchen_small_1/kitchen_small_1_"
                         + std::to_string(i + 1);
        predictions[i] = classifySceneImage(stem + ".png",
                                            stem + "_depth.png",
                                            vocabulary,
                                            models);
        if ((i + 1) % 30 == 0)
            std::cout << "  Scene BOW processed frame " << (i + 1) << std::endl;
    }

    return metricsFromDetections("1.c_bow_scenes", predictions, gt);
}

// evaluates a diagnostic upper bound for part 1.c.
std::vector<MetricRow> evaluateBowScenesGtCandidates(
        const cv::Mat& vocabulary,
        const std::vector<BowModel>& models,
        const std::vector<std::vector<Detection> >& gt) {
    std::vector<std::vector<Detection> > predictions(gt.size());

    for (size_t i = 0; i < gt.size(); i++) {
        std::string imagePath = DATASET_DIR + "/kitchen_small/kitchen_small_1/kitchen_small_1_"
                              + std::to_string(i + 1) + ".png";
        cv::Mat img = cv::imread(imagePath);
        if (img.empty())
            continue;
        cv::resize(img, img, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));

        for (size_t j = 0; j < gt[i].size(); j++) {
            cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
            cv::rectangle(mask, gt[i][j].box, cv::Scalar(255), cv::FILLED);

            cv::Mat hist = computeImageBowHist(imagePath, mask, vocabulary);
            std::vector<double> scores = scoreAgainstModels(hist, models);

            int bestClass = -1;
            double bestScore = 0.0;
            for (int c = 0; c < (int)scores.size(); c++) {
                if (scores[c] > bestScore) {
                    bestScore = scores[c];
                    bestClass = c;
                }
            }

            if (bestClass >= 0) {
                Detection d;
                d.classId = bestClass;
                d.className = CLASS_NAMES[bestClass];
                d.box = gt[i][j].box;
                d.score = bestScore;
                predictions[i].push_back(d);
            }
        }
    }

    return metricsFromDetections("1.c_bow_scene_gt_candidates", predictions, gt);
}

// evaluates part 1.d on object images.
std::vector<MetricRow> evaluateYoloObjects(const std::vector<ImageSample>& testSamples) {
    if (!fileExists(YOLO_MODEL_PATH)) {
        std::cerr << "Warning: YOLO model not found: " << YOLO_MODEL_PATH << std::endl;
        return std::vector<MetricRow>();
    }
    return loadYoloAndEvaluateObjects(testSamples);
}

// evaluates part 1.d on scene images.
std::vector<MetricRow> evaluateYoloScenes(const std::vector<std::vector<Detection> >& gt) {
    std::vector<std::vector<Detection> > predictions(gt.size());
    if (!fileExists(YOLO_MODEL_PATH)) {
        std::cerr << "Warning: YOLO model not found: " << YOLO_MODEL_PATH << std::endl;
        return std::vector<MetricRow>();
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(YOLO_MODEL_PATH);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    for (size_t i = 0; i < gt.size(); i++) {
        std::string path = DATASET_DIR + "/kitchen_small/kitchen_small_1/kitchen_small_1_"
                         + std::to_string(i + 1) + ".png";
        cv::Mat img = cv::imread(path);
        if (!img.empty()) {
            cv::resize(img, img, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));
            predictions[i] = runYoloOnImage(net, img);
        }

        if ((i + 1) % 30 == 0)
            std::cout << "  YOLO processed scene frame " << (i + 1) << std::endl;
    }

    return metricsFromDetections("1.d_scene_yolo", predictions, gt);
}

// writes metric text and csv files.
void writeMetrics(const std::string& txtPath,
                  const std::string& csvPath,
                  const std::vector<MetricRow>& rows,
                  const std::string& note) {
    std::ofstream csv(csvPath.c_str());
    csv << "part,class,threshold,tp,fp,fn,precision,recall\n";
    for (size_t i = 0; i < rows.size(); i++) {
        csv << rows[i].partName << "," << rows[i].className << ","
            << rows[i].threshold << "," << rows[i].tp << ","
            << rows[i].fp << "," << rows[i].fn << ","
            << rows[i].precision << "," << rows[i].recall << "\n";
    }

    std::ofstream out(txtPath.c_str());
    out << note << "\n\n";
    out << "Precision is plotted on Y-axis and Recall on X-axis.\n";
    out << "The best model is closest to the top-right corner (1,1).\n\n";

    out << std::left << std::setw(14) << "Class"
        << std::setw(10) << "Thr"
        << std::setw(8) << "TP"
        << std::setw(8) << "FP"
        << std::setw(8) << "FN"
        << std::setw(12) << "Precision"
        << std::setw(12) << "Recall" << "\n";
    out << std::string(72, '-') << "\n";

    out << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < rows.size(); i++) {
        out << std::left << std::setw(14) << rows[i].className
            << std::setw(10) << rows[i].threshold
            << std::setw(8) << rows[i].tp
            << std::setw(8) << rows[i].fp
            << std::setw(8) << rows[i].fn
            << std::setw(12) << rows[i].precision
            << std::setw(12) << rows[i].recall << "\n";
    }

    out << "\nDistance to ideal point (1,1):\n";
    double best = DBL_MAX;
    std::string bestClass;
    for (size_t c = 0; c < CLASS_NAMES.size(); c++) {
        double d = bestDistanceToTopRight(rows, CLASS_NAMES[c]);
        out << CLASS_NAMES[c] << ": " << d << "\n";
        if (d < best) {
            best = d;
            bestClass = CLASS_NAMES[c];
        }
    }
    if (!bestClass.empty())
        out << "Best class/model here: " << bestClass << "\n";
}

// writes the final part 1.e comparison.
void writeFinalSummary(const std::vector<MetricRow>& part1b,
                       const std::vector<MetricRow>& part1c,
                       const std::vector<MetricRow>& yoloObj,
                       const std::vector<MetricRow>& yoloScene) {
    std::ofstream out((OUTPUT_DIR + "/part1e_test_metrics.txt").c_str());
    out << "Part 1.e - Final test-set summary\n\n";
    out << "All runs use 75/25 split from split_summary.txt.\n";
    out << "The selected classical model is average BOW descriptor per class.\n";
    out << "Best/worst class is decided by closeness of the Precision-Recall ";
    out << "curve to the ideal point (Recall=1, Precision=1).\n\n";

    std::vector<std::pair<std::string, const std::vector<MetricRow>* > > parts;
    parts.push_back(std::make_pair("1.b BOW object images", &part1b));
    parts.push_back(std::make_pair("1.c BOW scene images", &part1c));
    parts.push_back(std::make_pair("1.d YOLO object images", &yoloObj));
    parts.push_back(std::make_pair("1.d YOLO scene images", &yoloScene));

    out << std::fixed << std::setprecision(3);
    for (size_t p = 0; p < parts.size(); p++) {
        out << parts[p].first << "\n";
        for (size_t c = 0; c < CLASS_NAMES.size(); c++)
            out << "  " << CLASS_NAMES[c] << ": distance_to_(1,1)="
                << bestDistanceToTopRight(*parts[p].second, CLASS_NAMES[c]) << "\n";
        out << "\n";
    }
}

// finds the best point on a pr curve.
double bestDistanceToTopRight(const std::vector<MetricRow>& rows,
                              const std::string& className) {
    double best = DBL_MAX;
    for (size_t i = 0; i < rows.size(); i++) {
        if (rows[i].className != className)
            continue;
        double d = std::sqrt((1.0 - rows[i].precision) * (1.0 - rows[i].precision) +
                             (1.0 - rows[i].recall) * (1.0 - rows[i].recall));
        best = std::min(best, d);
    }
    return best;
}
