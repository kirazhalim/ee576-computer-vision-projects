/*
 * Author: Abdulhalim Kiraz
 * course: ee576
 * project no: 3
 * file: project3_params.h
 * description: stores paths, split settings, bow settings, and yolo settings.
 */

#ifndef PROJECT3_PARAMS_H
#define PROJECT3_PARAMS_H

#include <string>
#include <vector>

const std::string DATASET_DIR     = "../dataset";
const std::string OUTPUT_DIR      = "../outputs";
const std::string YOLO_MODEL_PATH = "../yolov8n.onnx";

const std::vector<std::string> CLASS_NAMES = {
    "bowl",
    "coffee_mug",
    "soda_can"
};

const std::vector<std::string> CLASS_DIRS = {
    "bowl_1",
    "coffee_mug_1",
    "soda_can_1"
};

// coco class ids used by yolov8n.onnx.
const std::vector<int> YOLO_COCO_IDS = {
    45, // bowl
    41, // cup for coffee_mug
    39  // bottle for soda_can
};

const double TRAIN_RATIO = 0.75;
const unsigned int SPLIT_SEED = 5763;

const int SIFT_FEATURES = 250;
const int MAX_DESCRIPTORS_PER_IMAGE = 80;
const int BOW_VOCAB_SIZE = 80;
const int BOW_KMEANS_ATTEMPTS = 3;

const int TARGET_WIDTH  = 640;
const int TARGET_HEIGHT = 480;

const double IOU_THRESHOLD = 0.30;
const int MIN_SCENE_AREA = 700;
const int MAX_SCENE_AREA = 45000;

const int YOLO_INPUT_SIZE = 640;
const float YOLO_NMS_THR = 0.45f;

const std::vector<double> SCORE_THRESHOLDS = {
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90
};

#endif
