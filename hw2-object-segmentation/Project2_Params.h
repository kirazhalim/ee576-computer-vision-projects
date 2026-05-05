/*
 * Author: Abdulhalim Kiraz
 * Course: EE576
 * Project No: 2
 * File: Project2_Params.h
 * Description: Stores the constants for the project2 such as color map size,
 * segment filtering threshold, display layout, and YOLO parameters.
 */

#ifndef PROJECT2_PARAMS_H
#define PROJECT2_PARAMS_H

#include <string>

// Display grid settings: 2x2 arrangement
const int GRID_ROWS = 2;
const int GRID_COLS = 2;

// Resizing dimensions for all input images
const int TARGET_WIDTH  = 640;
const int TARGET_HEIGHT = 480;

// Number of colors in the color map (M)
const int NUM_COLORS = 32;

// Minimum segment size as a fraction of total image pixels (tau_min)
// A segment must have at least TAU_MIN * N1 * N2 pixels to be kept.
const double TAU_MIN = 0.005;

// Relative paths to the input images from the build directory.
// Images should be placed in the images/ folder next to the build directory.
const std::string IMG_BOWL    = "../images/bowl.png";
const std::string IMG_BANANA  = "../images/banana.png";
const std::string IMG_KITCHEN = "../images/kitchen.png";

// Path to the YOLOv8-seg ONNX model file.
const std::string YOLO_MODEL_PATH = "../yolov8n-seg.onnx";

// YOLO inference parameters
const float YOLO_CONF_THR = 0.40f;
const float YOLO_NMS_THR  = 0.45f;
const int   YOLO_INPUT_SZ = 640;

#endif
