/*
 * Author: Abdulhalim Kiraz
 * Student ID: 2021401219
 * Course: EE576
 * Project No: 4
 * File: Project4_Params.h
 * Description: Stores dataset paths, display settings, and algorithm
 * parameters used in Project 4.
 */

#ifndef PROJECT4_PARAMS_H
#define PROJECT4_PARAMS_H

#include <string>

// Display dimensions for each panel.
const int TARGET_WIDTH = 640;
const int TARGET_HEIGHT = 480;

// Relative paths from the build directory.
const std::string TUM_DATA_DIR = "../tum_freiburg3_sitting_static";
const std::string TUM_MASK_DIR = "../tum_freiburg3_sitting_static/masks";
const std::string TUM_IMAGE_PREFIX = "";
const std::string TUM_IMAGE_EXT = ".jpg";
const std::string TUM_EXCLUDED_SUFFIX = "";

const std::string KITCHEN_DATA_DIR = "../kitchen";
const std::string KITCHEN_IMAGE_PREFIX = "kitchen_small_1_";
const std::string KITCHEN_IMAGE_EXT = ".png";
const std::string KITCHEN_EXCLUDED_SUFFIX = "_depth";

// Keyboard controls.
const int UI_DELAY_MS = 0;
const char KEY_QUIT = 'q';
const char KEY_NEXT = 'n';
const char KEY_PREV = 'p';
const char KEY_SWITCH_DATASET = 'd';
const int KEY_SPACE = 32;
const int KEY_LEFT_ARROW = 81;
const int KEY_RIGHT_ARROW = 83;

// Label drawing settings.
const int LABEL_BAR_HEIGHT = 26;
const int LABEL_TEXT_X = 6;
const int LABEL_TEXT_Y = 18;
const double LABEL_FONT_SCALE = 0.55;
const int LABEL_FONT_THICKNESS = 1;
const int LABEL_BAR_COLOR = 30;
const int LABEL_TEXT_COLOR = 235;

// Display labels.
const std::string WINDOW_NAME = "EE576 Project 4";

// Mask and dense optical flow parameters.
const int MASK_THRESHOLD_VALUE = 10;
const int MASK_MAX_VALUE = 255;
const int MIN_REGION_AREA = 80;
const double FARNEBACK_PYR_SCALE = 0.5;
const int FARNEBACK_LEVELS = 3;
const int FARNEBACK_WINDOW_SIZE = 15;
const int FARNEBACK_ITERATIONS = 3;
const int FARNEBACK_POLY_N = 5;
const double FARNEBACK_POLY_SIGMA = 1.2;
const int FARNEBACK_FLAGS = 0;

// Automatic mask parameters for the kitchen sequence.
const int AUTO_MASK_THRESHOLD_VALUE = 18;
const int AUTO_MASK_BLUR_SIZE = 5;
const int AUTO_MASK_MORPH_SIZE = 5;
const int AUTO_MASK_DILATE_ITERATIONS = 1;

// Dense flow drawing settings.
const double DENSE_ARROW_SCALE = 8.0;
const int DENSE_ARROW_THICKNESS = 2;
const int REGION_BORDER_THICKNESS = 1;
const double MASK_OVERLAY_ALPHA = 0.30;
const int DENSE_ARROW_BLUE = 0;
const int DENSE_ARROW_GREEN = 0;
const int DENSE_ARROW_RED = 255;
const int REGION_BORDER_BLUE = 0;
const int REGION_BORDER_GREEN = 255;
const int REGION_BORDER_RED = 255;

// Sparse tracking parameters.
const int SIFT_FEATURE_COUNT = 250;
const int LK_WINDOW_WIDTH = 21;
const int LK_WINDOW_HEIGHT = 21;
const int LK_MAX_LEVEL = 3;
const int LK_TERM_COUNT = 30;
const double LK_TERM_EPS = 0.01;
const int MAX_SPARSE_TRACKS_TO_DRAW = 120;
const double SPARSE_ARROW_SCALE = 1.0;
const int SPARSE_ARROW_THICKNESS = 1;
const int SPARSE_POINT_RADIUS = 2;
const int SPARSE_ARROW_BLUE = 0;
const int SPARSE_ARROW_GREEN = 255;
const int SPARSE_ARROW_RED = 0;
const int SPARSE_POINT_BLUE = 0;
const int SPARSE_POINT_GREEN = 0;
const int SPARSE_POINT_RED = 255;
const int MIN_REGION_TRACKS = 2;
const double SPARSE_REGION_ARROW_SCALE = 8.0;
const int SPARSE_REGION_ARROW_THICKNESS = 2;
const int SPARSE_REGION_ARROW_BLUE = 255;
const int SPARSE_REGION_ARROW_GREEN = 0;
const int SPARSE_REGION_ARROW_RED = 0;

// Orientation comparison parameters.
const double MIN_VECTOR_NORM = 0.05;
const double RADIANS_TO_DEGREES = 180.0 / 3.14159265358979323846;

#endif
