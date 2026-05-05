/*
 * Author: Abdulhalim Kiraz
 * Course: EE576
 * Project No: 1
 * File: Project1_Params.h
 * Description: Stores project-wide constants for display layout, image size,
 * data settings, and user interface parameters.
 */

#ifndef PROJECT1_PARAMS_H
#define PROJECT1_PARAMS_H

#include <string>

// Display grid settings: 1x3 arrangement
const int GRID_ROWS = 1;
const int GRID_COLS = 3;

// Resizing dimensions
const int TARGET_HEIGHT = 480;
const int TARGET_WIDTH = 640;

// Image sequence indices
const int START_INDEX = 8;

// Minimum number of point pairs used in the project
const int MIN_CORRESPONDING_POINTS = 4;

// Relative path to the dataset folder used in this project.
// The folder corridor_human2-Small is expected to be located one level above the build directory.
const std::string DATA_DIR = "../corridor_human2-Small/";
const std::string IMG_EXT = ".jpg";

// UI parameters
const int UI_DELAY_MS = 10;
const int KEY_ESC = 27;

#endif