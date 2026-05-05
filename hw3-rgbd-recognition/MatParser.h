/*
 * Author: Abdulhalim Kiraz
 * course: ee576
 * project no: 3
 * file: matparser.h
 * description: function declaration for reading matlab scene boxes.
 */

#ifndef MATPARSER_H
#define MATPARSER_H

#include "Functions.h"

#include <string>
#include <vector>

// reads the scene ground truth from the matlab file.
std::vector<std::vector<Detection> > parseSceneGroundTruthMat(const std::string& matPath,
                                                              const std::string& csvPath);

#endif
