/*
 * Author: Abdulhalim Kiraz
 * course: ee576
 * project no: 3
 * file: main.cpp
 * description: runs the full hw3 pipeline and saves all metric summaries.
 */

#include "Functions.h"
#include "Project3_Params.h"
#include <iostream>

int main() {
    std::cout << "EE576 Project 3 - Object Representation and Recognition" << std::endl;

    makeOutputDir();

    std::vector<ImageSample> samples = loadObjectSamples();
    std::vector<ImageSample> trainSamples, testSamples;
    splitSamples(samples, trainSamples, testSamples);
    writeSplitSummary(trainSamples, testSamples);

    // part 1.a: learn the visual words and class models.
    std::cout << "\nPart 1.a - Building BOW vocabulary and class models..." << std::endl;
    cv::Mat vocabulary = buildVocabulary(trainSamples);
    std::vector<BowModel> models = trainBowModels(trainSamples, vocabulary);
    saveBowSummary(vocabulary, models);

    // part 1.b: test bow on object images.
    std::cout << "\nPart 1.b - Evaluating BOW on object test images..." << std::endl;
    std::vector<MetricRow> part1b = evaluateBowObjects(testSamples, vocabulary, models);
    writeMetrics(OUTPUT_DIR + "/part1b_metrics.txt",
                 OUTPUT_DIR + "/part1b_pr.csv",
                 part1b,
                 "Average BOW descriptors with Chi-square histogram distance. "
                 "Scores are 1/(1+distance). This follows the Lecture 8 "
                 "precision/recall evaluation idea.");

    // part 1.c: test bow after classical scene segmentation.
    std::cout << "\nPart 1.c - Evaluating BOW on segmented scene images..." << std::endl;
    std::vector<std::vector<Detection> > sceneGt = loadSceneGroundTruth();
    std::vector<MetricRow> part1c = evaluateBowScenes(vocabulary, models, sceneGt);
    writeMetrics(OUTPUT_DIR + "/part1c_metrics.txt",
                 OUTPUT_DIR + "/part1c_pr.csv",
                 part1c,
                 "Scene candidates are generated with thresholding, morphology, "
                 "and connected components, consistent with Lectures 5 and 6. "
                 "Candidates are then recognized with the BOW models.");

    std::vector<MetricRow> part1cGt =
        evaluateBowScenesGtCandidates(vocabulary, models, sceneGt);
    writeMetrics(OUTPUT_DIR + "/part1c_gt_candidates_metrics.txt",
                 OUTPUT_DIR + "/part1c_gt_candidates_pr.csv",
                 part1cGt,
                 "Diagnostic upper bound for Part 1.c. Parsed scene annotations "
                 "are used only as candidate boxes, then recognition is still done "
                 "with the object-trained BOW models. This separates segmentation "
                 "failure from recognition failure and should be reported as an "
                 "annotation-assisted analysis, not as the automatic segmentation result.");

    // part 1.d: run the deep baseline with the selected coco ids.
    std::cout << "\nPart 1.d - Evaluating YOLO baseline..." << std::endl;
    std::vector<MetricRow> yoloObj = evaluateYoloObjects(testSamples);
    writeMetrics(OUTPUT_DIR + "/part1d_object_metrics.txt",
                 OUTPUT_DIR + "/part1d_object_pr.csv",
                 yoloObj,
                 "YOLO baseline uses yolov8n.onnx through OpenCV DNN. "
                 "Class mapping: bowl=45, coffee_mug->cup=41, soda_can->bottle=39.");

    std::vector<MetricRow> yoloScene = evaluateYoloScenes(sceneGt);
    writeMetrics(OUTPUT_DIR + "/part1d_scene_metrics.txt",
                 OUTPUT_DIR + "/part1d_scene_pr.csv",
                 yoloScene,
                 "YOLO scene detections are matched against parsed kitchen_small_1.mat "
                 "ground-truth boxes using IoU >= 0.30.");

    // part 1.e: collect the final comparison.
    std::cout << "\nPart 1.e - Writing final test summary..." << std::endl;
    writeFinalSummary(part1b, part1c, yoloObj, yoloScene);

    std::cout << "\nDone. Outputs saved under: " << OUTPUT_DIR << std::endl;
    return 0;
}
