EE576 Project 3
Author: Abdulhalim Kiraz

Included files:
- Source code: main.cpp, Functions.cpp, Functions.h, MatParser.cpp, MatParser.h, Project3_Params.h
- Build file: CMakeLists.txt
- Report source: report_hw3.tex
- Final report: EE576Project3.pdf
- Plot scripts: plot_scripts/ (uses matplotlib to plot PR CSVs; Recall on X, Precision on Y)

To build and run, the project root should contain:
- dataset/ with Washington RGB-D data: object folders bowl_1, coffee_mug_1, soda_can_1; scene
  folder kitchen_small/kitchen_small_1/ (RGB-D frames) and kitchen_small/kitchen_small_1.mat
- yolov8n.onnx in the project root (if missing: python3 -c
  "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')")

Project3_Params.h uses ../dataset, ../outputs, and ../yolov8n.onnx from build/; run the binary
  from build/ after cmake && make. outputs/ is created on the first run.

MatParser.cpp reads scene ground-truth boxes from kitchen_small_1.mat and writes
  outputs/scene_gt_kitchen_small_1.csv (same data used for scene metrics).

Build instructions:
  mkdir build
  cd build
  cmake ..
  make
  ./rgbd_recognition
