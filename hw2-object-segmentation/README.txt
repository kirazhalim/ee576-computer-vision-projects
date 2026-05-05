EE576 Project 2
Author: Abdulhalim Kiraz

Included files:
- Source code: main.cpp, Functions.cpp, Functions.h, Project2_Params.h
- Build file: CMakeLists.txt

Note:
The dataset images are not included in this archive.
Three images should be placed in an images/ folder next to the build directory:
  images/apple.png    (Washington RGB-D object dataset - apple)
  images/banana.png   (Washington RGB-D object dataset - banana)
  images/kitchen.png  (Washington RGB-D scene dataset - kitchen)

The YOLOv8 model file is also not included.
Download yolov8n-seg.onnx and place it next to the build directory.
To export from Python: python3 -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt').export(format='onnx')"

Build instructions:
  mkdir build
  cd build
  cmake ..
  make
  ./object_segmentation
