# EE576 Computer Vision Projects

This repository collects C++/OpenCV computer vision coursework projects, including camera calibration, segmentation/object detection experiments, and RGB-D object/scene recognition.

## Projects

- `hw1-camera-calibration`: camera geometry, pseudo-inverse estimation, OpenCV matching, and result visualizations.
- `hw2-object-segmentation`: C++/OpenCV project with sample images.
- `hw3-rgbd-recognition`: RGB-D recognition pipeline with result metrics, PR curves, and plotting scripts.

## What Is Excluded

Large RGB-D datasets, build directories, virtual environments, archives, and YOLO/ONNX model files are excluded. The README files inside each original project describe the expected dataset/model placement when needed.

## Build Pattern

Each subproject follows the same basic CMake workflow:

```bash
mkdir build
cd build
cmake ..
make
```

Run the generated executable from the relevant build directory. Some projects require omitted datasets or model files to be restored before execution.
