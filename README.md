# EE576 Machine Vision Projects

This repository contains three C++/OpenCV projects from my EE576 machine vision coursework. I worked on camera calibration, object segmentation, and RGB-D object/scene recognition.

The repository includes 3 subprojects, CMake files, source code, selected images, reports, and result metrics. I kept the large datasets and model files out of the repository.

## Projects

- `hw1-camera-calibration`: camera calibration and projection experiments.
- `hw2-object-segmentation`: object segmentation with sample images.
- `hw3-rgbd-recognition`: RGB-D object and scene recognition with BOW and YOLO-based experiments.

## What I Did

- Implemented the main pipelines in C++ with OpenCV.
- Used CMake for each subproject.
- Added selected result images and precision-recall outputs.
- Compared classical BOW descriptors and YOLO-based results in the RGB-D project.
- Kept the final reports inside the related project folders.

## How to Build

Each subproject follows a similar pattern:

```bash
mkdir build
cd build
cmake ..
make
```

Some parts need external datasets or model files before running.
