# EE576 Machine Vision Projects

This repository collects C++/OpenCV computer vision coursework projects, including camera calibration, segmentation/object detection experiments, RGB-D object/scene recognition, and optical flow.

## Projects

- `hw1-camera-calibration`: camera geometry, pseudo-inverse estimation, OpenCV matching, and result visualizations.
- `hw2-object-segmentation`: C++/OpenCV project with sample images.
- `hw3-rgbd-recognition`: RGB-D recognition pipeline with result metrics, PR curves, and plotting scripts.
- `hw4-optical-flow`: dense optical flow, SIFT + Lucas-Kanade sparse tracking, moving-region masks, and flow comparison results.

## How to Build

Each subproject follows a similar pattern:

```bash
mkdir build
cd build
cmake ..
make
```

Some parts need external datasets or model files before running.
