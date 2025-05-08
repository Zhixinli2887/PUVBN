# GSV Projection and Template Matching

This repository contains Python scripts for processing Google Street View (GSV) images, performing template matching, and conducting bundle adjustment (BA) for geospatial analysis. The project leverages various libraries and tools for image processing, geospatial transformations, and 3D reconstruction.

## Features

- **GSV Image Processing**: Extracts metadata and processes GSV images for geospatial analysis.
- **Template Matching**: Matches templates to GSV images using a deep learning-based model.
- **Bundle Adjustment (BA)**: Performs BA to estimate camera positions and refine geospatial data.
- **Geospatial Transformations**: Converts coordinates between different spatial reference systems.
- **Integration with Metashape**: Uses Metashape for 3D reconstruction and camera alignment.

## Features

![workflow](https://github.com/user-attachments/assets/a6090841-6077-4c64-994a-c4518508bbd5)

## Datasets

The query images are open access at:
The GSVs are downloaded through https://svd360.com/

## Requirements

The project requires the following dependencies:

- Python 3.8+
- Libraries:
  - `osmnx`
  - `numpy`
  - `pandas`
  - `cv2` (OpenCV)
  - `Pillow`
  - `Metashape`
  - `pyproj`
  - `scipy`
  - `tqdm`
  - `py360convert`
  - `torch` (for deep learning models)
- Pre-trained deep learning model: VGG19 (used for template matching)

## Citations
```bash
@ARTICLE{Li2024-pa,
  title    = "Urban visual localization of block-wise monocular images with
              Google street views",
  author   = "Li, Zhixin and Li, Shuang and Anderson, John and Shan, Jie",
  journal  = "Remote Sens",
  month    =  feb,
  year     =  2024
}
```
