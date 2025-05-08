# GSV Projection and Template Matching

This repository contains Python scripts for processing Google Street View (GSV) images, performing template matching, and conducting bundle adjustment (BA) for geospatial analysis. The project leverages various libraries and tools for image processing, geospatial transformations, and 3D reconstruction.

## Features

- **GSV Image Processing**: Extracts metadata and processes GSV images for geospatial analysis.
- **Template Matching**: Matches templates to GSV images using a deep learning-based model.
- **Bundle Adjustment (BA)**: Performs BA to estimate camera positions and refine geospatial data.
- **Geospatial Transformations**: Converts coordinates between different spatial reference systems.
- **Integration with Metashape**: Uses Metashape for 3D reconstruction and camera alignment.

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

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
