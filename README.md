# BDD100K Object Detection Project

## Project Overview
This project focuses on object detection using the Berkeley Deep Drive (BDD100k) dataset, which includes 100,000 images with corresponding labels for 10 detection classes. The project is structured around three main tasks:

1. Data Analysis
2. Model Selection and Implementation
3. Evaluation and Visualization

## Task 1: Data Analysis

### Objective
The data analysis task involves analyzing the BDD object detection dataset with a focus on the 10 detection classes. This includes examining class distributions, train/validation splits, anomalies, patterns, and visualizing dataset statistics.

### Key Findings

#### Class Distribution
Our analysis revealed significant class imbalance in the dataset:
- The 'car' class dominates both training and validation sets, accounting for over 55% of all instances
- 'Traffic sign' and 'traffic light' are the next most frequent classes
- Classes such as 'train', 'motor', 'rider', and 'bike' are significantly underrepresented (often < 1%)
- The distribution patterns are consistent between training and validation sets

#### Image Attributes
We also analyzed image-level attributes:

**Weather Conditions:**
- Clear weather dominates (~53% of images)
- Overcast, undefined, snowy, and rainy conditions make up most of the remainder
- Foggy conditions are rare (<0.2%)

**Scene Types:**
- City street scenes are most common (~61-62%)
- Highway (~25%) and residential scenes (~12%) follow
- Other scenes (parking lots, tunnels, gas stations) are infrequent

**Time of Day:**
- Roughly balanced between daytime (~53%) and night (~40%)
- Dawn/dusk represents a smaller portion (~7%)

#### Object Attributes
Analysis of object-level attributes revealed:

- About 47% of objects are occluded, posing a challenge for detection models
- Only about 7% of objects are truncated (extending beyond image boundaries)

#### Object Size Analysis
The analysis of bounding box areas provided additional insights:

- Vehicle classes (cars, trucks, buses) occupy the vast majority (>90%) of labeled object pixel area
- Small but frequent objects (traffic signs, traffic lights) contribute much less to total labeled area
- This pattern is consistent across both training and validation sets

### Visualization
All visualizations are available in the `assets` directory, including:
- Bar charts showing class distributions
- Visualizations of image attribute distributions
- Treemaps showing pixel area per class

### Implementation
The data analysis is implemented in `data_analysis.py` and packaged in a Docker container for reproducibility.

## Getting Started
(Additional setup instructions will be provided in subsequent updates) 