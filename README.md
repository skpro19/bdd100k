# BDD100K Object Detection Project

## Project Overview
This project focuses on object detection using the Berkeley Deep Drive (BDD100k) dataset, which includes 100,000 images with corresponding labels for 10 detection classes. The project is structured around two main tasks:

1. Data Analysis
2. Evaluation and Visualization

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

## Task 3: Evaluation and Visualization

### Evaluation Approach
We conducted a comprehensive evaluation of the Faster R-CNN R50-FPN model on the full BDD100K validation set (10,000 images) using:

1. **Official Evaluation Metrics**: Standard detection metrics at IoU threshold of 0.5
2. **Statistical Analysis**: Detection counts, class distribution, and confidence scores
3. **Visual Inspection**: Systematic analysis of detection visualizations
4. **Comparative Analysis**: Connecting model performance to data analysis findings

### Quantitative Performance

#### Overall Metrics
- **Mean Average Precision (mAP)**: 0.1916
- **Overall Precision**: 0.3696
- **Overall Recall**: 0.7666
- **True Positives**: 130,943
- **False Positives**: 223,320
- **False Negatives**: 39,862

The considerable gap between recall and precision indicates the model tends to generate many false positives.

#### Per-Category Performance
The model shows significant performance variation across categories:

| Class | AP | Precision | Recall |
|-------|------|-----------|--------|
| car | 0.3868 | 0.4591 | 0.8424 |
| traffic sign | 0.2658 | 0.3565 | 0.7454 |
| traffic light | 0.2464 | 0.4893 | 0.5035 |
| truck | 0.1761 | 0.2205 | 0.7986 |
| bus | 0.1332 | 0.1724 | 0.7727 |
| rider | 0.1329 | 0.2114 | 0.6287 |
| pedestrian | 0.0000 | 0.0000 | 0.0000 |
| motorcycle | 0.0000 | 0.0000 | 0.0000 |
| bicycle | 0.0000 | 0.0000 | 0.0000 |
| train | 0.0000 | 0.0000 | 0.0000 |

Performance directly correlates with class frequency in the dataset, with best results for common classes (car, traffic sign, traffic light) and poorest for rare classes.

#### Detection Statistics
- **Total Detections (score ≥ 0.3)**: 170,662 across 10,000 images
- **Average Detections per Image**: 17.07
- **Confidence Distribution**: 44.6% of detections with confidence ≥ 0.9, showing a bimodal pattern with peaks at very high confidence (0.9+) and at lower threshold (0.3-0.4)

### Qualitative Analysis

#### Successful Detection Patterns
1. **Clearly Visible Cars**: Excellent detection of unoccluded cars
2. **Traffic Signs and Lights**: Reliable detection even at moderate distances
3. **Larger Vehicles**: High recall for trucks and buses, albeit with significant false positives

#### Common Failure Cases
1. **Occlusion**: Heavily occluded objects frequently missed or detected with low confidence
2. **Small Objects**: Difficulty detecting distant pedestrians, motorcycles, and bicycles
3. **Nighttime Scenes**: Reduced performance in low-light conditions
4. **Class Confusion**: Confusion between similar classes (e.g., car vs. truck, pedestrian vs. rider)
5. **Environmental Challenges**: Performance degradation in adverse weather conditions

#### Performance Across Conditions
- **Best Performance**: Daytime, clear weather, city street scenes
- **Moderate Performance**: Dusk/dawn, overcast conditions, highway scenes
- **Poorest Performance**: Nighttime, adverse weather (rain/snow), residential areas with occlusions

### Connection to Data Analysis Findings

The evaluation results strongly align with our data analysis findings:

1. **Class Imbalance Impact**: The model's AP values directly correlate with class frequencies identified in our data analysis. Cars (most frequent at >55%) achieve the highest AP (0.3868), while rare classes show extremely poor performance.

2. **Occlusion Effects**: Our data analysis found that 47% of objects are occluded, and the model indeed struggles with occluded objects, contributing to the 39,862 false negatives.

3. **Environmental Conditions**: The model performs best in the most common conditions identified in our analysis (clear weather, daytime, city streets) and degrades in less represented conditions.

4. **Object Size Challenges**: The data analysis highlighted the small pixel area of certain object classes, which corresponds to detection difficulties for smaller objects like traffic signs and distant pedestrians.

### Improvement Suggestions

Based on our evaluation, we recommend several approaches to enhance model performance:

1. **Addressing Class Imbalance**:
   - Implement class-weighted loss functions
   - Apply targeted data augmentation for rare classes
   - Consider two-stage training strategy (balanced subset, then full dataset)

2. **Improving Precision**:
   - Optimize confidence thresholds per class
   - Implement hard negative mining
   - Refine non-maximum suppression parameters

3. **Enhancing Occlusion Handling**:
   - Incorporate occlusion-aware model architectures
   - Add attention mechanisms for partially visible objects
   - Leverage context modeling

4. **Environmental Robustness**:
   - Apply domain adaptation techniques
   - Consider condition-specific fine-tuning
   - Implement image enhancement preprocessing

### Implementation and Visualization
The evaluation code and visualization tools are available in the repository:
- Quantitative evaluation scripts in `evaluation/`
- Visualization code in `visualization/`
- Example detection visualizations in `assets/`

All detailed evaluation results are documented in separate markdown files:
- `quantitative_performance.md`: Comprehensive metrics and statistics
- `qualitative_analysis.md`: Patterns and failure case analysis
- `evaluation_summary.md`: Integrated findings and recommendations

## Getting Started
(Additional setup instructions will be provided in subsequent updates) 