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

## Task 3: Evaluation and Visualization

### Objective
This task evaluates the performance of a chosen object detection model (Faster R-CNN R50-FPN 1x) on the BDD100K validation dataset (10,000 images). The evaluation includes quantitative metrics, qualitative analysis of predictions, and connections to the initial data analysis findings.

### Model Evaluated
- **Model**: Faster R-CNN with ResNet-50 backbone and FPN (Feature Pyramid Network)
- **Training**: Pre-trained on COCO, fine-tuned for 1x schedule (details assumed standard for the model).

### Key Evaluation Findings

#### Quantitative Performance (IoU Threshold = 0.5)
The model was evaluated on the BDD100K validation set using the standard detection metrics with an IoU threshold of 0.5.

**Official Evaluation Results:**

| Metric | Value |
|--------|-------|
| Mean Average Precision (mAP) | **0.1916** |

**Per-Category Performance:**

| Category | Average Precision (AP) | Precision | Recall | GT Count | Pred Count |
|----------|------------------------|-----------|--------|----------|------------|
| car | 0.3868 | 0.4591 | 0.8424 | 102,506 | 188,083 |
| traffic sign | 0.2658 | 0.3565 | 0.7454 | 34,908 | 72,975 |
| traffic light | 0.2464 | 0.4893 | 0.5035 | 26,885 | 27,664 |
| truck | 0.1761 | 0.2205 | 0.7986 | 4,245 | 15,375 |
| bus | 0.1332 | 0.1724 | 0.7727 | 1,597 | 7,156 |
| rider | 0.1329 | 0.2114 | 0.6287 | 649 | 1,930 |
| pedestrian | 0.0000 | 0.0000 | 0.0000 | 0 | 36,266 |
| motorcycle | 0.0000 | 0.0000 | 0.0000 | 0 | 1,588 |
| bicycle | 0.0000 | 0.0000 | 0.0000 | 0 | 3,226 |
| train | 0.0000 | 0.0000 | 0.0000 | 15 | 0 |

*Key Insights from Metrics:*
1.  **Overall Performance**: Moderate mAP (0.1916) indicates room for improvement.
2.  **Class Imbalance Impact**: Best performance on common classes (car, traffic sign, traffic light).
3.  **Precision vs. Recall Trade-off**: High recall but low precision for most classes, suggesting many false positives.
4.  **Zero-AP Classes**: `pedestrian`, `motorcycle`, `bicycle`, `train` show AP=0, indicating potential evaluation issues or complete failure.

**Overall Detection Statistics:**

| Metric | Value |
|--------|-------|
| Total ground truth boxes | 170,805 |
| Total prediction boxes | 354,263 |
| True positives | 130,943 |
| False positives | 223,320 |
| False negatives | 39,862 |
| Overall recall | 0.7666 |
| Overall precision | 0.3696 |

*Note: The high number of false positives confirms the tendency to over-predict.*

**Detection Performance Overview:**

| Metric | Value |
|--------|-------|
| Number of Images Evaluated | 10,000 |
| Total Detections (score ≥ 0.3) | 170,662 |
| Average Detections per Image | 17.07 |
| Median Detections per Image | 17.00 |
| Maximum Detections per Image | 56 |

*Note: Substantial detections per image reflect scene complexity.*

**Class Distribution of Detections:**

| Class | Count | Percentage | Average Confidence |
|-------|-------|------------|-------------------|
| car | 101,631 | 59.55% | 0.8024 |
| traffic sign | 32,656 | 19.13% | 0.7126 |
| pedestrian | 14,010 | 8.21% | 0.7093 |
| traffic light | 13,475 | 7.90% | 0.6926 |
| truck | 4,912 | 2.88% | 0.6375 |
| bus | 2,075 | 1.22% | 0.6656 |
| bicycle | 1,035 | 0.61% | 0.6451 |
| rider | 510 | 0.30% | 0.6863 |
| motorcycle | 358 | 0.21% | 0.6242 |
| train | 0 | 0.00% | N/A |

![Class Distribution](bdd100k-models/det/assets/class_distribution.png)

*Note: Detection distribution closely mirrors class imbalance.*

**Confidence Score Analysis:**

![Average Confidence by Class](bdd100k-models/det/assets/avg_score_by_class.png)

| Confidence Score | Number of Detections |
|------------------|----------------------|
| 0.9 - 1.0 | 76,186 (44.6%) |
| 0.8 - 0.9 | 17,322 (10.1%) |
| 0.7 - 0.8 | 13,662 (8.0%) |
| 0.6 - 0.7 | 12,963 (7.6%) |
| 0.5 - 0.6 | 13,758 (8.1%) |
| 0.4 - 0.5 | 15,933 (9.3%) |
| 0.3 - 0.4 | 20,838 (12.2%) |

![Confidence Score Distribution](bdd100k-models/det/assets/score_distribution.png)

*Note: Bimodal distribution with peaks at high (≥0.9) and low (0.3-0.4) confidence.*

**Detection Density Distribution:**

![Detections per Image](bdd100k-models/det/assets/detections_per_image.png)

*Note: Considerable variation in detections per image reflects scene diversity.*

#### Qualitative Analysis
Visual inspection revealed several patterns:
- **Successes**: Good detection of clearly visible cars, traffic signs, and traffic lights in favorable conditions (daytime, clear weather).
- **Common Failures**:
    - **False Positives**: Reflections, shadows, confusion between similar objects (e.g., trucks vs. buses).
    - **False Negatives**: Heavily occluded objects (>50% occlusion often missed), objects at image boundaries, rare classes (especially in challenging conditions), small/distant objects.
    - **Class Confusion**: Significant issues differentiating pedestrians, riders, and potentially bicycles/motorcycles.
- **Environmental Impact**: Performance degrades noticeably in nighttime scenes and adverse weather (rain, snow).

#### Connection to Data Analysis
- **Class Imbalance**: The model's performance directly mirrors the class distribution identified in Task 1. The dominance of cars and the rarity of other classes heavily influence detection rates and AP scores.
- **Occlusion**: The difficulty in detecting occluded objects aligns with the data analysis finding that ~47% of objects in the dataset are occluded.

### Visualization
Key visualizations summarizing the quantitative analysis include:
- Class distribution of detections: ![Class Distribution](bdd100k-models/det/assets/class_distribution.png)
- Average confidence score per class: ![Average Confidence by Class](bdd100k-models/det/assets/avg_score_by_class.png)
- Overall confidence score distribution: ![Confidence Score Distribution](bdd100k-models/det/assets/score_distribution.png)
- Distribution of detections per image: ![Detections per Image](bdd100k-models/det/assets/detections_per_image.png)

*(Note: Visualizations of ground truth vs. predictions showing specific success/failure cases are typically generated by accompanying scripts, e.g., `visualize_predictions.py` mentioned in the qualitative report).*

### Summary and Potential Improvements
- **Strengths**: High recall for common objects, handles varying scene density.
- **Weaknesses**: Low precision (many false positives), severe performance issues on rare classes, sensitivity to occlusion and environmental conditions.
- **Suggestions**: Address class imbalance (e.g., weighted loss, augmentation), improve precision (e.g., threshold tuning, hard negative mining), implement occlusion handling techniques, enhance environmental robustness (e.g., domain adaptation).

### Implementation Details
The detailed evaluation reports and analysis can be found in the `bdd100k-models/det/` directory:
- `quantitative_performance.md`
- `qualitative_analysis.md`
- `evaluation_summary.md`

## Getting Started
(Additional setup instructions will be provided in subsequent updates) 