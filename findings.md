# Data Analysis Findings

This section documents the key findings from analyzing the BDD100k dataset labels, focusing on the 10 specified object detection classes.

## Class Distribution Analysis

The distribution of object classes was analyzed for both the training and validation sets. The script `data_analysis.py` was used to parse the label files and count the occurrences of each category. 

The key observations are:
*   **Significant Class Imbalance:** There is a large imbalance in the dataset. The 'car' class dominates both splits, accounting for over 55% of all instances. Classes like 'traffic sign' and 'traffic light' are the next most frequent, while classes such as 'train', 'motor', 'rider', and 'bike' are significantly less common (often < 1%). This imbalance could affect model training and evaluation, potentially leading to bias towards more frequent classes.
*   **Similar Train/Validation Distributions:** The relative frequencies of classes are highly consistent between the training and validation sets. This suggests the validation set is a representative sample of the training data in terms of class distribution, which is good for reliable evaluation.

Below are bar charts showing the percentage distribution for each class in the training and validation sets.

### Training Set Class Distribution (% - Bar Chart)

![Training Set Class Distribution (%)](assets/class_distribution_train_bar_pct.png)

### Validation Set Class Distribution (% - Bar Chart)

![Validation Set Class Distribution (%)](assets/class_distribution_val_bar_pct.png)

## Image Attribute Analysis (Training Set)

The distribution of image-level attributes (weather, scene, time of day) was analyzed for the training set.

*   **Weather:** The dataset is dominated by 'clear' weather images (~53%), with 'overcast', 'undefined', 'snowy', and 'rainy' conditions making up the bulk of the remainder. 'Foggy' conditions are very rare.
*   **Scene:** 'City street' is the most common scene type (~62%), followed by 'highway' (~25%) and 'residential' (~12%). Other scenes like 'parking lot', 'tunnel', and 'gas stations' are infrequent.
*   **Time of Day:** Images are roughly balanced between 'daytime' (~53%) and 'night' (~40%), with a smaller portion captured at 'dawn/dusk' (~7%).

These distributions are important as model performance might vary significantly depending on these conditions.

Below are bar charts showing the percentage distribution for each attribute in the training set:

### Weather Distribution (Training Set %)

![Weather Distribution (Training Set %)](assets/image_attr_weather_dist_train.png)

### Scene Distribution (Training Set %)

![Scene Distribution (Training Set %)](assets/image_attr_scene_dist_train.png)

### Time of Day Distribution (Training Set %)

![Time of Day Distribution (Training Set %)](assets/image_attr_timeofday_dist_train.png)

