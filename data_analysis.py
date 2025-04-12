import json
import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set, Counter as TypingCounter

# Import the plotting functions from vis.py
from vis import plot_class_distribution, plot_treemap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
# Assuming the script is run from the project root
DATA_DIR = "data"
LABELS_DIR = os.path.join(DATA_DIR, "bdd100k_labels_release", "bdd100k", "labels")
TRAIN_LABELS_FILE = os.path.join(LABELS_DIR, "bdd100k_labels_images_train.json")
VAL_LABELS_FILE = os.path.join(LABELS_DIR, "bdd100k_labels_images_val.json")
OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the 10 object detection classes to focus on
OBJECT_DETECTION_CLASSES: Set[str] = {
    "traffic light", "traffic sign", "person", "car", "truck", 
    "bus", "bike", "motor", "rider", "train" 
    # Excludes 'lane' and 'drivable area' based on assignment instructions
}

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads JSON data from the specified file path.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A list of dictionaries representing the loaded JSON data.
        Returns an empty list if the file is not found or is invalid JSON.
    """
    if not os.path.exists(file_path):
        logging.error(f"Label file not found: {file_path}")
        return []
    
    logging.info(f"Loading labels from {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded {len(data)} entries from {os.path.basename(file_path)}.")
        return data
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return []

def analyze_class_distribution(data: List[Dict[str, Any]], dataset_name: str) -> TypingCounter[str]:
    """Analyzes and logs the distribution of specified object detection categories for a given dataset.

    Args:
        data: The list of dictionaries loaded from the JSON label file.
        dataset_name: Name of the dataset being analyzed (e.g., "Training Set", "Validation Set").

    Returns:
        A Counter object containing the counts for each object detection class.
        Returns an empty Counter if no data is provided or no relevant labels are found.
    """
    class_counts: TypingCounter[str] = Counter()

    if not data:
        logging.warning(f"No data provided for class distribution analysis for {dataset_name}.")
        return class_counts

    logging.info(f"Analyzing distribution for {len(OBJECT_DETECTION_CLASSES)} object detection classes in {dataset_name}...")

    total_labels_processed = 0
    object_detection_labels_found = 0

    for image_entry in data:
        labels = image_entry.get('labels', [])
        for label in labels:
            total_labels_processed += 1
            category = label.get('category')
            if category in OBJECT_DETECTION_CLASSES:
                class_counts[category] += 1
                object_detection_labels_found += 1

    logging.info(f"[{dataset_name}] Processed {total_labels_processed} total labels across all categories.")
    logging.info(f"[{dataset_name}] Found {object_detection_labels_found} labels matching the specified object detection classes.")
    
    logging.info(f"Object Detection Class Distribution ({dataset_name}):")
    if not class_counts:
        logging.info("  No labels matching the specified object detection classes found.")
        return class_counts
        
    # Sort by count descending for better readability
    sorted_counts = class_counts.most_common()
    total_filtered_labels = sum(class_counts.values())
    logging.info(f"  Total object detection labels: {total_filtered_labels}")
    for category, count in sorted_counts:
        percentage = (count / total_filtered_labels) * 100 if total_filtered_labels > 0 else 0
        logging.info(f"  - {category}: {count} ({percentage:.2f}%)")

    return class_counts

def analyze_image_attributes(data: List[Dict[str, Any]], dataset_name: str) -> Dict[str, TypingCounter[str]]:
    """Analyzes the distribution of image-level attributes (weather, scene, timeofday).

    Args:
        data: The list of dictionaries loaded from the JSON label file.
        dataset_name: Name of the dataset being analyzed.

    Returns:
        A dictionary where keys are attribute names ('weather', 'scene', 'timeofday') 
        and values are Counter objects with counts for each attribute value.
    """
    attribute_counts: Dict[str, TypingCounter[str]] = {
        'weather': Counter(),
        'scene': Counter(),
        'timeofday': Counter()
    }

    if not data:
        logging.warning(f"No data provided for image attribute analysis for {dataset_name}.")
        return attribute_counts

    logging.info(f"Analyzing image attributes for {dataset_name} ({len(data)} images)..." )

    images_processed = 0
    for image_entry in data:
        attributes = image_entry.get('attributes')
        if attributes:
            for attr_name in attribute_counts.keys():
                value = attributes.get(attr_name)
                if value:
                    attribute_counts[attr_name][value] += 1
            images_processed += 1
        else:
            logging.debug(f"Image {image_entry.get('name', 'N/A')} missing 'attributes' key.")

    logging.info(f"[{dataset_name}] Processed attributes for {images_processed} images.")

    # Log the distributions
    for attr_name, counts in attribute_counts.items():
        logging.info(f"Image Attribute Distribution - {attr_name.capitalize()} ({dataset_name}):")
        if not counts:
            logging.info(f"  No data found for attribute '{attr_name}'.")
            continue
        
        total = sum(counts.values())
        logging.info(f"  Total images with '{attr_name}' attribute: {total}")
        # Sort by count descending
        for value, count in counts.most_common():
            percentage = (count / total) * 100 if total > 0 else 0
            logging.info(f"  - {value}: {count} ({percentage:.2f}%)")

    return attribute_counts

def analyze_object_attributes(data: List[Dict[str, Any]], dataset_name: str) -> Dict[str, TypingCounter[str]]:
    """Analyzes the distribution of object-level attributes (occluded, truncated)
       across all objects in the specified detection classes.

    Args:
        data: The list of dictionaries loaded from the JSON label file.
        dataset_name: Name of the dataset being analyzed.

    Returns:
        A dictionary where keys are attribute names ('occluded', 'truncated') 
        and values are Counter objects counting True/False occurrences.
    """
    attribute_counts: Dict[str, TypingCounter[str]] = {
        'occluded': Counter(),
        'truncated': Counter()
        # Add other boolean attributes here if needed
    }

    if not data:
        logging.warning(f"No data provided for object attribute analysis for {dataset_name}.")
        return attribute_counts

    logging.info(f"Analyzing object attributes for {dataset_name}...")

    total_relevant_labels = 0
    for image_entry in data:
        labels = image_entry.get('labels', [])
        for label in labels:
            # Only consider labels belonging to the target object detection classes
            if label.get('category') in OBJECT_DETECTION_CLASSES:
                total_relevant_labels += 1
                attributes = label.get('attributes')
                if attributes:
                    for attr_name in attribute_counts.keys():
                        value = attributes.get(attr_name) 
                        # Convert boolean value to string for Counter key
                        if isinstance(value, bool):
                            attribute_counts[attr_name][str(value)] += 1
                        # else: (handle missing or non-boolean values if necessary)
                        #    attribute_counts[attr_name]['Unknown/Missing'] += 1 
                # else: (handle labels missing the 'attributes' dict)
                #     attribute_counts['occluded']['Unknown/Missing'] += 1
                #     attribute_counts['truncated']['Unknown/Missing'] += 1
    
    logging.info(f"[{dataset_name}] Processed attributes for {total_relevant_labels} relevant object labels.")

    # Log the distributions
    for attr_name, counts in attribute_counts.items():
        logging.info(f"Object Attribute Distribution - {attr_name.capitalize()} ({dataset_name}):")
        if not counts:
            logging.info(f"  No data found for attribute '{attr_name}'.")
            continue
            
        total = sum(counts.values())
        logging.info(f"  Total relevant labels with '{attr_name}' attribute: {total}")
        # Sort True/False for consistent reporting
        for value, count in sorted(counts.items()):
            percentage = (count / total) * 100 if total > 0 else 0
            logging.info(f"  - {value}: {count} ({percentage:.2f}%)")

    return attribute_counts

def analyze_total_area_per_class(data: List[Dict[str, Any]], dataset_name: str) -> Dict[str, float]:
    """Calculates the total pixel area occupied by each object class.

    Args:
        data: The list of dictionaries loaded from the JSON label file.
        dataset_name: Name of the dataset being analyzed.

    Returns:
        A dictionary mapping class names to the sum of their bounding box areas.
    """
    class_areas = defaultdict(float)

    if not data:
        logging.warning(f"No data provided for class area analysis for {dataset_name}.")
        return dict(class_areas)

    logging.info(f"Analyzing total area per class for {dataset_name}...")

    total_relevant_labels = 0
    labels_missing_box2d = 0
    invalid_boxes = 0

    for image_entry in data:
        labels = image_entry.get('labels', [])
        for label in labels:
            category = label.get('category')
            if category in OBJECT_DETECTION_CLASSES:
                total_relevant_labels += 1
                box2d = label.get('box2d')
                if box2d:
                    x1, y1, x2, y2 = box2d.get('x1'), box2d.get('y1'), box2d.get('x2'), box2d.get('y2')
                    # Basic validation
                    if None not in [x1, y1, x2, y2] and x2 > x1 and y2 > y1:
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        class_areas[category] += area
                    else:
                        invalid_boxes += 1
                        logging.debug(f"Invalid box2d coordinates found for label id {label.get('id', '?')} in image {image_entry.get('name')}: {box2d}")
                else:
                    labels_missing_box2d += 1
                    logging.debug(f"Label id {label.get('id', '?')} in image {image_entry.get('name')} missing 'box2d' key.")

    logging.info(f"[{dataset_name}] Calculated areas for {total_relevant_labels - labels_missing_box2d - invalid_boxes} valid boxes out of {total_relevant_labels} relevant labels.")
    if labels_missing_box2d > 0:
        logging.warning(f"[{dataset_name}] {labels_missing_box2d} relevant labels were missing 'box2d' information.")
    if invalid_boxes > 0:
        logging.warning(f"[{dataset_name}] {invalid_boxes} relevant labels had invalid 'box2d' coordinates (e.g., x2<=x1 or y2<=y1)." )

    # Log the total areas
    logging.info(f"Total Area per Class ({dataset_name}):")
    if not class_areas:
        logging.info("  No areas calculated.")
    else:
        # Sort by area descending for logging
        sorted_areas = sorted(class_areas.items(), key=lambda item: item[1], reverse=True)
        grand_total_area = sum(class_areas.values())
        logging.info(f"  Grand Total Area (all classes): {grand_total_area:,.0f} pixels")
        for category, total_area in sorted_areas:
            percentage = (total_area / grand_total_area) * 100 if grand_total_area > 0 else 0
            logging.info(f"  - {category}: {total_area:,.0f} pixels ({percentage:.2f}%)")

    return dict(class_areas)

def main() -> None:
    """Main function to run the data analysis and visualization."""
    # Analyze Training Set
    logging.info("--- Starting Analysis for Training Set ---")
    train_data = load_json_data(TRAIN_LABELS_FILE)
    train_counts = Counter()
    train_image_attr_counts: Dict[str, TypingCounter[str]] = {}
    train_object_attr_counts: Dict[str, TypingCounter[str]] = {}
    train_class_areas: Dict[str, float] = {}
    if train_data:
        # Class distribution analysis
        train_counts = analyze_class_distribution(train_data, dataset_name="Training Set")
        logging.info(f"Training set class distribution analysis complete.")
        # Image attribute analysis
        train_image_attr_counts = analyze_image_attributes(train_data, dataset_name="Training Set")
        logging.info(f"Training set image attribute analysis complete.")
        # Object attribute analysis
        train_object_attr_counts = analyze_object_attributes(train_data, dataset_name="Training Set")
        logging.info(f"Training set object attribute analysis complete.")
        # Class area analysis
        train_class_areas = analyze_total_area_per_class(train_data, dataset_name="Training Set")
        logging.info(f"Training set class area analysis complete.")
    else:
        logging.error("Could not load or process training data. Skipping analysis.")

    # --- Analyze Validation Set --- (Optional - currently only analyzing train for attributes)
    # logging.info("--- Starting Analysis for Validation Set ---")
    # val_data = load_json_data(VAL_LABELS_FILE)
    # val_counts = Counter()
    # val_image_attr_counts: Dict[str, TypingCounter[str]] = {}
    # if val_data:
    #     val_counts = analyze_class_distribution(val_data, dataset_name="Validation Set")
    #     logging.info(f"Validation set class distribution analysis complete.")
    #     val_image_attr_counts = analyze_image_attributes(val_data, dataset_name="Validation Set")
    #     logging.info(f"Validation set image attribute analysis complete.")
    # else:
    #     logging.error("Could not load or process validation data. Skipping analysis.")

    # --- Visualization Step --- 
    logging.info("--- Starting Visualization --- ")
    
    # -- Class Distribution Visualization --
    logging.info("- Visualizing Class Distributions (Bar Charts) -")
    if train_counts:
        plot_class_distribution(train_counts, 
                                title="Object Detection Class Distribution (Training Set - Percentage)", 
                                output_filename="class_distribution_train_bar_pct.png")
    else:
        logging.warning("No training class counts available for Bar visualization.")
    
    # Plot validation class distribution if loaded
    # if val_counts:
    #     plot_class_distribution(val_counts, 
    #                             title="Object Detection Class Distribution (Validation Set - Percentage)", 
    #                             output_filename="class_distribution_val_bar_pct.png")
    # else:
    #     logging.warning("No validation class counts available for Bar visualization.")

    # -- Image Attribute Distribution Visualization (Training Set Only) --
    logging.info("- Visualizing Image Attribute Distributions (Training Set) -")
    if train_image_attr_counts:
        for attr_name, counts in train_image_attr_counts.items():
            if counts:
                plot_class_distribution(counts, # Reusing the bar chart function
                                        title=f"Image Attribute Distribution: {attr_name.capitalize()} (Training Set)", 
                                        output_filename=f"image_attr_{attr_name}_dist_train.png")
            else:
                logging.warning(f"No data to plot for image attribute: {attr_name}")
    else:
        logging.warning("No training image attribute counts available for visualization.")

    # -- Object Attribute Distribution Visualization (Training Set Only) --
    logging.info("- Visualizing Object Attribute Distributions (Training Set) -")
    if train_object_attr_counts:
        for attr_name, counts in train_object_attr_counts.items():
            if counts:
                # Sort the counter by key ('False', 'True') for consistent bar order
                sorted_counts_dict = dict(sorted(counts.items()))
                sorted_counts = Counter(sorted_counts_dict)
                plot_class_distribution(sorted_counts, # Reusing the bar chart function
                                        title=f"Object Attribute Distribution: {attr_name.capitalize()} (Training Set)", 
                                        output_filename=f"object_attr_{attr_name}_dist_train.png")
            else:
                logging.warning(f"No data to plot for object attribute: {attr_name}")
    else:
        logging.warning("No training object attribute counts available for visualization.")

    # -- Class Area Treemap Visualization (Training Set Only) --
    logging.info("- Visualizing Total Class Area (Treemap - Training Set) -")
    if train_class_areas:
        plot_treemap(train_class_areas,
                     title="Total Pixel Area per Object Class (Training Set)",
                     output_filename="class_total_area_treemap_train.png")
    else:
        logging.warning("No training class area data available for Treemap visualization.")
    
    # -- Combined Pie Chart (Optional - currently commented) --
    # logging.info("- Visualizing Combined Pie Chart -") # Add logging if uncommenting
    # from vis import plot_combined_pie_charts # Need import if uncommenting
    # if train_counts or val_counts: # Ensure val_counts is defined if uncommenting
    #     plot_combined_pie_charts(train_counts, val_counts,
    #                              title="Object Detection Class Distribution Comparison (Pie Charts)",
    #                              output_filename="class_distribution_pie_combined.png")
    # else:
    #      logging.warning("No train or validation counts available for Pie visualization.")

if __name__ == "__main__":
    main() 