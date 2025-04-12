import json
import logging
import os
from collections import Counter
from typing import Dict, List, Any, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
# Assuming the script is run from the project root
DATA_DIR = "data"
LABELS_DIR = os.path.join(DATA_DIR, "bdd100k_labels_release", "bdd100k", "labels")
TRAIN_LABELS_FILE = os.path.join(LABELS_DIR, "bdd100k_labels_images_train.json")
VAL_LABELS_FILE = os.path.join(LABELS_DIR, "bdd100k_labels_images_val.json")

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

def analyze_class_distribution(data: List[Dict[str, Any]], dataset_name: str) -> None:
    """Analyzes and logs the distribution of specified object detection categories for a given dataset.

    Args:
        data: The list of dictionaries loaded from the JSON label file.
        dataset_name: Name of the dataset being analyzed (e.g., "Training Set", "Validation Set").
    """
    if not data:
        logging.warning(f"No data provided for class distribution analysis for {dataset_name}.")
        return

    logging.info(f"Analyzing distribution for {len(OBJECT_DETECTION_CLASSES)} object detection classes in {dataset_name}...")
    class_counts: Counter = Counter()

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
        return
        
    # Sort by count descending for better readability
    sorted_counts = class_counts.most_common()
    total_filtered_labels = sum(class_counts.values())
    logging.info(f"  Total object detection labels: {total_filtered_labels}")
    for category, count in sorted_counts:
        percentage = (count / total_filtered_labels) * 100 if total_filtered_labels > 0 else 0
        logging.info(f"  - {category}: {count} ({percentage:.2f}%)")

def main() -> None:
    """Main function to run the data analysis for both train and val sets."""
    # Analyze Training Set
    logging.info("--- Starting Analysis for Training Set ---")
    train_data = load_json_data(TRAIN_LABELS_FILE)
    if train_data:
        analyze_class_distribution(train_data, dataset_name="Training Set")
    else:
        logging.error("Could not load or process training data. Skipping analysis.")

    # Analyze Validation Set
    logging.info("--- Starting Analysis for Validation Set ---")
    val_data = load_json_data(VAL_LABELS_FILE)
    if val_data:
        analyze_class_distribution(val_data, dataset_name="Validation Set")
    else:
        logging.error("Could not load or process validation data. Skipping analysis.")

if __name__ == "__main__":
    main() 