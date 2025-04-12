import json
import logging
import os
from collections import Counter
from typing import Dict, List, Any, Set, Counter

# Import the plotting function from vis.py
from vis import plot_class_distribution

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

def analyze_class_distribution(data: List[Dict[str, Any]], dataset_name: str) -> Counter:
    """Analyzes and logs the distribution of specified object detection categories for a given dataset.

    Args:
        data: The list of dictionaries loaded from the JSON label file.
        dataset_name: Name of the dataset being analyzed (e.g., "Training Set", "Validation Set").

    Returns:
        A Counter object containing the counts for each object detection class.
        Returns an empty Counter if no data is provided or no relevant labels are found.
    """
    class_counts: Counter = Counter()

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

def main() -> None:
    """Main function to run the data analysis and visualization."""
    # Analyze Training Set
    logging.info("--- Starting Analysis for Training Set ---")
    train_data = load_json_data(TRAIN_LABELS_FILE)
    train_counts = Counter()
    if train_data:
        train_counts = analyze_class_distribution(train_data, dataset_name="Training Set")
        logging.info(f"Training set analysis complete. Class counts obtained.")
    else:
        logging.error("Could not load or process training data. Skipping analysis.")

    # Analyze Validation Set
    logging.info("--- Starting Analysis for Validation Set ---")
    val_data = load_json_data(VAL_LABELS_FILE)
    val_counts = Counter()
    if val_data:
        val_counts = analyze_class_distribution(val_data, dataset_name="Validation Set")
        logging.info(f"Validation set analysis complete. Class counts obtained.")
    else:
        logging.error("Could not load or process validation data. Skipping analysis.")

    # Visualization step - Individual Plots (using function from vis.py)
    logging.info("--- Starting Visualization --- ")
    if train_counts:
        plot_class_distribution(train_counts, 
                                title="Object Detection Class Distribution (Training Set)", 
                                output_filename="class_distribution_train.png")
    else:
        logging.warning("No training counts available for visualization.")
        
    if val_counts:
        plot_class_distribution(val_counts, 
                                title="Object Detection Class Distribution (Validation Set)", 
                                output_filename="class_distribution_val.png")
    else:
        logging.warning("No validation counts available for visualization.")
        
    # Ensure the comparison plot call remains commented out or removed
    # from vis import plot_comparison_distribution # Import if uncommenting
    # if train_counts or val_counts:
    #     plot_comparison_distribution(train_counts, val_counts, 
    #                                  title="Object Detection Class Distribution (Train vs. Validation)", 
    #                                  output_filename="class_distribution_comparison.png")
    # else:
    #     logging.warning("No training or validation counts available for comparison visualization.")

if __name__ == "__main__":
    main() 