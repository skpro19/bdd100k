import logging
import os
from collections import Counter
from typing import Dict, List, Any, Set
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configure logging (can be configured independently or inherit from main script)
# If run independently, this basic config helps. If imported, root logger config might apply.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
OUTPUT_DIR = "assets"
# Ensure output directory exists (important if vis.py is run standalone)
os.makedirs(OUTPUT_DIR, exist_ok=True) 


def plot_class_distribution(counts: Counter, title: str, output_filename: str) -> None:
    """Generates and saves a bar chart for class distribution.

    Args:
        counts: Counter object with class names and their counts.
        title: The title for the plot.
        output_filename: The filename to save the plot to (relative to OUTPUT_DIR).
    """
    if not counts:
        logging.warning(f"No data to plot for {title}. Skipping plot generation.")
        return

    logging.info(f"Generating plot: {title}")
    # Prepare data for plotting
    sorted_items = counts.most_common()
    categories = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Create the plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(categories, values, color=sns.color_palette("viridis", len(categories))) # Use seaborn colors
    plt.xlabel("Object Class")
    plt.ylabel("Frequency (Count)")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add grid lines
    plt.tight_layout()

    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{int(yval):,}', va='bottom', ha='center', fontsize=9) # Format large numbers

    # Save the plot
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        plt.savefig(save_path)
        logging.info(f"Saved class distribution plot to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save plot {save_path}: {e}")
    plt.close()


def plot_comparison_distribution(train_counts: Counter, val_counts: Counter, title: str, output_filename: str) -> None:
    """Generates and saves a grouped bar chart comparing train and validation class distributions.

    Args:
        train_counts: Counter object with training class counts.
        val_counts: Counter object with validation class counts.
        title: The title for the plot.
        output_filename: The filename to save the plot to (relative to OUTPUT_DIR).
    """
    if not train_counts and not val_counts:
        logging.warning("No train or validation counts provided for comparison plot. Skipping.")
        return

    # Combine counts into a Pandas DataFrame
    plot_data = []
    all_categories = sorted(list(set(train_counts.keys()) | set(val_counts.keys())))

    for category in all_categories:
        plot_data.append({'category': category, 'count': train_counts.get(category, 0), 'split': 'train'})
        plot_data.append({'category': category, 'count': val_counts.get(category, 0), 'split': 'validation'})
    
    df = pd.DataFrame(plot_data)

    if df.empty:
        logging.warning("DataFrame created for plotting is empty. Skipping plot generation.")
        return
        
    logging.info(f"Generating comparison plot: {title}")

    # Create the plot using seaborn
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df, x='category', y='count', hue='split', palette="viridis")

    # Add counts on top of bars (optional, can be cluttered)
    # for container in ax.containers:
    #    ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

    plt.xlabel("Object Class")
    plt.ylabel("Frequency (Count)")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dataset Split')
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        plt.savefig(save_path)
        logging.info(f"Saved comparison class distribution plot to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save plot {save_path}: {e}")
    plt.close()

# Example usage if run directly (optional)
if __name__ == '__main__':
    logging.info("vis.py executed directly. Running example plots (if data available).")
    # Example data (replace with actual data loading if needed for standalone testing)
    # train_example = Counter({'car': 100, 'person': 50, 'traffic light': 30})
    # val_example = Counter({'car': 90, 'person': 55, 'traffic light': 25, 'truck': 5})
    # plot_class_distribution(train_example, "Example Train Distribution", "example_train.png")
    # plot_comparison_distribution(train_example, val_example, "Example Comparison", "example_comparison.png")
    pass # No actual examples run by default 