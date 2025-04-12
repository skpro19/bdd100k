import logging
import os
from collections import Counter
from typing import Dict, List, Any, Set
import matplotlib.pyplot as plt
import numpy as np # Added import for numpy
import pandas as pd
import seaborn as sns
import squarify # Added import for treemaps

# Configure logging (can be configured independently or inherit from main script)
# If run independently, this basic config helps. If imported, root logger config might apply.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
OUTPUT_DIR = "assets"
# Ensure output directory exists (important if vis.py is run standalone)
os.makedirs(OUTPUT_DIR, exist_ok=True) 


def plot_class_distribution(counts: Counter, title: str, output_filename: str) -> None:
    """Generates and saves a bar chart for class distribution, showing percentages on bars.
       Ensures color consistency with pie charts using the same palette and mapping.
    """
    if not counts:
        logging.warning(f"No data to plot for {title}. Skipping plot generation.")
        return

    logging.info(f"Generating plot: {title}")
    
    # --- Data Preparation ---
    # Sort items by count for plotting order
    sorted_items = counts.most_common()
    categories_plot_order = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    total_count = sum(values)
    
    # --- Color Mapping (Consistent with Pie Chart) ---
    # Get all unique categories present in this dataset, sorted alphabetically for consistency
    all_categories_present = sorted(counts.keys())
    # Assign consistent colors using 'viridis' palette based on alphabetical order
    colors = sns.color_palette('viridis', n_colors=len(all_categories_present))
    category_to_color = {category: colors[i % len(colors)] for i, category in enumerate(all_categories_present)}
    # Get the colors for the bars in the desired plot order (sorted by frequency)
    bar_colors = [category_to_color[category] for category in categories_plot_order]

    # --- Plotting ---
    with plt.style.context('seaborn-v0_8-whitegrid'): 
        plt.figure(figsize=(12, 7))
        # Use the mapped colors for the bars
        bars = plt.bar(categories_plot_order, values, color=bar_colors) 
        plt.xlabel("Object Class")
        plt.ylabel("Frequency (Count)")
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Add percentages on top of bars
        for bar in bars:
            yval = bar.get_height()
            percentage = (yval / total_count) * 100 if total_count > 0 else 0
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{percentage:.1f}%', va='bottom', ha='center', fontsize=9)

        # --- Save Plot ---
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        try:
            plt.savefig(save_path)
            logging.info(f"Saved class distribution plot (with percentages) to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save plot {save_path}: {e}")
        finally:
             plt.close() 


def plot_pie_chart(counts: Counter, title: str, output_filename: str) -> None:
    """Generates and saves a pie chart for class distribution.

    Args:
        counts: Counter object with class names and their counts.
        title: The title for the plot.
        output_filename: The filename to save the plot to (relative to OUTPUT_DIR).
    """
    if not counts:
        logging.warning(f"No data to plot for {title}. Skipping pie chart generation.")
        return

    logging.info(f"Generating pie chart: {title}")
    
    # Prepare data - sort by count for potentially better layout if using explode
    sorted_items = counts.most_common()
    labels = [item[0] for item in sorted_items]
    sizes = [item[1] for item in sorted_items]
    total = sum(sizes)

    # Use seaborn color palette
    colors = sns.color_palette('viridis', len(labels))

    # Create the plot
    plt.figure(figsize=(10, 10)) # Pie charts often need more space
    
    # autopct shows percentages, lambda func prevents display for small slices
    # textprops can adjust font size
    wedges, texts, autotexts = plt.pie(sizes, 
                                       labels=None, # Don't draw labels directly on slices initially
                                       autopct=lambda p: f'{p:.1f}%' if p > 2 else '', # Show % only if > 2%
                                       startangle=90, 
                                       colors=colors, 
                                       pctdistance=0.85, # Position of the % labels
                                       wedgeprops={'edgecolor': 'white'}) # Add white border to slices
    
    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')  
    plt.title(title, pad=20)

    # Add a legend outside the pie chart
    # Use labels and corresponding sizes for the legend
    legend_labels = [f'{label} ({size:,} - {size/total:.1%})' for label, size in zip(labels, sizes)]
    plt.legend(wedges, legend_labels, 
               title="Classes", 
               loc="center left", 
               bbox_to_anchor=(1, 0, 0.5, 1), # Position legend outside
               fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.75, 1]) # Adjust layout to make space for legend

    # Save the plot
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        plt.savefig(save_path, bbox_inches='tight') # bbox_inches helps include the legend
        logging.info(f"Saved pie chart plot to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save plot {save_path}: {e}")
    plt.close()


def plot_combined_pie_charts(train_counts: Counter, val_counts: Counter, title: str, output_filename: str) -> None:
    """Generates and saves a figure with two pie charts (train & val) side-by-side.
    Uses a cleaner font, viridis color scheme, and simplified legend.
    """
    # --- Font & Style Setup ---
    # Try setting a sans-serif font family. Availability depends on user's system.
    # Using a context manager to temporarily change settings
    with plt.style.context('seaborn-v0_8-whitegrid'): # Use a seaborn style for base
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Verdana', 'Tahoma'] # Specify fallback fonts

        if not train_counts and not val_counts:
            logging.warning("No train or validation counts provided for combined pie chart. Skipping.")
            return

        logging.info(f"Generating combined pie chart: {title}")

        # --- Data Preparation ---
        all_categories = sorted(list(set(train_counts.keys()) | set(val_counts.keys())))
        
        # Assign consistent colors using 'viridis' palette
        colors = sns.color_palette('viridis', n_colors=len(all_categories))
        category_to_color = {category: colors[i % len(colors)] for i, category in enumerate(all_categories)}

        datasets = {
            'Training Set': train_counts,
            'Validation Set': val_counts
        }
        plot_data = {}
        for name, counts in datasets.items():
            if counts:
                sorted_items = sorted([(cat, counts.get(cat, 0)) for cat in all_categories if counts.get(cat, 0) > 0], key=lambda x: x[1], reverse=True)
                labels = [item[0] for item in sorted_items]
                sizes = [item[1] for item in sorted_items]
                pie_colors = [category_to_color[label] for label in labels] # Get colors in order
                plot_data[name] = {'labels': labels, 'sizes': sizes, 'colors': pie_colors}
            else:
                plot_data[name] = None

        # --- Plotting --- 
        fig, axes = plt.subplots(1, 2, figsize=(18, 9)) # 1 row, 2 columns
        fig.suptitle(title, fontsize=18, weight='bold') # Slightly larger, bold title

        legend_elements = []
        legend_labels_simple = [] # Store only category names for the legend

        for i, (name, data) in enumerate(plot_data.items()):
            ax = axes[i]
            if data:
                wedges, texts, autotexts = ax.pie(data['sizes'],
                                                  labels=None,
                                                  autopct=lambda p: f'{p:.1f}%' if p > 1.5 else '', # Only show % if > 1.5%
                                                  startangle=90, 
                                                  colors=data['colors'], 
                                                  pctdistance=0.85,
                                                  textprops={'fontsize': 10}, # Control font size of percentages
                                                  wedgeprops={'edgecolor': 'white'})
                ax.set_title(name, fontsize=14) # Title for each subplot
                ax.axis('equal') 

                # Collect legend info (only need to do once)
                if i == 0:
                     # Get simple labels (just category names) in the correct order
                     legend_labels_simple = [label for label in data['labels']]
                     # Map category to color rectangle for the legend
                     for label in data['labels']:
                          legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=category_to_color[label], edgecolor='none'))
            else:
                ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(name, fontsize=14)
                ax.axis('off')

        # Add a single combined legend with simple labels
        if legend_elements:
            fig.legend(legend_elements, legend_labels_simple, 
                       title="Classes", 
                       loc='center right', 
                       bbox_to_anchor=(0.98, 0.5), 
                       fontsize='large', # Larger legend font
                       title_fontsize='x-large') # Larger legend title

        fig.tight_layout(rect=[0, 0, 0.85, 0.95]) # Adjust rect to fit legend/title

        # --- Save Plot --- 
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        try:
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"Saved updated combined pie chart plot to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save plot {save_path}: {e}")
        finally:
            plt.close(fig) # Ensure figure is closed


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

def plot_treemap(class_areas: Dict[str, float], title: str, output_filename: str) -> None:
    """Generates and saves a treemap visualizing total area per class.

    Args:
        class_areas: Dictionary mapping class name to total bounding box area.
        title: The title for the plot.
        output_filename: The filename to save the treemap to (relative to OUTPUT_DIR).
    """
    if not class_areas:
        logging.warning(f"No data provided for treemap: {title}. Skipping plot generation.")
        return

    logging.info(f"Generating treemap: {title}")

    # Prepare data: sizes (areas) and labels (class names)
    # Sort by area descending for consistent layout
    sorted_items = sorted(class_areas.items(), key=lambda item: item[1], reverse=True)
    sizes = [item[1] for item in sorted_items]
    labels = [item[0] for item in sorted_items]
    total_area = sum(sizes)

    # Create labels for the plot (e.g., "ClassName\n{Percentage}%")
    plot_labels = [
        f"{label}\n({(size / total_area * 100):.1f}%)" 
        for label, size in zip(labels, sizes)
    ]

    # Use a consistent color palette
    colors = sns.color_palette('viridis', n_colors=len(labels))

    # Create the plot
    plt.figure(figsize=(14, 8))
    squarify.plot(sizes=sizes, 
                  label=plot_labels, 
                  color=colors, 
                  alpha=0.8, # Slight transparency
                  text_kwargs={'fontsize':10, 'color':'white', 'weight':'bold'} # Style text inside boxes
                  )
    plt.title(title, fontsize=16)
    plt.axis('off') # Hide axes for treemaps

    # Save the plot
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved treemap plot to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save treemap plot {save_path}: {e}")
    finally:
        plt.close()

def plot_spatial_heatmap(centers_x: List[float], centers_y: List[float], title: str, output_filename: str, 
                         img_width: int = 1280, img_height: int = 720, bins: int = 50) -> None:
    """Generates and saves a 2D histogram heatmap of object center locations.

    Args:
        centers_x: List of object center x-coordinates.
        centers_y: List of object center y-coordinates.
        title: The title for the plot.
        output_filename: The filename to save the heatmap to (relative to OUTPUT_DIR).
        img_width: The width of the original images (for setting range).
        img_height: The height of the original images (for setting range).
        bins: The number of bins to use for the histogram in each dimension.
    """
    if not centers_x or not centers_y:
        logging.warning(f"No data provided for spatial heatmap: {title}. Skipping plot generation.")
        return
    
    logging.info(f"Generating spatial heatmap: {title}")
    
    # Create the 2D histogram
    plt.figure(figsize=(10, 6)) # Adjust figure size as needed
    # Use hist2d for density. Consider cmin=1 to ignore empty bins.
    # Use cmap='inferno' or 'hot' for heatmap effect.
    heatmap, xedges, yedges, img = plt.hist2d(centers_x, centers_y, 
                                             bins=bins, 
                                             range=[[0, img_width], [0, img_height]],
                                             cmap='inferno', 
                                             cmin=1) # cmin avoids plotting bins with zero counts
    
    plt.colorbar(label='Object Center Density')
    plt.title(title)
    plt.xlabel("X-coordinate (pixels)")
    plt.ylabel("Y-coordinate (pixels)")
    
    # Set axis limits and invert y-axis to match image coordinates (0,0 at top-left)
    plt.xlim(0, img_width)
    plt.ylim(img_height, 0) 
    plt.gca().set_aspect('equal', adjustable='box') # Ensure correct aspect ratio
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved spatial heatmap plot to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save spatial heatmap plot {save_path}: {e}")
    finally:
        plt.close()

# Example usage if run directly (optional)
if __name__ == '__main__':
    logging.info("vis.py executed directly. Running example plots (if data available).")
    # Example data (replace with actual data loading if needed for standalone testing)
    # train_example = Counter({'car': 100, 'person': 50, 'traffic light': 30})
    # val_example = Counter({'car': 90, 'person': 55, 'traffic light': 25, 'truck': 5})
    # plot_class_distribution(train_example, "Example Train Distribution", "example_train.png")
    # plot_comparison_distribution(train_example, val_example, "Example Comparison", "example_comparison.png")
    # Example for combined pie:
    # train_ex = Counter({'car': 100, 'person': 50, 'light': 30})
    # val_ex = Counter({'car': 90, 'person': 55, 'light': 25, 'truck': 5})
    # plot_combined_pie_charts(train_ex, val_ex, "Example Combined Pies", "example_combined_pie.png")
    pass # No actual examples run by default 