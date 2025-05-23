import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

def plot_curves_from_h5(files, output_path=None, stat=True):
    """
    Reads data from multiple files (HDF5 or numpy), plots the average curves for each file,
    and displays relevant statistics.
    
    Parameters:
    files (list of str): List of file paths to the data files.
    output_path (str, optional): Path to save the plot. If None, plot is displayed.
    stat (bool): Whether to display statistics on the plot.
    """
    colors = ['blue', 'green', 'orange']  # Colors for each file
    sum_colors = ['purple', 'brown', 'gray']  # Colors for sum curves
    assert len(files) <= len(colors), "Provide at most 3 files (or extend the color list)."
    
    plt.figure(figsize=(12, 8))
    all_stats = []

    for i, file in enumerate(files):
        try:
            # Try loading as numpy file first
            data = np.load(file, allow_pickle=True)
            # Convert to numpy array if it's not already
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            # Ensure data is float type for plotting
            data = data.astype(float)
        except:
            try:
                # If that fails, try as h5py file
        with h5py.File(file, 'r') as h5_file:
                    data = np.array(h5_file['dataset']).astype(float)
            except Exception as e:
                print(f"Error loading file {file}: {str(e)}")
                continue
        
        # For Betti numbers, we have a 2D array where each row is [β₀, β₁]
        n_layers = data.shape[0]  # Number of layers
        x = np.arange(n_layers)  # x-coordinates
        
        # Compute mean and std for each Betti number across layers
        mean_curve_0 = data[:, 0].astype(float)
        mean_curve_1 = data[:, 1].astype(float)
        betti_sum = (data[:, 0] + data[:, 1]).astype(float)
        std_curve_0 = np.zeros_like(mean_curve_0)
        std_curve_1 = np.zeros_like(mean_curve_1)
        std_curve_sum = np.zeros_like(betti_sum)
        # For a single file, std is zero; for multiple runs, you would stack and compute std across axis=0
        # Plot β₀ (first column)
        plt.plot(x, mean_curve_0, color=colors[i], linewidth=2, marker='o', 
                label=f'β₀ (File {i+1})')
        plt.fill_between(x, mean_curve_0 - std_curve_0, mean_curve_0 + std_curve_0, color=colors[i], alpha=0.15)
        
        # Plot β₁ (second column)
        plt.plot(x, mean_curve_1, color=colors[i], linewidth=2, marker='s', 
                label=f'β₁ (File {i+1})', linestyle='--')
        plt.fill_between(x, mean_curve_1 - std_curve_1, mean_curve_1 + std_curve_1, color=colors[i], alpha=0.15)
        
        # Plot sum of Betti numbers
        plt.plot(x, betti_sum, color=sum_colors[i], linewidth=2, marker='^', 
                label=f'β₀+β₁ (File {i+1})', linestyle=':')
        plt.fill_between(x, betti_sum - std_curve_sum, betti_sum + std_curve_sum, color=sum_colors[i], alpha=0.15)
        
        # Calculate statistics for the stats box (mean/std/min/max across all layers)
        mean_val_0 = np.mean(mean_curve_0)
        mean_val_1 = np.mean(mean_curve_1)
        mean_val_sum = np.mean(betti_sum)
        std_val_0 = np.std(mean_curve_0)
        std_val_1 = np.std(mean_curve_1)
        std_val_sum = np.std(betti_sum)
        min_val_0 = np.min(mean_curve_0)
        min_val_1 = np.min(mean_curve_1)
        min_val_sum = np.min(betti_sum)
        max_val_0 = np.max(mean_curve_0)
        max_val_1 = np.max(mean_curve_1)
        max_val_sum = np.max(betti_sum)
        
        all_stats.append((mean_val_0, mean_val_1, mean_val_sum, std_val_0, std_val_1, std_val_sum,
                         min_val_0, min_val_1, min_val_sum, max_val_0, max_val_1, max_val_sum))

    # Add labels and grid
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Betti Numbers', fontsize=12)
    plt.xticks(x, labels=["input"] + [str(i) for i in range(1, n_layers - 1)] + ["output"])
    plt.grid(True)
    
    # Add legend
    plt.legend()

    if stat:
        # Prepare statistics text
        stats_text = "Statistics for Each File:\n\n"
        stats_text += "{:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "File", "Betti", "Mean", "Std Dev", "Min - Max"
        )
        stats_text += "-" * 50 + "\n"
        for i, stats in enumerate(all_stats):
            mean_0, mean_1, mean_sum, std_0, std_1, std_sum, min_0, min_1, min_sum, max_0, max_1, max_sum = stats
            stats_text += "{:<10} {:<10} {:<10.2f} {:<10.2f} {:.2f} - {:.2f}\n".format(
                f"File {i+1}", "β₀", mean_0, std_0, min_0, max_0
            )
            stats_text += "{:<10} {:<10} {:<10.2f} {:<10.2f} {:.2f} - {:.2f}\n".format(
                f"File {i+1}", "β₁", mean_1, std_1, min_1, max_1
            )
            stats_text += "{:<10} {:<10} {:<10.2f} {:<10.2f} {:.2f} - {:.2f}\n".format(
                f"File {i+1}", "β₀+β₁", mean_sum, std_sum, min_sum, max_sum
                )
        
        # Add the statistics text box to the plot
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='center', bbox=props)

    # Add title
    plt.title("Betti Numbers and Their Sum Across Layers", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
    plt.show()