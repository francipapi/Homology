import numpy as np
import matplotlib.pyplot as plt
import h5py

def plot_curves_from_h5(files, stat=True):
    """
    Reads data from multiple HDF5 files, plots the average curves for each file,
    and displays relevant statistics.
    
    Parameters:
    files (list of str): List of file paths to the HDF5 files.
    stat (bool): Whether to display statistics on the plot.
    """
    colors = ['blue', 'green', 'orange']  # Colors for each file
    assert len(files) <= len(colors), "Provide at most 3 files (or extend the color list)."
    
    plt.figure(figsize=(12, 8))
    all_stats = []

    for i, file in enumerate(files):
        with h5py.File(file, 'r') as h5_file:
            # Assuming data is stored in a dataset named 'data'
            data = np.array(h5_file['dataset'])
        
        n_layers = data.shape[1]  # Number of layers
        x = np.arange(n_layers)  # x-coordinates
        
        # Plot all curves for the file with transparency
        for curve in data:
            plt.plot(x, curve, color=colors[i], alpha=0.1)
        
        # Calculate statistics
        mean_curve = np.mean(data, axis=0)
        variance = np.var(data, axis=0)
        std_dev = np.std(data, axis=0)
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        all_stats.append((mean_curve, variance, std_dev, min_vals, max_vals))
        
        # Plot the mean curve
        plt.plot(x, mean_curve, color=colors[i], linewidth=2, marker='o', label=f'Mean Curve (File {i+1})')
        
        # Plot mean ± std deviation
        plt.fill_between(x, mean_curve - std_dev, mean_curve + std_dev, color=colors[i], alpha=0.1, label=f'±1 Std Dev (File {i+1})')

    # Add labels and grid
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(r'$\beta_0$', fontsize=14, rotation=0, labelpad=15)
    plt.xticks(x, labels=["input"] + [str(i) for i in range(1, n_layers - 1)] + ["output"])
    plt.grid(True)
    
    # Add legend
    plt.legend()

    if stat:
        # Prepare statistics text
        stats_text = "Statistics for Each File:\n\n"
        stats_text += "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "File", "Layer", "Mean", "Variance", "Std Dev", "Min - Max"
        )
        stats_text += "-" * 60 + "\n"
        for i, stats in enumerate(all_stats):
            mean_curve, variance, std_dev, min_vals, max_vals = stats
            for j in range(n_layers):
                layer_label = "input" if j == 0 else ("output" if j == n_layers - 1 else str(j))
                stats_text += "{:<10} {:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:.4f} - {:.4f}\n".format(
                    f"File {i+1}",
                    layer_label,
                    mean_curve[j],
                    variance[j],
                    std_dev[j],
                    min_vals[j],
                    max_vals[j]
                )
        
        # Add the statistics text box to the plot
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='center', bbox=props)

    # Add title
    plt.title("Average Curves and Statistics for Each File", fontsize=14)
    plt.tight_layout()
    plt.show()

# Example usage
files = ['homology_results_relu_30it', 'homology_results_relu_30it_25w', 'homology_results_relu_30it_40w']
plot_curves_from_h5(files)