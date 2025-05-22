# Current best estimates: 
# gen_easy k=10, d=3
# generate K=20, d=3


from src.topology import homology
from src.data import dataset
from src.visualization import graph
from src.models import train, train2
from src.utils import mps, fps_mlx
import os
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map
import h5py

def plot_curves_with_mean(data, stat=True):
    """
    Plots multiple curves with a highlighted mean curve and displays relevant statistics.
    
    Parameters:
    data (numpy.ndarray): An array of shape (n, 6), where each row is a curve.
    """
    # Ensure data is a NumPy array
    data = np.array(data)
    
    n_layers = data.shape[1]  # Number of layers (6 in this case)
    x = np.arange(n_layers)  # x-coordinates: Layer 0 to 5
    
    # Plot all curves with transparency
    plt.figure(figsize=(10, 8))
    for curve in data:
        plt.plot(x, curve, color='blue', alpha=0.15)
    
    # Calculate statistics
    mean_curve = np.mean(data, axis=0)
    variance = np.var(data, axis=0)
    std_dev = np.std(data, axis=0)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Plot the mean curve
    plt.plot(x, mean_curve, color='red', linewidth=2, marker='s', label='Mean Curve')
    
    # Optionally, plot mean ± std deviation
    plt.fill_between(x, mean_curve - std_dev, mean_curve + std_dev, color='red', alpha=0.1, label='±1 Std Dev')
    
    # Add labels and grid
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(r'$\beta_0$', fontsize=14, rotation=0, labelpad=15)
    plt.xticks(x, labels=["input"] + [str(i) for i in range(1, n_layers-1)] + ["output"])
    plt.grid(True)
    
    # Add legend
    plt.legend()
    
    if stat:
        # Prepare statistics text
        stats_text = "Layer-wise Statistics:\n\n"
        stats_text += "{:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "Layer", "Mean", "Variance", "Std Dev", "Min - Max"
        )
        stats_text += "-"*50 + "\n"
        for i in range(n_layers):
            layer_label = "input" if i == 0 else ("output" if i == n_layers-1 else str(i))
            stats_text += "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:.4f} - {:.4f}\n".format(
                layer_label,
                mean_curve[i],
                variance[i],
                std_dev[i],
                min_vals[i],
                max_vals[i]
            )
        
        # Add the statistics text box to the plot
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='center', bbox=props)
    
    # Add title
    plt.title("Curves, Mean Curve, and Layer-wise Statistics", fontsize=14)
    
    plt.tight_layout()
    plt.show()


def is_homogeneous(data):
    """
    Returns True if all elements in the input have the same shape, False otherwise.
    """
    try:
        shapes = [np.shape(item) for item in data]
        return len(set(shapes)) == 1
    except Exception:
        return False

def main():

    # Parameters determined from homology of the dataset 
    n=4000
    k=20
    d=3
    iter = 25
    result_all=[]
    X, y = data.generate(n, 3, 1)
    train_data = []
    for _ in range(iter):
        train_data.append([X,y])
    
    
    with Pool(processes=12) as pool:  # Outer multiprocessing pool
        train_results = pool.map(mps.train, train_data)
    '''
    train_results = process_map(train_mlx.train, train_data, max_workers=12, desc="Training Progress")
    '''

    filtered_train_results = [item for item in train_results if isinstance(item, np.ndarray)]

    dataset = []
    '''
    for i in train_results:
        partial = []
        for j in i:
            partial.append(data.farthest_point_sampling(j, 2000))
        dataset.append(partial)
    print(np.array(dataset).shape)
    '''
    start = time.time()
    for i in filtered_train_results:

        '''
        # Implementation with fps library using parallel processing on CPU 
        with Pool(processes=12) as pool:  # Outer multiprocessing pool
            sample_param = [[k, 2000] for k in i]
            sample_results = pool.map(data.parallel_farthest_point_sampling, sample_param)
        dataset.append(sample_results)
        '''
    
        # Custom implementation with GPU on MLX
        dataset.append([fps_mlx.farthest_point_sampling(k, 16000) for k in i])
        
    end = time.time()
    print("Homology Dataset size:", np.array(dataset).shape)
    print("Time to compute fps:", end-start)

    with h5py.File("trained_networks", "w") as hf:
        # Example: store your dataset
        hf.create_dataset("dataset", data=np.array(dataset))
        


    for sample in dataset: 
        mat = Parallel(n_jobs=-1)(
                delayed(graph.distance)(sample[i], k)
            for i in range(np.array(sample).shape[0])) 
        
        
        D = [[m,d] for m in mat]
        with Pool(processes=12) as pool:  # Outer multiprocessing pool
            results = pool.map(homology.multi_hom, D)
        if is_homogeneous(results):
            print(np.array(results).sum(axis=1))
            if np.array(results).sum(axis=1).max() <= 200:
                result_all.append(np.array(results).sum(axis=1))

    with h5py.File("homology_results", "w") as hf:
        # Example: store your dataset
        hf.create_dataset("dataset", data=np.array(result_all))
    
    plot_curves_with_mean(np.array(result_all), True)

if __name__ == "__main__":
    main()
