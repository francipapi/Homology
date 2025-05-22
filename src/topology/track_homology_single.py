from src.topology import homology
from src.data import dataset
from src.visualization import graph
from src.models import train
import os
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

def plot_curves_with_mean(data):
    """
    Plots multiple curves with a highlighted mean curve.

    Parameters:
    data (numpy.ndarray): An array of shape (n, 6), where each row is a curve.
    """
    n_layers = data.shape[1]  # Number of layers (6 in this case)
    x = np.arange(n_layers)  # x-coordinates: Layer 0 to 5

    # Plot all curves with transparency
    plt.figure(figsize=(8, 6))
    for curve in data:
        plt.plot(x, curve, color='blue', alpha=0.2)

    # Calculate and plot the mean curve
    mean_curve = np.mean(data, axis=0)
    plt.plot(x, mean_curve, color='blue', linewidth=2, marker='s', label='Mean Curve')

    # Add labels and grid
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(r'$\beta_0$', fontsize=14, rotation=0, labelpad=15)
    plt.xticks(x, labels=["input"] + [str(i) for i in range(1, n_layers-1)] + ["output"])
    plt.grid(True)
    plt.legend()
    plt.title("Curves and Mean Curve")

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
    k=10
    d=3
    result_all=[]
    X, y = data.gen_easy(n, 3, 1)
    # Data generation
    for _ in range(5):

        #X, y = data.gen_easy(n, 3, 1)
        dataset_raw=train.train(X, y, width=8, accuracy=0.999, epochs=100, layers=8, device="cpu")
        dataset = []
        for l in dataset_raw:
            dataset.append(data.farthest_point_sampling(l, 2000))

        mat = Parallel(n_jobs=-1)(
                delayed(graph.distance)(dataset[i], k)
            for i in range(8))
        
        
        D = [[m,d] for m in mat]
        with Pool(processes=8) as pool:  # Outer multiprocessing pool
            results = pool.map(homology.multi_hom, D)
        if is_homogeneous(results):
            print(np.array(results).sum(axis=1))
            if np.array(results).sum(axis=1).max() <= 30:
                result_all.append(np.array(results).sum(axis=1))
        
    plot_curves_with_mean(np.array(result_all))

if __name__ == "__main__":
    main()