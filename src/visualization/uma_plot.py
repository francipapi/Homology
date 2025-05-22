import umap.umap_
from src.data import dataset
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from src.models import train
import umap.plot
import pandas as pd
from src.visualization import graph
import h5py
from src.utils import mps



X, y = data.generate(4000, 3, 1)



dataset = mps.train([X,y])
y = y.reshape(-1,1)

with h5py.File("trained_networks", "w") as hf:
        # Example: store your dataset
        hf.create_dataset("dataset", data=np.array(dataset))
'''
with h5py.File("trained_networks", 'r') as f:
        dataset = np.array(f["dataset"])
        print("Shape:", dataset.shape)
'''
#scaled_data = StandardScaler().fit_transform(dataset[2])
#embedding_3d = umap.umap_.UMAP(n_components=2, n_neighbors=15, n_jobs=-1, metric='precomputed').fit(distance_matrix)
#embedding_3d = umap.umap_.UMAP(n_components=2, n_neighbors=15, n_jobs=-1).fit(dataset[7])
# Set up the subplot grid
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for idx, data in enumerate(dataset):
    reducer = umap.umap_.UMAP(n_neighbors=15, n_jobs=-1)
    embedding = reducer.fit_transform(data)
    
    ax = axes[idx]
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                         c=y, cmap='Spectral', s=5)
    ax.set_title(f'layer {idx}')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.grid(True)


plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.suptitle('UMAP Embeddings', fontsize=16, y=1.02)
plt.show()

#umap.plot.points(embedding_3d, labels=y, width=1200, height=1200)
#plt.show()