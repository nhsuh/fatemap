import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load Data ---
# We need the 50-dimensional data for accurate clustering
adata = sc.read('data/paul15_pca.h5ad')
data_50d = adata.obsm['X_pca'] 

print(f"High-Dim Shape for Clustering: {data_50d.shape}")

# --- Step 2: Define the Engine ---
def cluster_bin(indices, data_matrix, eps=10.0, min_samples=5):
    """
    Input:
      indices: List of cell indices in this bin (from Week 3)
      data_matrix: The 50D PC matrix
      eps: The 'radius' for DBSCAN (how close points must be)
      min_samples: Minimum points to form a cluster
    Returns:
      labels: Cluster ID for each point (-1 means noise)
    """
    if len(indices) == 0:
        return np.array([])
    
    # Extract the subset of data
    subset = data_matrix[indices]
    
    # Run DBSCAN
    # Note: eps=3.0 is a starting guess for PCA space. 
    # If you normalized differently, you might need 0.5 or 50.0.
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(subset)
    
    return clustering.labels_

# --- Step 3: Test on a "Tricky" Bin ---
# Let's artificially pick some points to test if it separates them.
# We'll pick 100 points from the start (stem) and 100 from the end (differentiated)
# In 50D space, these should definitely be two separate clusters.
indices_test = np.concatenate([np.arange(100), np.arange(2600, 2700)])

labels = cluster_bin(indices_test, data_50d, eps=10.0)

print(f"Test Input: {len(indices_test)} cells")
print(f"Clusters Found: {np.unique(labels)}")

# Visualization of the Test
# We plot these specific points in 2D to see if DBSCAN colored them differently
subset_2d = adata.obsm['X_pca'][indices_test, :2]
plt.figure(figsize=(6, 4))
plt.scatter(subset_2d[:, 0], subset_2d[:, 1], c=labels, cmap='tab10', s=20)
plt.title(f"Testing Nerve: Found {len(np.unique(labels)) - (1 if -1 in labels else 0)} Clusters")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.savefig('data/cluster_test.png')
print("Saved test plot to data/cluster_test.png")