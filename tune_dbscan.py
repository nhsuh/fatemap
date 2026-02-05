import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 1. Load the 50D Data
adata = sc.read('data/paul15_pca.h5ad')
data_50d = adata.obsm['X_pca']

print(f"Data Shape: {data_50d.shape}")

# 2. Compute Nearest Neighbors
# We look at the 5th nearest neighbor (since min_samples=5 in DBSCAN)
# This asks: "How far do I have to travel to find 5 friends?"
nbrs = NearestNeighbors(n_neighbors=5).fit(data_50d)
distances, indices = nbrs.kneighbors(data_50d)

# 3. Sort and Plot (The K-Distance Graph)
# We take the distance to the 5th neighbor (column index 4)
k_dist = np.sort(distances[:, 4])

# Calculate basic stats to give you a starting number
print(f"\n--- Distance Statistics (50D) ---")
print(f"Min neighbor dist: {np.min(k_dist):.4f}")
print(f"Max neighbor dist: {np.max(k_dist):.4f}")
print(f"Mean neighbor dist: {np.mean(k_dist):.4f}")
print(f"Median neighbor dist: {np.median(k_dist):.4f}")
print(f"90th Percentile: {np.percentile(k_dist, 90):.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_dist)
plt.ylabel("Distance to 5th Nearest Neighbor")
plt.xlabel("Points sorted by distance")
plt.title("DBSCAN Parameter Tuning (The Elbow Method)")
plt.grid(True)
plt.axhline(y=np.median(k_dist), color='r', linestyle='--', label='Median')
plt.legend()
plt.savefig('data/dbscan_tuning.png')
print("\nCheck 'data/dbscan_tuning.png'. Look for the 'Elbow' (where the curve shoots up).")