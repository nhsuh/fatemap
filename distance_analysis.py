import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

# --- Step 1: Load Data ---
# We use the raw data you downloaded during the week
adata = sc.datasets.paul15()

# Pre-processing (minimal) just to handle zeros
# We convert sparse matrix to dense for scipy processing
# (Note: For massive datasets >50k cells, we would not do .toarray())
data_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

print(f"Data Shape: {data_matrix.shape}") # Should be (2730, 3451)

# --- Step 2: Compute Euclidean Distance (L2) ---
# pdist calculates pairwise distances condensed, squareform expands it to N x N
print("Computing Euclidean Distance...")
dist_euclidean = squareform(pdist(data_matrix, metric='euclidean'))

# --- Step 3: Compute Cosine Distance ---
print("Computing Cosine Distance...")
dist_cosine = squareform(pdist(data_matrix, metric='cosine'))

# --- Step 4: Visualization (The Manifold Check) ---
# We will sort the matrix by cell type to see if structure emerges
# If the metric is good, cells of the same type should be close (blue blocks on diagonal)

# Get cell types and sort indices
obs_key = 'paul15_clusters'
cell_types = adata.obs[obs_key]
sort_indices = np.argsort(cell_types.values)

# Reorder the matrices
sorted_euclidean = dist_euclidean[sort_indices, :][:, sort_indices]
sorted_cosine = dist_cosine[sort_indices, :][:, sort_indices]

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Plot Euclidean
sns.heatmap(sorted_euclidean, ax=ax[0], cmap='viridis', cbar=True)
ax[0].set_title("Euclidean Distance (Sorted by Cell Type)")
ax[0].set_xlabel("Cells")
ax[0].set_ylabel("Cells")

# Plot Cosine
sns.heatmap(sorted_cosine, ax=ax[1], cmap='viridis', cbar=True)
ax[1].set_title("Cosine Distance (Sorted by Cell Type)")
ax[1].set_xlabel("Cells")
ax[1].set_ylabel("Cells")

plt.tight_layout()
plt.show()