import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load Clean Data ---
adata = sc.read('data/paul15_cleaned.h5ad')

# --- Step 2: Compute PCA (The Lens) ---
# We compute 50 components, but we will mostly look at the first 2-3
sc.tl.pca(adata, svd_solver='arpack')

# --- Step 3: Variance Ratio (The "Elbow") ---
# This tells us how much information we lose by projecting
sc.pl.pca_variance_ratio(adata, log=True, show=False)
plt.savefig('data/pca_variance.png')

# --- Step 4: The Projection (Visualizing the Lens) ---
# We plot PC1 vs PC2. This is the "Shadow" of the high-dim object.
# We color it by the 'paul15_clusters' to see if the lens separates the biology.
sc.pl.pca(adata, color='paul15_clusters', show=False)
plt.title("The Mapper Lens (PCA Projection)")
plt.savefig('data/lens_projection.png')

print("Projection complete.")
print("Check 'data/lens_projection.png' - Do you see a triangle/Y-shape?")

# --- Step 5: Save the Lens Coordinates ---
# We need these numbers for the custom Mapper implementation next week
# They are stored in adata.obsm['X_pca']
print(f"Lens Coordinates Shape: {adata.obsm['X_pca'].shape}")
adata.write('data/paul15_pca.h5ad')