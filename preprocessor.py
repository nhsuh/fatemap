import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

# --- Step 1: Load the Raw Data ---
# We load the file you saved in Week 1, or re-download if needed
try:
    adata = sc.read('data/paul15.h5ad')
    print("Loaded cached data.")
except:
    adata = sc.datasets.paul15()
    adata.write('data/paul15.h5ad')
    print("Loaded fresh data.")

print(f"Original Shape: {adata.shape}")

# --- Step 2: Quality Control (QC) ---
# We calculate QC metrics to see how "healthy" our cells are
# This adds columns to adata.obs and adata.var
sc.pp.calculate_qc_metrics(adata, inplace=True)

# Filter Genes: Keep genes found in at least 3 cells
# Explanation: If a gene is in <3 cells, it cannot form a topological "bridge"
sc.pp.filter_genes(adata, min_cells=3)
print(f"Shape after Gene Filtering: {adata.shape}")

# --- Step 3: Normalization ---
# Normalize every cell to 10,000 counts (TPM-like)
# This removes the "Magnitude" effect we saw in the Euclidean heatmap
sc.pp.normalize_total(adata, target_sum=1e4)

# Logarithmize
# This helps with the skewness of biological data
sc.pp.log1p(adata)

# --- Step 4: Feature Selection (HVGs) ---
# We identify genes that vary highly. These are the "dimensions that matter."
# We keep the top 2000 genes for our PCA Lens.
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# Plot the dispersion (variance) to visualize which genes we are keeping
sc.pl.highly_variable_genes(adata, show=False)
plt.savefig('data/hvg_plot.png') # Check this image in your folder
print("\nSaved HVG plot to data/hvg_plot.png")

# Actually slice the data to keep only these variable genes
# We save the "raw" data first just in case we need it later
adata.raw = adata 
adata = adata[:, adata.var.highly_variable]

print(f"Final Shape (Cells x High-Variance Genes): {adata.shape}")

# --- Step 5: Save Checkpoint ---
# We overwrite the file so Week 2 Weekend starts with clean data
adata.write('data/paul15_cleaned.h5ad')
print("Saved preprocessed data to 'data/paul15_cleaned.h5ad'")