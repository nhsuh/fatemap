import scanpy as sc
import numpy as np
import pandas as pd

# --- Step 1: Ingestion ---
print("1. Downloading and loading Paul15 dataset...")
# This fetches data from the cloud and caches it locally.
# Paul15 is a standard dataset of mouse bone marrow cells.
adata = sc.datasets.paul15()

# --- Step 2: Inspection of the AnnData Object ---
print("\n2. The AnnData Object (The Container):")
print(adata)
# You will see output like: AnnData object with n_obs × n_vars = 2730 × 3451

# --- Step 3: The Point Cloud (The X Matrix) ---
# In Topology, this is your metric space.
# Rows = Points (Cells)
# Cols = Dimensions (Genes)
print(f"\n3. Data Matrix Shape: {adata.X.shape}")
print(f"   (Rows/Cells: {adata.n_obs}, Cols/Genes: {adata.n_vars})")

# Check if it's a dense matrix or sparse matrix (usually floats)
print(f"   Data Type: {type(adata.X)}")

# --- Step 4: Metadata (obs and var) ---
# 'obs' (Observations) stores data about the CELLS (the points).
# This is where the 'Ground Truth' for cell type lives.
print("\n4. Cell Metadata (adata.obs):")
print(adata.obs.head())

# 'var' (Variables) stores data about the GENES (the dimensions).
print("\n5. Gene Metadata (adata.var):")
print(adata.var.head())

# --- Step 5: Accessing Specific Data Points ---
# Let's look at the expression of the 1st gene in the 1st cell.
# Note: Single-cell data is often stored as a Sparse Matrix to save RAM.
# We might need to convert to dense to view it easily.
cell_0_gene_0 = adata.X[0, 0]
print(f"\n6. Expression of Gene #0 in Cell #0: {cell_0_gene_0}")

print("\nSuccess. You have successfully ingested the manifold.")