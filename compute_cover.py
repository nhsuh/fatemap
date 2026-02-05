import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Step 1: Load Data ---
adata = sc.read('data/paul15_pca.h5ad')
data_2d = adata.obsm['X_pca'][:, :2]  # We only use the first 2 PCs
print(f"Loaded Lens Data: {data_2d.shape}")

# --- Step 2: The Mapper Cover Logic ---
class HypercubeCover:
    def __init__(self, resolution=15, overlap=0.5):
        """
        resolution: How many bins along the longest axis
        overlap: Percentage of overlap (0.0 to 1.0)
        """
        self.resolution = resolution
        self.overlap = overlap
        
    def fit(self, data):
        # 1. Find the bounds of the data
        self.min_vals = data.min(axis=0)
        self.max_vals = data.max(axis=0)
        
        # 2. Calculate bin width
        # The total range needs to be covered by 'resolution' number of bins
        # The math for bin_width w with overlap p: 
        # Range = w + (n-1)*(w*(1-p))
        data_range = self.max_vals - self.min_vals
        
        # Simple heuristic: Divide range by resolution, then expand by overlap factor
        # This isn't the 'perfect' math but works well for Mapper
        self.step_size = data_range / self.resolution
        self.bin_width = self.step_size / (1 - self.overlap)
        
        # 3. Generate intervals (The "Starts" of the bins)
        self.x_starts = np.arange(self.min_vals[0], self.max_vals[0], self.step_size[0])
        self.y_starts = np.arange(self.min_vals[1], self.max_vals[1], self.step_size[1])
        
        print(f"Created {len(self.x_starts)} x {len(self.y_starts)} grid.")
        print(f"Bin Width: {self.bin_width}")
        
        return self

    def transform(self, data):
        """
        Assigns every cell to a bin (or multiple bins).
        Returns: Dict { 'bin_id': [list of cell_indices] }
        """
        bins = {}
        
        # Iterate through every possible bin position
        for i, x_start in enumerate(self.x_starts):
            for j, y_start in enumerate(self.y_starts):
                
                # Define the box for this bin
                x_end = x_start + self.bin_width[0]
                y_end = y_start + self.bin_width[1]
                
                # Find points inside this box (Boolean masking)
                in_x = (data[:, 0] >= x_start) & (data[:, 0] <= x_end)
                in_y = (data[:, 1] >= y_start) & (data[:, 1] <= y_end)
                points_in_bin = np.where(in_x & in_y)[0]
                
                # Only save the bin if it's not empty
                if len(points_in_bin) > 0:
                    bin_id = f"{i}_{j}"
                    bins[bin_id] = points_in_bin.tolist()
                    
        return bins

# --- Step 3: Run the Cover ---
# We use Resolution=15 (15 blocks across) and 50% overlap
cover = HypercubeCover(resolution=15, overlap=0.5)
cover.fit(data_2d)
bins = cover.transform(data_2d)

print(f"Total Active Bins: {len(bins)}")

# --- Step 4: Visualize the Grid (Sanity Check) ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(data_2d[:, 0], data_2d[:, 1], s=5, c='gray', alpha=0.5)

# Draw the boxes
for i, x in enumerate(cover.x_starts):
    for j, y in enumerate(cover.y_starts):
        # Draw a rectangle for each bin
        # We only draw it if it actually contains data (roughly checking)
        bin_id = f"{i}_{j}"
        if bin_id in bins:
            rect = patches.Rectangle(
                (x, y), cover.bin_width[0], cover.bin_width[1], 
                linewidth=1, edgecolor='red', facecolor='none', alpha=0.3
            )
            ax.add_patch(rect)

plt.title("The Mapper Cover (Overlapping Hypercubes)")
plt.savefig('data/cover_grid.png')
print("Saved grid visualization to 'data/cover_grid.png'")