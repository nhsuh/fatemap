import scanpy as sc
import numpy as np
import networkx as nx
import json
import pandas as pd # Essential for calculating the majority vote
from sklearn.cluster import DBSCAN

# --- Step 1: Load Data ---
print("Loading data...")
try:
    adata = sc.read('data/paul15_pca.h5ad')
except Exception as e:
    print(f"Error loading data: {e}")
    print("Make sure you run compute_lens.py first!")
    exit()

# The Lens (for binning) - 2 Dimensions
data_2d = adata.obsm['X_pca'][:, :2] 

# The Metric Space (for clustering) - 50 Dimensions
data_50d = adata.obsm['X_pca']       

# The Biology (Ground Truth)
# We need this to "vote" on what a node represents
cell_types = adata.obs['paul15_clusters']

# --- Step 2: Define the Cover (Grid) ---
# Using your successful parameters
resolution = 15
overlap = 0.5 

# Calculate Grid Dimensions
x_min, y_min = data_2d.min(axis=0)
x_max, y_max = data_2d.max(axis=0)
x_step = (x_max - x_min) / resolution
y_step = (y_max - y_min) / resolution

# The actual width of a bin (accounting for overlap)
bin_w_x = x_step / (1 - overlap)
bin_w_y = y_step / (1 - overlap)

print(f"Grid Setup: {resolution}x{resolution} with {overlap*100}% overlap.")

# --- Step 3: The Mapper Loop ---
print("Starting Mapper Loop with Biological Annotation...")

nodes = [] 

# Grid Loop
x_starts = np.arange(x_min, x_max, x_step)
y_starts = np.arange(y_min, y_max, y_step)

for i, x in enumerate(x_starts):
    for j, y in enumerate(y_starts):
        
        # A. Binning (The Lens)
        # Find all cells that fall visually inside this square
        in_box = (
            (data_2d[:, 0] >= x) & (data_2d[:, 0] <= x + bin_w_x) &
            (data_2d[:, 1] >= y) & (data_2d[:, 1] <= y + bin_w_y)
        )
        indices = np.where(in_box)[0]
        
        # Skip empty/tiny parts of the grid
        if len(indices) < 5: 
            continue
            
        # B. Clustering (The Nerve)
        # We switch to 50D space to see the real topology
        subset_50d = data_50d[indices]
        
        # *** TUNED PARAMETER: eps=11.0 ***
        clustering = DBSCAN(eps=11.0, min_samples=3).fit(subset_50d)
        unique_labels = np.unique(clustering.labels_)
        
        # C. Node Creation
        for label in unique_labels:
            if label == -1: continue # Skip noise
            
            # Get the specific cells in this cluster
            member_mask = (clustering.labels_ == label)
            member_indices = indices[member_mask]
            
            # --- NEW: Identify the Biology ---
            # 1. Get the cell types for these specific cells
            node_cell_types = cell_types.iloc[member_indices]
            
            # 2. Find the most common type (The "Majority Vote")
            # If the node has 10 cells and 8 are "Erythrocyte", the node is Erythrocyte.
            if len(node_cell_types) > 0:
                majority_type = node_cell_types.mode()[0]
                
                # 3. Calculate "purity" (Confidence)
                # Count of majority / Total count
                count = node_cell_types.value_counts().max()
                purity = count / len(member_indices)
            else:
                majority_type = "Unknown"
                purity = 0.0

            # Create the Node Object
            node = {
                'id': f"node_{i}_{j}_{label}", # Unique ID
                'indices': member_indices.tolist(), # The payload
                'size': int(len(member_indices)),   # For visualization sizing
                'pos_x': float(np.mean(data_2d[member_indices, 0])),
                'pos_y': float(np.mean(data_2d[member_indices, 1])),
                
                # Metadata for the Frontend
                'type': str(majority_type), 
                'purity': float(purity)
            }
            nodes.append(node)

print(f"Mapper Complete. Generated {len(nodes)} biologically annotated nodes.")

# --- Step 4: Build Edges (Intersection) ---
print("Connecting the graph...")
G = nx.Graph()

# Add all nodes first
for n in nodes:
    G.add_node(n['id'], size=n['size'], pos_x=n['pos_x'], pos_y=n['pos_y'], type=n['type'], purity=n['purity'])

# Check for connections (The Glue)
for i in range(len(nodes)):
    for k in range(i + 1, len(nodes)):
        node_a = nodes[i]
        node_b = nodes[k]
        
        # Intersection Logic
        set_a = set(node_a['indices'])
        set_b = set(node_b['indices'])
        common_cells = set_a.intersection(set_b)
        
        if len(common_cells) > 0:
            G.add_edge(node_a['id'], node_b['id'], weight=len(common_cells))

print(f"Graph Built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# --- Step 5: Export ---
data_export = nx.node_link_data(G)

with open('data/mapper_graph.json', 'w') as f:
    json.dump(data_export, f)

print("\nSUCCESS: Saved 'data/mapper_graph.json'.")
print("Action: Copy this file to your web/public/data/ folder now.")