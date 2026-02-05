import scanpy as sc
import numpy as np
import networkx as nx
import json
from sklearn.cluster import DBSCAN

# --- Step 1: Load Data ---
print("Loading data...")
adata = sc.read('data/paul15_pca.h5ad')
data_2d = adata.obsm['X_pca'][:, :2] # The Lens (for binning)
data_50d = adata.obsm['X_pca']       # The Metric Space (for clustering)

# --- Step 2: Define the Cover (Grid) ---
# We use the resolution/overlap from Week 3 that looked good
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
print("Starting Mapper Loop (This might take a minute)...")

nodes = [] 
node_id_counter = 0

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
        
        # Skip empty parts of the grid
        if len(indices) < 5: 
            continue
            
        # B. Clustering (The Nerve)
        # We switch to 50D space to see the real topology
        subset_50d = data_50d[indices]
        
        # *** USING YOUR TUNED PARAMETER HERE ***
        # eps=11.0 (Median distance + buffer)
        clustering = DBSCAN(eps=11.0, min_samples=3).fit(subset_50d)
        unique_labels = np.unique(clustering.labels_)
        
        # C. Node Creation
        for label in unique_labels:
            if label == -1: continue # Skip noise
            
            # Get the specific cells in this cluster
            member_mask = (clustering.labels_ == label)
            member_indices = indices[member_mask]
            
            # Create the Node Object
            # We calculate the average 2D position so we can plot it nicely later
            avg_x = np.mean(data_2d[member_indices, 0])
            avg_y = np.mean(data_2d[member_indices, 1])
            
            node = {
                'id': f"node_{i}_{j}_{label}", # Unique ID
                'indices': member_indices.tolist(), # The payload
                'size': int(len(member_indices)),   # For visualization sizing
                'pos_x': float(avg_x),
                'pos_y': float(avg_y)
            }
            nodes.append(node)

print(f"Mapper Complete. Generated {len(nodes)} topological nodes.")

# --- Step 4: Build Edges (Intersection) ---
print("Connecting the graph...")
G = nx.Graph()

# Add all nodes first
for n in nodes:
    G.add_node(n['id'], size=n['size'], pos_x=n['pos_x'], pos_y=n['pos_y'])

# Check for connections (The Glue)
# This is O(N^2), but N is small (number of nodes, not cells)
for i in range(len(nodes)):
    for k in range(i + 1, len(nodes)):
        node_a = nodes[i]
        node_b = nodes[k]
        
        # Intersection Logic
        # If the same cell appears in Node A and Node B, they are connected.
        set_a = set(node_a['indices'])
        set_b = set(node_b['indices'])
        common_cells = set_a.intersection(set_b)
        
        if len(common_cells) > 0:
            # The weight is how many cells they share (strength of connection)
            G.add_edge(node_a['id'], node_b['id'], weight=len(common_cells))

print(f"Graph Built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# --- Step 5: Export ---
# Save as JSON for the Cytoscape frontend
data_export = nx.node_link_data(G)

with open('data/mapper_graph.json', 'w') as f:
    json.dump(data_export, f)

print("\nSUCCESS: Saved 'data/mapper_graph.json'.")
print("You are ready for Week 6 (Frontend Visualization).")