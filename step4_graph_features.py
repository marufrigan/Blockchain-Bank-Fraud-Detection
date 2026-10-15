import pandas as pd
import networkx as nx
import os

os.makedirs("output", exist_ok=True)
print("=== Building Graph from network_edges.csv ===")

edges = pd.read_csv("network_edges.csv")
print(f"Network edges shape: {edges.shape}")
print(f"Columns: {edges.columns.tolist()}")

G = nx.Graph()

# Add edges — use first two columns as source/target if standard names not found
src_col = "account_id_1" if "account_id_1" in edges.columns else edges.columns[0]
tgt_col = "account_id_2" if "account_id_2" in edges.columns else edges.columns[1]

for _, row in edges.iterrows():
    G.add_edge(str(row[src_col]), str(row[tgt_col]))

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Compute graph features
degree_centrality = nx.degree_centrality(G)
clustering_coeff = nx.clustering(G)

# Fraud ring membership
ring_col = "ring_id" if "ring_id" in edges.columns else None
ring_accounts = set()
if ring_col:
    ring_edges = edges[edges[ring_col].notna()]
    ring_accounts = set(ring_edges[src_col].astype(str)).union(
                    set(ring_edges[tgt_col].astype(str)))

graph_df = pd.DataFrame({
    "account_id": list(G.nodes()),
    "degree_centrality": [degree_centrality.get(n, 0) for n in G.nodes()],
    "clustering_coefficient": [clustering_coeff.get(n, 0) for n in G.nodes()],
    "is_ring_member": [1 if n in ring_accounts else 0 for n in G.nodes()]
})

graph_df.to_csv("output/graph_features.csv", index=False)
print(f"Graph features saved: {graph_df.shape}")
print(f"Ring members: {graph_df['is_ring_member'].sum()}")
