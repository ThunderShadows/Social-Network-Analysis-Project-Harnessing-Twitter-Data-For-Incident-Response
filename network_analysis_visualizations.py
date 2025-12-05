import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("SOCIAL NETWORK ANALYSIS - DISASTER HASHTAG NETWORK")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/8] Loading network data...")

# Load edges and nodes
edges_df = pd.read_csv('edges.csv')
nodes_df = pd.read_csv('nodes.csv')

print(f"   ✓ Loaded {len(edges_df)} edges")
print(f"   ✓ Loaded {len(nodes_df)} nodes")

# ============================================================================
# STEP 2: BUILD NETWORK
# ============================================================================
print("\n[2/8] Building network graph...")

# Create undirected graph
G = nx.Graph()

# Add nodes with attributes
for _, row in nodes_df.iterrows():
    G.add_node(row['Id'],
               label=row['Label'],
               frequency=row['Frequency'],
               disaster_count=row['disaster_count'],
               non_disaster_count=row['non_disaster_count'],
               disaster_ratio=row['disaster_ratio'],
               node_type=row['node_type'])

# Add edges with attributes
for _, row in edges_df.iterrows():
    if row['Source'] in G.nodes() and row['Target'] in G.nodes():
        G.add_edge(row['Source'], row['Target'],
                   weight=row['Weight'],
                   disaster_count=row['disaster_count'],
                   disaster_ratio=row['disaster_ratio'])

print(f"   ✓ Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"   ✓ Network type: {'Connected' if nx.is_connected(G) else 'Disconnected'}")

# ============================================================================
# STEP 3: CALCULATE NETWORK METRICS (REQUIRED BY PDF)
# ============================================================================
print("\n[3/8] Calculating network metrics...")

# Basic network statistics
print("\n   --- BASIC NETWORK STATISTICS ---")
density = nx.density(G)
print(f"   • Network Density: {density:.4f}")

# Get largest connected component for certain metrics
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G_connected = G.subgraph(largest_cc).copy()
    print(f"   • Largest component: {G_connected.number_of_nodes()} nodes")
else:
    G_connected = G

# Average degree
avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
print(f"   • Average Degree: {avg_degree:.2f}")

# Diameter and average path length (on connected component)
try:
    diameter = nx.diameter(G_connected)
    avg_path_length = nx.average_shortest_path_length(G_connected)
    print(f"   • Network Diameter: {diameter}")
    print(f"   • Average Path Length: {avg_path_length:.4f}")
except:
    print(f"   • Network Diameter: N/A (disconnected)")
    print(f"   • Average Path Length: N/A (disconnected)")

# Clustering coefficient
avg_clustering = nx.average_clustering(G)
print(f"   • Average Clustering Coefficient: {avg_clustering:.4f}")

# Transitivity
transitivity = nx.transitivity(G)
print(f"   • Transitivity: {transitivity:.4f}")

# ============================================================================
# CALCULATE CENTRALITY METRICS (REQUIRED)
# ============================================================================
print("\n   --- CENTRALITY METRICS ---")

# 1. Degree Centrality
degree_centrality = nx.degree_centrality(G)
print(f"   ✓ Degree Centrality calculated")

# 2. Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
print(f"   ✓ Betweenness Centrality calculated")

# 3. Closeness Centrality
closeness_centrality = nx.closeness_centrality(G)
print(f"   ✓ Closeness Centrality calculated")

# 4. Eigenvector Centrality
try:
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
    print(f"   ✓ Eigenvector Centrality calculated")
except:
    eigenvector_centrality = {node: 0 for node in G.nodes()}
    print(f"   ⚠ Eigenvector Centrality failed (using zeros)")

# 5. PageRank
pagerank = nx.pagerank(G, weight='weight')
print(f"   ✓ PageRank calculated")

# ============================================================================
# COMMUNITY DETECTION (REQUIRED)
# ============================================================================
print("\n   --- COMMUNITY DETECTION ---")

# Louvain method (using python-louvain if available, else greedy modularity)
try:
    import community.community_louvain as community_louvain
    communities = community_louvain.best_partition(G, weight='weight')
    modularity = community_louvain.modularity(communities, G, weight='weight')
    print(f"   ✓ Louvain Method: {len(set(communities.values()))} communities")
    print(f"   • Modularity Score: {modularity:.4f}")
except ImportError:
    # Fallback to greedy modularity communities
    communities_gen = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
    communities = {}
    for idx, comm in enumerate(communities_gen):
        for node in comm:
            communities[node] = idx
    print(f"   ✓ Greedy Modularity: {len(set(communities.values()))} communities")

# ============================================================================
# STEP 4: CREATE METRICS DATAFRAME
# ============================================================================
print("\n[4/8] Creating metrics dataframe...")

metrics_df = pd.DataFrame({
    'Node': list(G.nodes()),
    'Label': [G.nodes[node].get('label', node) for node in G.nodes()],
    'Degree': [G.degree(node) for node in G.nodes()],
    'Degree_Centrality': [degree_centrality[node] for node in G.nodes()],
    'Betweenness_Centrality': [betweenness_centrality[node] for node in G.nodes()],
    'Closeness_Centrality': [closeness_centrality[node] for node in G.nodes()],
    'Eigenvector_Centrality': [eigenvector_centrality[node] for node in G.nodes()],
    'PageRank': [pagerank[node] for node in G.nodes()],
    'Community': [communities[node] for node in G.nodes()],
    'Frequency': [G.nodes[node].get('frequency', 0) for node in G.nodes()],
    'Disaster_Ratio': [G.nodes[node].get('disaster_ratio', 0) for node in G.nodes()]
})

# Sort by degree
metrics_df = metrics_df.sort_values('Degree', ascending=False)

# Export metrics
metrics_df.to_csv('network_metrics_calculated.csv', index=False)
print(f"   ✓ Metrics saved to 'network_metrics_calculated.csv'")

# Show top 10 nodes by different metrics
print("\n   --- TOP 10 NODES BY DEGREE ---")
top_10_degree = metrics_df.nlargest(10, 'Degree')[['Label', 'Degree', 'PageRank']]
print(top_10_degree.to_string(index=False))

# ============================================================================
# STEP 5: VISUALIZATION 1 - NETWORK OVERVIEW WITH COMMUNITIES
# ============================================================================
print("\n[5/8] Creating Visualization 1: Network Overview with Communities...")

fig, ax = plt.subplots(figsize=(20, 20), facecolor='white')

# Use spring layout for better visualization
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Node colors by community
node_colors = [communities[node] for node in G.nodes()]
unique_communities = set(node_colors)

# Node sizes by degree
node_sizes = [G.degree(node) * 50 for node in G.nodes()]

# Draw network
nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       node_size=node_sizes,
                       cmap='tab20',
                       alpha=0.8,
                       ax=ax)

# Draw edges with transparency
nx.draw_networkx_edges(G, pos,
                       width=0.5,
                       alpha=0.3,
                       edge_color='gray',
                       ax=ax)

# Draw labels for top nodes only (degree > 15)
top_nodes = [node for node in G.nodes() if G.degree(node) > 15]
labels = {node: G.nodes[node].get('label', node) for node in top_nodes}
nx.draw_networkx_labels(G, pos,
                        labels=labels,
                        font_size=10,
                        font_weight='bold',
                        ax=ax)

ax.set_title('Disaster Hashtag Co-occurrence Network\nColored by Communities, Sized by Degree',
             fontsize=24, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig('viz_1_network_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: viz_1_network_overview.png")
plt.close()

# ============================================================================
# STEP 6: VISUALIZATION 2 - TOP NODES BY CENTRALITY
# ============================================================================
print("\n[6/8] Creating Visualization 2: Top Nodes by Centrality...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='white')
fig.suptitle('Top 15 Hashtags by Different Centrality Measures',
             fontsize=24, fontweight='bold', y=0.995)

# Top 15 for each metric
top_n = 15

# 1. Degree Centrality
ax = axes[0, 0]
top_degree = metrics_df.nlargest(top_n, 'Degree')
colors = plt.cm.Reds(np.linspace(0.4, 0.9, top_n))
ax.barh(range(top_n), top_degree['Degree'].values, color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_degree['Label'].values, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Degree', fontsize=12, fontweight='bold')
ax.set_title('Degree Centrality\n(Most Connected Nodes)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 2. Betweenness Centrality
ax = axes[0, 1]
top_between = metrics_df.nlargest(top_n, 'Betweenness_Centrality')
colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
ax.barh(range(top_n), top_between['Betweenness_Centrality'].values, color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_between['Label'].values, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Betweenness Centrality', fontsize=12, fontweight='bold')
ax.set_title('Betweenness Centrality\n(Bridge Nodes)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 3. Closeness Centrality
ax = axes[1, 0]
top_close = metrics_df.nlargest(top_n, 'Closeness_Centrality')
colors = plt.cm.Greens(np.linspace(0.4, 0.9, top_n))
ax.barh(range(top_n), top_close['Closeness_Centrality'].values, color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_close['Label'].values, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Closeness Centrality', fontsize=12, fontweight='bold')
ax.set_title('Closeness Centrality\n(Fast Information Spreaders)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 4. PageRank
ax = axes[1, 1]
top_pagerank = metrics_df.nlargest(top_n, 'PageRank')
colors = plt.cm.Purples(np.linspace(0.4, 0.9, top_n))
ax.barh(range(top_n), top_pagerank['PageRank'].values, color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_pagerank['Label'].values, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('PageRank', fontsize=12, fontweight='bold')
ax.set_title('PageRank\n(Most Influential Nodes)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('viz_2_centrality_rankings.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: viz_2_centrality_rankings.png")
plt.close()

# ============================================================================
# STEP 7: VISUALIZATION 3 - DISASTER VS NON-DISASTER COMPARISON
# ============================================================================
print("\n[7/8] Creating Visualization 3: Disaster vs Non-Disaster Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='white')
fig.suptitle('Disaster-Related vs Non-Disaster Hashtag Analysis',
             fontsize=22, fontweight='bold', y=0.995)

# Categorize nodes
disaster_nodes = metrics_df[metrics_df['Disaster_Ratio'] >= 0.8]
non_disaster_nodes = metrics_df[metrics_df['Disaster_Ratio'] < 0.2]
mixed_nodes = metrics_df[(metrics_df['Disaster_Ratio'] >= 0.2) & (metrics_df['Disaster_Ratio'] < 0.8)]

# 1. Distribution of Disaster Ratio
ax = axes[0, 0]
ax.hist(metrics_df['Disaster_Ratio'], bins=20, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax.set_xlabel('Disaster Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Hashtags', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Disaster Ratio', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2. Pie chart of node categories
ax = axes[0, 1]
categories = ['Disaster\n(≥0.8)', 'Mixed\n(0.2-0.8)', 'Non-Disaster\n(<0.2)']
sizes = [len(disaster_nodes), len(mixed_nodes), len(non_disaster_nodes)]
colors_pie = ['#ff6b6b', '#feca57', '#48dbfb']
explode = (0.05, 0, 0)
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=categories, colors=colors_pie,
                                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax.set_title('Hashtag Categories', fontsize=14, fontweight='bold')

# 3. Average degree by category
ax = axes[1, 0]
categories_data = ['Disaster', 'Mixed', 'Non-Disaster']
avg_degrees = [
    disaster_nodes['Degree'].mean() if len(disaster_nodes) > 0 else 0,
    mixed_nodes['Degree'].mean() if len(mixed_nodes) > 0 else 0,
    non_disaster_nodes['Degree'].mean() if len(non_disaster_nodes) > 0 else 0
]
bars = ax.bar(categories_data, avg_degrees, color=colors_pie, edgecolor='black', linewidth=2)
ax.set_ylabel('Average Degree', fontsize=12, fontweight='bold')
ax.set_title('Average Degree by Category', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# 4. Top disaster hashtags network
ax = axes[1, 1]
ax.axis('off')

# Create subgraph of top disaster nodes
top_disaster = disaster_nodes.nlargest(20, 'Degree')['Node'].values
G_disaster = G.subgraph(top_disaster)

if G_disaster.number_of_nodes() > 0:
    pos_disaster = nx.spring_layout(G_disaster, k=1.5, iterations=50, seed=42)

    node_sizes_disaster = [G_disaster.degree(node) * 100 for node in G_disaster.nodes()]
    node_colors_disaster = [communities[node] for node in G_disaster.nodes()]

    nx.draw_networkx_nodes(G_disaster, pos_disaster,
                           node_color=node_colors_disaster,
                           node_size=node_sizes_disaster,
                           cmap='Set3',
                           alpha=0.8,
                           ax=ax)

    nx.draw_networkx_edges(G_disaster, pos_disaster,
                           width=1,
                           alpha=0.5,
                           edge_color='gray',
                           ax=ax)

    labels_disaster = {node: G.nodes[node].get('label', node) for node in G_disaster.nodes()}
    nx.draw_networkx_labels(G_disaster, pos_disaster,
                            labels=labels_disaster,
                            font_size=8,
                            font_weight='bold',
                            ax=ax)

    ax.set_title('Top 20 Disaster Hashtags Network', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('viz_3_disaster_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: viz_3_disaster_analysis.png")
plt.close()

# ============================================================================
# STEP 8: VISUALIZATION 4 - COMMUNITY ANALYSIS
# ============================================================================
print("\n[8/8] Creating Visualization 4: Community Analysis...")

fig = plt.figure(figsize=(22, 12), facecolor='white')
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Community Detection and Analysis',
             fontsize=24, fontweight='bold', y=0.98)

# 1. Main network with communities (large plot)
ax1 = fig.add_subplot(gs[:, 0])
ax1.axis('off')

# Filter to show only well-connected nodes
degree_threshold = 5
G_filtered = G.subgraph([n for n in G.nodes() if G.degree(n) >= degree_threshold])

pos_comm = nx.spring_layout(G_filtered, k=1.5, iterations=50, seed=42)

node_colors_comm = [communities[node] for node in G_filtered.nodes()]
node_sizes_comm = [G_filtered.degree(node) * 80 for node in G_filtered.nodes()]

nx.draw_networkx_nodes(G_filtered, pos_comm,
                       node_color=node_colors_comm,
                       node_size=node_sizes_comm,
                       cmap='tab20',
                       alpha=0.8,
                       ax=ax1)

nx.draw_networkx_edges(G_filtered, pos_comm,
                       width=0.5,
                       alpha=0.3,
                       edge_color='gray',
                       ax=ax1)

# Label top nodes
top_nodes_filtered = [node for node in G_filtered.nodes() if G_filtered.degree(node) > 15]
labels_filtered = {node: G.nodes[node].get('label', node) for node in top_nodes_filtered}
nx.draw_networkx_labels(G_filtered, pos_comm,
                        labels=labels_filtered,
                        font_size=9,
                        font_weight='bold',
                        ax=ax1)

ax1.set_title(f'Network Communities\n(Showing nodes with degree ≥ {degree_threshold})',
              fontsize=16, fontweight='bold', pad=10)

# 2. Community sizes
ax2 = fig.add_subplot(gs[0, 1])
community_sizes = Counter(communities.values())
comm_labels = [f'C{i}' for i in sorted(community_sizes.keys())]
comm_counts = [community_sizes[i] for i in sorted(community_sizes.keys())]

colors_bar = plt.cm.tab20(np.linspace(0, 1, len(comm_labels)))
bars = ax2.bar(comm_labels, comm_counts, color=colors_bar, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Community', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
ax2.set_title('Community Sizes', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. Top communities pie chart
ax3 = fig.add_subplot(gs[0, 2])
top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:8]
other_count = sum([count for comm, count in community_sizes.items()
                   if comm not in [c[0] for c in top_communities]])

pie_labels = [f'Community {comm}' for comm, _ in top_communities]
pie_sizes = [count for _, count in top_communities]

if other_count > 0:
    pie_labels.append('Others')
    pie_sizes.append(other_count)

colors_pie2 = plt.cm.tab20(np.linspace(0, 1, len(pie_labels)))
wedges, texts, autotexts = ax3.pie(pie_sizes, labels=pie_labels, colors=colors_pie2,
                                     autopct='%1.1f%%', startangle=90,
                                     textprops={'fontsize': 9})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax3.set_title('Top Communities Distribution', fontsize=14, fontweight='bold')

# 4. Average centrality by community (top 5 communities)
ax4 = fig.add_subplot(gs[1, 1])
top_5_communities = [comm for comm, _ in top_communities[:5]]
metrics_df['Community_Label'] = 'C' + metrics_df['Community'].astype(str)

avg_centrality_by_comm = []
for comm in top_5_communities:
    comm_nodes = metrics_df[metrics_df['Community'] == comm]
    avg_centrality_by_comm.append({
        'Community': f'C{comm}',
        'Avg_Degree_Centrality': comm_nodes['Degree_Centrality'].mean(),
        'Avg_PageRank': comm_nodes['PageRank'].mean() * 100  # Scale for visibility
    })

centrality_df = pd.DataFrame(avg_centrality_by_comm)
x = np.arange(len(centrality_df))
width = 0.35

bars1 = ax4.bar(x - width/2, centrality_df['Avg_Degree_Centrality'],
                width, label='Avg Degree Centrality', color='steelblue', edgecolor='black')
bars2 = ax4.bar(x + width/2, centrality_df['Avg_PageRank'],
                width, label='Avg PageRank (×100)', color='coral', edgecolor='black')

ax4.set_xlabel('Community', fontsize=12, fontweight='bold')
ax4.set_ylabel('Average Centrality', fontsize=12, fontweight='bold')
ax4.set_title('Average Centrality by Community (Top 5)', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(centrality_df['Community'])
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# 5. Top hashtags from largest communities
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')

# Get top 3 communities
top_3_communities = [comm for comm, _ in top_communities[:3]]
text_y = 0.95

ax5.text(0.5, text_y, 'Top Hashtags per Community',
         ha='center', va='top', fontsize=14, fontweight='bold',
         transform=ax5.transAxes)

text_y -= 0.1

for idx, comm in enumerate(top_3_communities):
    comm_nodes = metrics_df[metrics_df['Community'] == comm].nlargest(5, 'Degree')

    ax5.text(0.05, text_y, f'Community {comm} (n={community_sizes[comm]}):',
             ha='left', va='top', fontsize=12, fontweight='bold',
             color=colors_bar[comm], transform=ax5.transAxes)

    text_y -= 0.05

    for _, row in comm_nodes.iterrows():
        ax5.text(0.1, text_y, f"• {row['Label']} (deg: {row['Degree']})",
                 ha='left', va='top', fontsize=10,
                 transform=ax5.transAxes)
        text_y -= 0.04

    text_y -= 0.03

plt.savefig('viz_4_community_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: viz_4_community_analysis.png")
plt.close()

# ============================================================================
# VISUALIZATION 5 - DEGREE DISTRIBUTION
# ============================================================================
print("\n[BONUS] Creating Visualization 5: Degree Distribution...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='white')
fig.suptitle('Network Degree Distribution Analysis',
             fontsize=22, fontweight='bold', y=1.02)

degrees = [G.degree(n) for n in G.nodes()]

# 1. Histogram
ax = axes[0]
ax.hist(degrees, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Degree', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Degree Distribution (Linear Scale)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.axvline(np.mean(degrees), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(degrees):.2f}')
ax.axvline(np.median(degrees), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(degrees):.2f}')
ax.legend()

# 2. Log-log plot (check for power law)
ax = axes[1]
degree_counts = Counter(degrees)
x_vals = sorted(degree_counts.keys())
y_vals = [degree_counts[x] for x in x_vals]

ax.scatter(x_vals, y_vals, alpha=0.7, s=50, color='darkgreen')
ax.set_xlabel('Degree (log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
ax.set_title('Degree Distribution (Log-Log Scale)', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which="both", ls="-")

# 3. Box plot by disaster ratio category
ax = axes[2]
metrics_df['Category'] = metrics_df['Disaster_Ratio'].apply(
    lambda x: 'Disaster' if x >= 0.8 else ('Non-Disaster' if x < 0.2 else 'Mixed')
)

categories_order = ['Disaster', 'Mixed', 'Non-Disaster']
data_to_plot = [metrics_df[metrics_df['Category'] == cat]['Degree'].values
                for cat in categories_order]

bp = ax.boxplot(data_to_plot, labels=categories_order, patch_artist=True,
                medianprops=dict(color='red', linewidth=2),
                boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5))

ax.set_ylabel('Degree', fontsize=12, fontweight='bold')
ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_title('Degree Distribution by Category', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('viz_5_degree_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: viz_5_degree_distribution.png")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("\n GENERATED VISUALIZATIONS:")
print("   1. viz_1_network_overview.png - Full network with communities")
print("   2. viz_2_centrality_rankings.png - Top nodes by centrality metrics")
print("   3. viz_3_disaster_analysis.png - Disaster vs non-disaster analysis")
print("   4. viz_4_community_analysis.png - Community detection results")
print("   5. viz_5_degree_distribution.png - Degree distribution analysis")
print("\n GENERATED DATA FILES:")
print("   • network_metrics_calculated.csv - All calculated metrics")

print("\n KEY FINDINGS:")
print(f"   • Total Nodes: {G.number_of_nodes()}")
print(f"   • Total Edges: {G.number_of_edges()}")
print(f"   • Network Density: {density:.4f}")
print(f"   • Average Degree: {avg_degree:.2f}")
print(f"   • Number of Communities: {len(set(communities.values()))}")
print(f"   • Clustering Coefficient: {avg_clustering:.4f}")
print(f"   • Disaster-related Hashtags (≥80%): {len(disaster_nodes)}")
print(f"   • Non-disaster Hashtags (<20%): {len(non_disaster_nodes)}")

print("\n TOP 5 MOST INFLUENTIAL HASHTAGS (by PageRank):")
for idx, row in metrics_df.nlargest(5, 'PageRank').iterrows():
    print(f"   {row['Label']:<25} - PageRank: {row['PageRank']:.6f}, Degree: {row['Degree']}")

