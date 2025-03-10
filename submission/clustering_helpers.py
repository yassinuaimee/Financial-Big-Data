import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import networkx as nx 
import numpy as np
from numpy import linalg as LA
from community import community_louvain
import matplotlib.colors as mcolors
from correlation_helpers import *


def compute_C_minus_C0(lambdas,v,lambda_plus,removeMarketMode=True):
    N=len(lambdas)
    C_clean=np.zeros((N, N))
    
    order = np.argsort(lambdas)
    lambdas,v = lambdas[order],v[:,order]
    
    v_m=np.matrix(v)

    # note that the eivenvalues are sorted
    for i in range(1*removeMarketMode,N):                            
        if lambdas[i]>lambda_plus: 
            C_clean=C_clean+lambdas[i] * np.dot(v_m[:,i],v_m[:,i].T)  
    return C_clean    



def create_graph(corr,sparsify=False,epsilon=1e-4):
    G=nx.Graph()
    for u in range(np.abs(corr).shape[0]):
        for v in range(np.abs(corr).shape[1]):
            if (u!=v):
                if(sparsify):  
                    if(np.abs(corr)[u,v]>epsilon):
                        G.add_edge(u,v,weight=np.abs(corr)[u,v])
                else:
                    G.add_edge(u,v,weight=np.abs(corr)[u,v])    
    return G

def plot_graph(G,partitions):
    pos = nx.spring_layout(G)  # Spring layout for clear visualization

    node_colors = [partitions[node] for node in G.nodes()]  # Cluster assignments as colors
    
    plt.figure(figsize=(6, 6))
    nx.draw(
        G, pos, with_labels=True, node_color=node_colors, 
        cmap=plt.cm.Set3, node_size=500, font_size=10
    )
    plt.title("Graph Visualization with Louvain Clustering")
    plt.show()

def plot_clustering(G,paritions,drop_small=False,draw_label=False):
    communities = {}
    for node, cluster in paritions.items():
        if cluster not in communities:
            communities[cluster] = []
        communities[cluster].append(node)

    if drop_small:
        subgraphs = {cluster: G.subgraph(nodes) for cluster, nodes in communities.items() if len(nodes) > 2}
    else:
        subgraphs = {cluster: G.subgraph(nodes) for cluster, nodes in communities.items()} 
    
    # Layout the communities modularly
    fig, ax = plt.subplots(figsize=(10, 6))
    pos_dict = {}
    cluster_positions = nx.circular_layout(list(range(len(subgraphs))), scale=10)
    
    # Color map for time-of-day
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=9, vmax=16)  # Assuming trading hours are 09:00 to 16:00

    for i, (cluster, subgraph) in enumerate(subgraphs.items()):
        cluster_pos = nx.circular_layout(subgraph, center=cluster_positions[i], scale=2)
        pos_dict.update(cluster_pos)

        node_colors = [
            cmap(norm(int(G.nodes[node]["timestamp"].hour)))
            for node in subgraph.nodes
        ]

        nx.draw(
            subgraph, cluster_pos, ax=ax,
            node_color=node_colors, node_size=100, edge_color="gray",
            with_labels=False, alpha=0.8
        )
        
        if draw_label:
             hour_labels = {node: str(G.nodes[node]["timestamp"].hour) for node in subgraph.nodes}

             nx.draw_networkx_labels(subgraph, cluster_pos, labels=hour_labels, font_size=10, font_color="black")

        x, y = zip(*[cluster_pos[node] for node in subgraph.nodes])
        cluster_center = (sum(x) / len(x), sum(y) / len(y))
        ax.text(
            cluster_center[0], cluster_center[1], f"Cluster {cluster}",
            fontsize=12, fontweight="bold", color="black", ha="center", va="center"
        )

    
    # Add a color bar for time-of-day
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
    cbar.set_label("Time of Day (Hour)")

    ax.set_title("Separated Communities with Time-of-Day Legend", fontsize=14)
    plt.axis("off")
    plt.show()
