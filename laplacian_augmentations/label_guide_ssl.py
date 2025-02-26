####### If we want to create the two views only using different centrality measures and detach the Laplacian-Max-Min Spectral Augmentations Module.
####### SAME NODES USED DURING LOGISTIC REGRESSION EVALUATION, AS STANDARD PROTOCOL

import torch
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx

def laplace_v1(dataset, percentage=0.5):
    data = dataset[0]  # PyG datasets often have a single graph at index 0
    G = to_networkx(data, to_undirected=True)
    degree_centrality = nx.degree_centrality(G)  # Returns a dictionary with node -> centrality
    num_nodes = len(degree_centrality)
    num_top_nodes = max(1, int(percentage * num_nodes / 100))  # Calculate the number of top nodes based on the percentage
    # Sort by centrality and get the top percentage of nodes based on centrality
    top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:num_top_nodes]
    node_classes = data.y.numpy()  # Node class labels
    for node in top_nodes:
        node_class = node_classes[node]
        same_class_nodes = [n for n, c in enumerate(node_classes) if c == node_class and n != node]
        # Add edges between the current node and nodes of the same class
        for other_node in same_class_nodes:
            if not G.has_edge(node, other_node):  # Avoid adding duplicate edges
                G.add_edge(node, other_node)
    modified_data = from_networkx(G)
    modified_data.x = data.x
    modified_data.y = data.y
    
    print(50*'-')
    print('laplace_v1 has been used')
    print(f"Percentage of top nodes used: {percentage}%")
    print(f"Length of top nodes list: {len(top_nodes)}")
    print(f"Length of non-top nodes list: {num_nodes - len(top_nodes)}")
    print(f"Number of different classes in top nodes: {len(set(node_classes[node] for node in top_nodes))}")
    print(50*'-')

    return modified_data

def laplace_v2_dc(dataset, percentage=2):
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    degree_centrality = nx.degree_centrality(G) # Returns a dictionary with node -> centrality
    total_nodes = len(degree_centrality)
    num_top_nodes = max(1, int(total_nodes * (percentage / 100)))
    top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:num_top_nodes]
    # Identify non-top-centrality nodes
    non_top_nodes = [n for n in range(total_nodes) if n not in top_nodes]
    print(50*'-')
    print('laplace_v2_dc has been used')
    print(f"Percentage of top nodes used: {percentage}%")
    print(f"Length of top nodes list: {len(top_nodes)}")
    print(f"Length of non-top nodes list: {len(non_top_nodes)}")
    node_classes = data.y
    unique_classes = set(node_classes[node] for node in top_nodes)
    print(f"Number of different classes in top nodes: {len(unique_classes)}")
    print(50*'-')
    # restrict edge creation to only top nodes belonging to the same class
    for node in top_nodes:
        node_class = node_classes[node]
        same_class_top_nodes = [n for n in top_nodes if node_classes[n] == node_class and n != node]
        # add edges between nodes within this top set that belong to the same class
        for other_node in same_class_top_nodes:
            if not G.has_edge(node, other_node):  # avoid adding duplicate edges
                G.add_edge(node, other_node)
    # remove edges between nodes of different classes in the top-centrality set
    for node in top_nodes:
        node_class = node_classes[node]
        different_class_top_nodes = [n for n in top_nodes if node_classes[n] != node_class]
        for other_node in different_class_top_nodes:
            if G.has_edge(node, other_node):  # Remove edge if it exists
                G.remove_edge(node, other_node)
    modified_data = from_networkx(G)
    modified_data.x = data.x
    modified_data.y = data.y

    return modified_data

def laplace_v2_pc(dataset, percentage=2):
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    pagerank_centrality = nx.pagerank(G) # Returns a dictionary with node -> centrality
    total_nodes = len(pagerank_centrality)
    num_top_nodes = max(1, int(total_nodes * (percentage / 100)))
    top_nodes = sorted(pagerank_centrality, key=pagerank_centrality.get, reverse=True)[:num_top_nodes]
    # Identify non-top-centrality nodes based on centrality
    non_top_nodes = [n for n in range(total_nodes) if n not in top_nodes]
    print(50*'-')
    print('laplace_v2_pc has been used')
    print(f"Percentage of top nodes used: {percentage}%")
    print(f"Length of top nodes list: {len(top_nodes)}")
    print(f"Length of non-top nodes list: {len(non_top_nodes)}")
    node_classes = data.y
    unique_classes = set(node_classes[node] for node in top_nodes)
    print(f"Number of different classes in top nodes: {len(unique_classes)}")
    print(50*'-')
    # restrict edge creation to only top nodes belonging to the same class
    for node in top_nodes:
        node_class = node_classes[node]
        same_class_top_nodes = [n for n in top_nodes if node_classes[n] == node_class and n != node]
        # add edges between nodes within this top set that belong to the same class
        for other_node in same_class_top_nodes:
            if not G.has_edge(node, other_node):  # avoid adding duplicate edges
                G.add_edge(node, other_node)
    # remove edges between nodes of different classes in the top-centrality set
    for node in top_nodes:
        node_class = node_classes[node]
        different_class_top_nodes = [n for n in top_nodes if node_classes[n] != node_class]
        for other_node in different_class_top_nodes:
            if G.has_edge(node, other_node):  # Remove edge if it exists
                G.remove_edge(node, other_node)
    modified_data = from_networkx(G)
    modified_data.x = data.x
    modified_data.y = data.y

    return modified_data

def laplace_v3(dataset, percentage=2):
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    degree_centrality = nx.degree_centrality(G)  # Compute degree centrality
    total_nodes = len(degree_centrality)
    num_top_nodes = max(1, int(total_nodes * (percentage / 100)))  # Determine the number of top nodes
    top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:num_top_nodes] # based on centrality
    non_top_nodes = [n for n in range(total_nodes) if n not in top_nodes]
    
    print(50 * '-')
    print('laplace_v3 has been used')
    print(f"Percentage of top nodes used: {percentage}%")
    print(f"Length of top nodes list: {len(top_nodes)}")
    print(f"Length of non-top nodes list: {len(non_top_nodes)}")
    print(50 * '-')
    
    # Add edges to connect all top nodes to each other
    for i, node in enumerate(top_nodes):
        for other_node in top_nodes[i + 1:]:  # Avoid duplicate edges
            if not G.has_edge(node, other_node):  # Add edge if it doesn't already exist
                G.add_edge(node, other_node)
    
    modified_data = from_networkx(G)
    modified_data.x = data.x
    modified_data.y = data.y

    return modified_data
