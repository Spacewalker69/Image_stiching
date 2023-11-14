import cv2
import numpy as np
from scipy.spatial import KDTree
import networkx as nx
import matplotlib.pyplot as plt
import os

def plot_graph(graph, features):
    G = nx.Graph()
    for (img1, img2), weight in graph.items():
        G.add_edge(img1, img2, weight=weight)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Original Graph")
    plt.show()

def plot_minimum_spanning_tree(T, features):
    pos = nx.spring_layout(T)
    nx.draw(T, pos, with_labels=True, font_weight='bold', node_size=700, node_color='lightcoral')
    labels = nx.get_edge_attributes(T, 'weight')
    nx.draw_networkx_edge_labels(T, pos, edge_labels=labels)
    plt.title("Minimum Spanning Tree")
    plt.show()

def display_images_with_nodes(features, graph):
    for img_path, (keypoints, _) in features.items():
        img = cv2.imread(img_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Image: {os.path.basename(img_path)}")

        # Plot keypoints on the image
        for point in keypoints:
            plt.scatter(point.pt[0], point.pt[1], s=50, c='red', marker='x')

        # Highlight connected nodes in the graph
        for (img1, img2) in graph.keys():
            if img_path == img1 or img_path == img2:
                plt.plot([keypoints[0].pt[0], keypoints[1].pt[0]],
                         [keypoints[0].pt[1], keypoints[1].pt[1]], 'k-', color='blue')

        plt.show()