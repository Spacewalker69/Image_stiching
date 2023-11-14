import cv2
import numpy as np
from scipy.spatial import KDTree
import networkx as nx
import matplotlib.pyplot as plt
import os


def compute_features(images):
    orb = cv2.ORB_create()
    features = {}
    for img_path in images:
        img = cv2.imread(img_path)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        features[img_path] = (keypoints, descriptors)
    return features


def construct_graph(features):
    graph = {}
    for img1, data1 in features.items():
        for img2, data2 in features.items():
            if img1 != img2:
                tree = KDTree(data2[1])
                matches = tree.query_ball_point(data1[1], r=0.1)  # r is a threshold

                # Calculate total number of matches
                total_matches = sum(len(match) for match in matches)

                # Use total_matches to compute weight
                weight = 1 / total_matches if total_matches > 0 else float('inf')
                graph[(img1, img2)] = weight
    return graph
def compute_mst(graph):
    G = nx.Graph()
    for (img1, img2), weight in graph.items():
        G.add_edge(img1, img2, weight=weight)
    T = nx.minimum_spanning_tree(G)
    return T
def extract_triplets(T):
    triplets = []
    for node in T.nodes():
        grandchildren = [n for child in T.neighbors(node) for n in T.neighbors(child) if n != node]
        for grandchild in grandchildren:
            triplets.append((node, next(T.neighbors(node)), grandchild))
    return triplets

def get_image_paths(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

