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

'''
def compute_features(images):
    # Create MSER and SIFT objects
    mser = cv2.MSER_create()
    sift = cv2.SIFT_create()

    features = {}
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect MSER regions
        regions, _ = mser.detectRegions(gray)

        # Convert MSER regions to keypoints (this is an approximation)
        keypoints = [cv2.KeyPoint(float(x), float(y), 1) for region in regions for x, y in region]

        # Compute SIFT descriptors
        _, descriptors = sift.compute(gray, keypoints)

        features[img_path] = (keypoints, descriptors)
    return features

    
def compute_features(images):
    mser = cv2.MSER_create()
    sift = cv2.SIFT_create()
    features = {}

    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # MSER works on grayscale images
        keypoints = mser.detect(gray, None)
        
        # Compute SIFT descriptors for the keypoints detected by MSER
        keypoints, descriptors = sift.compute(gray, keypoints)
        
        features[img_path] = {'keypoints': keypoints, 'descriptors': descriptors}
        del img
    
    return features
'''
'''def compute_features(images):
    # Check if SURF is available
    try:
        surf = cv2.xfeatures2d.SURF_create()
    except:
        raise ValueError("SURF not available.")
    
    features = {}

    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale directly
        keypoints, descriptors = surf.detectAndCompute(img, None)
        features[img_path] = {'keypoints': keypoints, 'descriptors': descriptors}
        del img  # Release memory

    return features
'''

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




def estimate_motion_parameters(matches, keypoints1, keypoints2, W, H):
    if len(matches) < 5:
        # Not enough matches, cannot compute essential matrix
        return None, None
    # Extract the locations of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Convert image points to spherical coordinates
    spherical_coords1 = [image_to_spherical(pt[0], pt[1], W, H) for pt in points1]
    spherical_coords2 = [image_to_spherical(pt[0], pt[1], W, H) for pt in points2]

    # Convert spherical coordinates to camera frame coordinates
    camera_coords1 = [spherical_to_camera(theta, phi) for theta, phi in spherical_coords1]
    camera_coords2 = [spherical_to_camera(theta, phi) for theta, phi in spherical_coords2]
    #print(np.float32(camera_coords1).shape)
    #print(np.float32(camera_coords2).shape)
    camera_coords1_array = np.float32(camera_coords1)[:, :2]
    camera_coords2_array = np.float32(camera_coords2)[:, :2]

    # Find the essential matrix
    E, mask = cv2.findEssentialMat(camera_coords1_array, camera_coords2_array, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    #E, mask = cv2.findEssentialMat(np.float32(camera_coords1), np.float32(camera_coords2), method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Decompose the essential matrix into R and t
    _, R, t, _ = cv2.recoverPose(E, np.float32(camera_coords1_array), np.float32(camera_coords2_array))
    #_, R, t, _ = cv2.recoverPose(E, np.float32(camera_coords1), np.float32(camera_coords2))

    # Compute the direction of motion β and the relative rotation α
    β = np.arctan2(t[1], t[0])
    α = np.arctan2(R[1, 0], R[0, 0])

    return β, α


def find_local_pose_for_triplets(triplets, features):
    # For each triplet, estimate the local pose
    local_poses = {}
    for triplet in triplets:
        img1, img2, img3 = triplet
        keypoints1, descriptors1 = features[img1]
        keypoints2, descriptors2 = features[img2]
        keypoints3, descriptors3 = features[img3]
        # Get the dimensions for each image
        W1, H1 = cv2.imread(img1).shape[1], cv2.imread(img1).shape[0]
        W2, H2 = cv2.imread(img2).shape[1], cv2.imread(img2).shape[0]
        W3, H3 = cv2.imread(img3).shape[1], cv2.imread(img3).shape[0]
        
        # Match features between img1 and img2, and between img2 and img3
        matches12 = match_features(descriptors1, descriptors2)
        matches23 = match_features(descriptors2, descriptors3)
        
        # Estimate motion parameters for each pair within the triplet
        β12, α12 = estimate_motion_parameters(matches12, keypoints1, keypoints2,W1,H1)
        β23, α23 = estimate_motion_parameters(matches23, keypoints2, keypoints3,W2,H2)
        
        # Store the estimated poses for the triplet
        local_poses[triplet] = {'β12': β12, 'α12': α12, 'β23': β23, 'α23': α23}
    
    return local_poses

# The match_features function would perform feature matching between two sets of descriptors
def match_features(descriptors1, descriptors2):
    # Create a matcher and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    # Sort matches based on their distance
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def image_to_spherical(u, v, W, H):
    # Convert image coordinates to spherical coordinates
    theta = -2 * np.pi * u / W
    phi = np.pi * (H - 2 * v) / (2 * H)
    return theta, phi

def spherical_to_camera(theta, phi):
    # Convert spherical coordinates to camera frame coordinates
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    return x, y, z

def propagate_scale_and_pose(local_poses, triplets):
    # Initialize the dictionary to hold the global poses and scales
    global_poses = {triplets[0][0]: np.eye(4)}  # Initialize with the pose of the first image as identity
    global_scales = {}  # This will store the scales

    for i in range(len(triplets) - 1):
        current_triplet = triplets[i]
        next_triplet = triplets[i + 1]

        # Find the shared image in current and next triplet
        shared_images = set(current_triplet) & set(next_triplet)
        if not shared_images:
            continue  # No shared image, skip this pair of triplets
        shared_image = shared_images.pop()  # Get the shared image

        # Ensure the shared image has a global pose computed
        if shared_image not in global_poses:
            # Compute the pose for the shared image if not present
            # For now, we'll just continue, but you'll need to add logic to compute this pose based on your application's needs
            continue

        pose_shared_global = global_poses[shared_image]
        
        # Propagate scale and pose to the next image using the shared image
        for img in current_triplet:
            if img != shared_image and img in next_triplet:
                # Compute local scale and pose
                local_pose = local_poses[current_triplet]['β12'], local_poses[current_triplet]['α12']
                # Compute global pose for the next image
                global_pose_next = pose_shared_global @ local_pose
                global_poses[img] = global_pose_next
                # Compute scale for the next image
                scale_next = np.linalg.norm(global_pose_next[:3, 3])
                global_scales[img] = scale_next

    return global_scales, global_poses



def traverse_mst(node, T, local_poses, global_poses, visited):
    """
    Traverse the MST to propagate pose and scale using triplets.
    node: Current node in the traversal
    T: Minimum spanning tree
    local_poses: Local poses calculated for each triplet
    global_poses: Dictionary to store the global pose of each node
    visited: Set to keep track of visited nodes
    """
    if node in visited:
        return
    visited.add(node)
    
    for neighbor in T.neighbors(node):
        if neighbor not in visited:
            # Find a relevant triplet that contains both node and neighbor
            for triplet, pose_data in local_poses.items():
                if node in triplet and neighbor in triplet:
                    # Assuming pose_data contains relevant transformation matrix
                    transformation = pose_data['transformation']
                    break
            else:
                print(f"No matching triplet found for nodes {node} and {neighbor}")
                continue  # Skip to the next neighbor

            # Compute and update global pose for neighbor
            if node in global_poses:
                global_poses[neighbor] = global_poses[node] @ transformation
            else:
                global_poses[neighbor] = transformation  # or some default pose

# Specify the folder containing images
folder_path = '1'
image_paths = get_image_paths(folder_path)
print (image_paths)
#image_paths = ["Images/20180109213443.xK7FQt_thumb.jpg", "Images/20180109213527.eVHVKN_thumb.jpg"]  # List of image paths
features = compute_features(image_paths)
graph = construct_graph(features)

T= compute_mst(graph)
#print(T)
triplets = extract_triplets(T)
print(triplets)
#local_poses = find_local_pose_for_triplets(triplets, features)
# Use the new function to propagate scale and pose
#scales, poses = propagate_scale_and_pose(local_poses, triplets)

# Print results
#print("Scales:", scales)
#print("Poses:", poses)
#for i,y in local_poses.items():
 ##  print(y)

#print(local_poses[""])
# Extract unique image names from the keys of the local_poses dictionary
#root_idx = 0  # Assuming the first image in triplets list is the root
#global_poses = {triplets[root_idx][0]: np.eye(4)}  # Initialize root pose as identity matrix
#visited = set()
#traverse_mst(triplets[root_idx][0], T, local_poses, global_poses, visited)

