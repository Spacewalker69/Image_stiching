import cv2
import numpy as np
from scipy.spatial import KDTree
import networkx as nx
import matplotlib.pyplot as plt
import os


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
    b = np.arctan2(t[1], t[0])
    a = np.arctan2(R[1, 0], R[0, 0])

    return b, a , R , t
def triangulate_point(P1, P2, pts1, pts2):
    # Check if points are valid
    if pts1.size == 0 or pts2.size == 0:
        return None
    pts1 = pts1.reshape(2, -1)
    pts2 = pts2.reshape(2, -1)
    point_4d_hom = cv2.triangulatePoints(P1, P2, pts1, pts2)
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    return point_4d[:3, :].T

def find_local_pose_for_triplets(triplets, features):
    # For each triplet, estimate the local pose
    local_poses = {}
    for triplet in triplets:
        img1, img2, img3 = triplet
        keypoints1, descriptors1 = features[img1]
        keypoints2, descriptors2 = features[img2]
        keypoints3, descriptors3 = features[img3]
        K = np.eye(3)
        # Get the dimensions for each image
        W1, H1 = cv2.imread(img1).shape[1], cv2.imread(img1).shape[0]
        W2, H2 = cv2.imread(img2).shape[1], cv2.imread(img2).shape[0]
        W3, H3 = cv2.imread(img3).shape[1], cv2.imread(img3).shape[0]
        
        # Match features between img1 and img2, and between img2 and img3
        matches12 = match_features(descriptors1, descriptors2)
        matches23 = match_features(descriptors1, descriptors3)
        
        # Estimate motion parameters for each pair within the triplet
        b12, a12, R12, t12 = estimate_motion_parameters(matches12, keypoints1, keypoints2,W1,H1)
        b13, a13,R13,t13 = estimate_motion_parameters(matches23, keypoints2, keypoints3,W2,H2)
        
        pts1, pts2, pts3 = get_common_point(keypoints1, keypoints2, keypoints3, matches12, matches23)
        
        # Create projection matrices for each camera
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R12, t12))  # R12 and t12 from decomposing E for img1 and img2
        P3 = K @ np.hstack((R13, t13))  # R13 and t13 from decomposing E for img1 and img3

        # Triangulate the same point from two pairs of images
        point_12 = triangulate_point(P1, P2, pts1, pts2)
        point_13 = triangulate_point(P1, P3, pts1, pts3)

        # Compute relative scale
        print(point_12)
        print(point_13)
        scale_12 = np.linalg.norm(point_12[0][1] - point_12[0][0])
        scale_13 = np.linalg.norm(point_13[0][1] - point_13[0][0])
        relative_scale = scale_12 / scale_13 if scale_13 != 0 else 0

        # Store the estimated poses and scale for the triplet
        local_poses[triplet] = {'b12': b12, 'a12': a12, 'b13': b13, 'a13': a13, 'scale': relative_scale}

    return local_poses

def get_common_point(kp1, kp2, kp3, matches12, matches23):
    # Mapping keypoints from image 1 to image 3 via image 2
    map_12 = {m.queryIdx: m.trainIdx for m in matches12}  # Map from img1 to img2 keypoints
    map_23 = {m.queryIdx: m.trainIdx for m in matches23}  # Map from img2 to img3 keypoints

    for m12 in matches12:
        if m12.trainIdx in map_23:
            # Found a keypoint in img1 that maps to img2 and img3
            idx1 = m12.queryIdx  # Index in img1
            idx2 = m12.trainIdx  # Index in img2
            idx3 = map_23[idx2]  # Index in img3

            # Get the matching points
            pt1 = kp1[idx1].pt
            pt2 = kp2[idx2].pt
            pt3 = kp3[idx3].pt
            pts1 = np.float32([pt1]).reshape(-1, 1, 2)
            pts2 = np.float32([pt2]).reshape(-1, 1, 2)
            pts3 = np.float32([pt3]).reshape(-1, 1, 2)

            return np.array([pts1]), np.array([pts2]), np.array([pts3])
        

    return np.empty((0, 1, 2)), np.empty((0, 1, 2)), np.empty((0, 1, 2))  # Return None if no common point is found

# The match_features function would perform feature matching between two sets of descriptors
def match_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Estimate Essential Matrix for each pair
def estimate_essential_matrix(kp1, kp2, matches, camera_matrix):
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(points1, points2, camera_matrix)
    return E
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