import cv2
import numpy as np
import os
from scipy.optimize import least_squares


def match_features(descriptors1, descriptors2):
    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def estimate_homography(matches, keypoints1, keypoints2):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def calibrate_camera(homographies):
    # Assuming all homographies are from the same camera and the camera is only rotating (no translation)

    # Initialize lists to hold values
    h1_h2_cross_products = []
    v_12 = []
    v_11_minus_v_22 = []

    for H in homographies:
        # Ensure H is normalized
        H /= H[2, 2]

        # Decompose homography matrix
        h1 = H[:, 0]
        h2 = H[:, 1]

        # Cross product of h1 and h2
        h1_h2_cross_products.append(np.cross(h1, h2))

        # Compute elements of V (for the intrinsic matrix)
        v_12.append([h1[0]*h2[0], h1[0]*h2[1] + h1[1]*h2[0], h1[1]*h2[1], h1[2]*h2[0] + h1[0]*h2[2], h1[2]*h2[1] + h1[1]*h2[2], h1[2]*h2[2]])
        v_11_minus_v_22.append([h1[0]*h1[0] - h2[0]*h2[0], 2*(h1[0]*h1[1] - h2[0]*h2[1]), h1[1]*h1[1] - h2[1]*h2[1], 2*(h1[2]*h1[0] - h2[2]*h2[0]), 2*(h1[2]*h1[1] - h2[2]*h2[1]), h1[2]*h1[2] - h2[2]*h2[2]])

    # Stack V
    V = np.vstack(v_12 + v_11_minus_v_22)

    # Solve Vb = 0 by SVD
    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1] # Solution is the last row of V^T

    # Extract intrinsic parameters
    w = b[0]*b[2]*b[5] - b[1]**2*b[5] - b[0]*b[4]**2 + 2*b[1]*b[3]*b[4] - b[2]*b[3]**2
    d = b[0]*b[2] - b[1]**2

    alpha = np.sqrt(w / d)
    beta = np.sqrt(w / (d * b[0]))
    gamma = -b[1] * alpha**2 * beta / w
    uc = (gamma * b[3] / beta) - (b[0] * b[4] / w)
    vc = (b[0] * b[3] / w) - (alpha**2 * b[3] / gamma)

    # Construct intrinsic matrix
    A = np.array([[alpha, gamma, uc],
                  [0, beta, vc],
                  [0, 0, 1]])

    return A