import cv2
import numpy as np

def extract_points_from_matches(kp1, kp2, matches):
    """
    Convert matched keypoints into coordinate arrays
    kp1, kp2: keypoints from SIFT
    matches: list of good matches
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    return src_pts, dst_pts


def compute_homography(src_pts, dst_pts):
    if len(src_pts) < 4 or len(dst_pts) < 4:
        raise ValueError("Need at least 4 points for homography")
    """
    Compute homography using RANSAC
    """
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask