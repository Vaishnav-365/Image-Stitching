import cv2
import numpy as np

from modules.feature import get_features, validate_features
from modules.matcher import match_features
from modules.homography import compute_homography, extract_points_from_matches
from modules.transform import compute_global_homographies, normalize_homography

def compute_pairwise_homographies(images):
    features = []
    pairwise_H = {}
    matches_dict = {}

    print("\n========== FEATURE EXTRACTION ==========")

    # =========================
    # FEATURE EXTRACTION
    # =========================
    for i, img in enumerate(images):
        kp, des, _ = get_features(img)

        if des is None or len(des) < 10:
            raise ValueError(f"Not enough features detected in image {i}")

        print(f"[Image {i}] Keypoints: {len(kp)}")

        features.append((kp, des))

    print("\n========== MATCHING + HOMOGRAPHY ==========")

    # =========================
    # MATCHING + HOMOGRAPHY
    # =========================
    for i in range(len(images) - 1):
        kp1, des1 = features[i]
        kp2, des2 = features[i + 1]

        matches = match_features(des1, des2)
        matches_dict[(i, i + 1)] = matches

        print(f"\n[Pair {i} → {i+1}] Total Matches: {len(matches)}")

        if len(matches) < 10:
            raise ValueError(f"Not enough matches between image {i} and {i+1}")

        src_pts, dst_pts = extract_points_from_matches(kp1, kp2, matches)

        H, mask = compute_homography(src_pts, dst_pts)

        if H is None:
            raise ValueError(f"Homography computation failed between {i} and {i+1}")

        # =========================
        # INLIER REFINEMENT
        # =========================
        inlier_src = src_pts[mask.ravel() == 1]
        inlier_dst = dst_pts[mask.ravel() == 1]

        inliers = int(mask.sum())
        inlier_ratio = inliers / len(matches)

        print(f"[Pair {i} → {i+1}] Inliers: {inliers}")
        print(f"[Pair {i} → {i+1}] Inlier Ratio: {inlier_ratio:.3f}")

        if inlier_ratio < 0.5:
            print(f"[WARNING] Weak homography for pair {i}-{i+1}")

        # refine homography
        H_refined, _ = compute_homography(inlier_src, inlier_dst)
        H = H_refined if H_refined is not None else H

        H = normalize_homography(H)

        print(f"[Pair {i} → {i+1}] Homography:\n{H}")

        # =========================
        # VALIDATION (corners)
        # =========================
        h, w = images[i].shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        transformed = cv2.perspectiveTransform(corners, H)

        print(f"[Pair {i} → {i+1}] Transformed corners:\n{transformed}")

        if np.any(np.isnan(transformed)):
            raise ValueError("Invalid homography (NaN detected)")

        if abs(H[2][0]) > 0.01 or abs(H[2][1]) > 0.01:
            print(f"[WARNING] Strong perspective distortion in pair {i}-{i+1}")

        pairwise_H[(i, i + 1)] = H

    return features, matches_dict, pairwise_H

def warp_all_images(images, global_H):
    #Compute bounds
    all_corners = []
    for i,img in enumerate(images):
        h,w = img.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        transformed = cv2.perspectiveTransform(corners,global_H[i])
        all_corners.append(transformed)

    all_corners = np.concatenate(all_corners, axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    translation = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])

    width = xmax - xmin
    height = ymax - ymin

    panorama = np.zeros((height, width, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        H = translation @ global_H[i]
        warped = cv2.warpPerspective(img, H, (width, height))
        mask = np.any(warped != 0, axis=2)
        panorama[mask] = warped[mask]
    
    return panorama

def build_panorama(images, ref_index=1):
    print("\n========== BUILDING PANORAMA ==========")

    # Get everything
    features, matches_dict, pairwise_H = compute_pairwise_homographies(images)

    print("\n========== GLOBAL HOMOGRAPHIES ==========")

    global_H = compute_global_homographies(pairwise_H, ref_index)

    for i, H in global_H.items():
        print(f"[Image {i} → Ref]:\n{H}")

    print("\n========== WARPING IMAGES ==========")

    panorama = warp_all_images(images, global_H)

    print(f"\nFinal Panorama Size: {panorama.shape}")

    return panorama, features, matches_dict, pairwise_H, global_H
