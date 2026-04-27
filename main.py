from modules.feature import (
    load_image,
    resize_if_needed,
    show_image,
    save_image,
    draw_keypoints,
    get_features,
    validate_features,
    log_feature_info
)

from modules.matcher import match_features, draw_matches
from modules.homography import extract_points_from_matches, compute_homography
from modules.stitch import stitch_images
from modules.transform import chain_homographies, normalize_homography, compute_global_homographies

import cv2
import numpy as np


def main():
    img_paths = [
        "images/img1.png",
        "images/img2.png",
        "images/img3.png"
    ]

    images = [resize_if_needed(load_image(p)) for p in img_paths]

    for i, img in enumerate(images):
        print(f"Image {i} loaded successfully. Shape:", img.shape)

    # Feature extraction
    features = []

    for i, img in enumerate(images):
        kp, des, gray = get_features(img)

        if not validate_features(des, des):
            print(f"Feature validation failed for image {i}")
            return

        log_feature_info(f"Image {i}", kp, des)

        kp_img = draw_keypoints(img, kp)
        save_image(f"outputs/keypoints_{i}.jpg", kp_img)
        show_image(f"Keypoints Image {i}", kp_img)
        features.append((kp, des, gray))

    # =========================
    # PAIRWISE HOMOGRAPHY + MATCH VIS
    # =========================
    pairwise_H = {}

    for i in range(len(images) - 1):
        kp1, des1, _ = features[i]
        kp2, des2, _ = features[i + 1]

        matches = match_features(des1, des2)

        if len(matches) < 10:
            print(f"Not enough matches between image {i} and {i+1}")
            return
        
        #Draw matches
        match_img = draw_matches(images[i], kp1, images[i + 1], kp2, matches)
        save_image(f"outputs/matches_{i}_{i+1}.jpg", match_img)
        show_image(f"Feature Matches {i} → {i+1}", match_img)

        # Homography
        src_pts, dst_pts = extract_points_from_matches(kp1, kp2, matches)
        H, mask = compute_homography(src_pts, dst_pts)

        if H is None:
            print(f"Homography failed between image {i} and {i+1}")
            return
        
        # Inlier refinement
        inlier_src=src_pts[mask.ravel() == 1]
        inlier_dst=dst_pts[mask.ravel() == 1]

        H_refined, _ = compute_homography(inlier_src, inlier_dst)
        H = H_refined if H_refined is not None else H

        H = normalize_homography(H)

        pairwise_H[(i, i + 1)] = H

        print(f"H[{i} → {i+1}]:\n", H)
        print(f"Inliers: {int(mask.sum())}")
        print(f"Inlier Ratio: {mask.sum()/len(matches):.3f}")

    # =========================
    # GLOBAL HOMOGRAPHIES
    # =========================
    ref_index = 1  # middle image

    global_H = compute_global_homographies(pairwise_H, ref_index)

    print("\nGlobal Homographies (relative to Image 1):")

    for k, v in global_H.items():
        print(f"Image {k} → Ref:\n", v)

    # Validate homography using corner transform
    for i, img in enumerate(images):
        h, w = img.shape[:2]

        corners = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(corners, global_H[i])

        print(f"\nTransformed corners for Image {i}:\n", transformed)

    if np.any(np.isnan(transformed)):
        print("Invalid homography (NaN detected)")
        return
    
    print("\nAll transformations look valid")

    # =========================
    # CORRECT MULTI-IMAGE STITCH
    # =========================
    print("\nStarting corrected multi-image stitching...")

    # Step 1: Compute all transformed corners
    all_corners = []

    for i, img in enumerate(images):
        h, w = img.shape[:2]

        corners = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(corners, global_H[i])
        all_corners.append(transformed)

    # Step 2: Find global bounding box
    all_corners = np.concatenate(all_corners, axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

    print("Canvas bounds:", x_min, y_min, x_max, y_max)

    # Step 3: Compute translation to shift everything into view
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    # Step 4: Canvas size
    width = x_max - x_min
    height = y_max - y_min

    # Step 5: Create empty canvas
    panorama = np.zeros((height, width, 3), dtype=np.uint8)

    # Step 6: Warp all images into canvas
    for i, img in enumerate(images):
        H = global_H[i]

        H_translated = translation @ H

        warped = cv2.warpPerspective(img, H_translated, (width, height))

        # Simple overlay (Person D will improve this)
        mask = (warped > 0)

        panorama[mask] = warped[mask]

    # Save & show
    save_image("outputs/panorama.jpg", panorama)
    show_image("Final Panorama", panorama)

    print("Final Panorama Size:", panorama.shape)

if __name__ == "__main__":
    main()