from modules.feature import (
    load_image,
    resize_if_needed,
    show_image,
    save_image,
    draw_keypoints,
    get_features,
    validate_features,
    extract_matched_points,
    log_feature_info
)

from modules.matcher import match_features, draw_matches
from modules.homography import extract_points_from_matches, compute_homography
from modules.stitch import stitch_images

import cv2
import numpy as np


def main():
    img1_path = "images/left.jpg"
    img2_path = "images/right.jpg"

    # Load images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    img1 = resize_if_needed(img1)
    img2 = resize_if_needed(img2)

    print("Image 1 loaded successfully. Shape:", img1.shape)
    print("Image 2 loaded successfully. Shape:", img2.shape)

    # Feature extraction
    kp1, des1, gray1 = get_features(img1)
    kp2, des2, gray2 = get_features(img2)

    if not validate_features(des1, des2):
        print("Feature validation failed. Exiting.")
        return

    log_feature_info("Image 1", kp1, des1)
    log_feature_info("Image 2", kp2, des2)

    print("Keypoints in Image 1:", len(kp1))
    print("Keypoints in Image 2:", len(kp2))

    # Draw keypoints
    img1_kp = draw_keypoints(img1, kp1)
    img2_kp = draw_keypoints(img2, kp2)

    save_image("outputs/final_keypoints1.jpg", img1_kp)
    save_image("outputs/final_keypoints2.jpg", img2_kp)

    # Feature matching
    good_matches = match_features(des1, des2)

    print("Good matches after ratio test:", len(good_matches))
    print("Total matches:", len(good_matches))

    if len(good_matches) < 10:
        print("Not enough matches!")
        return

    # Save/display match visualization
    match_img = draw_matches(img1, kp1, img2, kp2, good_matches)
    save_image("outputs/final_matches.jpg", match_img)

    # Homography
    src_pts, dst_pts = extract_points_from_matches(kp1, kp2, good_matches)
    H, mask = compute_homography(src_pts, dst_pts)
    if H is None:
        print("Homography failed! Cannot stitch.")
        return

    print("Homography Matrix:\n", H)
    print("Mapping: Image1 → Image2")
    print("Inliers:", int(mask.sum()))
    print("Inlier Ratio:", mask.sum() / len(good_matches))

    # RANSAC inlier matches
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
    print("Good matches after RANSAC:", len(inlier_matches))
    
    print("Starting stitching...")

    stitched = stitch_images(img1, img2, H)

    # Save result
    save_image("outputs/panorama.jpg", stitched)

    

    inlier_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    save_image("outputs/final_inlier_matches.jpg", inlier_img)

    # Display results
    show_image("Left Image", img1)
    show_image("Right Image", img2)
    show_image("Gray Left Image", gray1)
    show_image("Gray Right Image", gray2)
    show_image("Keypoints Image 1", img1_kp)
    show_image("Keypoints Image 2", img2_kp)
    show_image("Feature Matches", match_img)
    show_image("RANSAC Inlier Matches", inlier_img)
    # Show result
    show_image("Stitched Panorama", stitched)


if __name__ == "__main__":
    main()