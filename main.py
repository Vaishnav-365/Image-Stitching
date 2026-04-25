from modules.feature import (
    load_image,
    resize_if_needed,
    convert_to_gray,
    show_image,
    detect_sift_features,
    save_image,
    extract_matched_points,
    draw_keypoints
)
from modules.matcher import match_features, draw_matches
from modules.matcher import match_features
from modules.feature import get_features, validate_features
from modules.feature import preprocess_for_sift
from modules.homography import extract_points_from_matches, compute_homography
import cv2
import numpy as np

def main():
    img1_path = "images/left.jpg"
    img2_path = "images/right.jpg"

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    img1 = resize_if_needed(img1)
    img2 = resize_if_needed(img2)

    kp1, des1, gray1 = get_features(img1)
    kp2, des2, gray2 = get_features(img2)

    if not validate_features(des1, des2):
        print("Feature validation failed. Exiting.")
        return

    print("Image 1 loaded successfully. Shape:", img1.shape)
    print("Image 2 loaded successfully. Shape:", img2.shape)



    good_matches = match_features(des1, des2)

    src_pts, dst_pts = extract_matched_points(kp1, kp2, good_matches)

    print("Matched source points:", len(src_pts))
    print("Matched destination points:", len(dst_pts))


    print("Good matches after ratio test:", len(good_matches))

    match_img = draw_matches(img1, kp1, img2, kp2, good_matches)

    show_image("Feature Matches", match_img)

    matches = match_features(des1, des2)

    print("Total matches:", len(matches))

    if len(matches) < 10:
        print("Not enough matches!")
        return

    src_pts, dst_pts = extract_points_from_matches(kp1, kp2, matches)

    print("Keypoints in Image 1:", len(kp1))
    print("Keypoints in Image 2:", len(kp2))

    img1_kp = draw_keypoints(img1, kp1)
    img2_kp = draw_keypoints(img2, kp2)

    save_image("outputs/final_keypoints1.jpg", draw_keypoints(img1, kp1))
    save_image("outputs/final_keypoints2.jpg", draw_keypoints(img2, kp2))
    save_image("outputs/final_matches.jpg", match_img)
    
    show_image("Left Image", img1)
    show_image("Right Image", img2)
    show_image("Gray Left Image", gray1)
    show_image("Gray Right Image", gray2)
    show_image("Keypoints Image 1", img1_kp)
    show_image("Keypoints Image 2", img2_kp)

    H, mask = compute_homography(src_pts, dst_pts)

    np.savetxt("H.txt", H)

    print("Homography Matrix:\n", H)
    print("Mapping: Image1 → Image2")
    print("Inliers:", mask.sum())

    matched_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    show_image("Matches", matched_img)

    inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
    print("Good matches after RANSAC:", len(inlier_matches))

if __name__ == "__main__":
    main()
