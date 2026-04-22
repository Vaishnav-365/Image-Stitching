from modules.feature import (
    load_image,
    resize_if_needed,
    convert_to_gray,
    show_image,
    detect_sift_features,
    draw_keypoints
)
from modules.matcher import match_features
from modules.homography import extract_points_from_matches, compute_homography
import cv2

def main():
    img1_path = "images/left.jpg"
    img2_path = "images/right.jpg"

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    img1 = resize_if_needed(img1)
    img2 = resize_if_needed(img2)

    gray1 = convert_to_gray(img1)
    gray2 = convert_to_gray(img2)

    print("Image 1 loaded successfully. Shape:", img1.shape)
    print("Image 2 loaded successfully. Shape:", img2.shape)

    kp1, des1 = detect_sift_features(gray1)
    kp2, des2 = detect_sift_features(gray2)

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

    show_image("Left Image", img1)
    show_image("Right Image", img2)
    show_image("Gray Left Image", gray1)
    show_image("Gray Right Image", gray2)
    show_image("Keypoints Image 1", img1_kp)
    show_image("Keypoints Image 2", img2_kp)

    H, mask = compute_homography(src_pts, dst_pts)

    np.savetxt("H.txt", H)

    print("Homography Matrix:\n", H)
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
