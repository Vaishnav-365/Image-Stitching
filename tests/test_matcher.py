import cv2
from modules.matcher import match_features
from modules.feature import load_image, convert_to_gray, detect_sift_features

def main():
    img1 = load_image("images/left.jpg")
    img2 = load_image("images/right.jpg")

    gray1 = convert_to_gray(img1)
    gray2 = convert_to_gray(img2)

    kp1, desc1 = detect_sift_features(gray1)
    kp2, desc2 = detect_sift_features(gray2)

    matches = match_features(desc1, desc2)

    print("Total good matches:", len(matches))

    img_matches = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()