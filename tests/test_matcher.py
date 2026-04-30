import os
import cv2
from modules.feature import load_image, get_features
from modules.matcher import match_features, draw_matches


BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def load(img_name):
    return load_image(os.path.join(BASE_DIR, "images", img_name))


def match_pair(name, imgA, imgB):
    print(f"\n===== {name} =====")

    kpA, descA, _ = get_features(imgA)
    kpB, descB, _ = get_features(imgB)

    matches = match_features(
        descA,
        descB,
        kp1=kpA,
        kp2=kpB,
        debug=True
    )

    print(f"[Result] Matches: {len(matches)}")

    vis = draw_matches(imgA, kpA, imgB, kpB, matches, max_matches=50)

    return vis


def main():
    img1 = load("img1.png")
    img2 = load("img2.png")
    img3 = load("img3.png")

    vis1 = match_pair("LEFT ↔ MIDDLE", img1, img2)
    vis2 = match_pair("MIDDLE ↔ RIGHT", img2, img3)

    cv2.imshow("Left-Middle Matches", vis1)
    cv2.imshow("Middle-Right Matches", vis2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()