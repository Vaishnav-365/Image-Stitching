import cv2

def match_features(desc1, desc2, ratio_thresh=0.75):

    if desc1 is None or desc2 is None:
        return []

    if len(desc1) == 0 or len(desc2) == 0:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches
def draw_matches(img1, kp1, img2, kp2, matches, max_matches=50):
    selected_matches = matches[:max_matches]

    match_image = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        selected_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return match_image