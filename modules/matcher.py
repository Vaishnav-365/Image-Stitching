import cv2

def match_features(
    desc1,
    desc2,
    ratio_thresh=0.75,
    max_matches=200,
    distance_thresh=300,
    debug=True
):
    """
    Args:
        desc1, desc2 : SIFT descriptors
        ratio_thresh : Lowe's ratio test threshold
        max_matches  : cap on number of matches
        distance_thresh : absolute distance cutoff
        debug        : print debug info

    Returns:
        List of cv2.DMatch objects
    """

    if desc1 is None or desc2 is None:
        return []

    if len(desc1) == 0 or len(desc2) == 0:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_thresh * n.distance and m.distance < distance_thresh:
            good_matches.append(m)

    if debug:
        print(f"[Matcher] Raw KNN matches: {len(knn_matches)}")
        print(f"[Matcher] After ratio + distance filter: {len(good_matches)}")

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    good_matches = good_matches[:min(max_matches, len(good_matches))]

    if debug:
        print(f"[Matcher] Final matches used: {len(good_matches)}")

    if len(good_matches) < 10:
        print("[Matcher WARNING] Too few matches — homography may fail")

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
