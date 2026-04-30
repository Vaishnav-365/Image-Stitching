import cv2


def compute_match_stats(matches):
    if not matches:
        return None

    distances = [m.distance for m in matches]

    return {
        "avg": sum(distances) / len(distances),
        "min": min(distances),
        "max": max(distances)
    }


def geometric_filter(matches, kp1, kp2, max_vertical_diff=80):
    """
    Relaxed geometric filter for panorama scenes
    """
    if kp1 is None or kp2 is None:
        return matches

    filtered = []

    for m in matches:
        y1 = kp1[m.queryIdx].pt[1]
        y2 = kp2[m.trainIdx].pt[1]

        if abs(y1 - y2) < max_vertical_diff:
            filtered.append(m)

    return filtered


def match_features(
    desc1,
    desc2,
    ratio_thresh=0.72,        # FIXED (was too strict)
    max_matches=150,
    distance_thresh=250,
    cross_check=False,
    kp1=None,
    kp2=None,
    debug=True
):
    """
    Stable feature matcher for panorama stitching
    """

    # ---- Safety checks ----
    if desc1 is None or desc2 is None:
        return []

    if len(desc1) < 2 or len(desc2) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)

    # ---- STRICT MODE ----
    if cross_check:
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        if debug:
            print(f"[Matcher] Cross-check matches: {len(matches)}")

    # ---- FAST MODE ----
    else:
        knn_matches = bf.knnMatch(desc1, desc2, k=2)

        good_matches = []

        for pair in knn_matches:
            if len(pair) < 2:
                continue

            m, n = pair

            # FIXED: cleaner filtering
            if m.distance < ratio_thresh * n.distance and m.distance < distance_thresh:
                good_matches.append(m)

        if debug:
            print(f"[Matcher] Raw KNN matches: {len(knn_matches)}")
            print(f"[Matcher] After ratio filter: {len(good_matches)}")

        matches = sorted(good_matches, key=lambda x: x.distance)

    # ---- Geometric filtering ----
    #matches = geometric_filter(matches, kp1, kp2)

    if debug:
        print(f"[Matcher] After geometric filter: {len(matches)}")

    # ---- Limit matches ----
    matches = matches[:min(max_matches, len(matches))]

    # ---- Stats ----
    stats = compute_match_stats(matches)

    if debug:
        print(f"[Matcher] Final matches used: {len(matches)}")

        if stats:
            print(
                f"[Matcher] Distance stats → "
                f"avg: {stats['avg']:.2f}, "
                f"min: {stats['min']:.2f}, "
                f"max: {stats['max']:.2f}"
            )

    if len(matches) < 10:
        print("[Matcher WARNING] Too few matches — homography may fail")

    return matches


def draw_matches(img1, kp1, img2, kp2, matches, max_matches=50):
    selected_matches = matches[:max_matches]

    return cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        selected_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )