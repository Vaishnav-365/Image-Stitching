import cv2
from matcher import match_features

# load images
img1 = cv2.imread("../images/left.jpg")
img2 = cv2.imread("../images/right.jpg")

# SIFT
sift = cv2.SIFT_create()
kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2, None)

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
