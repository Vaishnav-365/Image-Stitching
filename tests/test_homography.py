from modules.homography import compute_homography
import numpy as np
import cv2

src = np.float32([[0,0],[1,0],[1,1],[0,1]])
dst = np.float32([[1,1],[2,1],[2,2],[1,2]])

H, mask = compute_homography(src, dst)

print("Homography Matrix:\n", H)
print("Inliers:\n", mask)

pts = np.float32([[0,0],[1,0],[1,1],[0,1]]).reshape(-1,1,2)
transformed = cv2.perspectiveTransform(pts, H)

print("Transformed points:\n", transformed)