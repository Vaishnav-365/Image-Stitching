from modules.homography import compute_homography
import numpy as np

src = np.float32([[0,0],[1,0],[1,1],[0,1]])
dst = np.float32([[1,1],[2,1],[2,2],[1,2]])

H, mask = compute_homography(src, dst)

print("Homography Matrix:\n", H)
print("Inliers:\n", mask)