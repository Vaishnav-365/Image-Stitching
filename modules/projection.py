import numpy as np
import cv2

def cylindrical_projection(img, f=700):
    """
    Warp image to cylindrical coordinates

    f = focal length (controls curvature)
    """

    h, w = img.shape[:2]
    cyl = np.zeros_like(img)

    cx = w // 2
    cy = h // 2

    for y in range(h):
        for x in range(w):
            theta = (x - cx) / f
            h_ = (y - cy) / f

            X = np.sin(theta)
            Y = h_
            Z = np.cos(theta)

            x_ = int(f * X / Z + cx)
            y_ = int(f * Y / Z + cy)

            if 0 <= x_ < w and 0 <= y_ < h:
                cyl[y, x] = img[y_, x_]

    return cyl