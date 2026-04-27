import cv2
import numpy as np


def warp_images(img1, img2, H):
    """
    Warp img1 to img2 using homography H and create a panorama canvas.
    """

    # Get image shapes
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get corners of img1
    corners_img1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)

    # Warp corners
    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)

    # Corners of img2
    corners_img2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(-1, 1, 2)

    # Combine all corners
    all_corners = np.vstack((warped_corners_img1, corners_img2))

    # Find bounding box
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    # Translation matrix (to avoid negative coords)
    translation = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ])

    # Warp img1
    result = cv2.warpPerspective(
        img1,
        translation @ H,
        (xmax - xmin, ymax - ymin)
    )

    # Place img2 into result
    result[-ymin:h2 - ymin, -xmin:w2 - xmin] = img2

    return result


def feather_blend(base, overlay):
    """
    Simple feather blending (average where both images exist)
    """

    blended = base.copy()

    for y in range(base.shape[0]):
        for x in range(base.shape[1]):

            pix1 = base[y, x]
            pix2 = overlay[y, x]

            if not np.array_equal(pix2, [0, 0, 0]):
                if np.array_equal(pix1, [0, 0, 0]):
                    blended[y, x] = pix2
                else:
                    blended[y, x] = (pix1 // 2 + pix2 // 2)

    return blended


def stitch_images(img1, img2, H):
    """
    Complete stitching pipeline:
    Warp + blend
    """

    # Warp first image
    warped = warp_images(img1, img2, H)

    # Create overlay image (same size)
    overlay = warped.copy()

    # Blend (simple)
    result = feather_blend(warped, overlay)

    return result