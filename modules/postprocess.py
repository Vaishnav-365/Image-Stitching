import cv2
import numpy as np

def crop_black_borders(image):
    """
    Automatically crop black borders from stitched panorama.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to separate black vs non-black
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Largest contour = actual panorama
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    cropped = image[y:y+h, x:x+w]

    return cropped

def sharpen_image(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5,-1],
        [0, -1, 0]
    ])
    return cv2.filter2D(image, -1, kernel)