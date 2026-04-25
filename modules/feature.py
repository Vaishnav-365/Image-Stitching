import cv2
import os


def load_image(image_path):
    """
    Load an image from disk.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def resize_if_needed(image, max_width=1000):
    """
    Resize image if it is too large, while keeping aspect ratio.
    """
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(image, (new_width, new_height))
    return resized


def convert_to_gray(image):
    """
    Convert BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_image(window_name, image):
    """
    Display an image.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(output_path, image):
    """
    Save image to disk.
    """
    folder = os.path.dirname(output_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    cv2.imwrite(output_path, image)


def detect_sift_features(image, nfeatures=2000):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def draw_keypoints(image, keypoints):
    output_image = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return output_image
def preprocess_for_sift(image):
    gray = convert_to_gray(image)

    # improves contrast slightly
    gray = cv2.equalizeHist(gray)

    return gray
def extract_matched_points(kp1, kp2, matches):
    src_pts = []
    dst_pts = []

    for match in matches:
        src_pts.append(kp1[match.queryIdx].pt)
        dst_pts.append(kp2[match.trainIdx].pt)

    return src_pts, dst_pts