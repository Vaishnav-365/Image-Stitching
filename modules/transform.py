import numpy as np

def chain_homographies(H1, H2):
    """
    Chain two homographies:
    H1: Image1 → Image2
    H2: Image2 → Image3

    Returns:
    H_combined: Image1 → Image3
    """
    return H2 @ H1

def get_identity_homography():
    return np.eye(3)

def normalize_homography(H):
    return H / H[2, 2]

def compute_global_homographies(pairwise_H, ref_index):
    """
    Convert pairwise homographies into global homographies
    relative to a reference image.

    Args:
        pairwise_H: dict of (i, j) → H (i → j)
        ref_index: index of reference image

    Returns:
        global_H: dict of i → H (i → ref)
    """

    global_H = {}

    # Reference image → identity
    global_H[ref_index] = np.eye(3)

    # Forward direction
    for i in range(ref_index - 1, -1, -1):
        H = pairwise_H[(i, i + 1)]
        global_H[i] = global_H[i + 1] @ H

    # Backward direction
    for i in range(ref_index + 1, len(pairwise_H) + 1):
        H = pairwise_H[(i - 1, i)]
        global_H[i] = np.linalg.inv(H) @ global_H[i - 1]
        global_H[i] = normalize_homography(global_H[i])

    return global_H