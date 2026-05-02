from modules.feature import (
    load_image,
    resize_if_needed,
    show_image,
    save_image,
    draw_keypoints,
)

from modules.matcher import draw_matches
from modules.pipeline import build_panorama
from modules.projection import cylindrical_projection

def main():
    img_paths = [
        "images/img1.png",
        "images/img2.png",
        "images/img3.png"
    ]

    images = [resize_if_needed(load_image(p)) for p in img_paths]

    for i, img in enumerate(images):
        print(f"Image {i} loaded successfully. Shape:", img.shape)

    for i, img in enumerate(images):
        cyl = cylindrical_projection(img)
        save_image(f"outputs/cylindrical_{i}.jpg", cyl)
        show_image(f"Cylindrical Image {i}", cyl)

    print("\nBuilding panorama using pipeline...")
    
    panorama, features, matches_dict, pairwise_H, global_H = build_panorama(images)

    for i, (kp, _) in enumerate(features):
        kp_img = draw_keypoints(images[i], kp)

        save_image(f"outputs/keypoints_{i}.jpg", kp_img)
        show_image(f"Keypoints Image {i}", kp_img)

    for (i, j), matches in matches_dict.items():
        kp1, _ = features[i]
        kp2, _ = features[j]

        match_img = draw_matches(images[i], kp1, images[j], kp2, matches)

        save_image(f"outputs/matches_{i}_{j}.jpg", match_img)
        show_image(f"Matches {i} → {j}", match_img)

    panorama, features, matches_dict, pairwise_H, global_H = build_panorama(images)

    if panorama is None:
        print("Panorama generation failed. Check logs.")
        return
    save_image("outputs/panorama.jpg", panorama)
    show_image("Final Panorama", panorama)

    print("Final Panorama Size:", panorama.shape)

if __name__ == "__main__":
    main()