from modules.feature import load_image, resize_if_needed, convert_to_gray, show_image


def main():
    img1_path = "images/left.jpg"
    img2_path = "images/right.jpg"

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    img1 = resize_if_needed(img1)
    img2 = resize_if_needed(img2)

    gray1 = convert_to_gray(img1)
    gray2 = convert_to_gray(img2)

    print("Image 1 loaded successfully. Shape:", img1.shape)
    print("Image 2 loaded successfully. Shape:", img2.shape)

    show_image("Left Image", img1)
    show_image("Right Image", img2)
    show_image("Gray Left Image", gray1)
    show_image("Gray Right Image", gray2)


if __name__ == "__main__":
    main()