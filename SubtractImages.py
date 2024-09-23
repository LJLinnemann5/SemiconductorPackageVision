import cv2
import numpy as np

def subtract_images(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    if image1 is None:
        raise ValueError(f"Image 1 not found: {image1_path}")
    if image2 is None:
        raise ValueError(f"Image 2 not found: {image2_path}")

    if image1.shape != image2.shape:
        raise ValueError("Images must be the same size for subtraction")

    b1, g1, r1 = cv2.split(image1)
    b2, g2, r2 = cv2.split(image2)

    b_result = cv2.absdiff(b1, b2)
    g_result = cv2.absdiff(g1, g2)
    r_result = cv2.absdiff(r1, r2)

    result = cv2.merge([b_result, g_result, r_result])

    return result

def save_image(image, save_path):
    cv2.imwrite(save_path, image)
    print(f"Image saved to: {save_path}")

def main():
    image2_path = "/Users/lucaslinnemann/Desktop/LucasFiles/Images/beltbackground_chip_lowlight.jpg"
    image1_path = "/Users/lucaslinnemann/Desktop/LucasFiles/Images/beltbackground_nochip_lowlight.jpg"
    result_image_path = "/Users/lucaslinnemann/Desktop/LucasFiles/SubImages/lowbeltsubtracted2_image.jpg"

    result_image = subtract_images(image1_path, image2_path)
    save_image(result_image, result_image_path)

if __name__ == "__main__":
    main()
