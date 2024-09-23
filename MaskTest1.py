import cv2
import numpy as np
import os

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    
    return edges

def detect_and_draw_lines(image, edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=100)
   
    # If lines are found, draw them
    if lines is not None:
        line1 = lines[0][0]
        line2 = lines[1][0]
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        # Draw the lines on the image
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(image, (x3, y3), (x4, y4), (0, 255, 0), 2)
        p1 = (x1, y1)
        p2 = (x2, y2)
        p3 = (x4, y4)
        p4 = (x3, y3)
        # Draw the vertical connecting lines to form a rectangle
        cv2.line(image, p1, p4, (0, 255, 0), 2)
        cv2.line(image, p2, p3, (0, 255, 0), 2)
        return np.array([p1, p2, p3, p4])
    return None

def mask_image(image, polygon):
    # Create a black mask with the same dimensions as the image
    mask = np.zeros_like(image)
    # Fill the polygon with white (the area to keep)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

if __name__ == "__main__":
    image_path = "/Users/lucaslinnemann/Desktop/LucasFiles/SubImages/subtracted_image.jpg"
    save_directory = "/Users/lucaslinnemann/Desktop/LucasFiles/Images/SemiCondFindImages/"
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        raise ValueError(f"Image not found: {image_path}")
    edges = preprocess_image(original_image)
    polygon = detect_and_draw_lines(original_image, edges)

    # If a valid area is detected, mask the image
    if polygon is not None:

        masked_image = mask_image(original_image, polygon)
        # Show the original image with detected lines
        cv2.imshow("Detected Lines", original_image)
        # Show the masked image
        cv2.imshow("Masked Image", masked_image)
        save_path = os.path.join(save_directory, "masked_image.jpg")
        cv2.imwrite(save_path, masked_image)
        print(f"Masked image saved at: {save_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No valid area detected.")