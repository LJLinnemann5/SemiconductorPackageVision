import cv2
import numpy as np
import os

def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    
    return edges

def detect_and_draw_lines(image, edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=100)
   
    if lines is not None:
        
        line1 = lines[0][0]
        line2 = lines[1][0]
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(image, (x3, y3), (x4, y4), (0, 255, 0), 2)
        # Define the polygon points from detected lines
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
    # Create a black mask with the same dimensions image
    mask = np.zeros_like(image)
    # Fill the polygon with white (the area to keep)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def apply_mask_to_another_image(image, mask):
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

if __name__ == "__main__":

    image1_path = "/Users/lucaslinnemann/Desktop/LucasFiles/SubImages/subtracted_image.jpg"
    image2_path = "/Users/lucaslinnemann/Desktop/LucasFiles/Images/blackbackground_chip_midlight.jpg" 
    save_directory = "/Users/lucaslinnemann/Desktop/LucasFiles/SemiCondFindImages/"
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    image1 = cv2.imread(image1_path)
    if image1 is None:
        raise ValueError(f"Image not found: {image1_path}")
    
    edges = preprocess_image(image1)

    # Detect lines and highlight them in the first image
    polygon = detect_and_draw_lines(image1, edges)

    # If a valid area is detected, mask the first image
    if polygon is not None:
        mask = np.zeros_like(image1)
        cv2.fillPoly(mask, [polygon], (255, 255, 255))  

        image2 = cv2.imread(image2_path)
        if image2 is None:
            raise ValueError(f"Second image not found: {image2_path}")
        
        if image1.shape != image2.shape:
            raise ValueError("The second image must be the same size as the first image.")
        
        masked_image_on_second = apply_mask_to_another_image(image2, mask)
        cv2.imshow("Masked Image on Second Image", masked_image_on_second)

        key = cv2.waitKey(0)
        if key == ord('s'):

            save_path = os.path.join(save_directory, "masked_image_on_second.jpg")
            cv2.imwrite(save_path, masked_image_on_second)
            print(f"Masked image saved at: {save_path}")

        cv2.destroyAllWindows()
    else:
        print("No valid area detected.")