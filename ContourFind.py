import cv2
import numpy as np

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged = cv2.Canny(blurred_image, 50, 150)
    dilated = cv2.dilate(edged, None, iterations=1)

    return dilated

def detect_rectangles(image, min_width, min_height, max_width, max_height):
    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours to detect rectangles
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            # Get the bounding box of the rectangle
            x, y, w, h = cv2.boundingRect(approx)

            # Filter rectangles by size
            if min_width <= w <= max_width and min_height <= h <= max_height:
                # Draw the rectangle on the original image
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return original_image

if __name__ == "__main__":
    image_path = "/Users/lucaslinnemann/Desktop/LucasFiles/SubImages/subtracted_image.jpg"
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        raise ValueError(f"Image not found: {image_path}")
    preprocessed_image = preprocess_image(original_image)

    # Define the minimum and maximum width and height for rectangles
    min_width, min_height = 10, 10
    max_width, max_height = 2000, 2000
    result_image = detect_rectangles(preprocessed_image, min_width, min_height, max_width, max_height)
    cv2.imshow("Detected Rectangles", result_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
