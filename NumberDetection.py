import cv2
import pytesseract

def detect_numbers_in_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

    # Perform text detection using Tesseract (restrict to numeric characters)
    custom_config = r'-c tessedit_char_whitelist=0123456789 --psm 6'
    text = pytesseract.image_to_string(threshold_image, config=custom_config)
    boxes = pytesseract.image_to_boxes(threshold_image, config=custom_config)
    h, w, _ = image.shape
    for box in boxes.splitlines():
        box = box.split(' ')
        x, y, w_box, h_box = int(box[1]), int(box[2]), int(box[3]), int(box[4])
        cv2.rectangle(image, (x, h - y), (w_box, h - h_box), (0, 255, 0), 2)
    return image, text

if __name__ == "__main__":
    image_path = "/Users/lucaslinnemann/Desktop/LucasFiles/ThresholdPhotos/thresholded_image.jpg"  # Update this path with your image path

    try:
        # Detect numbers and display the image with highlighted numbers
        highlighted_image, detected_text = detect_numbers_in_image(image_path)
        print("Detected Numbers:\n", detected_text)
        cv2.imshow("Highlighted Numbers", highlighted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ValueError as e:
        print(e)