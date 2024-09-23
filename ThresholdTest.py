import cv2
import os

def apply_threshold(image, threshold_value, max_value):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)
    return thresholded_image

def apply_color_threshold(image, threshold_value, max_value):
    b_channel, g_channel, r_channel = cv2.split(image)
    _, b_thresh = cv2.threshold(b_channel, threshold_value, max_value, cv2.THRESH_BINARY)
    _, g_thresh = cv2.threshold(g_channel, threshold_value, max_value, cv2.THRESH_BINARY)
    _, r_thresh = cv2.threshold(r_channel, threshold_value, max_value, cv2.THRESH_BINARY)
    color_thresholded_image = cv2.merge([b_thresh, g_thresh, r_thresh])
    return color_thresholded_image

def on_trackbar(val):
    threshold_value = cv2.getTrackbarPos("Threshold", "Thresholded Image")
    max_value = cv2.getTrackbarPos("Max Value", "Thresholded Image")
    # Update grayscale threshold image
    thresholded_image = apply_threshold(image, threshold_value, max_value)
    cv2.imshow("Thresholded Image", thresholded_image)
    # Update color threshold image
    color_thresholded_image = apply_color_threshold(image, threshold_value, max_value)
    cv2.imshow("Color Thresholded Image", color_thresholded_image)
    return thresholded_image, color_thresholded_image

if __name__ == "__main__":
    image_path = "/Users/lucaslinnemann/Desktop/LucasFiles/SemiCondFindImages/masked_image_on_second.jpg"
    save_dir = "/Users/lucaslinnemann/Desktop/LucasFiles/ThresholdPhotos"
    os.makedirs(save_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    cv2.namedWindow("Thresholded Image")
    cv2.namedWindow("Color Thresholded Image")
    cv2.createTrackbar("Threshold", "Thresholded Image", 0, 255, on_trackbar)
    cv2.createTrackbar("Max Value", "Thresholded Image", 255, 255, on_trackbar)
    thresholded_image, color_thresholded_image = on_trackbar(0)

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Save both images when 's' is pressed
            gray_save_path = os.path.join(save_dir, "thresholded_image.jpg")
            color_save_path = os.path.join(save_dir, "color_thresholded_image.jpg")

            cv2.imwrite(gray_save_path, thresholded_image)
            cv2.imwrite(color_save_path, color_thresholded_image)

            print(f"Images saved: {gray_save_path}, {color_save_path}")

        elif key == 27:
            break

        # Update images in the loop as trackbar values change
        thresholded_image, color_thresholded_image = on_trackbar(0)

    cv2.destroyAllWindows()
