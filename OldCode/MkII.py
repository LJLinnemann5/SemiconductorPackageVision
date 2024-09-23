import cv2
from picamera2 import Picamera2
import pytesseract
import numpy as np
import time
import threading
import os
from libcamera import controls

cv2.startWindowThread()
picam2 = Picamera2()
picam2.set_controls({"AeEnable":False,"AwbEnable":False,"ExposureTime": 4000,"AnalogueGain":10.0,"FrameRate": 1})
picam2.exposure_mode = 'on'
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (1000, 480)}))
picam2.start()
zoom_factor = 3


picam2.framerate = 1/8

picam2.exposure_mode = 'off'
#picam2.camera_configuration()['sensor']
# autofocus ??
#picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

config = ('-l eng --oem 1 --psm 3')

lower_bound = 100
upper_bound = 250

def threshold(image):
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 17, 2)
    return thresh

def draw_histogram(image, mask=None):
    # Calculate histogram
    hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
    
    # Normalize the histogram
    hist = cv2.normalize(hist, hist, alpha=0, beta=300, norm_type=cv2.NORM_MINMAX)
    hist = hist.flatten()
    
    # Create an image to draw the histogram (with axes)
    hist_img = np.zeros((320, 400, 3), dtype=np.uint8)
    
    # Draw X and Y axes
    cv2.line(hist_img, (50, 300), (350, 300), (255, 255, 255), 2)  # X-axis
    cv2.line(hist_img, (50, 300), (50, 20), (255, 255, 255), 2)    # Y-axis

    # Draw the histogram
    for x in range(256):
        cv2.line(hist_img, (x + 50, 300), (x + 50, 300 - int(hist[x])), (255, 255, 255), 1)
    
    # Draw X-axis labels (0, 128, 255)
    cv2.putText(hist_img, '0', (40, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(hist_img, '128', (170, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(hist_img, '255', (340, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw Y-axis labels (Frequency example labels)
    cv2.putText(hist_img, '0', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(hist_img, '150', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(hist_img, '300', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return hist_img

def procimg(im):
    result = cv2.medianBlur(im, 3)
    result = cv2.equalizeHist(result)
    result = threshold(result)
    
    return result
while True:
    im = picam2.capture_array()
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    
    '''
    # Create a mask
    mask = cv2.inRange(gray, lower_bound, upper_bound)
    mask = ~mask
    
    #create subtraction mask
    submask = cv2.inRange(gray,0,upper_bound)
    
    
    
    
    result = cv2.bitwise_and(gray, gray, mask=mask)
    result = ~result
    
    result = threshold(result)
    '''
    result = ~procimg(gray)
    
    
    # OCR using pytesseract
    d = pytesseract.image_to_data(result, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:  # Check for positive confidence
            text = d['text'][i]
            x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            if text and text.strip() != "":
                # Draw bounding box
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Draw text
                cv2.putText(im, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    print(d)
    
    # Draw histogram of the mask
    hist_img = draw_histogram(result)
    
    # Show the images
    cv2.imshow('OCR', im)
    cv2.imshow('MASK', result)
    cv2.imshow('Gray Histogram', hist_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
