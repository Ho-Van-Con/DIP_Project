import cv2              
import numpy as np
import argparse
import pytesseract
import imutils

from imutils.object_detection import non_max_suppression

image = cv2.imread("Image/example_02.jpg")
h,w = image.shape[:2]   # find height and width of image
width = int(1000)
height = int(width * (h/w))
image = cv2.resize(src=image, dsize=(width,height))
image = cv2.rectangle(image,(360,105),(950,155),(0,255,0),2)
image = cv2.rectangle(image,(535,165),(785,205),(0,255,0),2)
cv2.imshow("Original Image", image)
cv2.waitKey(0)  # pause execution until any key pressed