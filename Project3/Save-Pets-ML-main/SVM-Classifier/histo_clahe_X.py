import cv2
import numpy as np

def histo_clahe(img_dir):
    # Read the image from the given directory
    img = cv2.imread(img_dir)
    height, width, channel = img.shape

    # Resize the image if its size is too large. 인스타 이미지 사이즈 640x640
    # (larger than 600x600)
    while height >= 600 or width >= 600:
        img = cv2.resize(img, (int(width / 2), int(height / 2)))
        height, width, channel = img.shape

    # Convert the image to YUV color space
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Apply histogram equalization to the brightness (Y) channel
    img_eq = img_yuv.copy()
    img_eq[:, :, 0] = cv2.equalizeHist(img_eq[:, :, 0])
    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

    # Apply CLAHE to the brightness (Y) channel
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    img_clahe = img_yuv.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe[:, :, 0] = clahe.apply(img_clahe[:, :, 0])
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

    return img_clahe[:, :, :] #  resulting CLAHE-processed image (img_clahe).
'''
this function is intended to enhance the contrast and equalize the brightness of an input image, 
which can be useful for improving the quality and feature visibility of images before further processing, such as feature extraction.
'''