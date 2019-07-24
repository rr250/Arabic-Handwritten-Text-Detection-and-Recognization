from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)


def preprocess(img, imgSize, dataAugmentation=False):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"
    #img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        stretch = (random.random() - 0.5) # -0.5 .. +0.5
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
        img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5
    
    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img
    
    # transpose for TF
    img = cv2.transpose(target)
    #cv2.imwrite( filePath[:-4]+'_pre1.png', img );
    # normalize
    
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    #cv2.imwrite( filePath[:-4]+'_pre2.png', img );
    img = img / s if s>0 else img
    #cv2.imwrite( filePath[:-4]+'_pre3.png', img );
    
    return img

def plot_image(image,xlabel):
    plt.figure(figsize=(30,20))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('off')
    plt.xlabel(xlabel, fontsize=18)
    plt.show()

def denoise(img):
    return denoise_tv_chambolle(img, weight=0.1, multichannel=True)
    
def dilation(img):
    kernel = np.ones((5,5), np.uint8) 
    return cv2.dilate(img, kernel, iterations=1)

def erotion(img):
    kernel = np.ones((5,5), np.uint8) 
    return cv2.erode(img, kernel, iterations=1) 
    
def edgeDetection(img):
    slice1Copy = np.uint8(img)
    return cv2.Canny(slice1Copy,100,200)
    
def hough_transform(img):
    # Convert the image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, 50, 200)
    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, max_slider, minLineLength=10, maxLineGap=250)
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # Show result
    #cv2.imshow("Result Image", img)
    return img
    
def binarization(img):
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return thresh1