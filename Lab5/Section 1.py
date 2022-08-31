#%% Simple Thresholding
import cv2
import os
from matplotlib import pyplot as plt

VISUALISE = True
images_path = 'C:/Users/Amar/Desktop/School/Artificial Vision/Labs/Lab5/Images_Thresholding'
image_name = 'Gradient horizontal.png'

# Obtaining relative path for image
image_relname = os.path.join(images_path, image_name)
imgori = cv2.imread(image_relname, 0)

# Visualise the image in grayscale using matplotlib
# with a title and without ticks in the x and y axis
if VISUALISE:
    plt.imshow(imgori, cmap='gray')
    plt.title('Original Grayscale gradient')
    plt.xticks([])
    plt.yticks([])
 
#%% Apply the different kinds of threshold
NAIVE_THRESH = 127
ret1, im_thresh1 = cv2.threshold(imgori, NAIVE_THRESH, 255, cv2.THRESH_BINARY)
ret2, im_thresh2 = cv2.threshold(
 imgori, NAIVE_THRESH, 255, cv2.THRESH_BINARY_INV)
ret3, im_thresh3 = cv2.threshold(imgori, NAIVE_THRESH, 255, cv2.THRESH_TRUNC)
ret4, im_thresh4 = cv2.threshold(imgori, NAIVE_THRESH, 255, cv2.THRESH_TOZERO)
ret5, im_thresh5 = cv2.threshold(
 imgori, NAIVE_THRESH, 255, cv2.THRESH_TOZERO_INV)
titles = ['Original Image', 'BINARY',
 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']

images = [imgori, im_thresh1, im_thresh2, im_thresh3, im_thresh4, im_thresh5]
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()
