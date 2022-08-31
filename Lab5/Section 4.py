#%% OTSU'S THRESHOLD
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

images_path = 'C:/Users/Amar/Desktop/School/Artificial Vision/Labs/Lab5/Images_Thresholding'
image_name = 'Original noisy image.png'

# Obtaining relative path for image
image_relname = os.path.join(images_path, image_name)
FLAG_COLOR = 0 # Return a grayscale image
img_ori = cv2.imread(image_relname, FLAG_COLOR)

# Global thresholding of the original image
thr_glob, imag_thr_global = cv2.threshold(img_ori, 127, 255, cv2.THRESH_BINARY)

# Otsu's thresholding of the original image
thr_Otsu, imag_thr_Otsu = cv2.threshold(
 img_ori, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
img_blur = cv2.GaussianBlur(img_ori, (5, 5), 0)
thr_Gaus, imag_thr_Gauss = cv2.threshold(
 img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img_ori, 0, imag_thr_global,
 img_ori, 0, imag_thr_Otsu,
 img_blur, 0, imag_thr_Gauss]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
 'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
 'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
plt.figure(figsize=(10, 10))

for i in range(3):
 plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
 plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
 plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
 plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
 plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
 plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
plt.savefig('Section 4.png')
plt.show()
