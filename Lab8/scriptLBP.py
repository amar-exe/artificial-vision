#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:45:03 2018
Updated on Wed May 05 17:55:00 2021

@author: victor
"""
import numpy as np
from matplotlib import pyplot as plt
# import imageio
import cv2
import functions_LBP as lbp


# image = imageio.imread('LBPImageTest.jpg')
image = cv2.imread('Fabric_0002.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_lbp = lbp.LBP_image(image)

plt.figure()
plt.subplot(2, 2, 1), plt.imshow(image), plt.axis('off')
plt.title('Original image')
plt.subplot(2, 2, 2), plt.imshow(image_lbp.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.title('LBP image')
plt.subplot(2, 2, 3), plt.hist(image_lbp.ravel(), bins=20)
plt.title('20 bins histogram')
plt.subplot(2, 2, 4), plt.hist(image_lbp.ravel(), bins=10)
plt.title('10 bins histogram')
plt.show()
