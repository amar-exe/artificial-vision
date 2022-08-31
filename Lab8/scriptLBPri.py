#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:45:03 2018
Updated on Wed May 05 18:14:00 2021

@author: victor
"""
import numpy as np
from matplotlib import pyplot as plt
# import imageio
from skimage import transform
import cv2
import functions_LBP as lbp


# image = imageio.imread('LBPImageTest.jpg').astype(np.float)
image = cv2.imread('Fabric_0002.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_lbp = lbp.LBPri_image(image)
image_rot = transform.rotate(image, 90, preserve_range=True).astype(np.uint8)
image_rot_lbp = lbp.LBPri_image(image_rot)


plt.figure()
plt.subplot(2, 2, 1), plt.imshow(image), plt.axis('off')
plt.title('Original image')
plt.subplot(2, 2, 2), plt.imshow(image_rot), plt.axis('off')
plt.title('Rotated image')
plt.subplot(2, 2, 3), plt.hist(image_lbp.ravel(), bins=20)
plt.title('Original LBP histogram')
plt.subplot(2, 2, 4), plt.hist(image_rot_lbp.ravel(), bins=20)
plt.title('Rotated LBP histogram')
plt.show()
