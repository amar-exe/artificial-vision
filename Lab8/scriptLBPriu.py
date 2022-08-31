#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:45:03 2018
Updated on Wed May 05 18:31:00 2021

@author: victor
"""
import numpy as np
from matplotlib import pyplot as plt
# import imageio
from skimage import exposure
import cv2
import functions_LBP as lbp


# image = imageio.imread('LBPImageTest.jpg').astype(np.float)
image = cv2.imread('Fabric_0002.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_lbp = lbp.LBPriu_image(image)


plt.figure()
plt.subplot(1, 3, 1), plt.imshow(image), plt.axis('off')
plt.title('Original image')
image_lbp_eq = exposure.equalize_hist(image_lbp.astype(np.uint8))
plt.subplot(1, 3, 2), plt.imshow(image_lbp_eq, cmap='gray'), plt.axis('off')
plt.title('LBPriu image')
plt.subplot(1, 3, 3), plt.hist(image_lbp.ravel(), bins=20)
plt.title('LBPriu 20 bins histogram')
plt.show()
