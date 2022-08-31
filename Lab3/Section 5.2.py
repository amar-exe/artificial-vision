# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 19:43:50 2022

@author: Amar
"""
import os
import cv2
from matplotlib import pyplot as plt

os.chdir("/Users/Amar/Desktop/School/Artificial Vision/Labs/Lab3/images_Leon/")
cwd = os.getcwd()

for i in os.listdir(cwd):
    img = cv2.imread(i)
    
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    cv2.destroyAllWindows()
    
    
