# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 17:05:35 2022

@author: Amar
"""
import os
import cv2


os.chdir("/Users/Amar/Desktop/School/Artificial Vision/Labs/Lab3/images_Leon/")
cwd = os.getcwd()
print(cwd)

for i in os.listdir(cwd):
    split_tup = os.path.splitext(i)
    
    if((len(split_tup[0])>1) and split_tup[1] == ".jpg"):
        print(split_tup[0])
     
    img = cv2.imread(i)
    cv2.imshow(i, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
        
