# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 17:05:35 2022

@author: Amar
"""

def vis_images(cwd, extension):
    import os
    import cv2


    os.chdir(cwd)

    for i in os.listdir(cwd):
        split_tup = os.path.splitext(i)
        
        if((len(split_tup[0])>1) and split_tup[1] == extension):
            print(split_tup[0])

         
        img = cv2.imread(i)
        cv2.imshow(i, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
vis_images(
    
    "/Users/Amar/Desktop/School/Artificial Vision/Labs/Lab3/images_Leon/", ".jpg")
