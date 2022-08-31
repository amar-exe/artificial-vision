"""
otsu_thres_hist(img_ori, siz_gau=13, title='Original image filtered')
This function plots the indicated image (img_ori) -after applying a Gaussian
filter- with the title indicated in the title input variable, its histogram (with title
'Histogram') and its binary image after binarizing it (with 'Otsu's Thresholding'
title). The function should receive the image (img_ori), apply Gaussian filtering, and
then compute the Otsu threshold. In other words, the function should return three
images: the input image after using Gaussian filter with a title indicated by the input
argument title, the image of the histogram and the image of the binary output resulting
after applying Otsu. It should return something similar to the third row presented in
the previous script in " Figure 5.6. Example of global and Otsu's thresholding
"
img_ori: the image
siz_gau: Integer. Size of the Gaussian window applied. It will be siz_gau x siz_gau
title: a string to be used for the input image
"""

def otsu_thres_hist(img_ori, siz_gau=13, title='Original image filtered'):
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    
    plt.figure(figsize=(10,2))
    
    img_original = cv2.imread(img_ori)
    
    #Gaussian
    img_blur = cv2.GaussianBlur(img_original, (siz_gau, siz_gau), 0)
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_blur, 'gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    
    #Histogram
    histogram, bin_edges = np.histogram(img_original, bins=256, range=(0, 255))
    
    plt.subplot(1, 3, 2), plt.hist(img_blur.ravel(), 256)
    plt.title("Histogram")
    plt.xticks([]), plt.yticks([])
    
    #Otsu's
    img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_blur = img_blur.astype('uint8')
    ret3,th3 = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    plt.subplot(1, 3, 3)
    plt.imshow(th3, 'gray')
    plt.title("Otsu's Thresholding")
    plt.xticks([]), plt.yticks([])
    plt.savefig("Deliverable3")
    
otsu_thres_hist('C:/Users/Amar/Desktop/School/Artificial Vision/Labs/Lab5/Images_Thresholding/Original noisy image.png')
    
