#%% FUNCTION TO ADD NOISE TO AN IMAGE
# MODIFIED FROM http://stackoverflow.com/questions/22937589/...
# how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
import numpy as np
import os
import cv2

def noisy(image, noise_typ):
    '''
    Parameters
    ----------
     image : ndarray
     Input image data. Will be converted to float.
     noise_typ : str
     One of the following strings, selecting the type of noise to add:
         'gauss' Gaussian-distributed additive noise.
         'poisson' Poisson-distributed noise generated from the data.
         's&p' Replaces random pixels with 0 or 1.
         'speckle' Multiplicative noise using out = image + n*image, where
         n is uniform noise with specified mean & variance.
         '''
# 1_GAUSSIAN NOISE --------------------------------------
    if noise_typ == "gauss":
        if len(image.shape) == 3:
            FLAG_RGB = True
            row, col, ch = image.shape
        else:
            FLAG_RGB = False
            row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        
        if FLAG_RGB:
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
        else:
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
        img_noisy = image + gauss
        return img_noisy
        
# 2_SALT AND PEPPER NOISE -------------------------------
    elif noise_typ == 's&p':
        if len(image.shape) == 3:
            FLAG_RGB = True
            row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
                 
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1
 
# Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

# 3_POISSON NOISE ---------------------------------------
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        img_noisy = np.random.poisson(image * vals) / float(vals)
        return img_noisy
 
# 4_SPECKLE NOISE --------------------------------------
    elif noise_typ == "speckle":
        if len(image.shape) == 3:
            FLAG_RGB = True
            row, col, ch = image.shape
        else:
            FLAG_RGB = False
            row, col = image.shape
        if FLAG_RGB:
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
        else:
            gauss = np.random.randn(row, col)
            gauss = gauss.reshape(row, col)
        img_noisy = image + image * gauss
        return img_noisy

#%% TRYING noisy FUNCTION 
from matplotlib import pyplot as plt

# Create an numpy array, size 400 x 600, with zeros
imgarray = np.zeros([400,600],dtype=np.uint8)

# Slicing the array to put the inside square white
imgarray [125:275,150:450] = 255
types = ['gauss', 's&p', 'poisson', 'speckle']

noisy_gau = noisy(imgarray,'gauss')
noisy_sAp = noisy(imgarray,'s&p')
noisy_poi = noisy(imgarray,'poisson')
noisy_spe = noisy(imgarray,'speckle')

plt.figure(figsize=(10,10))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(noisy(imgarray, types[i]), 'gray')
    plt.title(types[i])
    plt.xticks([]), plt.yticks([])

plt.savefig('Comparison types of Noise.png')
plt.show()
