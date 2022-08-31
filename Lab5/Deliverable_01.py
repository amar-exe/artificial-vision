#%% Simple Thresholding
import cv2
import os
from matplotlib import pyplot as plt

VISUALISE = True
images_path = 'C:/Users/Amar/Desktop/School/Artificial Vision/Labs/Lab5/Images_Thresholding'
image_name = 'numbers_thresholds.png'

# Obtaining relative path for image
image_relname = os.path.join(images_path, image_name)
imgori = cv2.imread(image_relname, 0)


#%% Apply the different values of threshold
NAIVE_THRESH = [32, 64, 100, 128, 180, 200, 255]

for i in range(7):
    ret, im_thresh = cv2.threshold(imgori, NAIVE_THRESH[i], 255, cv2.THRESH_BINARY)
    
    plt.subplot(2, 4, i + 1)
    plt.imshow(im_thresh, 'gray')
    plt.title("Thr: " + str(NAIVE_THRESH[i]))
    plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 8)
plt.imshow(imgori, 'gray')
plt.title("Original")
plt.xticks([]), plt.yticks([])

plt.savefig("Deliverable1")
plt.show()
