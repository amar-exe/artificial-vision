from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 
import cv2

path = "/Users/Amar/Desktop/School/Artificial Vision/Labs/Lab4/lena512color.tiff"

img = Image.open(path, mode='r')
imgg = cv2.imread(path, cv2.IMREAD_UNCHANGED)

plt.imshow(img)
plt.show()

img_grayscale = img.convert("L")
img_grayscale_array = np.asarray(img_grayscale)
plt.imshow(img_grayscale_array, cmap = "gray")

img_height = imgg.shape[0]
img_width = imgg.shape[1]
img_channels = imgg.shape[2]

print("Image height: ", img_height)
print("Image width: ", img_width)
print("Number of channels: ", img_channels)
