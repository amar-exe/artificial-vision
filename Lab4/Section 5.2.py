import cv2
from matplotlib import pyplot as plt

path = "/Users/Amar/Desktop/School/Artificial Vision/Labs/Lab4/lena512color.tiff"

img = cv2.imread(path)

#Create Figure for Subplots
fig = plt.figure(figsize=(7, 5))

rows = 2
cols = 3


#Original
fig.add_subplot(rows, cols, 1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title("Original")



#Replicate
fig.add_subplot(rows, cols, 2)

img_replicate = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_REPLICATE)

plt.imshow(img_replicate)
plt.axis('off')
plt.title("Replicate")

#Reflect
fig.add_subplot(rows, cols, 3)

img_reflect = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_REFLECT)

plt.imshow(img_reflect)
plt.axis('off')
plt.title("Reflect")

#Reflect101
fig.add_subplot(rows, cols, 4)

img_reflect_101 = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_REFLECT_101)

plt.imshow(img_reflect_101)
plt.axis('off')
plt.title("Reflect101")

#Wrap
fig.add_subplot(rows, cols, 5)

img_wrap = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_WRAP)

plt.imshow(img_wrap)
plt.axis('off')
plt.title("Wrap")

#Constant
fig.add_subplot(rows, cols, 6)

img_constant = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[255, 0, 0])
plt.imshow(img_constant)
plt.axis('off')
plt.title("Constant")
