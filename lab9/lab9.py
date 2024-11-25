import cv2
import numpy as np
from matplotlib import pyplot as plt

#importing ATU image
img = cv2.imread('ATU1.jpg',)

# Deep Copy of Image
imgHarris = img.copy()

# Defining columns and rows
ncols = 2
nrows = 4

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris Corner detection funcion
dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)

# Original image
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Greyscale image
plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

# Corner Dection image
plt.subplot(nrows, ncols,3),plt.imshow(gray_image, cmap = 'gray')
plt.title('Corner Detection'), plt.xticks([]), plt.yticks([])

# Thresholds
threshold = 0.5; #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(2, 10, 50),-1)


plt.show()