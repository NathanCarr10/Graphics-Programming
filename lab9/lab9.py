import cv2
import numpy as np
from matplotlib import pyplot as plt

#importing ATU image
img = cv2.imread('ATU1.jpg',)

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Original image
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Greyscale image
plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.show()