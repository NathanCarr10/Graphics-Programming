import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',)

# Load the input image
img = cv2.imread('ATU.jpg')
# cv2.imshow('Original', img)

#Creating Kernal Array
kernel = np.array([[0, 1, 0], 
                   [1, 1.5, 1], 
                   [0, 1, 0]])

cv2.waitKey(0)

# Defining Kernal Sizes
KernelSizeHeight = 15
KernelSizeWidth = 15

BlurredImage = cv2.GaussianBlur(img,(KernelSizeWidth, KernelSizeHeight),0)

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale', gray_image)

# Defining columns and rows
ncols = 1
nrows = 3

# Original image
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Greyscale image
plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

# Blurred image
plt.subplot(nrows, ncols,3),plt.imshow(cv2.cvtColor(BlurredImage, cv2.COLOR_BGR2RGB))
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])

plt.show()

imgOut = cv2.GaussianBlur(imgIn,(KernelSizeWidth, KernelSizeHeight),0)


# Keep image on screen
# cv2.waitKey(0)  

# Window shown waits for any key pressing event
# cv2.destroyAllWindows()