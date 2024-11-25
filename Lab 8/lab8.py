import cv2
import numpy as np
from matplotlib import pyplot as plt


# Load the input image
img = cv2.imread('ShopStreet.jpg')
# cv2.imshow('Original', img)

#Creating Kernal Array
kernel = np.array([[-1, -1, -1], 
                   [-1, 8, -1], 
                   [-1, -1, -1]])

#cv2.waitKey(0)

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale', gray_image)

# Defining Kernal Sizes
KernelSizeHeight = 21
KernelSizeWidth = 21

# 2nd Blured Image Kernal Sizes
KernelSizeHeight1 = 41
KernelSizeWidth1 = 41

# 3x3 Blurred image 
BlurredImage = cv2.GaussianBlur(gray_image,(KernelSizeWidth, KernelSizeHeight),0)

# 13x13 Blurred image
BlurredImage1 = cv2.GaussianBlur(gray_image,(KernelSizeWidth1, KernelSizeHeight1),0)

# Defining columns and rows
ncols = 2
nrows = 4

# Original image
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Greyscale image
plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

# Blurred image 3x3
plt.subplot(nrows, ncols,3),plt.imshow(BlurredImage, cmap = 'gray')
plt.title('Blurred Image 3x3'), plt.xticks([]), plt.yticks([])

# Blurred image 13x13
plt.subplot(nrows, ncols,4),plt.imshow(BlurredImage1, cmap = 'gray')
plt.title('Blurred Image 13x13'), plt.xticks([]), plt.yticks([])

# Sobal Code
sobelHorizontal = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5) # x dir
sobelVertical = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5) # y dir

# Sobal Image 1
plt.subplot(nrows, ncols,5),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Sobal Image 1'), plt.xticks([]), plt.yticks([])

# Sobal Image 2
plt.subplot(nrows, ncols,6),plt.imshow(sobelVertical, cmap = 'gray')
plt.title('Sobal Image 2'), plt.xticks([]), plt.yticks([])

# Canny Definition
cannyThreshold = 100
cannyParam2 = 200
canny = cv2.Canny(gray_image,cannyThreshold,cannyParam2)

# Canny image
plt.subplot(nrows, ncols,7),plt.imshow(canny, cmap = 'gray')
plt.title('Canny Image'), plt.xticks([]), plt.yticks([])

plt.show()

# Keep image on screen
# cv2.waitKey(0)  

# Window shown waits for any key pressing event
# cv2.destroyAllWindows()