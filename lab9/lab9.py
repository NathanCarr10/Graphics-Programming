import cv2
import numpy as np
from matplotlib import pyplot as plt

#importing ATU image
img = cv2.imread('ATU1.jpg',)

# Deep Copy of Image
imgHarris = img.copy()

# mgShiTomasi image copy
imgShiTomasi = img.copy()

# Defining columns and rows
ncols = 2
nrows = 4

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris Corner detection funcion
dst = cv2.cornerHarris(gray_image, blockSize=2, aperture_size=3, k=0.04)

# Thresholds
threshold = 0.5; #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(0, 255, 0),-1)

# Original image
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Greyscale image
plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

# Corner Dection/ Harris image
plt.subplot(nrows, ncols,3),plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris'), plt.xticks([]), plt.yticks([])            


# Good Features To Track
corners = cv2.goodFeaturesToTrack(gray_image, maxCorners, qualityLevel, minDistance)

# GFTT corners loop through the corners array
for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(x,y),5,(255, 0, 0),-1)  

#ShiTomasi image plot
plt.subplot(nrows, ncols, 4)plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB))
plt.title('Shi-Tomasi Corners')plt.xticks([]), plt.yticks([])


plt.show()