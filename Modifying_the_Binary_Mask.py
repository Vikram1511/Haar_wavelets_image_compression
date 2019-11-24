import numpy as np
import sys
import cv2

file = sys.argv[1]

image = cv2.imread(file)

cv2.imshow("original", image)
cv2.waitKey(0)

arr = np.array(image)
height, width, ch = image.shape

for i in range(0, height):
	for j in range(0, width):
		for k in range(0, ch):
			if(arr[i][j][k] != 0):
				arr[i][j][k] = arr[i][j][k]/255

cv2.imwrite("New_mask.png", arr)
cv2.waitKey(0)
