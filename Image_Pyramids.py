import sys
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt 
# https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html


yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])

rgb_from_yuv = np.linalg.inv(yuv_from_rgb)

gaussian_kernel = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
def yuv2rgb(yuv):
    return np.clip(np.dot(yuv,rgb_from_yuv),0,1)

def gaussian_matrix():
	A = np.zeros([5,5])
	W = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
	A[2] = W
	A[0][2]=A[4][2] = 0.05
	A[1][2]=A[3][2] = 0.25
	return A

def padded_image(image):
	borderType = cv2.BORDER_CONSTANT
	top = bottom = left = right = 2
	padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType)

	return padded_image

def gaussian_blur(img,up=False,mask=False):
	image = img.copy()
	height, width, ch = image.shape
	print(image.shape)
	image = padded_image(image)
	output = np.zeros((height, width, ch), dtype=str(image.dtype))
	kernel = gaussian_kernel
	for i in range(ch):
		for y in range(2, height+2):
			for x in range(2, width+2):
				roi = image[y-2:y+3, x-2:x+3, i]
				k = (roi*kernel).sum()
				if(up==False):
					output[y-2,x-2,i] = k/(kernel.sum())
				else:
					output[y-2,x-2,i] = 4*(k/(kernel.sum()))
	output = rescale_intensity(output, in_range=(output.min(), output.max()),out_range=(0,255))
	return output

def pyrDown(img):
	image = img.copy()
	image =gaussian_blur(image)
	m,n,c = image.shape

	output = np.zeros((int(m/2), n, c), dtype=str(image.dtype))
	for j in range(int(m/2)):
		output[j,:,:] = image[2*j+1,:,:]

	output_new = np.zeros((int(m/2),int(n/2),c),dtype=str(image.dtype))
	for i in range(int(n/2)):
		output_new[:,i,:] = output[:,2*i+1,:]
	return output_new

def pyrUp(img):
	image = img.copy()
	m = image.shape[0]
	n = image.shape[1]
	scale = 2
	width = int(image.shape[1]*scale)
	height = int(image.shape[0]*scale)
	output = np.zeros((height,n,image.shape[2]),dtype=str(image.dtype))
	for i in range(int(height/2)):
		output[2*i+1,:,:] = image[i,:,:]
	
	output_new = np.zeros((height,width,image.shape[2]),dtype=str(image.dtype))
	for i in range(int(width/2)):
		output_new[:,2*i+1,:]= output[:,i,:]
	result = gaussian_blur(output_new,up=True)
	# dim = (width, height)
	# resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return result

def gaussian_pyramid(image,levels):
	assert np.power(2,levels)<=image.shape[0] and np.power(2,levels)<=image.shape[1]
	layer = image.copy()
	arr = [layer]
	for i in range(levels):
		layer = pyrDown(layer)
		arr.append(layer)
	return arr
	
def print_gaussian(gaussian_pyramid):
	for i in range(len(gaussian_pyramid)):
		cv2.imshow(str(i), gaussian_pyramid[i])
		cv2.waitKey(0)
		
def print_laplacian_pyramid(laplacian_pyramid):
    for i in range(len(laplacian_pyramid)):
        cv2.imshow(str(i), laplacian_pyramid[i])
        cv2.waitKey(0)

def laplacian_pyramid(image,levels):
	assert np.power(2,levels)<=image.shape[0] and np.power(2,levels)<=image.shape[1]
	layer = image.copy()
	gp = gaussian_pyramid(layer,levels)

	lp = []
	for i in range(1, levels+1):
		expanded_image = pyrUp(gp[i])
		j = i-1
		laplacian = cv2.subtract(gp[j],expanded_image)
		lp.append(laplacian)
	lp.append(gp[-1])
	return lp,gp

def reconstructed(lp):
	levels = len(lp)
	corrected_image = lp[-1]
	for i in range(levels-2,-1,-1):
		expanded_image = pyrUp(corrected_image)
		# expanded_image = gaussian_blur(expanded_image)
		corrected_image = cv2.add(expanded_image,lp[i])
	return corrected_image

def blending_with_overlapping_regions(lp1, lp2):
	LS = []
	for la,lb in zip(lp1, lp2):
		rows, cols, dpt = la.shape
		ls = np.hstack((la[:,0:int(cols/2),:], lb[:, int(cols/2):,:]))
		LS.append(ls)
	image = reconstructed(LS)
	return image

def blending_with_Arbitrary_regions(image1_l, image2_l, mask_g):
	LS = []
	for la,lb,gp in zip(image1_l, image2_l, mask_g):
		gp = rescale_intensity(gp,in_range=(gp.min(),gp.max()),out_range=(0,1))
		cv2.imshow('gp',gp)
		cv2.waitKey(0)
		rows, cols, dpt = la.shape
		gla =  gp*la
		glb = (1-gp)*lb
		ls = cv2.add(gla,glb)
		cv2.imshow("ls",ls)
		cv2.waitKey(0)
		LS.append(ls)
	image = reconstructed(LS)
	return image


file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]

image1 = cv2.imread(file1)
image2 = cv2.imread(file2)
image3 = cv2.imread(file3)

# cv2.imshow("Original1", image1)
# cv2.imshow("Original2", image2)
# cv2.waitKey(0)

a1,b1 = laplacian_pyramid(image1, 3)
a2,b2 = laplacian_pyramid(image2, 3)
# print_laplacian_pyramid(a2)
# print_laplacian_pyramid(a1)

# b3 = [image3]
# for i in range(3):
# 	l = cv2.pyrDown(image3)
# 	image3=l
# 	b3.append(l)
# b3 = gaussian_pyramid(image3,3)

reconstructed1 = reconstructed(a1)
reconstructed2 = reconstructed(a2)
cv2.imshow("reconstructed1", reconstructed1)
cv2.imshow("reconstructed2", reconstructed2)
cv2.waitKey(0)

blended_image = blending_with_overlapping_regions(a1, a2)
cv2.imshow("blended_image", blended_image)
plt.imsave("blended_image.png",blended_image)
cv2.waitKey(0)

def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, -1)
    # We need to mask the negative differences
    # since we are looking for values above
    if np.all(mask):
        c = np.abs(diff).argmin()
        return c # returns min index of the nearest if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

def hist_match(original, specified):
 
    oldshape = original.shape
    original = original.ravel()
    specified = specified.ravel()
 
    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(original, return_inverse=True,return_counts=True)
    t_values, t_counts = np.unique(specified, return_counts=True)
 
    # Calculate s_k for original image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    # Calculate s_k for specified image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
 
    # Round the values
    sour = np.around(s_quantiles*255)
    temp = np.around(t_quantiles*255)
    
    # Map the rounded values
    b=[]
    for data in sour[:]:
        b.append(find_nearest_above(temp,data))
    b= np.array(b,dtype='uint8')
 
    return b[bin_idx].reshape(oldshape)

file1 = sys.argv[1]
file2 = sys.argv[2]


image1 = cv2.imread(file1)
image2 = cv2.imread(file2)

print(image1.shape)
print(image2.shape)
cv2.imshow("Original1", image1)
cv2.imshow("Original2", image2)
cv2.waitKey(0)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# The image is read and converted to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# The face or faces in an image are detected
# This section requires the most adjustments to get accuracy on face being detected.
copy1 = image1.copy()
copy2 = image2.copy()

faces1 = faceCascade.detectMultiScale(
    gray1,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(1,1),
    flags = cv2.CASCADE_SCALE_IMAGE
)

faces2 = faceCascade.detectMultiScale(
    gray2,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(1,1),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Detected {0} faces!".format(len(faces1)))

# This draws a green rectangle around the faces detected

# for (x, y, w, h) in faces1:
#     cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 255, 0), 0)

# for (x, y, w, h) in faces2:
#     cv2.rectangle(image2, (x, y), (x+w, y+h), (0, 255, 0), 0)

# mask = cv2.subtract(image2, copy2)
a = hist_match(image1, image2)

# image2 = cv2.subtract(image2, mask)

l1,g1 = laplacian_pyramid(a,5)
l2,g2 = laplacian_pyramid(image2,5)

a = gaussian_blur(a)

cv2.imshow("A", a)
cv2.waitKey(0)

cv2.imshow("Mask", mask)
cv2.waitKey(0)
# l3=[]
# for i in range(5):
# 	l3.append(cv2.add(l1[i],l2[i]))

# print_laplacian_pyramid(l3)
# reconstructed = reconstructed(l3)
# reconstructed = cv2.subtract(reconstructed, mask)
# cv2.imshow("reconstructed", reconstructed)

# cv2.waitKey(0)



# a = hist_match(image1, image2)
# cv2.imshow("hist_match", a)
# cv2.waitKey(0)



# l2,g2 = laplacian_pyramid(image2, 5)

# cv2.imshow("a", l2[0])
# cv2.waitKey(0)
# # a1,b1 = laplacian_pyramid(image1, 5)
# # a2,b2 = laplacian_pyramid(image2, 5)
# # a3 = []
# # for i in range(len(a1)):
# 	a = hist_match(b1[i], b2[i])
# 	a3.append(a)

# for i in range(len(a3)):
# 	a3[i] = cv2.add(a3[i], a2[i])

# for i in range(len(a3)):
# 	cv2.imshow(str(i), a3[i])
# 	cv2.waitKey(0)

# reconstructed = reconstructed(a3)
# cv2.imshow("re", reconstructed)
# cv2.waitKey(0)


# a,b = laplacian_pyramid(image,3)

# for i in range(len(a)):
# 	print(b[i].shape)
# 	cv2.imshow("image_pyramid"+str(i),b[i])
# cv2.waitKey(0)

# for i in range(len(a)):
# 	print(a[i].shape)
# 	cv2.imshow("image_pyramid"+str(i),a[i])
# cv2.waitKey(0)

# c = reconstructed(a)
# cv2.imshow("reconstructed", c)
# cv2.waitKey(0)



# a,b = laplacian_pyramid(image,3)

# for i in range(len(a)):
# 	print(b[i].shape)
# 	cv2.imshow("image_pyramid"+str(i),b[i])
# cv2.waitKey(0)

# for i in range(len(a)):
# 	print(a[i].shape)
# 	cv2.imshow("image_pyramid"+str(i),a[i])
# cv2.waitKey(0)

# c = reconstructed(a)
# cv2.imshow("reconstructed", c)
# cv2.waitKey(0)
