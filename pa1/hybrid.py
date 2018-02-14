import cv2
import numpy as np

"""Splits 3D img into 3 separate channels"""
def split_RGB(img):
 	(x,y,z) = img.shape
 	R_array = np.zeros((x,y))
 	G_array = np.zeros((x,y))
 	B_array = np.zeros((x,y))
 	for i in range(x):
 		for j in range(y):
 			for k in range(z):
 				if k == 0:
 					R_array[i][j] = img[i][j][k]
 				elif k == 1:
 					G_array[i][j] = img[i][j][k]
 				else:
 					B_array[i][j] = img[i][j][k] 

 	# R = img[np.array(range(x)), np.array(range(y)), np.array(range(1))]
 	# G = img[np.array(range(x)), np.array(range(y)), np.array(range(1, 2))]
 	# B = img[np.array(range(x)), np.array(range(y)), np.array(range(2, 3))]
 	# print "IN SPLIT RGB"
 	# print str(R) + "R"
 	# print str(G) + "G"
 	# print str(B) + "B"
 	return [R_array,G_array,B_array]

"""Applies cross correlation to each channel in RGB"""
def cross_correlation_RGB(channel_list, kernel):
	new_list = []
	for x in channel_list:
		new_list.append(cross_correlation_2d(x, kernel))
	return new_list

"""Recombines 3 RGB channels into one array"""
def RGB_recombine(channel_list):
	(x,y) = (channel_list[0]).shape
	new_image = np.zeros((x, y, 3))
	for c in range(len(channel_list)):
		temp = channel_list[c]
		for i in range(x):
			for j in range(y):
				new_image[i][j][c] = temp[i][j]
	#new_array= np.concatenate(channel_list, axis = 2)
	return new_image

def cross_correlation_2d(img, kernel):
	'''Given a kernel of arbitrary m x n dimensions, with both m and n being
	odd, compute the cross correlation of the given image with the given
	kernel, such that the output is of the same dimensions as the image and that
	you assume the pixels out of the bounds of the image to be zero. Note that
	you need to apply the kernel to each channel separately, if the given image
	is an RGB image.

	Inputs:
		img:    Either an RGB image (height x width x 3) or a grayscale image
				(height x width) as a numpy array.
		kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
				equal).

	Output:
		Return an image of the same dimensions as the input image (same width,
		height and the number of color channels)
	'''
	(kern_m, kern_n) = kernel.shape
	# TODO-BLOCK-BEGIN
	if (img.ndim == 3):
		RGB_list = split_RGB(img)
		RGB_cross = cross_correlation_RGB(RGB_list, kernel)
		new_image = RGB_recombine(RGB_cross)
		# (x, y, z) = img.shape
		# new_image = np.zeros((x, y, z))
		# up = (kern_m -1)/2
		# out = (kern_n -1)/2
		# for k in range(z):
		# 	for i in range(x):
		# 		for j in range(y):
				
					
		# 			#kern_matrix_temp = kernel[row_idx_kern[:, None], col_idx_kern]
		# 			#	kern_matrix = kern_matrix_temp[:, np.newaxis]
				
		# 			if (i-up<0 and j-out<0): #top left corner
		# 				row_idx_dot = np.array(range(0, i+up+1))
		# 				col_idx_dot = np.array(range(0, j+out+1))
		# 				dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
		# 				row_idx_kern = np.array(range(abs(i-up), kern_m))
		# 				col_idx_kern = np.array(range(abs(j-out), kern_n))
		# 				kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
		# 				average = np.sum(dot_matrix[:,:,k]* kern_matrix)
		# 				new_image[i][j][k] = average
		
		# 			elif (i-up<0 and j+out>=y): #top right corner
		# 				row_idx_dot = np.array(range(0, i+up+1))
		# 				col_idx_dot = np.array(range(j-out, y))
		# 				dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
		# 				row_idx_kern = np.array(range(abs(i-up), kern_m))
		# 				col_idx_kern = np.array(range(0, kern_n-((j+out)-(y-1))))
		# 				kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
		# 				average = np.sum(dot_matrix[:,:,k]* kern_matrix)
		# 				new_image[i][j][k] = average
		# 				#   return #
		
		# 			elif (i+up>=x and j-out<0): #bottom left corner
		# 				row_idx_dot = np.array(range(i-up, x))
		# 				col_idx_dot = np.array(range(0, j+out+1))
		# 				dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
		# 				row_idx_kern = np.array(range(0, kern_m-(i+up-(x-1))))
		# 				col_idx_kern = np.array(range(abs(j-out), kern_n))
		# 				kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
		# 				average = np.sum(dot_matrix[:,:,k]* kern_matrix)
		# 				new_image[i][j] = average
		
		# 			elif (i+up>=x and j+out>=y): #bottom right corner
		# 				row_idx_dot = np.array(range(i-up, x))
		# 				col_idx_dot = np.array(range(j-out, y))
		# 				dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
		# 				row_idx_kern = np.array(range(0, kern_m-(i+up-(x-1))))
		# 				col_idx_kern = np.array(range(0, kern_n-(j+out-(y-1))))
		# 				kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
		# 				average = np.sum(dot_matrix[:,:,k]* kern_matrix)
		# 				new_image[i][j][k] = average
		
		# 			elif i-up<0: #top
		# 				row_idx_dot = np.array(range(0, i+up+1))
		# 				col_idx_dot = np.array(range(j-out, j+out+1))
		# 				dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
		# 				row_idx_kern = np.array(range(abs(i-up), kern_m))
		# 				col_idx_kern = np.array(range(0, kern_n))
		# 				kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
		# 				average = np.sum(dot_matrix[:,:,k]* kern_matrix)
		# 				new_image[i][j][k] = average
		
		# 			elif j-out<0: #left
		# 				row_idx_dot = np.array(range(i-up, i+up+1))
		# 				col_idx_dot = np.array(range(0, j+out+1))
		# 				dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
		# 				row_idx_kern = np.array(range(0, kern_m))
					# 	col_idx_kern = np.array(range(abs(j-out), kern_n))
					# 	kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
					# 	average = np.sum(dot_matrix[:,:,k]* kern_matrix)
					# 	new_image[i][j][k] = average
		
					# elif i+up>=x: #bottom
					# 	row_idx_dot = np.array(range(i-up, x))
					# 	col_idx_dot = np.array(range(j-out, j+out+1))
					# 	dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
					# 	row_idx_kern = np.array(range(0, kern_m-(i+up-(x-1))))
					# 	col_idx_kern = np.array(range(0, kern_n))
					# 	kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
					# 	average = np.sum(dot_matrix[:,:,k]* kern_matrix)
					# 	new_image[i][j][k] = average
		
					# elif j+out>=y: #right
					# 	row_idx_dot = np.array(range(i-up, i+up+1))
					# 	col_idx_dot = np.array(range(j-out, y))
					# 	dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
					# 	row_idx_kern = np.array(range(0, kern_m))
					# 	col_idx_kern = np.array(range(0, kern_n- (j+out-(y-1))))
					# 	kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
					# 	average = np.sum(dot_matrix[:,:,k]* kern_matrix)
					# 	new_image[i][j][k] = average
		
					# else:
					# 	row_idx_dot = np.array(range(i-up, i+up+1))
					# 	col_idx_dot = np.array(range(j-out, j+out+1))
					# 	dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
					# 	average = np.sum(dot_matrix[:,:,k]* kernel)
					# 	new_image[i][j][k] = average


	else:
		(x,y) = img.shape
		new_image = np.zeros((x,y))
		up = (kern_m -1)/2
		out = (kern_n -1)/2
		for i in range(x):
			for j in range(y):
				#dot_matrix = np.zeroes((kern_m, kern_n), dtype= int)
				
					if (i-up<0 and j-out<0): #top left corner
						row_idx_dot = np.array(range(0, min([i+up+1, x])))
						#row_idx_dot = np.array(range(0, i+up+1))
						col_idx_dot = np.array(range(0, min([j+out+1, y])))
						#col_idx_dot = np.array(range(0, j+out+1))
						dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
						#dot_matrix = img[row_idx_dot, col_idx_dot]
						#if matrix is too big
						if (i+up<x):
							row_idx_kern = np.array(range(abs(i-up), kern_m))
						else:
							row_idx_kern = np.array(range(abs(i-up), kern_m-(i+up-x)-1))
						if (j+out<y):
							col_idx_kern = np.array(range(abs(j-out), kern_n)) 
						else:
							#print "kernel is too big"
							col_idx_kern = np.array(range(abs(j-out), kern_n-(j+out-y)-1))
						#np.array(range(abs(j-out)), kern_n-(j+out-y))
						kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
						average = np.sum(dot_matrix* kern_matrix)
						new_image[i][j] = average

					elif (i-up<0 and j+out>=y): #top right corner
						row_idx_dot = np.array(range(0, min([i+up+1, x])))
						#row_idx_dot = np.array(range(0, i+up+1))
						#col_idx_dot = np.array(range(j-out, y))
						col_idx_dot = np.array(range(max([j-out,0]), y))
						dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
						#if matrix is too big
						if (i+up<x):
							row_idx_kern = np.array(range(abs(i-up), kern_m))
						else:
							row_idx_kern = np.array(range(abs(i-up), kern_m-(i+up-x)-1))
						if (j-out>=0):
							col_idx_kern = np.array(range(0, kern_n-((j+out)-(y-1))))
						else:
							col_idx_kern = np.array(range(abs(j-out), kern_n-((j+out)-y)))
						kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
						average = np.sum(dot_matrix* kern_matrix)
						new_image[i][j] = average
						#   return #

					elif (i+up>=x and j-out<0): #bottom left corner
						row_idx_dot = np.array(range(max([i-up,0]), x))
						col_idx_dot = np.array(range(0, min([j+out+1, y])))
						dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
						if (i-up>=0):
							row_idx_kern = np.array(range(0, kern_m-(i+up-(x-1))))
						else:
							row_idx_kern = np.array(range(abs(i-up), kern_m-(i+up-(x-1))))
						if (j+out<y):
							col_idx_kern = np.array(range(abs(j-out), kern_n))
						else:
							#col_idx_kern = np.array(range(abs(j-out), kern_n))
							col_idx_kern = np.array(range(abs(j-out), kern_n-(j+out-y)-1))
						kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
						average = np.sum(dot_matrix* kern_matrix)
						new_image[i][j] = average

					elif (i+up>=x and j+out>=y): #bottom right corner
						row_idx_dot = np.array(range(i-up, x))
						col_idx_dot = np.array(range(j-out, y))
						dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
						row_idx_kern = np.array(range(0, kern_m-(i+up-(x-1))))
						col_idx_kern = np.array(range(0, kern_n-(j+out-(y-1))))
						kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
						average = np.sum(dot_matrix* kern_matrix)
						new_image[i][j] = average

					elif i-up<0: #top
						row_idx_dot = np.array(range(0, i+up+1))
						col_idx_dot = np.array(range(j-out, j+out+1))
						dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
						row_idx_kern = np.array(range(abs(i-up), kern_m))
						col_idx_kern = np.array(range(0, kern_n))
						kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
						average = np.sum(dot_matrix* kern_matrix)
						new_image[i][j] = average

					elif j-out<0: #left
						row_idx_dot = np.array(range(i-up, i+up+1))
						col_idx_dot = np.array(range(0, j+out+1))
						dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
						row_idx_kern = np.array(range(0, kern_m))
						col_idx_kern = np.array(range(abs(j-out), kern_n))
						kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
						average = np.sum(dot_matrix* kern_matrix)
						new_image[i][j] = average

					elif i+up>=x: #bottom
						row_idx_dot = np.array(range(i-up, x))
						col_idx_dot = np.array(range(j-out, j+out+1))
						dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
						row_idx_kern = np.array(range(0, kern_m-(i+up-(x-1))))
						col_idx_kern = np.array(range(0, kern_n))
						kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
						average = np.sum(dot_matrix* kern_matrix)
						new_image[i][j] = average

					elif j+out>=y: #right
						row_idx_dot = np.array(range(i-up, i+up+1))
						col_idx_dot = np.array(range(j-out, y))
						dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
						row_idx_kern = np.array(range(0, kern_m))
						col_idx_kern = np.array(range(0, kern_n- (j+out-(y-1))))
						kern_matrix = kernel[row_idx_kern[:, None], col_idx_kern]
						average = np.sum(dot_matrix* kern_matrix)
						new_image[i][j] = average

					else:
						row_idx_dot = np.array(range(i-up, i+up+1))
						col_idx_dot = np.array(range(j-out, j+out+1))
						dot_matrix = img[row_idx_dot[:, None], col_idx_dot]
						average = np.sum(dot_matrix* kernel)
						new_image[i][j] = average

	#print kernel
	#print img
	#print str(new_image)+"myimage"
	#print str(new_image.shape) + "myimage shape"
	return new_image



	# TODO-BLOCK-END

def convolve_2d(img, kernel):
	'''Use cross_correlation_2d() to carry out a 2D convolution.

	Inputs:
		img:    Either an RGB image (height x width x 3) or a grayscale image
				(height x width) as a numpy array.
		kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
				equal).

	Output:
		Return an image of the same dimensions as the input image (same width,
		height and the number of color channels)
	'''
	# TODO-BLOCK-BEGIN
	kern_lr = np.fliplr(kernel)
	kern_flipped = np.flipud(kern_lr)
	return cross_correlation_2d(img, kern_flipped)

	# TODO-BLOCK-END


def gaussian_blur_kernel_2d(sigma, width, height):
	'''Return a Gaussian blur kernel of the given dimensions and with the given
	sigma. Note that width and height are different.

	Input:
		sigma:  The parameter that controls the radius of the Gaussian blur.
				Note that, in our case, it is a circular Gaussian (symmetric
				across height and width).
		width:  The width of the kernel.
		height: The height of the kernel.

	Output:
		Return a kernel of dimensions width x height such that convolving it
		with an image results in a Gaussian-blurred image.
	'''
	# TODO-BLOCK-BEGIN
	kern = np.zeros((height, width))
	center_row = int(height/2) + 1 if height%2 == 1 else int(height/2)
	center_column = int(width/2) + 1 if width%2 == 1 else int(width/2)
	for i in range(height):
		for j in range(width):
			#constant = 1/(2*np.pi*sigma**2)
			val = np.exp(-1.0*
				((i-(center_row-1))**2+((j-(center_column-1))**2))/(2*sigma**2))
			kern[i][j] = val
	s = np.sum(kern.flatten())


	#divide the whole kernel by the sum of all the values
	#print kern.shape
	return np.transpose(kern/s)
	

	# TODO-BLOCK-END

def low_pass(img, sigma, size):
	'''Filter the image as if its filtered with a low pass filter of the given
	sigma and a square kernel of the given size. A low pass filter supresses
	the higher frequency components (finer details) of the image.

	Output:
		Return an image of the same dimensions as the input image (same width,
		height and the number of color channels)
	'''

	# TODO-BLOCK-BEGIN
	kern = np.transpose(gaussian_blur_kernel_2d(sigma, size, size))
	return convolve_2d(img, kern)
	# TODO-BLOCK-END

def high_pass(img, sigma, size):
	'''Filter the image as if its filtered with a high pass filter of the given
	sigma and a square kernel of the given size. A high pass filter suppresses
	the lower frequency components (coarse details) of the image.

	Output:
		Return an image of the same dimensions as the input image (same width,
		height and the number of color channels)
	'''
	# TODO-BLOCK-BEGIN
	#kern = np.transpose(gaussian_blur_kernel_2d(sigma, size, size))
	
	#print convolve_2d(img, kern)
	return img - low_pass(img, sigma, size)
	# TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
		high_low2, mixin_ratio):
	'''This function adds two images to create a hybrid image, based on
	parameters specified by the user.'''
	high_low1 = high_low1.lower()
	high_low2 = high_low2.lower()

	if img1.dtype == np.uint8:
		img1 = img1.astype(np.float32) / 255.0
		img2 = img2.astype(np.float32) / 255.0

	if high_low1 == 'low':
		img1 = low_pass(img1, sigma1, size1)
	else:
		img1 = high_pass(img1, sigma1, size1)

	if high_low2 == 'low':
		img2 = low_pass(img2, sigma2, size2)
	else:
		img2 = high_pass(img2, sigma2, size2)

	img1 *= 2 * (1 - mixin_ratio)
	img2 *= 2 * mixin_ratio
	hybrid_img = (img1 + img2)
	return (hybrid_img * 255).clip(0, 255).astype(np.uint8)



