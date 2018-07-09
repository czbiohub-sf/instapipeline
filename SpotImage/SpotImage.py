""" This module contains SpotImage class.
"""

import numpy as np
import math
import cv2
from skimage.restoration import estimate_sigma

# ------- #

class SpotImage():

	"""
	The SpotImage tool generates synthetic images for workers to annotate. 
	The user can specify the following features of the image:

	-	background image
	-	color mapping
	-	image size
	-	number of spots
	-	spot shape
	-	SNR distribution (from which to sample SNR of each spot)

	Implementation:
	->	The tool generates a list of spots. 
			Each spot has a random location and a patch.
			Each spot has an SNR (sampled from the SNR distribution) 
	->	The list of spots is converted to a 2D pixel array.
	->	The background image is scaled and color-mapped.
	->	The 2D pixel array is added to the background image to produce the final image.

	"""

	# list of SNR distributions handled
	spot_shapes = ['2D_Gaussian']
	snr_distrs = ['Gaussian']						

	"""
	Constructor
	"""
	def __init__(self, bg_img, color_map, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params):
		if (spot_shape_params[0] not in self.spot_shapes):
			raise ValueError('Invalid spot shape name entered.')
		if (snr_distr_params[0] not in self.snr_distrs):
			raise ValueError('Invalid SNR distribution name entered.')
		self.bg_img = bg_img
		self.color_map = color_map
		self.img_sz = img_sz
		self.patch_sz = patch_sz
		self.num_spots = num_spots
		self.spot_shape_params = spot_shape_params
		self.snr_distr_params = snr_distr_params
		self.margin = math.floor(self.patch_sz/2)		# setting margin such that no patches hang off the edges

	"""
	Samples SNR from the specified distribution.
	"""
	def get_snr(self):
		if (self.snr_distr_params[0] == 'Gaussian'):
			mu = self.snr_distr_params[1]
			sigma = self.snr_distr_params[2]
			snr = np.random.normal(mu, sigma)
		return snr

	"""
	Leaving get_spot_x() and get_spot_y() as separate functions for now
	in case we want to sample them from independent distributions later.
	"""
	def get_spot_x(self):
		return np.random.random_integers(self.margin, self.img_sz - self.margin - 1)		# discrete uniform distribution

	def get_spot_y(self):
		return np.random.random_integers(self.margin, self.img_sz - self.margin - 1)

	"""
	Returns:
		list of spots
			each spot has a random location and a patch of intensities
	"""
	def get_spot_list(self):
		spot_list = [[x=self.get_spot_x(), y=self.get_spot_y(), self.get_patch(x,y)] for i in range(self.num_spots)]
		return spot_list

	"""
	Returns one 2D square array with a spot
		Spot obeys spot_shape_params
		Spot has SNR sampled from SNR distr
	"""
	def get_patch(self, x, y):
		patch = np.zeros([self.patch_sz, self.patch_sz])
		sigma = self.get_sigma(x,y)			
		
		for row in patch:
			for i in range(self.patch_sz):
				# Temporarily populate patch with rand ints just
				# to get the logic of the rest of this class.
				row[i] = np.random.random_integers(0,400)

				# if (self.spot_shape_params[0] == '2D_Gaussian'):
				# 	mu = self.spot_shape_params[1]
				# 	sigma = self.spot_shape_params[2]

		return patch

	"""
	Gets sigma from background patch centered on (x,y) 
	"""
	def get_sigma(self, x, y):
		origin_x = x - self.margin
		origin_y = y - self.margin
		patch = np.zeros([self.patch_sz, self.patch_sz])
		for row in range(self.patch_sz):
			for col in range(self.patch_sz):
				patch[row][col] = self.img_to_array()[origin_y + row][origin_x + col]
		sigma = estimate_sigma(patch, multichannel=True, average_sigmas=True)			# get noise from equiv. patch on background
		return sigma

	"""
	Returns image as a grayscale array, squished down to img_sz x img_sz.
	"""
	def img_to_array(self):
		img = cv2.imread(self.bg_img)					# img is a numpy 2D array
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		resized_img = cv2.resize(img, (self.img_sz, self.img_sz))
		return resized_img

	def generate_spot_image(self):
		bg_array = self.img_to_array()
		spot_list = self.get_spot_list()
		spot_array = self.spot_list_to_spot_array(spot_list)
		spot_img = np.add(bg_array, spot_array)	
		cv2.imwrite("spot_img.png", spot_img)		

	def spot_list_to_spot_array(self, spot_list):
		spot_array = np.zeros([self.img_sz, self.img_sz])
		for spot in spot_list:
			self.add_spot(spot, spot_array)
		return spot_array

	"""
	Inputs:
		spot = array with 3 elems [spot_x, spot_y, patch]
				where patch is a 2D square array
		spot_array = array to which to add the spot
	Return:
		spot_array with patch added with center at spot_x, spot_y
	"""
	def add_spot(self, spot, spot_array):
		spot_x = spot[0]
		spot_y = spot[1]
		patch = spot[2]
		array_origin_x = spot_x - math.floor(self.patch_sz/2)
		array_origin_y = spot_y - math.floor(self.patch_sz/2)
		for row_ind in range(self.patch_sz):
			for col_ind in range(self.patch_sz):
				spot_array[array_origin_y + row_ind][array_origin_x + col_ind] = patch[row_ind][col_ind]
		return spot_array







