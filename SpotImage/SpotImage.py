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
	spot_index = 1				

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
		x_list = [self.get_spot_x() for i in range(self.num_spots)]
		y_list = [self.get_spot_y() for i in range(self.num_spots)]
		spot_list = [[x_list[i], y_list[i], self.get_patch(x_list[i], y_list[i])] for i in range(self.num_spots)]
		return spot_list

	"""
	Returns one 2D square array with a spot
		Spot obeys spot_shape_params
		Spot has SNR sampled from SNR distr
	"""
	def get_patch(self, x, y):
		patch = np.zeros([self.patch_sz, self.patch_sz])
		snr = self.get_snr()					# get snr from snr distribution
		sigma = self.get_sigma(x,y)				# get sigma corresp. to noise at equiv. patch on background
		max_intensity = snr*sigma
		x_0 = y_0 = math.floor(self.patch_sz/2)

		if (self.spot_shape_params[0] == '2D_Gaussian'):
			spot_sigma = self.spot_shape_params[1]
			for j in range(self.patch_sz):
				for i in range(self.patch_sz):
					x_dist = i - x_0
					y_dist = j - y_0
					exp_num = x_dist**2 + y_dist**2
					exp_den = 2*(spot_sigma**2)
					exp_quantity = exp_num/exp_den
					patch[i][j] = max_intensity*np.exp(-exp_quantity)

		cv2.imwrite("patch.png", patch)

		print("spot", self.spot_index, "/", self.num_spots)
		self.spot_index = self.spot_index + 1

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

	"""
	Returns spot_array generated from spot_list.
	"""
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

	def generate_spot_image(self):
		print("Generating...")
		bg_array = self.img_to_array()
		spot_list = self.get_spot_list()
		spot_array = self.spot_list_to_spot_array(spot_list)
		cv2.imwrite("spot_array.png", spot_array)				# for debugging
		spot_img = np.add(bg_array, spot_array)	
		cv2.imwrite("spot_img.png", spot_img)	







