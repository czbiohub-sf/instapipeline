""" This module contains the SpotImage class.
"""

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma

# ------- #

class SpotImage():

	"""
	The SpotImage tool generates synthetic images for workers to annotate. 
	The user can parameterize the following features of the image:

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

	spot_shapes = ['2D_Gaussian']	# list of spot shapes handled
	snr_distrs = ['Gaussian']		# list of SNR distributions handled
	spot_index = 1					# initialize the counter for the console output progress readout

	"""
	Constructor
	"""
	def __init__(self, bg_img, cmap, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params):
		if (spot_shape_params[0] not in self.spot_shapes):
			raise ValueError('Invalid spot shape name entered.')
		if (snr_distr_params[0] not in self.snr_distrs):
			raise ValueError('Invalid SNR distribution name entered.')
		if (patch_sz > img_sz):
			raise ValueError('Patch size is greater than image size.')
		self.bg_img = bg_img
		self.cmap = cmap
		self.img_sz = img_sz
		self.patch_sz = patch_sz
		self.num_spots = num_spots
		self.spot_shape_params = spot_shape_params
		self.snr_distr_params = snr_distr_params
		self.margin = math.floor(self.patch_sz/2)		# setting margin such that no patches hang off the edges

	"""
	Generate a spot image.
	"""
	def generate_spot_image(self, plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename, show_progress):
		print("Generating...")
		bg_array = self.img_to_array()
		if show_progress:
			self.show_progress = True
		else:
			self.show_progress = False
		spot_list = self.get_spot_list()
		spot_array = self.spot_list_to_spot_array(spot_list)
		spot_img = np.add(bg_array, spot_array)	
		if plot_spots:	
			plt.imshow(spot_array, cmap = self.cmap)
			plt.show()
		if plot_img:
			plt.imshow(spot_img, cmap = self.cmap)
			plt.show()
		if save_spots:
			cv2.imwrite(spots_filename, spot_array)
		if save_img:
			cv2.imwrite(spot_img_filename, spot_img)	

	"""
	Returns an image as a grayscale array, squished down to img_sz x img_sz.
	"""
	def img_to_array(self):
		img = cv2.imread(self.bg_img)					# img is a numpy 2D array
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		resized_img = cv2.resize(img, (self.img_sz, self.img_sz))
		return resized_img	

	"""
	Generate a list of random spots. 
	Each spot has a random location and a patch of intensity values.
	"""
	def get_spot_list(self):

		# Generate a random list of num_spots coordinates
		coord_list = [self.get_spot_coord() for i in range(self.num_spots)]

		spot_list = [[coord_list[i], self.get_patch(coord_list[i][0], coord_list[i][1])] for i in range(self.num_spots)]
		return spot_list

	def get_spot_coord(self):
		rand_x = np.random.random_integers(self.margin, self.img_sz - self.margin - 1)
		rand_y = np.random.random_integers(self.margin, self.img_sz - self.margin - 1)
		return [rand_x, rand_y]

	"""
	Generate one 2D square array with one spot.
		The spot obeys spot_shape_params.
		The spot has an SNR sampled from the SNR distribution.
	"""
	def get_patch(self, x, y):
		patch = np.zeros([self.patch_sz, self.patch_sz])
		snr = self.get_snr()					# get snr from snr distribution
		sigma = self.get_noise(x,y)				# get sigma corresp. to noise at equiv. patch on background
		max_intensity = snr*sigma
		x_0 = y_0 = math.floor(self.patch_sz/2)
		if (self.spot_shape_params[0] == '2D_Gaussian'):
			if (len(self.spot_shape_params) < 2):
				raise ValueError('Spot sigma required for 2D Gaussian spot shape.')
			spot_sigma = self.spot_shape_params[1]
			for j in range(self.patch_sz):
				for i in range(self.patch_sz):
					x_dist = i - x_0
					y_dist = j - y_0
					exp_num = x_dist**2 + y_dist**2
					exp_den = 2*(spot_sigma**2)
					exp_quantity = exp_num/exp_den
					patch[i][j] = max_intensity*np.exp(-exp_quantity)
		if self.show_progress:
			print("spot", self.spot_index, "/", self.num_spots)
		self.spot_index = self.spot_index + 1
		return patch

	"""
	Sample an SNR from the specified distribution.
	"""
	def get_snr(self):
		if (self.snr_distr_params[0] == 'Gaussian'):
			if (len(self.snr_distr_params) < 3):
				raise ValueError('Mu and sigma required for Gaussian SNR distribution.')
			mu = self.snr_distr_params[1]
			sigma = self.snr_distr_params[2]
			snr = np.random.normal(mu, sigma)
		return snr

	"""
	Get a noise (sigma) value from a square patch on the background 
	of size patch_sz and centered on (x,y).
	"""
	def get_noise(self, x, y):
		origin_x = x - self.margin
		origin_y = y - self.margin
		patch = np.zeros([self.patch_sz, self.patch_sz])
		for row in range(self.patch_sz):
			for col in range(self.patch_sz):
				patch[row][col] = self.img_to_array()[origin_y + row][origin_x + col]
		sigma = estimate_sigma(patch, multichannel=True, average_sigmas=True)			# get noise from equiv. patch on background
		return sigma

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
		spot_x = spot[0][0]
		spot_y = spot[0][1]
		patch = spot[1]
		array_origin_x = spot_x - math.floor(self.patch_sz/2)
		array_origin_y = spot_y - math.floor(self.patch_sz/2)
		for row_ind in range(self.patch_sz):
			for col_ind in range(self.patch_sz):
				spot_array_val = spot_array[array_origin_y + row_ind][array_origin_x + col_ind]		# the pre-existing value at that location in spot_array
				spot_array[array_origin_y + row_ind][array_origin_x + col_ind] = spot_array_val + patch[row_ind][col_ind]
		return spot_array

