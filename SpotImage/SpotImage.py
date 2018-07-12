""" This module contains the SpotImage class.
"""

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage import filters
from skimage.restoration import estimate_sigma

from PIL import ImageEnhance, Image

# ------- #

class SpotImage():

	"""
	The SpotImage tool generates synthetic spot images. 
	The user can parameterize the following features of the image:

	-	background image (bg_img_filename)
	-	color mapping (cmap)
	-	image size (img_sz)
	-	number of spots (num_spots)
	-	spot size and shape (patch_sz, spot_shape_params)
	-	SNR distribution from which to sample SNR of each spot (snr_distr_params)
	-	how much to constrain spots to brighter areas (intensity_threshold)

	Implementation:
	->	The background image is scaled and color-mapped in a 2D pixel array.
	->	The tool generates a list of spots. 
			Each spot has a random location.
			Each spot has an SNR (sampled from the SNR distribution) 
	->	The list of spots is converted to a 2D pixel array of spots.
			Each spot is represented as a patch: a 2D pixel array with one spot.
			The patches consolidated into one 2D pixel array of spots
	->	The 2D pixel array of spots is added to the background array to produce the final image.

	"""

	spot_shapes = ['2D_Gauss']	# list of spot shapes handled
	snr_distrs = ['Gauss']		# list of SNR distributions handled

	"""
	Constructor
	"""
	def __init__(self, bg_img_filename, cmap, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params, intensity_threshold):

		if (spot_shape_params[0] not in self.spot_shapes):
			raise ValueError('Invalid spot shape name entered.')
		if (snr_distr_params[0] not in self.snr_distrs):
			raise ValueError('Invalid SNR distribution name entered.')
		if (patch_sz > img_sz):
			raise ValueError('Patch size is greater than image size.')
		self.bg_img_filename = bg_img_filename
		self.cmap = cmap
		self.img_sz = img_sz
		self.patch_sz = patch_sz
		self.num_spots = num_spots
		self.spot_shape_params = spot_shape_params
		self.snr_distr_params = snr_distr_params

		self.margin = math.floor(self.patch_sz/2)			# setting margin such that no patches hang off the edges
		self.bg_array = self.img_to_array(bg_img_filename)
		self.threshold = filters.threshold_otsu(self.bg_array) + intensity_threshold
		self.valid_coords = self.get_valid_coords()			# set of coordinates where beads may be placed

	"""
	Generate a spot image.
	The spot_array and spot_img are saved as attributes of the SpotImage object
	for later access.
	"""
	def generate_spot_image(self, plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename):
		self.spot_array = self.generate_spot_array()
		self.spot_img = np.add(self.bg_array, self.spot_array)

		if plot_spots:	
			plt.imshow(self.spot_array, cmap = self.cmap)
			plt.title("".join(self.bg_img_filename.rsplit(self.bg_img_filename[-4:])) + "_nspots" + str(self.num_spots) + "_spot_sig" + str(self.spot_shape_params[1]) + "_snr" + str(self.snr_distr_params[1]) + "_" + str(self.snr_distr_params[2]) + "_spot_array")
			plt.show()
		if plot_img:
			plt.imshow(self.spot_img, cmap = self.cmap)
			plt.title("".join(self.bg_img_filename.rsplit(self.bg_img_filename[-4:])) + "_nspots" + str(self.num_spots) + "_spot_sig" + str(self.spot_shape_params[1]) + "_snr" + str(self.snr_distr_params[1]) + "_" + str(self.snr_distr_params[2]) + "_spot_img")
			plt.show()
		if save_spots:
			cv2.imwrite(spots_filename, self.spot_array)
		if save_img:
			cv2.imwrite(spot_img_filename, self.spot_img)

	# def generate_spot_image(self, plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename):
	# 	self.spot_array = self.generate_spot_array()
	# 	self.spot_img = np.add(self.bg_array, self.spot_array)

	# 	if plot_spots:	
	# 		plt.imshow(self.spot_array, cmap = self.cmap)
	# 		plt.title("".join(self.bg_img_filename.rsplit(self.bg_img_filename[-4:])) + "_nspots" + str(self.num_spots) + "_spot_sig" + str(self.spot_shape_params[1]) + "_snr" + str(self.snr_distr_params[1]) + "_" + str(self.snr_distr_params[2]) + "_spot_array")
	# 		plt.show()
	# 	if plot_img:
	# 		plt.imshow(self.spot_img, cmap = self.cmap)
	# 		plt.title("".join(self.bg_img_filename.rsplit(self.bg_img_filename[-4:])) + "_nspots" + str(self.num_spots) + "_spot_sig" + str(self.spot_shape_params[1]) + "_snr" + str(self.snr_distr_params[1]) + "_" + str(self.snr_distr_params[2]) + "_spot_img")
	# 		plt.show()

		# hist,bins = np.histogram(self.spot_img.flatten(),256,[0,256])
		# cdf = hist.cumsum()
		# cdf_m = np.ma.masked_equal(cdf,0)
		# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
		# cdf = np.ma.filled(cdf_m,0).astype('uint8')
		# print(cdf)

		# if save_spots:
		# 	spot_array_tr = cdf[self.spot_array]
		# 	cv2.imwrite(spots_filename, spot_array_tr)
		# if save_img:
		# 	spot_img_tr = cdf[self.spot_img]
		# 	cv2.imwrite(spot_img_filename, self.spot_img)
		# if save_spots:
		# 	cv2.imwrite(spots_filename, self.spot_array)
		# if save_img:
		# 	cv2.imwrite(spot_img_filename, self.spot_img)
		# 	image = Image.open(spot_img_filename)
		# 	contrast = ImageEnhance.Contrast(image)
		# 	for i in range (10):
		# 		val = i/2
		# 		contrast.enhance(val).save(spot_img_filename + str(val) + '.tif',"TIF")

	"""
	Save csv files of spot image data for later reference
	as ground truth values.
	"""
	def get_spot_array_csv(self, csv_filename):
		np.savetxt(csv_filename, self.spot_array, delimiter=",")

	def get_spot_img_csv(self, csv_filename):
		np.savetxt(csv_filename, self.spot_img, delimiter=",")

	def get_coord_snr_list_csv(self, csv_filename):
		coord_snr_list = [None]*self.num_spots
		for i in range(self.num_spots):
			spot = [self.coord_list[i][0], self.coord_list[i][1], self.snr_list[i]]
			coord_snr_list[i] = spot
		np.savetxt(csv_filename, coord_snr_list, delimiter=",")

	"""
	Returns an image as an array of gray values, squished down to img_sz x img_sz.
	"""
	def img_to_array(self, img_filename):

		# image = Image.open(img_filename)
		# contrast = ImageEnhance.Contrast(image)
		
		# pil_img = contrast.enhance(5)
		# img = np.array(pil_img)

		img = cv2.imread(img_filename)					# img is a numpy 2D array
		img_cvt = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)	
		resized_img = cv2.resize(img_cvt, (self.img_sz, self.img_sz))
		print()
		return resized_img	

	"""
	Returns the set of coordinates where spots may be added.
	Coordinates below self.threshold are excluded from this set.
	"""
	def get_valid_coords(self):
		valid_coords = []
		# valid_array = np.zeros([self.img_sz, self.img_sz])		# for visualizing the valid coordinates
		for row_ind in range(self.margin, self.img_sz - self.margin):
			for col_ind in range(self.margin, self.img_sz - self.margin):
				if (self.bg_array[row_ind][col_ind] >= self.threshold):
					valid_coords.append([col_ind,row_ind])
		# 			valid_array[row_ind][col_ind] = 1				
		# plt.imshow(valid_array, cmap = self.cmap)
		# plt.show()
		return(valid_coords)	

	"""
	Generate a list of random spots. 
	Each spot has a random location and a patch of intensity values.
	"""
	def get_spot_list(self):
		self.coord_list = [self.get_spot_coord() for i in range(self.num_spots)]
		self.snr_list = [self.get_snr() for i in range(self.num_spots)]
		spot_list = [[self.coord_list[i], self.get_patch(self.coord_list[i][0], self.coord_list[i][1], self.snr_list[i])] for i in range(self.num_spots)]
		return spot_list

	"""
	Select a random spot coordinate from the list of valid spots.
	"""
	def get_spot_coord(self):
		return random.choice(self.valid_coords)

	"""
	Generate one 2D square array with one spot.
		The spot obeys spot_shape_params.
		The spot has an SNR sampled from the SNR distribution.
	"""
	def get_patch(self, x, y, snr):
		patch = np.zeros([self.patch_sz, self.patch_sz])
		sigma = self.get_noise(x,y)				# get sigma corresp. to noise at equiv. patch on background
		max_intensity = snr*sigma
		x_0 = y_0 = math.floor(self.patch_sz/2)
		if (self.spot_shape_params[0] == '2D_Gauss'):
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
					value = max_intensity*np.exp(-exp_quantity)		
					if (value < 0):				
						value = 0
					patch[i][j] = value
		return patch

	"""
	Sample an SNR from the specified SNR distribution.
	"""
	def get_snr(self):
		if (self.snr_distr_params[0] == 'Gauss'):
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
				patch[row][col] = self.bg_array[origin_y + row][origin_x + col]
		sigma = estimate_sigma(patch, multichannel=True, average_sigmas=True)			# get noise from equiv. patch on background
		return sigma

	"""
	Returns spot_array generated from spot_list.
	"""
	def generate_spot_array(self):
		spot_array = np.zeros([self.img_sz, self.img_sz])
		spot_list = self.get_spot_list()
		for spot in spot_list:
			self.add_spot(spot, spot_array)
		return spot_array

	"""
	Adds one spot to spot_array.
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