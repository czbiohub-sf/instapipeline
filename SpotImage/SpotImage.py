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
from sklearn.neighbors import KDTree
from matplotlib.lines import Line2D


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
	-	minimum SNR of a spot (snr_threshold)
	-	how much to constrain spots to brighter areas (bg_intensity_threshold)

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
	increment = 500				# quantity by which to increment the total coordinate list

	"""
	Constructor
	"""
	def __init__(self, bg_img_filename, cmap, img_sz, patch_sz, spot_shape_params, brightness_bias, brightness_bias_dial, biasing_method, global_intensity_dial):

		if (spot_shape_params[0] not in self.spot_shapes):
			raise ValueError('Invalid spot shape name entered.')
		if (patch_sz > img_sz):
			raise ValueError('Patch size is greater than image size.')
		self.bg_img_filename = bg_img_filename
		self.cmap = cmap
		self.img_sz = img_sz
		self.patch_sz = patch_sz
		self.spot_shape_params = spot_shape_params

		# sole purpose of these parameters is to determine what goes into self.total_coord_list 
		self.brightness_bias = brightness_bias
		self.brightness_bias_dial = brightness_bias_dial
		self.biasing_method = biasing_method
		self.global_intensity_dial = global_intensity_dial

		self.margin = math.floor(self.patch_sz/2)				# setting margin such that no patches hang off the edges
		self.bg_array = self.img_to_array(bg_img_filename)
		self.min_bg_intensity = np.amin(self.bg_array)
		self.max_bg_intensity = np.amax(self.bg_array)
		self.threshold = filters.threshold_otsu(self.bg_array)
		self.valid_coords = self.get_valid_coords()				# set of coordinates where beads may be placed

		self.spot_counter = 0
		self.total_coord_list = [self.get_spot_coord() for i in range(self.increment)]

	"""
	Generate a spot image.
	The spot_array and spot_img are saved as attributes of the SpotImage object
	for later access.
	"""
	def generate_spot_image(self, num_spots, snr_distr_params, snr_threshold, plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename, density):
		
		if (snr_distr_params[0] not in self.snr_distrs):
			raise ValueError('Invalid SNR distribution name entered.')

		# assign class attributes that determine what goes in self.coord_list
		self.num_spots = num_spots
		if (density != None):
			self.num_spots = math.floor(density * len(self.valid_coords))
			self.density = density

		# assign class attributes that determine what goes in self.snr_list
		self.snr_distr_params = snr_distr_params
		self.snr_threshold = snr_threshold

		# assign class attributes: spot_list, coord_list, snr_list, spot_array, spot_img
		self.spot_list = self.generate_spot_list()			# generate_spot_list also updates self.coord_list and self.snr_list
		self.spot_array = self.generate_spot_array()
		self.spot_img = np.add(self.bg_array, self.spot_array)

		if plot_spots:	
			plt.imshow(self.spot_array, cmap = self.cmap)
			plt.title(spots_filename)
			plt.show()
			if save_spots:
				plt.imsave(spots_filename, self.spot_array, cmap = self.cmap)
		if plot_img:
			plt.imshow(self.spot_img, cmap = self.cmap)
			plt.title(spot_img_filename)
			plt.show()
			if save_img:
				plt.imsave(spot_img_filename, self.spot_img, cmap = self.cmap)

	def get_coord_snr_list(self):
		coord_snr_list = [None]*self.num_spots
		for i in range(self.num_spots):
			spot = [self.coord_list[i][0], self.coord_list[i][1], self.snr_list[i]]
			coord_snr_list[i] = spot
		return coord_snr_list

	def plot_coords(self):
		fig = plt.figure(figsize=(4,4))
		for coord in self.coord_list:
			plt.scatter(coord[0], coord[1], facecolors = 'b', s = 10)
		plt.axis('equal')
		plt.xlim(0, self.img_sz)
		plt.ylim(0, self.img_sz)
		plt.title('Coordinates')
		plt.show()

	"""
	Save csv file of spot image data for later reference
	as ground truth values.
	"""
	def get_coord_snr_list_csv(self, csv_filename):
		np.savetxt(csv_filename, self.get_coord_snr_list(), delimiter=",", comments='', header = "col,row,snr")

	"""
	Returns an image as an array of gray values, squished down to img_sz x img_sz.
	"""
	def img_to_array(self, img_filename):
		img = cv2.imread(img_filename)					# img is a numpy 2D array
		img_cvt = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)	
		resized_img = cv2.resize(img_cvt, (self.img_sz, self.img_sz))
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
				if (self.bg_array[row_ind][col_ind] >= self.threshold + self.global_intensity_dial):
					valid_coords.append([col_ind,row_ind])
		# 			valid_array[row_ind][col_ind] = 1				
		# plt.imshow(valid_array, cmap = self.cmap)
		# plt.show()
		return(valid_coords)	

	"""
	Generate a list of random spots. 
	Each spot has a random location and a patch of intensity values.
	"""
	def generate_spot_list(self):
		self.coord_list = self.total_coord_list
		while (self.num_spots > len(self.total_coord_list)):
			self.total_coord_list += [self.get_spot_coord() for i in range(self.increment)]
		self.snr_list = [self.get_snr() for i in range(len(self.total_coord_list))]
		spot_list = [[self.coord_list[i], self.get_patch(self.coord_list[i][0], self.coord_list[i][1], self.snr_list[i])] for i in range(self.num_spots)]
		return spot_list

	"""
	Select a random spot coordinate from the list of valid spots.
	"""
	def get_spot_coord(self):
		coord = random.choice(self.valid_coords)

		if self.brightness_bias:
			"""
			Method 1:
				While the background intensity is less than a randomly generated number
				between backgound_intensity_min and backgound_intensity_max + brightness_bias_dial, 
				reassign the coordinate. 
			"""
			if (self.biasing_method == 1):
				bg_intensity = self.bg_array[coord[0], coord[1]] 
				while(bg_intensity < random.randint(self.min_bg_intensity, self.max_bg_intensity + self.brightness_bias_dial)):
					coord = random.choice(self.valid_coords)
					bg_intensities = []
					for i in range(-1,2):
						for j in range(-1,2):
							bg_intensities.append(self.bg_array[coord[0]+i, coord[1]+j])
					bg_intensity = np.median(bg_intensities)
			
			"""
			Method 2: Threshold Dilation
				If the intensity of the kernel on the background is closer to the 
				validity threshold than the max background intensity, the coordinate
				is accepted 1/(brightness_bias_dial) of the time and reassigned the
				rest of the time.
			"""
			if (self.biasing_method == 2):
				bg_intensities = []
				for i in range(-1,2):
					for j in range(-1,2):
						bg_intensities.append(self.bg_array[coord[0]+i, coord[1]+j])
				bg_intensity = np.median(bg_intensities)
				while (bg_intensity < ((self.threshold + self.max_bg_intensity)/2)):
					if (random.randint(0,self.brightness_bias_dial) != 1):		# a fraction of the time
						coord = random.choice(self.valid_coords)
						bg_intensities = []
						for i in range(-1,2):
							for j in range(-1,2):
								bg_intensities.append(self.bg_array[coord[0]+i, coord[1]+j])
						bg_intensity = np.median(bg_intensities)
					else:
						break

			"""
			Method 3: 
				Increase global intensity dial for fraction (2/3) of all spots.
			"""
			if (self.biasing_method == 3):
				if (self.spot_counter > (self.num_spots * (2/3))):
					while (self.bg_array[coord[0],coord[1]] < (self.threshold + self.global_intensity_dial)):
						coord = random.choice(self.valid_coords)
						# if(coord[0] < 200):
						# 	coord[0] += 100
						# 	print(self.spot_counter)
						# 	print(coord)

		self.spot_counter += 1
		return coord

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
	Enforces that the SNR must be at least 3 (since there 
	is no reason to expect people to detect a signal with SNR < 3).
	"""
	def get_snr(self):
		if (self.snr_distr_params[0] == 'Gauss'):
			if (len(self.snr_distr_params) < 3):
				raise ValueError('Mu and sigma required for Gaussian SNR distribution.')
			mu = self.snr_distr_params[1]
			sigma = self.snr_distr_params[2]
			snr = np.random.normal(mu, sigma)
			if (snr < self.snr_threshold):
				snr = self.snr_threshold
		return snr

	"""
	Get a noise (sigma) value from a square patch on the background 
	of size patch_sz and centered on (x,y).
	"""
	def get_noise(self, x, y):
		origin_x = x - self.margin - 1
		origin_y = y - self.margin - 1
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
		for i in range(self.num_spots):
			self.add_spot(self.spot_list[i], spot_array)
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

	def plot_spot_nnd(self):
		coord_list = []
		for i in range(self.num_spots):
			coord_list.append(self.coord_list[i])

		spots_kdt = KDTree(coord_list, leaf_size=2, metric='euclidean')

		NND_list = []
		for coord in coord_list:
			coord = [coord]
			dist, ind = spots_kdt.query(coord, k=2)
			NND_list.append(dist[0][1])

		step_size = 2
		if(max(NND_list) > 30):
			step_size = 5

		plt.hist(NND_list, bins = np.arange(0,max(NND_list)+step_size,step=step_size)-step_size/2)
		plt.title("NND between spots with density = " + str(self.density) + ", " + str(self.num_spots) + " spots")
		plt.xlabel("Nearest Neighbor Distance (NND)")
		plt.ylabel("Number of Points")
		plt.xticks(np.arange(0,max(NND_list)+step_size,step=step_size))

		mean = math.floor(np.mean(NND_list))

		plt.axvline(x=mean, color='orange')
		label = "mean NND = " + str(mean)
		mean_line = Line2D([0],[0], color='orange', label=label)
		plt.legend(handles=[mean_line])

		plt.show()