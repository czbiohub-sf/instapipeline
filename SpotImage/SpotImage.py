""" This module contains the SpotImage class.
"""

import cv2, math, random
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from skimage.restoration import estimate_sigma
from sklearn.neighbors import KDTree
from matplotlib.lines import Line2D

# ------- #

class SpotImage():
	"""
	The SpotImage tool generates synthetic spot images. 
	The user can parameterize the following features of the image:

	-	path to background image (bg_img_path)
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
	def __init__(self, bg_img_path=None, cmap='gray', img_sz=300, patch_sz=11, spot_shape_params=['2D_Gauss', 1.75], global_intensity_dial=0):
		if bg_img_path is None:
			raise ValueError('Path to background image required.')
		if patch_sz > img_sz:
			raise ValueError('Patch size is greater than image size.')
		if spot_shape_params[0] not in self.spot_shapes:
			raise ValueError('Invalid spot shape name entered.')
		if (len(spot_shape_params) < 2):
			raise ValueError('Spot sigma required for 2D Gaussian spot shape.')

		self.bg_img_path = bg_img_path
		self.cmap = cmap
		self.img_sz = img_sz
		self.patch_sz = patch_sz
		self.spot_shape_params = spot_shape_params
		self.margin = math.floor(self.patch_sz/2)				# setting the margin such that no patches hang off the edges
		self.bg_array = self.img_to_array(bg_img_path)
		self.threshold = filters.threshold_otsu(self.bg_array)
		self.global_intensity_dial = global_intensity_dial		# adds to self.threshold
		self.valid_coords = self.get_valid_coords()				# set of coordinates where spots may be placed
		self.total_coord_list = [self.get_spot_coord() for i in range(self.increment)]

	"""
	Returns an image as an array of gray values, squished down to img_sz x img_sz.
	"""
	def img_to_array(self, img_filename):
		img = cv2.imread(img_filename)					# img is a numpy 2D array
		img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)	
		return cv2.resize(img_gray, (self.img_sz, self.img_sz))

	"""
	Returns the set of coordinates where spots may be added.
	Coordinates below self.threshold are excluded from this set.
	"""
	def get_valid_coords(self):
		valid_coords = []
		for row_ind in range(self.margin, self.img_sz - self.margin):
			for col_ind in range(self.margin, self.img_sz - self.margin):
				if (self.bg_array[row_ind][col_ind] >= self.threshold + self.global_intensity_dial):
					valid_coords.append([col_ind,row_ind])
		return valid_coords

	"""
	Select a random spot coordinate from the list of valid spots.
	"""
	def get_spot_coord(self):
		return random.choice(self.valid_coords)

	"""
	Generate a spot image.
	"""
	def generate_spot_image(self, num_spots=None, density=None, snr_distr_params=['Gauss', 10, 2], snr_threshold=3, plot_spots=False, plot_img=False, save_spots=False, save_img=False, spots_filename=None, spot_img_filename=None):
		
		# Step 1: Check that snr distribution params are valid

		if snr_distr_params[0] not in self.snr_distrs:
			raise ValueError('Invalid SNR distribution name entered.')
		self.snr_distr_params = snr_distr_params
		self.snr_threshold = snr_threshold

		# Step 2: Set self.coord_list

		if num_spots is None and density is None:
			raise ValueError('Specify num_spots and/or density.')

		elif num_spots is not None and density is not None:
			self.num_spots = math.floor(density * len(self.valid_coords))

			# constrict valid region to get a good ballpart number of spots
			while(self.num_spots > num_spots + 75):								# to do: make magic number global variable
				self.global_intensity_dial += 0.05								# to do: make magic number global variable
				self.valid_coords = self.get_valid_coords()
				self.num_spots = math.floor(density * len(self.valid_coords))
			self.density = density

			# get list of all coordinates being held right now
			coords = [self.total_coord_list[i] for i in range(self.num_spots)]

			# get NNDs of all coordinates being held right now
			coords_kdt = KDTree(coords, leaf_size=2, metric='euclidean')
			NND_list = []
			for coord in coords:
				coord = [coord]
				dist, ind = coords_kdt.query(coord, k=2)
				NND_list.append(dist[0][1])

			# remove the spots with the largest NNDs
			num_to_remove = self.num_spots - num_spots

				# Source: https://stackoverflow.com/questions/16878715/how-to-find-the-index-of-n-largest-elements-in-a-list-or-np-array-python?lq=1
			ordered_indices = sorted(range(len(NND_list)), key=lambda x: NND_list[x])
			indices_to_remove = sorted(range(len(NND_list)), key=lambda x: NND_list[x])[-num_to_remove:]

			coords_keeping = []
			for i in range(len(coords)):
				if i not in indices_to_remove:
					coords_keeping.append(coords[i])

			self.coord_list = coords_keeping
			self.num_spots = num_spots

		else: # if (num_spots is not None and density is None) or (num_spots is None and density is not None)

			# if num_spots is not None and density is None
			if num_spots is not None:	
				self.num_spots = num_spots
				self.density = round(float(self.num_spots)/(len(self.valid_coords)), 3)

			# if num_spots is None and density is not None
			else: 						
				self.num_spots = math.floor(density * len(self.valid_coords))
				self.density = density

			self.coord_list = self.total_coord_list
			while (self.num_spots > len(self.total_coord_list)):
				self.total_coord_list += [self.get_spot_coord() for i in range(self.increment)]

		# Step 3: Set self.spot_list, self.spot_array, and self.spot_img

		self.spot_list = self.generate_spot_list()
		self.spot_array = self.generate_spot_array()
		self.spot_img = np.add(self.bg_array, self.spot_array)

		# Step 4: Plot and save spot image and spot array as desired

		if plot_spots:
			plt.imshow(self.spot_array, cmap=self.cmap)
			plt.title(spots_filename)
			plt.show()
			if save_spots:
				plt.imsave(spots_filename, self.spot_array, cmap=self.cmap)
		if plot_img:
			plt.imshow(self.spot_img, cmap=self.cmap)
			plt.title(spot_img_filename)
			plt.show()
			if save_img:
				plt.imsave(spot_img_filename, self.spot_img, cmap=self.cmap)


	"""
	Generate a list of random spots. 
	Each spot has a random location and a patch of intensity values.
	"""
	def generate_spot_list(self):
		self.snr_list = [self.get_snr() for i in range(len(self.total_coord_list))]
		spot_list = [[self.coord_list[i], self.get_patch(self.coord_list[i], self.snr_list[i])] for i in range(self.num_spots)]
		return spot_list

	"""
	Returns spot_array generated from spot_list.
	"""
	def generate_spot_array(self):
		spot_array = np.zeros([self.img_sz, self.img_sz])
		for i in range(self.num_spots):
			self.add_spot(self.spot_list[i], spot_array)
		return spot_array

	"""
	Sample an SNR from the specified SNR distribution.
	Enforces that the SNR must be at least self.snr_threshold (since there 
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
	Generate one 2D square array with one spot.
		The spot obeys spot_shape_params.
		The spot has an SNR sampled from the SNR distribution.
	"""
	def get_patch(self, coord, snr):
		x = coord[0]
		y = coord[1]
		patch = np.zeros([self.patch_sz, self.patch_sz])
		sigma = self.get_noise(x,y)				# get sigma corresp. to noise at equiv. patch on background
		max_intensity = snr*sigma
		x_0 = y_0 = math.floor(self.patch_sz/2)
		if (self.spot_shape_params[0] == '2D_Gauss'):
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
		return estimate_sigma(patch, multichannel=True, average_sigmas=True)			# get noise from equiv. patch on background

	"""
	Save csv file of spot image data for later reference
	as ground truth values.
	"""
	def get_coord_snr_list_csv(self, csv_filename):
		np.savetxt(csv_filename, self.get_coord_snr_list(), delimiter=',', comments='', header='col,row,snr')

	"""
	Consolidate coordinates and SNR into one list.
	"""
	def get_coord_snr_list(self):
		coord_snr_list = [None]*self.num_spots
		for i in range(self.num_spots):
			coord_snr_list[i] = [self.coord_list[i][0], self.coord_list[i][1], self.snr_list[i]]
		return coord_snr_list

	"""
	Plots a histogram of nearest neighbor distance for each spot.
	"""
	def plot_spot_nnd(self):
		coord_list = []
		for i in range(self.num_spots):
			coord_list.append(self.coord_list[i])

		spots_kdt = KDTree(self.coord_list, leaf_size=2, metric='euclidean')
		NND_list = []
		for coord in coord_list:
			coord = [coord]
			dist, ind = spots_kdt.query(coord, k=2)
			NND_list.append(dist[0][1])

		if max(NND_list) > 50:
			step_size = 10
		elif max(NND_list) > 30:
			step_size = 5
		else:
			step_size = 2

		mean = np.mean(NND_list)
		plt.axvline(x=mean, color='orange')
		label = 'mean NND = ' + str(math.floor(mean))
		plt.legend(handles=[Line2D([0],[0], color='orange', label=label)])
		plt.hist(NND_list, bins=np.arange(0, max(NND_list) + step_size, step_size) - step_size/2)
		plt.title('NND between spots with density = ' + str(self.density) + ', ' + str(self.num_spots) + ' spots')
		plt.xlabel('Nearest Neighbor Distance (NND)')
		plt.ylabel('Number of Points')
		plt.xticks(np.arange(0,max(NND_list)+step_size,step=step_size))
		plt.show()

	"""
	Calculate mean nearest neighbor distance from spot to spot for the image.
	"""	
	def get_mean_spot_nnd(self):
		coord_list = []
		for i in range(self.num_spots):
			coord_list.append(self.coord_list[i])
		spots_kdt = KDTree(coord_list, leaf_size=2, metric='euclidean')
		NND_list = []
		for coord in coord_list:
			dist, ind = spots_kdt.query([coord], k=2)
			NND_list.append(dist[0][1])
		return np.mean(NND_list)

	"""
	Plot the coordinates of the most recently generated spot image.
	"""
	def plot_coords(self):
		fig = plt.figure(figsize=(4,4))
		for i in range(self.num_spots):
			plt.scatter(coord_list[i][0], coord_list[i][1], facecolors='b', s=10)
		plt.axis('equal')
		plt.xlim(0, self.img_sz)
		plt.ylim(0, self.img_sz)
		plt.title('Coordinates')
		plt.show()


