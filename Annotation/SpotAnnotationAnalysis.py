""" This module contains the SpotAnnotationAnalysis class.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import genfromtxt
import pandas as pd
import scipy
import sklearn as skl
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

from sklearn.neighbors import KDTree

from QuantiusAnnotation import QuantiusAnnotation
from BaseAnnotation import BaseAnnotation

# ------- #

class SpotAnnotationAnalysis():
	""" The SpotAnnotationAnalysis class provides tools for
	annotation analysis.
	SpotAnnotationAnalysis takes a BaseAnnotation object in as an 
	input and saves it as a property of the class.
	"""

	"""
	constructor takes in a BaseAnnotation object and saves it as 
	a property of the SpotAnnotationAnalysis class
	"""
	def __init__(self, ba_obj):
		self.ba = ba_obj

	# list of clustering algs handled
	clustering_algs = ['AffinityPropagation']

	# list of colors used for plotting different turkers
	colors = ['#14FF14', '#00994D', '#CC33FF', '#FF33CC', 
    '#33CCFF', '#009999', '#FF3399', '#FF3366', 
    '#33FFCC', '#B88A00', '#F5B800', '#FF6633',
    '#33FF66', '#66FF33', '#CCFF33', '#FFCC33',
    '#EB4E00', '#FF667F', '#FF66CC', '#9966FF', 
    '#CCFF66', '#667FFF', '#FF99FF', '#FF1F8F',
    '#9999FF', '#99FFCC', '#FF9999', '#E5FFFF',
    '#8A00B8', '#E5FFFF']

	# inputs: 
	# 	string name of clustering alg to use
	# 	dataframe
	# output:
	# 	array containing coordinates of cluster centers (column 0 is x vals, column 1 is y vals)
	def get_clusters(self, clustering_alg, df, clustering_params):
		if (clustering_alg not in self.clustering_algs):
			raise ValueError('Invalid clustering algorithm name entered.')
		# If AffinityPropagation is selected:
		if (clustering_alg == 'AffinityPropagation'):
			cluster_centers_list = []

			if(len(clustering_params) != 1):
				raise ValueError('Please enter a list containing the preference parameter.')

			coords = self.ba.get_coords(df)
			af = AffinityPropagation(preference = clustering_params[0]).fit(coords)
			cluster_centers_indices = af.cluster_centers_indices_
			n_clusters_ = len(cluster_centers_indices)
			for k in range(n_clusters_):
				cluster_centers = coords[cluster_centers_indices[k]]	# np array
				cluster_centers_list.append(cluster_centers)
		""" Convert list of cluster centers (each of which is a 2-element, 1D np array) to a 2D np array for convenience """
		to_return = np.empty([len(cluster_centers_list), 2], dtype = float)
		for i in range(len(cluster_centers_list)):
			to_return[i][0] = cluster_centers_list[i][0]
			to_return[i][1] = cluster_centers_list[i][1]
		return to_return

	""" 
	Input:
		string name of csv file containing reference points, aka "ground truth" values
	Output:
		k-d tree containing the same reference points
	"""
	def csv_to_kdt(self, csv_filename):

		ref_anno = pd.read_csv(csv_filename)
		ref_points = ref_anno.loc[:, ['row', 'col']].as_matrix() - 1

		ref_kdt = KDTree(ref_points, leaf_size=2, metric='euclidean')	# kdt is a kd tree with all the reference points

		return ref_kdt

	""" 
	Inputs:
		dataframe
		k-d tree with reference points, aka "ground truth" values
		img_filename to crop to
	Output:
		List containing one list for each worker.
			Each list is comprised of, for each of the worker's
			points, the distance to the nearest neighbor (found in
			the k-d tree of references).
	"""
	def calc_distances(self, df, ref_kdt, img_filename):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)

		to_return = []

		for worker in worker_list:			
			anno = self.ba.slice_by_worker(anno_one_crop, worker)	
			coords = self.ba.get_coords(anno)

			dist, ind = ref_kdt.query(coords, k=1)
			dist_list = dist.tolist()
			values = []

			for i in range(len(dist_list)):
				values.append(dist_list[i][0])

			to_return.append(values)

		return to_return

	"""
	Inputs:
		dataframe
		img_filename
	Output:
		list containing one list for each worker with time spent on each click
	Notes:
		Time spent on click_k = timestamp_k - timestamp_(k-1)
		Starts at index 1 of returned list. (Length of times_spent is same as length of timestamps.)
		Will not get a time spent on the first click
	"""
	def calc_time_per_click(self, df, img_filename):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)

		to_return = []

		for worker in worker_list:
			timestamps = self.ba.get_timestamps(anno_one_crop, worker)
			times_spent = [None]*len(timestamps)
			for i in range (1,len(timestamps)):
				times_spent[i] = timestamps[i] - timestamps[i-1]
			to_return.append(times_spent)

		return to_return

	# Plots all coordinates for all workers for one cropping. 
	# Inputs: 
	# 	pandas df (can contain many different croppings)
	# 		(I am thinking taking in a df gives more versatility than taking in a QuantiusAnnotation object.)
	# 	string img_filename (the only cropping to include)
	# 	int size of worker marker
	# 	int size of cluster centroid marker
	# 	bool whether to plot workers
	# 	bool whether to plot cluster centroids
	# 	string name of clustering algorithm
	# 	list of clustering parameters
	def plot_annotations(self, df, img_filename, worker_marker_size, cluster_marker_size, show_workers, show_clusters, clustering_alg, clustering_params):
		# fig = plt.figure(figsize=(14,12))		# for jupyter notebook
		fig = plt.figure(figsize = (12,7))
		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)

		if show_workers:
			handle_list = []
			for worker, color in zip(worker_list, self.colors):			# For each worker, use a different color.
			    anno = self.ba.slice_by_worker(anno_one_crop, worker)		
			    coords = self.ba.get_coords(anno)
			    x_coords = coords[:,0]
			    y_coords = coords[:,1]
			    y_coords_flipped = self.ba.flip(y_coords, 300)
			    handle = plt.scatter(x_coords, y_coords_flipped, s = worker_marker_size, facecolors = color, alpha = 0.5, label = worker)
			    handle_list.append(handle)
			plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.2, 1.015))
			if not show_clusters:
				plt.title('Worker Annotations')

		if show_clusters:
			cluster_centers = self.get_clusters(clustering_alg, anno_one_crop, clustering_params) # cluster_centers is a list of np arrays
			x_coords = cluster_centers[:,0]
			y_coords = cluster_centers[:,1]
			y_coords_flipped = self.ba.flip(y_coords, 300)
			plt.scatter(x_coords, y_coords_flipped, s = cluster_marker_size, facecolors = 'none', edgecolors = '#ffffff')
			if not show_workers:
				plt.title('Cluster Centroids')

		if show_workers and show_clusters:
			plt.title('Worker Annotations and Cluster Centroids')

		img = mpimg.imread(img_filename)
		plt.imshow(img, cmap = 'gray')

		plt.show()

	"""
	Plots the average time spent per click for all workers 
	in the dataframe.

	Input:
		dataframe
	Output:
		none
	"""
	def plot_avg_time_per_click(self, df):
		avg_list = []
		for worker in self.ba.get_workers(df):
			avg_time = self.ba.get_avg_time_per_click(df, worker)
			avg_list.append(avg_time)
		n_bins = 10
		# fig = plt.figure(figsize=(14,12))		# for jupyter notebook
		fig = plt.figure(figsize = (12,7))
		plt.hist(avg_list, n_bins)
		plt.title('Average time spent per click')
		plt.xlabel('Time [units]')
		plt.ylabel('Quantity of workers')
		plt.show()

	"""
	For each annotation (each click) in a dataframe, 
	plot nearest neighbor distance (nnd) vs. time spent. 
	Each point represents one annotation (one click). 
	All workers on one plot, colored by worker ID. 

	Inputs:
		dataframe
		img_filename (the cropping)
		csv_filename (contains reference data)
	Output:
		none
	"""
	def plot_nnd_vs_time_spent(self, df, img_filename, csv_filename):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)
		ref_kdt = self.csv_to_kdt(csv_filename)
		dist_list = self.calc_distances(anno_one_crop, ref_kdt, img_filename)	# list containing one list for each worker
		time_list = self.calc_time_per_click(anno_one_crop, img_filename)	# list containing one list for each worker

		# fig = plt.figure(figsize=(14,12))		# for jupyter notebook
		fig = plt.figure(figsize = (10,7))

		handle_list = []
		for i in range(len(worker_list)):			# for each worker
			color = self.colors[i]
			x_coords = time_list[i]
			y_coords = dist_list[i]
			handle = plt.scatter(x_coords, y_coords, s = 8, facecolors = color, alpha = 0.5, label = worker_list[i])
			handle_list.append(handle)

		plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.2, 1.015))
		plt.subplots_adjust(left=0.1, right=0.75)
		plt.title('Nearest Neighbor Distance (NND) vs. Time Spent For Each Click [units]')
		plt.xlabel('Time Spent [units]')
		plt.ylabel('Nearest Neighbor Distance (NND) [units]')
		plt.show()

	"""
	For each annotation (each click) in a dataframe, 
	plot nearest neighbor distance (nnd) vs. worker index. 
	Each point represents one annotation (one click). 

	Inputs:
		dataframe
		img_filename (the cropping)
		csv_filename (contains reference data)
	Output:
		none
	"""
	def plot_nnd_vs_worker_index(self, df, img_filename, csv_filename):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)
		ref_kdt = self.csv_to_kdt(csv_filename)
		dist_list = self.calc_distances(anno_one_crop, ref_kdt, img_filename)	# list containing one list for each worker

		# fig = plt.figure(figsize=(14,12))		# for jupyter notebook
		fig = plt.figure(figsize = (10,7))

		averages = []
		for i in range(len(worker_list)):			# for each worker
			x_coords = [i]*len(dist_list[i])
			y_coords = dist_list[i]
			plt.scatter(x_coords, y_coords, s = 4, alpha = 0.5, facecolors = 'c')
			average_dist = np.average(y_coords)
			averages.append(average_dist)
		
		handle = plt.scatter(range(len(worker_list)), averages, s = 20, facecolors = 'b', marker = '_', label = 'Average NND')

		plt.legend(handles = [handle], loc = 9, bbox_to_anchor = (1.15, 0.55))
		plt.subplots_adjust(left=0.1, right=0.8)
		plt.title('Nearest Neighbor Distance (NND) vs. Worker Index For Each Click')
		plt.xlabel('Worker Index')
		plt.ylabel('Nearest Neighbor Distance (NND) [units]')
		plt.xticks(np.arange(0, len(worker_list), step=1))
		plt.show()

	"""
	For each annotation (each click) in a dataframe, 
	plot time spent on click vs. worker index. 
	Each point represents one annotation (one click). 

	Inputs:
		dataframe
		img_filename (the cropping)
	Output:
		none
	"""
	def plot_time_spent_vs_worker_index(self, df, img_filename):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)
		time_list = self.calc_time_per_click(anno_one_crop, img_filename)	# list containing one list for each worker

		# fig = plt.figure(figsize=(14,12))		# for jupyter notebook
		fig = plt.figure(figsize = (10,7))

		averages = []
		for i in range(len(worker_list)):		# for each worker
			x_coords = [i]*len(time_list[i])
			y_coords = time_list[i]
			y_coords.pop(0)						# discard initial fencepost in time_list
			x_coords.pop(0)						# discard corresponding initial entry
			plt.scatter(x_coords, y_coords, s = 4, alpha = 0.5, facecolors = 'c')
			average_time = np.average(time_list[i])
			averages.append(average_time)

		handle = plt.scatter(range(len(worker_list)), averages, s = 20, facecolors = 'b', marker = '_', label = 'Average time spent')

		plt.legend(handles = [handle], loc = 9, bbox_to_anchor = (1.15, 0.55))
		plt.subplots_adjust(left=0.1, right=0.8)
		plt.title('Time Spent [units] vs. Worker Index')
		plt.xlabel('Worker Index')
		plt.ylabel('Time Spent [units]')
		plt.xticks(np.arange(0, len(worker_list), step=1))
		plt.show()

	"""
	For one worker in a dataframe,
	plot time spent on click vs. index of that click. 
	Each point represents one annotation (one click).

	Inputs:
		dataframe
		img_filename (the cropping)
		uid (worker ID)
	Output:
		none
	"""
	def plot_time_spent_vs_click_index(self, df, img_filename, uid):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)

		index = np.where(worker_list == uid)
		i = index[0][0]		# because np.where() returns a tuple containing an array
		time_list = self.calc_time_per_click(anno_one_crop, img_filename)	# list containing one list for each worker
		worker_time_list = time_list[i]

		x_coords = range(len(worker_time_list))
		y_coords = worker_time_list

		# fig = plt.figure(figsize=(14,12))		# for jupyter notebook
		fig = plt.figure(figsize = (10,7))
		handle = plt.scatter(x_coords, y_coords, s = 4, alpha = 0.5, facecolors = 'c', label = 'One click')
		
		plt.title('Time Spent [units] vs. Click Index for Worker ' + uid)
		plt.xlabel('Click Index')
		plt.ylabel('Time Spent [units]')
		plt.legend(handles = [handle], loc = 9, bbox_to_anchor = (1.15, 0.55))
		plt.subplots_adjust(left=0.1, right=0.8)
		plt.xticks(np.arange(0, len(worker_time_list), step=10))
		plt.show()



