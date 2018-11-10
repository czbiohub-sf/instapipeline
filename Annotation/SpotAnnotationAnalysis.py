""" 
This module contains the SpotAnnotationAnalysis class.
"""

import util

import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy

from numpy import genfromtxt
from matplotlib.lines import Line2D
from skimage import filters
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.neighbors import KDTree

# ------- #

class SpotAnnotationAnalysis():
	""" Tools for annotation analysis

	SpotAnnotationAnalysis takes in a BaseAnnotation 
	object as input and saves it as a property of
	the class.
	"""

	# list of clustering algs handled
	clustering_algs = ['AffinityPropagation']
	declumping_algs = ['KMeans']

	# list of colors used for plotting different turkers
	colors = ['#3399FF', '#CC33FF', '#FFFF00', '#FF33CC', 
	'#9966FF', '#009999', '#99E3FF', '#B88A00', 
	'#33FFCC', '#FF3366', '#F5B800', '#FF6633',
	'#FF9966', '#FF9ECE', '#CCFF33', '#FF667F',
	'#EB4E00', '#FFCC33', '#FF66CC', '#33CCFF', 
	'#ACFF07', '#667FFF', '#FF99FF', '#FF1F8F',
	'#9999FF', '#99FFCC', '#FF9999', '#91FFFF',
	'#8A00B8', '#91BBFF', '#FFB71C', '#FF1C76']


	def __init__(self, ba_obj):
		"""
		Take in a BaseAnnotation object and save it as 
		a property of the SpotAnnotationAnalysis class
		"""
		self.ba = ba_obj
		self.clusters_done = []
		self.cluster_objects = []

	def get_clusters(self, df, clustering_params):
		""" Cluster all annotations in df and arrange result as a dataframe. 
		Verifies clustering parameters and calls self.get_cluster_object()
		to check whether identical clustering has already been accomplished.

		Parameters
		----------
		df : pandas dataframe
		clustering_params : list of clustering parameters
			first element is string name of clustering algorithm
			subsequent elements are additional parameters

		Returns
		-------
		to_return : pandas dataframe (centroid_x | centroid_y | members)
			centroid_x = x coord of cluster centroid
			centroid_y = y coord of cluster centroid
			members = list of annotations belonging to the cluster
				each member is a list of properties of the annotation 
				i.e. [x coord, y coord, time spent, worker ID]
		"""

		clustering_alg = clustering_params[0]
		if (clustering_alg not in self.clustering_algs):
			raise ValueError('Invalid clustering algorithm name entered.')

		if (clustering_alg == 'AffinityPropagation'):											# If AffinityPropagation is selected:
			cluster_centroids_list = []																# Initialize a list of cluster centroids

			if(len(clustering_params) != 2):														# Check that there's only one clustering parameter
				raise ValueError('Please enter a list containing the preference parameter.')

			click_properties = util.get_click_properties(df)
			coords = click_properties[:,:2]															# Get all the coordinates from the annotation dataframe (dissociated from timestamps)

			af = self.get_cluster_object(coords, clustering_params)
			cluster_centers_indices = af.cluster_centers_indices_									# Get the indices of the cluster centers (list)
			num_clusters = len(cluster_centers_indices)
			cluster_members_lists = [[] for i in range(num_clusters)]
			labels = af.labels_																		# Each point that was in coords now has a label saying which cluster it belongs to.
			for label_index, click_property in zip(labels, click_properties):
				cluster_members_lists[label_index].append(click_property)
			for cluster_centers_index in cluster_centers_indices:
				cluster_centers = coords[cluster_centers_index]
				cluster_centroids_list.append(cluster_centers)

		centroid_IDs = range(num_clusters)
		column_names = ['centroid_x', 'centroid_y', 'members']
		to_return = pd.DataFrame(index = centroid_IDs, columns = column_names)

		for i in range(num_clusters):
			to_return['centroid_x'][i] = cluster_centroids_list[i][0]
			to_return['centroid_y'][i] = cluster_centroids_list[i][1]
			to_return['members'][i] = cluster_members_lists[i]

		return to_return

	"""
	Checks to see whether the cluster object has already been generated
	for the given df and clustering parameters, and returns or calculates
	appropriately.
	""" 
	def get_cluster_object(self, coords, clustering_params):
		""" Checks whether the cluster object has already been generated
		for the given df and clustering parameters, and returns or calculates
		appropriately.

		Parameters
		----------
		coords : np_array
			each row is an annotation [x_coord, y_coord]
		clustering_params : list of clustering parameters
			first element is string name of clustering algorithm
			subsequent elements are additional parameters

		Returns
		-------
		af : sklearn.cluster.AffinityPropagation object
		"""

		if (clustering_params[0] == 'AffinityPropagation'):

			for i in range(len(self.clusters_done)):
				coords_done = self.clusters_done[i][0]
				clustering_params_done = self.clusters_done[i][1]
				if ((np.array_equal(coords, coords_done)) and clustering_params == clustering_params_done):
					return self.cluster_objects[i]

			af = AffinityPropagation(preference = clustering_params[1]).fit(coords)				
			self.clusters_done.append([coords, clustering_params])
			self.cluster_objects.append(af)

			return af

	def plot_cluster_size_threshold(self, clusters, threshold):
		""" Visualize cluster sizes in a histogram with threshold demarcated.
		
		Parameters
		----------
		clusters : pandas dataframe (centroid_x | centroid_y | members) ~ output of get_clusters()
			centroid_x = x coord of cluster centroid
			centroid_y = y coord of cluster centroid
			members = list of annotations belonging to the cluster
				each member is a list of properties of the annotation 
				i.e. [x coord, y coord, time spent, worker ID]
		threshold : value to show threshold demarcation on histogram
		
		Returns
		-------
		none
		"""
		fig = plt.figure()
		hist_list = []
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			worker_list = [member[3] for member in members]
			hist_list.append(len(np.unique(worker_list)))
		width = max(hist_list)
		plt.hist(hist_list, bins=np.arange(0,width+4,2)-1)
		plt.axvline(x=threshold, color='b')
		plt.legend(handles=[Line2D([0],[0], color='b', label='cluster size threshold')])
		plt.title('Find Cluster Size Threshold')
		plt.xlabel('Number of unique annotators for cluster')
		plt.ylabel('Number of clusters')
		plt.show()

	def visualize_clusters(self, clusters, show_workers, show_centroids, show_ref_points, worker_marker_size, cluster_marker_size, ref_marker_size, csv_filepath, img_filepath, img_height, x_bounds, y_bounds, plot_title, bigger_window_size):
		""" Visualize clusters, each with a different color.
		
		Parameters
		----------
		clusters : pandas dataframe (centroid_x | centroid_y | members) ~ output of get_clusters()
			centroid_x = x coord of cluster centroid
			centroid_y = y coord of cluster centroid
			members = list of annotations belonging to the cluster
				each member is a list of properties of the annotation 
				i.e. [x coord, y coord, time spent, worker ID]
		show_workers : bool whether to plot workers
		show_centroids : bool whether to plot cluster centroids
		worker_marker_size, cluster_marker_size : plot parameters
		img_filepath : path to image file
		img_height : height of image in pixels
		plot_title : title of plot
		bigger_window_size : bool whether to use bigger window size (for jupyter notebook)

		Returns
		-------
		none
		"""
		fig = plt.figure(figsize = (12,7))
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		if x_bounds:
			plt.xlim(x_bounds[0], x_bounds[1])
		if y_bounds:
			plt.ylim(y_bounds[0], y_bounds[1])
		img = mpimg.imread(img_filepath)
		plt.imshow(img, cmap = 'gray')
		if show_workers:
			for color, member_list in zip(self.colors*10, clusters['members'].values):
				for member in member_list:
					plt.scatter([member[0]], [img_height-member[1]], s = worker_marker_size, facecolors = color, edgecolors = 'None')

		if show_ref_points:
			ref_df = pd.read_csv(csv_filepath)								
			ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()
			for point in ref_points:													
				plt.scatter([point[0]], [point[1]], s = ref_marker_size, facecolors = 'y')
			plt.legend(handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor='y', label='reference points')], loc = 9, bbox_to_anchor = (1.2, 1.015))	
		
		if show_centroids:
			plt.scatter(clusters['centroid_x'].values, util.flip(clusters['centroid_y'].values, img_height), s = cluster_marker_size, facecolors = 'none', edgecolors = '#ffffff')
		plt.title(plot_title)
		plt.show()

	def plot_clumpiness_threshold(self, clusters):
		""" Get cluster clumpiness threshold, visualize cluster clumpiness in a histogram with threshold demarcated.
		
		Parameters
		----------
		clusters : pandas dataframe (centroid_x | centroid_y | members) ~ output of get_clusters()
			centroid_x = x coord of cluster centroid
			centroid_y = y coord of cluster centroid
			members = list of annotations belonging to the cluster
				each member is a list of properties of the annotation 
				i.e. [x coord, y coord, time spent, worker ID]
		
		Returns
		-------
		threshold : the fraction of workers who contribute to the threshold cluster value only once
		"""
		single_fraction_list = []
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			workers = [member[3] for member in row.iloc[0]['members']]
			unique_workers = np.unique(workers)
			num_instances_list = [workers.count(unique_worker) for unique_worker in unique_workers]
			single_fraction_list.append(num_instances_list.count(1)/len(unique_workers))

		fig = plt.figure()
		(n, bins, patches) = plt.hist(single_fraction_list, bins = np.arange(0,1.2,0.1)-0.05)

		# calculate threshold
		total_counts_rev = list(reversed(n))
		threshold, prev_count, bin_width = 0, 0, 0.1
		for i in range(len(total_counts_rev)):
			count = total_counts_rev[i]
			if (count != 0):
				if((count < prev_count/3) and (count != 0) and (prev_count != 0)):
					threshold = bin_width*10-i*bin_width-bin_width/2
			prev_count = count

		threshold_line = Line2D([0],[0], color='orange', label='clumpiness threshold')
		plt.legend(handles=[threshold_line])
		plt.axvline(x=threshold, color='orange')
		plt.xticks(np.arange(0, bin_width*11, bin_width))
		plt.xlabel('Fraction of contributing workers who contribute only once')
		plt.ylabel('Number of clusters')
		plt.title('Finding the Clumpiness Threshold')
		plt.show()
		return threshold

	def declump(self, clusters, i, declumping_params):
		""" Declump the cluster at the ith index of clusters, a df only containing clumpy clusters.
		
		Parameters
		----------
		clusters : pandas dataframe (centroid_x | centroid_y | members) ~ output of sort_clusters_by_clumpiness()
			centroid_x = x coord of cluster centroid
			centroid_y = y coord of cluster centroid
			members = list of annotations belonging to the cluster
				each member is a list of properties of the annotation 
				i.e. [x coord, y coord, time spent, worker ID]
		i : index of cluster in clusters to declump
		declumping_params : list of clustering parameters
			first element is string name of declumping algorithm
			subsequent elements are additional parameters
		
		Returns
		-------
		declumped_clusters : pandas df containing resulting declumped clusters

		"""
		if (declumping_params[0] not in self.declumping_algs):
			raise ValueError('Invalid declumping algorithm name entered.')

		row = clusters.iloc[[i]]
		members = row.iloc[0]['members']
		workers = [member[3] for member in members]
		x_coords = [member[0] for member in members]
		y_coords = [member[1] for member in members]
		unique_workers = np.unique(workers)
		coords = np.stack((x_coords, y_coords), axis = -1)

		if (declumping_params[0] == 'KMeans'):
			k = declumping_params[1]
			km = KMeans(n_clusters=k).fit(coords)
			centers = km.cluster_centers_
			labels = km.labels_
			num_subclusters = k
		
		subclusters_list = [[center[0], center[1], []] for center in centers]
		for coord, label in zip(coords, labels):
			subclusters_list[label][2].append(coord)

		subclusters = pd.DataFrame(index = range(num_subclusters), columns = ['centroid_x','centroid_y','members'])

		for ind in range(num_subclusters):
			subclusters['centroid_x'][ind] = subclusters_list[ind][0]
			subclusters['centroid_y'][ind] = subclusters_list[ind][1]
			subclusters['members'][ind] = subclusters_list[ind][2]			

		return subclusters



