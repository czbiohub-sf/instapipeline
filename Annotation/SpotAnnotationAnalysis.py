""" 
This module contains the SpotAnnotationAnalysis class.
"""

from QuantiusAnnotation import QuantiusAnnotation
from BaseAnnotation import BaseAnnotation

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

	def csv_to_kdt(self, csv_filepath, img_height):
		""" Fit reference spot coordinates to a k-d tree

		Parameters
		----------
		csv_filepath : string filepath to csv file containing reference points
		img_height : height of image

		Returns
		-------
		ref_kdt : sklearn.neighbors.kd_tree.KDTree object containing reference points 
					y-coordinates are flipped about img_height 
		"""
		ref_df = pd.read_csv(csv_filepath)
		ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()

		for i in range(len(ref_points)):
			point = ref_points[i]
			first_elem = point[0]
			second_elem = img_height - point[1]
			point = np.array([first_elem, second_elem])
			ref_points[i] = point

		ref_kdt = KDTree(ref_points, leaf_size=2, metric='euclidean')	# kdt is a kd tree with all the reference points
		return ref_kdt

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

			click_properties = self.ba.get_click_properties(df)
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

	def get_pair_scores(self, df):
		""" Calculate pair scores for each pair of workers in df.

		Parameters
		----------
		df : pandas dataframe

		Returns
		-------
		pair_scores : pandas dataframe
			indices and columns of the dataframe are worker IDs
			contents are pair scores
			pair score between worker_A and worker_B = ((avg A->B NND) + (avg B->A NND))/2
		"""

		worker_list = self.ba.get_workers(df)
		pair_scores = pd.DataFrame(index = worker_list, columns = worker_list)
		for worker in worker_list:
			worker_df = self.ba.slice_by_worker(df, worker)
			worker_coords = self.ba.get_click_properties(worker_df)[:,:2]
			worker_kdt = KDTree(worker_coords, leaf_size=2, metric='euclidean')

			for other_worker in worker_list:
				if worker == other_worker:
					pair_scores[worker][other_worker] = 0
					continue

				other_worker_df = self.ba.slice_by_worker(df, other_worker)
				other_worker_coords = self.ba.get_click_properties(other_worker_df)[:,:2]
				other_worker_kdt = KDTree(other_worker_coords, leaf_size=2, metric='euclidean')

				list_A = [None]*len(worker_coords)
				for i in range(len(worker_coords)):
					dist, ind = other_worker_kdt.query([worker_coords[i]], k=1)
					list_A[i] = dist[0][0]

				list_B = [None]*len(other_worker_coords)
				for j in range(len(other_worker_coords)):
					dist, ind = worker_kdt.query([other_worker_coords[j]], k=1)
					list_B[j] = dist[0][0]

				pair_scores[worker][other_worker] = (np.mean(list_A) + np.mean(list_B))/2

		return pair_scores

	def get_worker_pair_scores(self, df):
		""" Calculate the total pairwise score for each workers in df.

		Parameters
		----------
		df : pandas dataframe

		Returns
		-------
		worker_scores : pandas dataframe
			indices of the dataframe are worker IDs
			column header of dataframe is "score" 
			"score" is the sum of the worker's pairwise scores
		"""
		worker_list = self.ba.get_workers(df)
		pair_scores = self.get_pair_scores(df)
		worker_scores = pd.DataFrame(index = worker_list, columns = ["score"])
		for worker in worker_list:
			worker_scores["score"][worker] = sum(pair_scores[worker].values)
		return worker_scores

	def get_worker_pair_score_threshold(self, df):
		""" Calculate a pairwise score threshold for all workers in
		df using Otsu's method. Assumes a bimodal distribution.

		Parameters
		----------
		df : pandas dataframe

		Returns
		-------
		pairwise score threshold value
		"""
		worker_pairwise_scores = self.get_worker_pair_scores(df)	# score workers based on pairwise matching (this step does not use clusters)
		worker_scores_list = worker_pairwise_scores['score'].tolist()	# get IDs of all workers
		return filters.threshold_otsu(np.asarray(worker_scores_list))	# threshold otsu

	def slice_by_worker_pair_score(self, df, threshold):
		""" Drop all annotations in df by workers with average pairwise 
		score greater than threshold

		Parameters
		----------
		df : pandas dataframe
		threshold : pairwise score threshold

		Returns
		-------
		df : pandas dataframe
		"""

		worker_pair_scores = self.get_worker_pair_scores(df)					# df with all workers. index = worker_ids, values = scores
		high_scores = worker_pair_scores[worker_pair_scores.score > threshold]	# df with only bad workers
		high_scoring_workers = high_scores.index.values
		for worker in high_scoring_workers:
			df = df[df.worker_id != worker]
		return df

	def get_cluster_size_threshold(self, clusters):
		""" Calculate a cluster size threshold for all clusters
		using K-means in 1D. Assumes a bimodal distribution.

		Parameters
		----------
		clusters : pandas dataframe 
			(centroid_x | centroid_y | members)

		Returns
		-------
		cluster size threshold
		"""
		total_list = []
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			worker_list = [member[3] for member in members]
			num_members = len(np.unique(worker_list))
			total_list.append(num_members)
		total_array = np.asarray(total_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		return (cluster_centers[0][0]+cluster_centers[1][0])/2

	def sort_clusters_by_size(self, clusters, threshold):
		""" Sort clusters by quantity of unique annotators.

		Parameters
		----------
		clusters : pandas dataframe 
			(centroid_x | centroid_y | members)
		threshold : threshold quantity of unique annotators

		Returns
		-------
		small_clusters : pandas dataframe containing clusters 
			for which num unique annotators < threshold
			(centroid_x | centroid_y | members)
		large_clusters : pandas dataframe containing clusters 
			for which num unique annotators >= threshold
			(centroid_x | centroid_y | members)
		"""
		small_clusters_list = []
		large_clusters_list = []
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			centroid_x = row.iloc[0]['centroid_x']
			centroid_y = row.iloc[0]['centroid_y']

			worker_list = []
			for member in members:
				worker_list.append(member[3])
			num_members = len(np.unique(worker_list))

			if (num_members < threshold):
				small_clusters_list.append([centroid_x, centroid_y, members])
			else:
				large_clusters_list.append([centroid_x, centroid_y, members])

		small_clusters = pd.DataFrame(index = range(len(small_clusters_list)), columns = ['centroid_x','centroid_y','members'])
		large_clusters = pd.DataFrame(index = range(len(large_clusters_list)), columns = ['centroid_x','centroid_y','members'])

		for i in range(len(small_clusters_list)):
			small_clusters['centroid_x'][i] = small_clusters_list[i][0]
			small_clusters['centroid_y'][i] = small_clusters_list[i][1]
			small_clusters['members'][i] = small_clusters_list[i][2]

		for i in range(len(large_clusters_list)):
			large_clusters['centroid_x'][i] = large_clusters_list[i][0]
			large_clusters['centroid_y'][i] = large_clusters_list[i][1]
			large_clusters['members'][i] = large_clusters_list[i][2]

		return small_clusters, large_clusters

	def get_clumpiness_threshold(self, clusters, bin_size, cutoff_fraction):
		""" Calculate a clumpiness threshold for all clusters
		by finding the value between the tail and the main mode. 
		Assumes a left-skewed unimodal distribution.

		Protocol for finding threshold:
		Sort all clusters into bins.
			e.g. if bin_size = 0.1, then sort clusters into bins 100-95%, 95-85%, ..., 5-0% 
			(% of contributors contributed only once to this cluster)
		Find all values between two adjacent bins where the number of clusters in the higher-value 
			bin is at least cutoff_fraction times greater than the number of clusters in the lower-value bin, 
			and neither bin contains zero clusters.
		threshold is the lowest of these values minus 0.1 (in order to move one bin to the left, 
			to minimize the number of clusters which are actually single in the group of clusters 
			detected as clumpy), or 0 if no such values exist.

		Parameters
		----------
		clusters : pandas dataframe 
			(centroid_x | centroid_y | members)
		bin_size : see protocol
		cutoff_fraction : see protocol

		Returns
		-------
		clumpiness threshold
		"""
		single_fraction_list = []
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			x_coords = []
			y_coords = []
			workers = []
			for member in members:
				x_coords.append(member[0])
				y_coords.append(member[1])
				workers.append(member[3])

			# Calculate replication of unique workers for each cluster
			unique_workers = np.unique(workers)
			num_instances_list = []
			for unique_worker in unique_workers:
				num_instances_list.append(workers.count(unique_worker))
			singles = num_instances_list.count(1)
			single_fraction = singles/len(unique_workers)
			single_fraction_list.append(single_fraction)

		(n, bins, patches) = plt.hist(single_fraction_list, bins=np.arange(0, 1+2*bin_size, bin_size) - bin_size/2)
		total_counts_reversed = list(reversed(n))

		threshold = 0
		prev_count = 0
		for i in range(len(total_counts_reversed)):
			count = total_counts_reversed[i]
			if (count != 0):
				if((count < prev_count/cutoff_fraction) and (count != 0) and (prev_count != 0)):
					threshold = 1 - i*bin_size - bin_size/2
			prev_count = count
		return threshold

	def sort_clusters_by_clumpiness(self, clusters, threshold):
		""" Sort clusters by fraction of contributors who contribute once
		to the cluster.

		Parameters
		----------
		clusters : pandas dataframe 
			(centroid_x | centroid_y | members)
		threshold : threshold fraction of contributors who only contribute once

		Returns
		-------
		clumpy_clusters : pandas dataframe containing clusters 
			for which fraction of contributors who only contribute once < threshold
			(centroid_x | centroid_y | members)
		nonclumpy_clusters : pandas dataframe containing clusters 
			for which fraction of contributors who only contribute once >= threshold
			(centroid_x | centroid_y | members)
		"""
		clumpy_clusters_list = []
		nonclumpy_clusters_list = []
		clumpy_counter = 0
		nonclumpy_counter = 0
		for j in range(len(clusters.index)):
			row = clusters.iloc[[j]]
			members = row.iloc[0]['members']
			centroid_x = row.iloc[0]['centroid_x']
			centroid_y = row.iloc[0]['centroid_y']

			workers = []
			for member in members:
				workers.append(member[3])
			unique_workers = np.unique(workers)

			num_instances_list = []
			for unique_worker in unique_workers:
				num_instances_list.append(workers.count(unique_worker))
			singles = num_instances_list.count(1)
			single_fraction = singles/len(unique_workers)

			if (single_fraction < threshold):
				clumpy_clusters_list.append([centroid_x, centroid_y, members])
				clumpy_counter += 1
			else:
				nonclumpy_clusters_list.append([centroid_x, centroid_y, members])
				nonclumpy_counter += 1

		clumpy_clusters = pd.DataFrame(index = range(clumpy_counter), columns = ['centroid_x','centroid_y','members'])
		nonclumpy_clusters = pd.DataFrame(index = range(nonclumpy_counter), columns = ['centroid_x','centroid_y','members'])

		for k in range(clumpy_counter):
			clumpy_clusters['centroid_x'][k] = clumpy_clusters_list[k][0]
			clumpy_clusters['centroid_y'][k] = clumpy_clusters_list[k][1]
			clumpy_clusters['members'][k] = clumpy_clusters_list[k][2]

		for m in range(nonclumpy_counter):
			nonclumpy_clusters['centroid_x'][m] = nonclumpy_clusters_list[m][0]
			nonclumpy_clusters['centroid_y'][m] = nonclumpy_clusters_list[m][1]
			nonclumpy_clusters['members'][m] = nonclumpy_clusters_list[m][2]

		return clumpy_clusters, nonclumpy_clusters

	def get_time_per_click(self, df):
		""" Get time spent on each annotation.

		Parameters
		----------
		df : pandas dataframe 
			(timestamp | x | y | annotation_type | height | width image_filename | time_when_completed | worker_id)

		Returns
		-------
		time_spent_list : list of the amount of time spent on all clicks in df
			except the first click (fencepost)
			len(time_spent_list) = num rows in df
			time_spent_list[0] = None
			units are miliseconds
		"""
		timestamps = self.ba.get_timestamps(df)
		time_spent_list = [None]*len(timestamps)
		for i in range (1,len(timestamps)):
			x = timestamps[i] - timestamps[i-1]
			time_spent_list[i] = x[0]
		return time_spent_list

	def get_nnd_per_click(self, df, ref_kdt):
		""" Get the distance to the nearest neighbor (found in
			the k-d tree of reference points).

		Parameters
		----------
		df : pandas dataframe 
			(timestamp | x | y | annotation_type | height | width image_filename | time_when_completed | worker_id)

		Returns
		-------
		list of distances to the nearest neighbor (found in
			the k-d tree of reference points)
		"""
		coords = self.ba.get_click_properties(df)[:,:2]
		dist, ind = ref_kdt.query(coords, k=1)
		dist_list = dist.tolist()
		return [dist[0] for dist in dist_list]

	def get_avg_time_per_click(self, df, uid):
		""" Get the average amount of time that a worker spent on one click.

		Parameters
		----------
		df : pandas dataframe 
			(timestamp | x | y | annotation_type | height | width image_filename | time_when_completed | worker_id)
		uid : string worker ID

		Returns
		-------
		the average time that the worker spent per click
		"""		

		worker_timestamps = self.get_timestamps(df, uid)
		time_spent = max(worker_timestamps) - min(worker_timestamps)
		num_clicks = len(worker_timestamps)
		return time_spent[0]/num_clicks

	def get_cluster_means(self, clusters):
		""" Get the mean x and y of each cluster.
		(Different from cluster centroids, which are the exemplar
		annotation for each cluster.)

		Parameters
		----------
		clusters : pandas dataframe 
			(centroid_x | centroid_y | members)

		Returns
		-------
		numpy array of coords
		"""
		mean_coords = []
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			x_coords = []
			y_coords = []
			for member in members:
				x_coords.append(member[0])
				y_coords.append(member[1])
			mean_coord = [np.mean(x_coords), np.mean(y_coords)]
			mean_coords.append(mean_coord)
		return np.asarray(mean_coords)

	def centroid_and_ref_df(self, clusters, csv_filepath, img_height):
		""" Assemble a dataframe of centroids found with annotation and reference data consolidated.
		
		Parameters
		----------
		df : Pandas Dataframe with annotation data (should already be cropped)
		clusters : pandas dataframe (centroid_x | centroid_y | members) ~ output of get_clusters()
			centroid_x = x coord of cluster centroid
			centroid_y = y coord of cluster centroid
			members = list of annotations belonging to the cluster
				each member is a list of properties of the annotation 
				i.e. [x coord, y coord, time spent, worker ID]
		csv_filepath : contains reference data

		Returns
		-------
		this dataframe: centroid_x | centroid_y | x of nearest ref | y of nearest ref | NN_dist | members
			* (the index is the Cluster ID)
			centroid_x = x coord of cluster centroid
			centroid_y = y coord of cluster centroid
			NN_x = x coord of nearest neighbor reference
			NN_y = y coord of nearest neighbor reference
			NN_dist = distance from centroid to nearest neighbor reference
			members = list of annotations belonging to cluster
				each annotation is a list of click properties: x_coord | y_coord | time_spent | worker_ID
		"""

		ref_kdt = self.csv_to_kdt(csv_filepath, img_height)
		ref_array = np.asarray(ref_kdt.data)

		centroid_IDs = range(clusters.shape[0])
		column_names = ['centroid_x', 'centroid_y', 'NN_x', 'NN_y', 'NN_dist', 'members']
		to_return = pd.DataFrame(index = centroid_IDs, columns = column_names)

		for i in centroid_IDs:

			to_return['centroid_x'][i] = clusters['centroid_x'][i]
			to_return['centroid_y'][i] = clusters['centroid_y'][i]

			coords = [[to_return['centroid_x'][i], to_return['centroid_y'][i]]]

			dist, ind = ref_kdt.query(coords, k=1)
			index = ind[0][0]
			nearest_neighbor = ref_array[index]

			to_return['NN_x'][i] = nearest_neighbor[0]
			to_return['NN_y'][i] = nearest_neighbor[1]
			to_return['NN_dist'][i] = dist[0][0]
			to_return['members'][i] = clusters['members'][i]		

		return to_return

	def get_cluster_correctness(self, df, correctness_threshold):
		""" Assemble a dataframe of centroids found with annotation and reference data consolidated.
		
		Parameters
		----------
		centroid_and_ref_df : outputted by centroid_and_ref_df()
			centroid_x | centroid_y | x of nearest ref | y of nearest ref | NN_dist | members (x | y | time_spent | worker_id)
			* the index is the Centroid ID
		correctness_threshold : tolerance for correctness in pixels, None if correctness will not be visualized
			for each centroid, if NN_dist <= threshold, centroid is "correct"

		Returns
		-------
		2-column array with a row for each centroid
			column 0 = Centroid ID
			column 1 = True if centroid is "correct", False if centroid is "incorrect"
		"""

		num_centroids = df.shape[0]
		to_return = np.empty([num_centroids, 2])
		for i in range(num_centroids):
			to_return[i] = i
			NN_dist = df['NN_dist'][i]
			if (NN_dist <= correctness_threshold):
				to_return[i][1] = True
			else:
				to_return[i][1] = False
		return to_return
	
	def plot_annotations(self, df, show_workers, show_correctness_workers, show_centroids, show_correctness_centroids, show_ref_points, show_NN_inc, centroid_and_ref_df, correctness_threshold, worker_marker_size, cluster_marker_size, img_filepath, csv_filepath, bigger_window_size):
		""" Quick visualization of worker annotations, clusters, and/or annotation and cluster "correctness." 
		
		Parameters
		----------
		df : pandas dataframe with annotation data for one crop only
		show_workers : bool whether to plot workers
		show_centroids : bool whether to plot cluster centroids
		show_ref_points : bool whether to plot reference annotations
		show_NN_inc : bool whether to show nearest neighbor for all "incorrect" centroids
		centroid_and_ref_df = pandas dataframe outputted by centroid_and_ref_df()
			centroid_x | centroid_y | x of nearest ref | y of nearest ref | NN_dist | members
		correctness_threshold : tolerance for correctness in pixels, None if correctness will not be visualized
		worker_marker_size, cluster_marker_size : plot parameters
		img_filepath, csv_filepath : paths to image and reference csv files
		bigger_window_size : bool whether to use bigger window size (for jupyter notebook)

		Returns
		-------
		none
		"""


		fig = plt.figure(figsize = (12,7))
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))

		handle_list = []
		img_height = df['height'].values[0]

		if correctness_threshold is not None:
			cluster_correctness = self.get_cluster_correctness(centroid_and_ref_df, correctness_threshold)

		if show_workers:
			if show_correctness_workers:
				member_lists = centroid_and_ref_df['members'].values
				for member_list, correctness in zip(member_lists, cluster_correctness):
					if correctness[1]:
						color = 'g'
					else:
						color = 'm'
					for member in member_list:
						coords = member[:2]
						plt.scatter([coords[0]], self.ba.flip([coords[1]], img_height), s = worker_marker_size, facecolors = color, alpha = 0.5)
				handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='g', label='anno of correct cluster'))
				handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='m', label='anno of incorrect cluster'))
			else:
				worker_list = self.ba.get_workers(df)
				for worker, color in zip(worker_list, self.colors):			# For each worker, use a different color.
					anno_one_worker = self.ba.slice_by_worker(df, worker)		
					coords = self.ba.get_click_properties(anno_one_worker)[:,:2]
					x_coords = coords[:,0]
					y_coords = coords[:,1]
					y_coords_flipped = self.ba.flip(y_coords, img_height)
					handle = plt.scatter(x_coords, y_coords_flipped, s = worker_marker_size, facecolors = color, alpha = 0.5, label = worker)
					handle_list.append(handle)
			if not show_centroids:
				plt.title('Worker Annotations')	

		if show_centroids:
			x_coords = centroid_and_ref_df['centroid_x'].values
			y_coords = centroid_and_ref_df['centroid_y'].values
			y_coords_flipped = self.ba.flip(y_coords, img_height)
			color_index = 0		
			if show_correctness_centroids:
				for i in range(len(centroid_and_ref_df.index)):		
					if (cluster_correctness[i][1]):
						color = 'g'								
					else:
						if show_NN_inc:
							color = self.colors[color_index]							
							color_index = (color_index+1)%len(self.colors)
							plt.scatter([centroid_and_ref_df['NN_x'].values[i]], [img_height-centroid_and_ref_df['NN_y'].values[i]], s = worker_marker_size*2, facecolors = color, edgecolors = color)
						else:
							color = 'm'
					plt.scatter(x_coords[i], y_coords_flipped[i], s = cluster_marker_size, facecolors = 'none', edgecolors = color)					
				handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=None, markeredgecolor='g', label='centroid of correct cluster'))
				handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=None, markeredgecolor='m', label='centroid of incorrect cluster'))
			else:
				plt.scatter(x_coords, y_coords_flipped, s = cluster_marker_size, facecolors = 'none', edgecolors = 'cyan')
				handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=None, markeredgecolor='cyan', label='cluster centroid'))
			if not show_workers:
				plt.title('Cluster Centroids')

		if show_workers and show_centroids:
			plt.title('Worker Annotations and Cluster Centroids')

		if show_ref_points:
			ref_df = pd.read_csv(csv_filepath)								
			ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()
			for point in ref_points:													
				plt.scatter([point[0]], [point[1]], s = 20, facecolors = 'y')
			handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='y', label='reference points'))
		
		img = mpimg.imread(img_filepath)
		plt.imshow(img, cmap = 'gray')
		plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.2, 1.015))	
		plt.show()

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
			plt.scatter(clusters['centroid_x'].values, self.ba.flip(clusters['centroid_y'].values, img_height), s = cluster_marker_size, facecolors = 'none', edgecolors = '#ffffff')
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

	def calc_fpr_tpr(self, clusters, csv_filepath, correctness_threshold, plot_tpr, plot_fpr, img_filepath, img_height, cluster_marker_size, bigger_window_size):
		""" Compare the centroids in the clusters dataframe with the reference
		values to calculate the false positive and true positive rates.
		
		Parameters
		----------
		clusters : pandas dataframe (centroid_x | centroid_y | members) ~ output of sort_clusters_by_clumpiness()
			centroid_x = x coord of cluster centroid
			centroid_y = y coord of cluster centroid
			members = list of annotations belonging to the cluster
				each member is a list of properties of the annotation 
				i.e. [x coord, y coord, time spent, worker ID]
		csv_filepath : string filepath to csv file containing reference points
		correctness_threshold : tolerance for correctness in pixels
		plot_tpr : boolean whether to visualize tpr
		plot_fpr : boolean whether to visualize fpr
		img_filepath : string filepath to image file
		img_heit : height of image in pixels
		cluster_marker_size : plotting parameter
		bigger_window_size : bool whether to use bigger window size (for jupyter notebook)
		
		Returns
		-------
		tpr : num spots detected / num spots total
		fpr : num clusters don’t correspond with a spot / num clusters total
		"""

		if plot_tpr or plot_fpr:
			fig = plt.figure(figsize = (12,7))
			if bigger_window_size:
				fig = plt.figure(figsize=(14,12))

		ref_df = pd.read_csv(csv_filepath)
		ref_coords = ref_df.loc[:, ['col', 'row']].as_matrix()
		for i in range(len(ref_coords)):
			point = ref_coords[i]
			first_elem = point[0]
			second_elem = img_height - point[1]
			point = np.array([first_elem, second_elem])
			ref_coords[i] = point
		centroid_coords = clusters.loc[:, ['centroid_x', 'centroid_y']].as_matrix()
		centroid_kdt = KDTree(centroid_coords, leaf_size=2, metric='euclidean')	# kdt is a kd tree with all the reference points

		# calc tpr
		num_spots_detected = 0
		for ref_coord in ref_coords:
			dist, ind = centroid_kdt.query([ref_coord], k=1)
			if dist[0] <= correctness_threshold:
				num_spots_detected += 1
				if plot_tpr:
					plt.scatter([ref_coord[0]], self.ba.flip([ref_coord[1]], img_height), s = cluster_marker_size, facecolors = 'g')
			else:
				if plot_tpr:
					plt.scatter([ref_coord[0]], self.ba.flip([ref_coord[1]], img_height), s = cluster_marker_size, facecolors = 'm')
		num_spots_total = len(ref_coords)
		tpr = num_spots_detected/num_spots_total

		# calc fpr
		ref_kdt = self.csv_to_kdt(csv_filepath, img_height)
		num_centroids_wout_spot = 0
		for centroid_coord in centroid_coords:
			dist, ind = ref_kdt.query([centroid_coord], k=1)
			if dist[0] > correctness_threshold:
				num_centroids_wout_spot += 1
				if plot_fpr:
					plt.scatter([centroid_coord[0]], self.ba.flip([centroid_coord[1]], img_height), s = cluster_marker_size, edgecolors = 'm', facecolors = 'none')
			else:
				if plot_fpr:
					plt.scatter([centroid_coord[0]], self.ba.flip([centroid_coord[1]], img_height), s = cluster_marker_size, edgecolors = 'g', facecolors = 'none')
		num_centroids_total = len(centroid_coords)
		fpr = num_centroids_wout_spot/num_centroids_total

		handle_list = []

		if plot_tpr:
			plt.title("TPR = " + str(round(tpr, 2)))
			handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='g', label='detected spot'))
			handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='m', label='undetected spot'))

		if plot_fpr:
			plt.title("FPR = " + str(round(fpr, 2)))
			handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=None, markeredgecolor='g', label='correct centroid'))
			handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=None, markeredgecolor='m', label='incorrect centroid'))

		if plot_tpr or plot_fpr:
			img = mpimg.imread(img_filepath)
			plt.imshow(img, cmap = 'gray')
			plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.2, 1.015))
			plt.show()	

		return tpr, fpr

