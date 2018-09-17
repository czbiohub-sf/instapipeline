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
		counter = 0
		for worker in worker_list:
			counter += 1
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














# start here






















	"""
	The list should contain, for each “putatively incorrect” cluster, 
	the fraction of the cluster’s annotations which are from workers 
	who are in many “putatively correct” clusters.
	"""
	def plot_fraction_from_crowd_per_cluster(self, clusters, crowd, show_correctness, correctness_threshold, csv_filepath, img_height, plot_title, bigger_window_size):
	
		correct_list = []
		incorrect_list = []
		total_list = []
		anno_and_ref_df = self.anno_and_ref_to_df_input_clusters(clusters, csv_filepath, img_height)
		cluster_correctness = self.get_cluster_correctness(anno_and_ref_df, correctness_threshold)
		for i in range(len(clusters.index)):

			# get list of unique members in that cluster
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			worker_list = []
			for member in members:
				worker_list.append(member[3])
			unique_workers = np.unique(worker_list)

			# get fraction of good crowd workers in that list
			numerator = 0
			for worker in unique_workers:
				if(worker in crowd):
					numerator += 1
			denominator = len(unique_workers)
			fract_members = math.floor((numerator/denominator)*100)

			if (cluster_correctness[i][1]):		
				correct_list.append(fract_members)
			else:
				incorrect_list.append(fract_members)
			total_list.append(fract_members)

		width = 100

		fig = plt.figure()
		
		y,x,_ = plt.hist([correct_list, incorrect_list], bins = np.arange(0,width+20,10)-5, stacked = True, color = ['g','m'])

		# # threshold otsu
		# threshold_otsu = filters.threshold_otsu(np.asarray(total_list))

		# treshold kmeans
		total_array = np.asarray(total_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

		# plt.axvline(x=threshold_otsu, color='r')
		plt.axvline(x=threshold_kmeans, color='b')

		g_patch = mpatches.Patch(color='g', label='correct clusters')
		m_patch = mpatches.Patch(color='m', label='incorrect clusters')
		# otsu_line = Line2D([0],[0], color='r', label='otsu threshold')
		kmeans_line = Line2D([0],[0], color='b', label='k-means threshold')
		plt.legend(handles=[g_patch, m_patch, kmeans_line])
		ymin, ymax = plt.ylim()
		y_step = 5
		if ymax<10:
			y_step = 1
		if (ymax>50):
			y_step = 10
		plt.xlabel("Percent of cluster’s annotations from good crowd [%]")
		plt.xticks(np.arange(0,width+10,step=10))
		plt.yticks(np.arange(0,ymax+2, step = y_step))
		plt.ylabel("Number of clusters")
		plt.title(plot_title)
		plt.show()

	def anno_and_ref_to_df_input_clusters(self, clusters, csv_filepath, img_height):

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

	def sort_workers_by_membership_in_large_clusters(self, df, large_clusters):
		other_crowd = []
		good_crowd = []

		worker_list = self.ba.get_workers(df)

		# find threshold (kmeans)
		total_list = []
		for uid in worker_list:
			pc_clusters_found = self.get_pc_clusters_found(large_clusters, uid)
			total_list.append(pc_clusters_found)
		total_array = np.asarray(total_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

		# given threshold, sort all workers
		for uid in worker_list:
			pc_clusters_found = self.get_pc_clusters_found(large_clusters, uid)
			if(pc_clusters_found) > threshold_kmeans:
				good_crowd.append(uid)
			else:
				other_crowd.append(uid)

		return other_crowd, good_crowd

	def plot_workers_pc_yield(self, df, large_clusters, plot_title):
		worker_list = self.ba.get_workers(df)
		hist_list = []
		for uid in worker_list:
			pc_clusters_found = self.get_pc_clusters_found(large_clusters, uid)
			hist_list.append(pc_clusters_found)

		step_size = 5
		if (max(hist_list) > 50):
			step_size = 10
		if (max(hist_list) > 100):
			step_size = 20

		y,x,_ = plt.hist(hist_list, bins=np.arange(0,max(hist_list)+step_size*2, step=step_size)-step_size/2)
		
		# threshold otsu
		threshold_otsu = filters.threshold_otsu(np.asarray(hist_list))

		# threshold kmeans
		total_array = np.asarray(hist_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

		plt.axvline(x=threshold_otsu, color='r')
		plt.axvline(x=threshold_kmeans, color='b')

		otsu_line = Line2D([0],[0], color='r', label='otsu threshold')
		kmeans_line = Line2D([0],[0], color='b', label='k-means threshold')
		plt.legend(handles=[otsu_line, kmeans_line])

		plt.xticks(np.arange(0,max(hist_list)+step_size*2, step=step_size))
		plt.yticks(np.arange(0,y.max()+1))

		plt.title(plot_title)
		plt.xlabel("Quantity of putatively correct clusters found by a worker")
		plt.ylabel("Quantity of workers")
		plt.show()

	def get_pc_clusters_found(self, large_clusters, uid):
		counter = 0
		for i in range(len(large_clusters.index)):
			row = large_clusters.iloc[[i]]
			members = row.iloc[0]['members']
			for member in members:
				worker = member[3]
				if(uid==worker):
					counter+=1
					break
		return counter

	def plot_snr_vs_members(self, df, clustering_params, csv_filepath, img_height, img_filename, correctness_threshold):

		clusters = self.get_clusters(df, clustering_params)			# this dataframe: centroid_x | centroid_y | members
		ref_df = pd.read_csv(csv_filepath)
		ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()	
		snr_val_list = ref_df.loc[:, ['snr']].as_matrix()	

		for i in range(len(ref_points)):			# flip vertical axis
			point = ref_points[i]
			first_elem = point[0]
			second_elem = img_height - point[1]
			point = np.array([first_elem, second_elem])
			ref_points[i] = point

		anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		centroid_coords = anno_and_ref_df.loc[:, ['centroid_x', 'centroid_y']].as_matrix()		
		centroids_kdt = KDTree(centroid_coords, leaf_size=2, metric='euclidean')

		snr_list = []
		num_members_list = []

		# for each spot
		for i in range(len(ref_points)):
			ref_point = ref_points[i]

			# get SNR
			snr = snr_val_list[i][0]	
			# get nearest neighbor centroid
			dist, ind = centroids_kdt.query([ref_point], k=1)
			if (dist[0][0] <= correctness_threshold):			# if the spot is detected
				centroid_coords_index = ind[0][0]
				nearest_centroid = centroid_coords[centroid_coords_index]
				nearest_centroid_x = nearest_centroid[0]
				nearest_cluster = clusters.loc[clusters['centroid_x']==nearest_centroid_x]
				members = nearest_cluster.iloc[0]['members']

				worker_list = []
				for member in members:
					worker_list.append(member[3])
				num_members = len(np.unique(worker_list))
				num_members_list.append(num_members)
				snr_list.append(snr)

		legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor='g', label='one detected spot')]
		plt.legend(handles = legend_elements)
		plt.scatter(num_members_list, snr_list, color = 'g', s = 20)
		plt.title("SNR vs. number of unique workers annotating")
		plt.xlabel("Number of unique workers annotating")
		plt.xticks(np.arange(0,30, step=2))
		plt.yticks(np.arange(min(snr_list)-1,max(snr_list)+1, step=2))
		plt.ylabel("SNR")
		plt.show()

	def plot_annotations_and_snr_per_cluster(self, df, clustering_params, show_correctness, correctness_threshold, csv_filepath, img_filename, img_height, plot_title, bigger_window_size):
		clusters = self.get_clusters(df, clustering_params)			# this dataframe: centroid_x | centroid_y | members
		
		correct_list = []
		incorrect_list = []
		total_list = []
		anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		cluster_correctness = self.get_cluster_correctness(anno_and_ref_df, correctness_threshold)
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			worker_list = []
			for member in members:
				worker_list.append(member[3])
			num_members = len(np.unique(worker_list))
			if (cluster_correctness[i][1]):		# if cluster is correct
				correct_list.append(num_members)
			else:
				incorrect_list.append(num_members)
			total_list.append(num_members)
		width = max(correct_list)
		if (max(incorrect_list) > width):
			width = max(incorrect_list)

		fig, ax1 = plt.subplots(figsize = (10,5))
		
		y,x,_ = ax1.hist([correct_list, incorrect_list], bins = np.arange(0,width+4,2)-1, stacked = True, color = ['g','m'])

		# threshold otsu
		threshold_otsu = filters.threshold_otsu(np.asarray(total_list))

		# treshold kmeans
		total_array = np.asarray(total_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

		ax1.axvline(x=threshold_otsu, color='r')
		ax1.axvline(x=threshold_kmeans, color='b')

		# NEXT PLOT			
		ref_df = pd.read_csv(csv_filepath)
		ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()	
		snr_val_list = ref_df.loc[:, ['snr']].as_matrix()	

		for i in range(len(ref_points)):			# flip vertical axis
			point = ref_points[i]
			first_elem = point[0]
			second_elem = img_height - point[1]
			point = np.array([first_elem, second_elem])
			ref_points[i] = point

		anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		centroid_coords = anno_and_ref_df.loc[:, ['centroid_x', 'centroid_y']].as_matrix()		
		centroids_kdt = KDTree(centroid_coords, leaf_size=2, metric='euclidean')

		snr_list = []
		num_members_list = []

		# for each spot
		for i in range(len(ref_points)):
			ref_point = ref_points[i]

			# get SNR
			snr = snr_val_list[i][0]	
			# get nearest neighbor centroid
			dist, ind = centroids_kdt.query([ref_point], k=1)
			if (dist[0][0] <= correctness_threshold):			# if the spot is detected
				centroid_coords_index = ind[0][0]
				nearest_centroid = centroid_coords[centroid_coords_index]
				nearest_centroid_x = nearest_centroid[0]
				nearest_cluster = clusters.loc[clusters['centroid_x']==nearest_centroid_x]
				members = nearest_cluster.iloc[0]['members']

				worker_list = []
				for member in members:
					worker_list.append(member[3])
				num_members = len(np.unique(worker_list))
				num_members_list.append(num_members)
				snr_list.append(snr)

		ax1.set_xlabel("Number of unique workers annotating")
		ax1.set_ylabel("Number of clusters")

		ax2 = ax1.twinx()
		ax2.scatter(num_members_list, snr_list, color = 'y', s = 20)
		ax2.set_ylabel("SNR")

		g_patch = mpatches.Patch(color='g', label='correct clusters')
		m_patch = mpatches.Patch(color='m', label='incorrect clusters')
		otsu_line = Line2D([0],[0], color='r', label='otsu threshold')
		kmeans_line = Line2D([0],[0], color='b', label='k-means threshold')
		snr_dot = Line2D([0],[0], marker='o', color='w', markerfacecolor='y', label='SNR for one detected spot')
		plt.legend(handles=[g_patch, m_patch, otsu_line, kmeans_line, snr_dot], bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

		plt.title(plot_title)
		fig.tight_layout()
		plt.show()

	def plot_annotations_per_cluster(self, df, clustering_params, show_correctness, correctness_threshold, csv_filepath, img_filename, plot_title, bigger_window_size):
		clusters = self.get_clusters(df, clustering_params)
		if not show_correctness:
			hist_list = []
			for i in range(len(clusters.index)):
				row = clusters.iloc[[i]]
				members = row.iloc[0]['members']
				worker_list = []
				for member in members:
					worker_list.append(member[3])
				num_members = len(np.unique(worker_list))
				hist_list.append(num_members)
			plt.title(plot_title)
			y,x,_ = plt.hist(hist_list, bins=np.arange(0,max(hist_list)+4,2)-1)
			width = max(hist_list)
		else:
			correct_list = []
			incorrect_list = []
			total_list = []
			anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
			cluster_correctness = self.get_cluster_correctness(anno_and_ref_df, correctness_threshold)
			for i in range(len(clusters.index)):
				row = clusters.iloc[[i]]
				members = row.iloc[0]['members']
				worker_list = []
				for member in members:
					worker_list.append(member[3])
				num_members = len(np.unique(worker_list))
				if (cluster_correctness[i][1]):		# if cluster is correct
					correct_list.append(num_members)
				else:
					incorrect_list.append(num_members)
				total_list.append(num_members)
			width = max(correct_list)
			if (max(incorrect_list) > width):
				width = max(incorrect_list)

			fig = plt.figure()
			
			y,x,_ = plt.hist([correct_list, incorrect_list], bins = np.arange(0,width+4,2)-1, stacked = True, color = ['g','m'])

			# threshold otsu
			threshold_otsu = filters.threshold_otsu(np.asarray(total_list))

			# treshold kmeans
			total_array = np.asarray(total_list)
			km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
			cluster_centers = km.cluster_centers_
			threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

			plt.axvline(x=threshold_otsu, color='r')
			plt.axvline(x=threshold_kmeans, color='b')

			g_patch = mpatches.Patch(color='g', label='clusters near ref spot')
			m_patch = mpatches.Patch(color='m', label='clusters far from any ref spot')
			otsu_line = Line2D([0],[0], color='r', label='otsu threshold')
			kmeans_line = Line2D([0],[0], color='b', label='k-means threshold')
			plt.legend(handles=[g_patch, m_patch, otsu_line, kmeans_line])

		plt.xlabel("Number of unique workers annotating")
		plt.xticks(np.arange(0,width+2,step=2))
		plt.ylabel("Number of clusters")
		ymin, ymax = plt.ylim()
		if(ymax < 30):
			plt.yticks(np.arange(0,ymax+1,step=3))
		plt.title(plot_title)
		plt.show()

	def plot_worker_pairwise_scores(self, df):
		worker_scores = self.get_worker_pairwise_scores(df)
		worker_scores = worker_scores["score"].values
		worker_scores_list = []
		for score in worker_scores:
			worker_scores_list.append(score)

		worker_list = self.ba.get_workers(df)

		fig = plt.figure(figsize = (10,7))

		handle_list = []
		for i in range(len(worker_list)):
			score = worker_scores_list[i]
			handle = plt.bar(i, score, color = self.colors[i], label = (str(i) + ". " + worker_list[i]))
			handle_list.append(handle)

		plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.15, 1.015))
		plt.subplots_adjust(left=0.1, right=0.8)
		plt.title('Pairwise Score [s] vs. Worker Index')
		plt.xlabel('Worker Index')
		plt.ylabel('Pairwise Score')
		plt.xticks(np.arange(0, len(worker_list), step=1))
		plt.show()

	def plot_worker_pairwise_scores_hist(self, df, plot_title, bigger_window_size):

		# get worker scores as list
		worker_scores = self.get_worker_pairwise_scores(df)
		worker_scores = worker_scores["score"].values
		worker_scores_list = []
		for score in worker_scores:
			worker_scores_list.append(score)

		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		step_size = 20
		low = math.floor((min(worker_scores_list)-100)/100)*100

		y,x,_ = plt.hist(worker_scores_list, bins=np.arange(low,max(worker_scores_list)+step_size*2, step=step_size)-step_size/2)

		# threshold otsu
		threshold_otsu = filters.threshold_otsu(np.asarray(worker_scores_list))

		# threshold kmeans
		total_array = np.asarray(worker_scores_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

		# threshold 3rd quartile
		threshold_q3 = np.mean(worker_scores_list) + 1.5*np.std(worker_scores_list)

		plt.axvline(x=threshold_otsu, color='r')
		plt.axvline(x=threshold_kmeans, color='b')
#		plt.axvline(x=threshold_q3, color='g')

		otsu_line = Line2D([0],[0], color='r', label='otsu threshold')
		kmeans_line = Line2D([0],[0], color='b', label='k-means threshold')
#		q3_line = Line2D([0],[0], color='g', label='q3 threshold')
		plt.legend(handles=[otsu_line, kmeans_line])

		plt.title(plot_title)
		plt.xlabel('Sum of pairwise NND averages')
		plt.ylabel('Quantity of workers')
		width = max(worker_scores_list) - low
		if(width>2000):
			x_step = 200
		elif (width>1000):
			x_step = 100
		else:
			x_step = 50
		plt.xticks(np.arange(low,max(worker_scores_list)+x_step*2,step=x_step))
		plt.yticks(np.arange(0,y.max()+1))
		plt.show()

	def plot_error_rate_vs_spotted(self, df, clustering_params, correctness_threshold, csv_filepath, img_filename, plot_title, bigger_window_size):
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		worker_list = self.ba.get_workers(df)

		num_good_clusters_list = []
		error_rate_list = []

		for worker in worker_list:
			num_good_clusters_list.append(self.get_worker_num_good_clusters(worker, clusters, correctness_threshold))
			error_rate_list.append(self.get_worker_error_rate(worker, clusters, correctness_threshold) * 100)
		
		plt.scatter(num_good_clusters_list, error_rate_list, facecolors='c', s=20)
		legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor='c', label='one worker')]
		plt.legend(handles = legend_elements)
		plt.title("Error rate vs. number of good clusters found")
		plt.xlabel("Number of good clusters found by the worker")
		plt.ylabel("Worker's error rate [%]")

		if (max(num_good_clusters_list)<=60):
			x_step = 5
		elif(max(num_good_clusters_list)<=110):
			x_step = 10
		else:
			x_step = 20
		plt.xticks(np.arange(0,max(num_good_clusters_list)+2,step=x_step))

		if (max(error_rate_list)<=20):
			y_step = 1
		elif(max(error_rate_list)<=40):
			y_step = 2
		else:
			y_step = 5
#		plt.yticks(np.arange(0,max(error_rate_list)+1,step=y_step))
		plt.yticks(np.arange(0,101,step=y_step))

		plt.show()

	"""
	For one worker, get the number of good clusters that worker found.
	Inputted df "clusters" is generated by anno_and_ref_to_df()
	"""
	def get_worker_num_good_clusters(self, uid, clusters, correctness_threshold):
		counter = 0
		cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)
		for i in range(len(clusters.index)):
			if not (cluster_correctness[i][1]):
				continue
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			for member in members:
				worker = member[3]
				if(uid==worker):
					counter += 1
					break
		return counter

	"""
	error_rate = (number of bad clusters the worker is a member in)/(number of clusters the worker is a member in)
	"""
	def get_worker_error_rate(self, uid, clusters, correctness_threshold):
		num_bad = num_total = 0
		cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			for member in members:
				worker = member[3]
				if(uid==worker):
					num_total += 1
					if (cluster_correctness[i][1] == False): # if it's a bad cluster
						num_bad += 1
		return num_bad/num_total

	def get_worker_correct_rate(self, uid, clusters, correctness_threshold):
		return (1 - self.get_worker_error_rate(uid, clusters, correctness_threshold))

	def plot_workers_correct_rate(self, df, clustering_params, correctness_threshold, csv_filepath, img_filename, plot_title, bigger_window_size):
		
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		worker_list = self.ba.get_workers(df)
		
		correct_rates = []
		for worker in worker_list:
			correct_rate = self.get_worker_correct_rate(worker, clusters, correctness_threshold)
			correct_rates.append(correct_rate*100)

		y,x,_ = plt.hist(correct_rates, bins=np.arange(0,105, step=1)-0.5, color = 'g')

		plt.title(plot_title)
		plt.xticks(np.arange(0,105, step=5))
		plt.yticks(np.arange(0,y.max()+1, step=1))
		plt.xlabel("Fraction of the worker's annotations that were in a good cluster [%]")
		plt.ylabel("Quantity of workers")
		plt.show()

	# """
	# Input "clusters" is a df: centroid_x | centroid_y | members.
	# """
	# def sort_clusters_by_size(self, clusters):

	# 	# Find threshold (k-means).
	# 	total_list = []
	# 	for i in range(len(clusters.index)):
	# 		row = clusters.iloc[[i]]
	# 		members = row.iloc[0]['members']
	# 		worker_list = []
	# 		for member in members:
	# 			worker_list.append(member[3])
	# 		num_members = len(np.unique(worker_list))
	# 		total_list.append(num_members)
	# 	total_array = np.asarray(total_list)
	# 	km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
	# 	cluster_centers = km.cluster_centers_
	# 	threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

	# 	# Given threshold, sort.
	# 	small_clusters_list = []
	# 	large_clusters_list = []
	# 	small_counter = 0
	# 	large_counter = 0
	# 	for j in range(len(clusters.index)):
	# 		row = clusters.iloc[[j]]
	# 		members = row.iloc[0]['members']
	# 		centroid_x = row.iloc[0]['centroid_x']
	# 		centroid_y = row.iloc[0]['centroid_y']

	# 		worker_list = []
	# 		for member in members:
	# 			worker_list.append(member[3])
	# 		num_members = len(np.unique(worker_list))

	# 		if (num_members < threshold_kmeans):
	# 			small_clusters_list.append([centroid_x, centroid_y, members])
	# 			small_counter += 1
	# 		else:
	# 			large_clusters_list.append([centroid_x, centroid_y, members])
	# 			large_counter += 1

	# 	small_clusters = pd.DataFrame(index = range(small_counter), columns = ['centroid_x','centroid_y','members'])
	# 	large_clusters = pd.DataFrame(index = range(large_counter), columns = ['centroid_x','centroid_y','members'])

	# 	for k in range(small_counter):
	# 		small_clusters['centroid_x'][k] = small_clusters_list[k][0]
	# 		small_clusters['centroid_y'][k] = small_clusters_list[k][1]
	# 		small_clusters['members'][k] = small_clusters_list[k][2]

	# 	for m in range(large_counter):
	# 		large_clusters['centroid_x'][m] = large_clusters_list[m][0]
	# 		large_clusters['centroid_y'][m] = large_clusters_list[m][1]
	# 		large_clusters['members'][m] = large_clusters_list[m][2]

	# 	return small_clusters, large_clusters


	"""
	For a given dataset, take the subset of reference spots with an SNR > n 
	and calculate the fraction of that subset that were detected by the turkers. 
	Build a curve by varying n (3 through SNR_max) to see how high the minimum 
	SNR needs to be for 100% of the spots to be detected.
	"""
	def plot_snr_sensitivity(self, df, clustering_params, csv_filepath, img_height, img_filename, correctness_threshold, plot_title, bigger_window_size):
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		centroid_coords = anno_and_ref_df.loc[:, ['centroid_x', 'centroid_y']].as_matrix()		
		centroids_kdt = KDTree(centroid_coords, leaf_size=2, metric='euclidean')

		ref_df = pd.read_csv(csv_filepath)
		ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()	
		for i in range(len(ref_points)):			# flip vertical axis
			point = ref_points[i]
			first_elem = point[0]
			second_elem = img_height - point[1]
			point = np.array([first_elem, second_elem])
			ref_points[i] = point

		snr_val_list = ref_df.loc[:, ['snr']].as_matrix()	

		snr_min = math.floor(min([snr_val_list[i][0] for i in range(len(snr_val_list))]))
		snr_max = math.ceil(max([snr_val_list[i][0] for i in range(len(snr_val_list))]))
		n_list = range(snr_min,snr_max)
		fraction_list = []
		for n in n_list:
			spots_detected = 0
			spots_total = 0
			# for each spot
			for i in range(len(ref_points)):
				# get SNR
				snr = snr_val_list[i][0]
				if(snr<n):
					continue
				spots_total += 1
				ref_point = ref_points[i]
				# get nearest neighbor centroid
				dist, ind = centroids_kdt.query([ref_point], k=1)
				if (dist[0][0] <= correctness_threshold):
					spots_detected += 1
			if(spots_total == 0):
				fraction_list.append(0)
			else:
				fraction_list.append((spots_detected/spots_total)*100)
			#print ('min_SNR ={0:2d}, spots_detected ={1:3d}, spots_total ={2:3d}'.format(n, spots_detected, spots_total))
		plt.scatter(n_list, fraction_list, facecolors = 'g', s = 20)
		plt.plot(n_list, fraction_list, color = 'green')

		plt.title(plot_title)
		plt.xlabel("Minimum SNR of spots in subset")
		plt.ylabel("Fraction of subset of spots detected by workers [%]")
		plt.show()


	"""
	For each spot, plot SNR vs. number of annotations in the corresponding cluster.
	"""
	def plot_snr_vs_membership(self, df, clustering_params, csv_filepath, img_height, img_filename, correctness_threshold, bigger_window_size):

		clusters = self.get_clusters(df, clustering_params)			# this dataframe: centroid_x | centroid_y | members
		
		ref_df = pd.read_csv(csv_filepath)
		ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()	
		snr_val_list = ref_df.loc[:, ['snr']].as_matrix()	

		for i in range(len(ref_points)):			# flip vertical axis
			point = ref_points[i]
			first_elem = point[0]
			second_elem = img_height - point[1]
			point = np.array([first_elem, second_elem])
			ref_points[i] = point

		anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		centroid_coords = anno_and_ref_df.loc[:, ['centroid_x', 'centroid_y']].as_matrix()		
		centroids_kdt = KDTree(centroid_coords, leaf_size=2, metric='euclidean')

		counter_undetected = counter_detected = 0
		snr_undetected = []
		snr_detected = []

		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		# for each spot
		for i in range(len(ref_points)):
			ref_point = ref_points[i]

			# get SNR
			snr = snr_val_list[i][0]	

			# get nearest neighbor centroid
			dist, ind = centroids_kdt.query([ref_point], k=1)
			if (dist[0][0] > correctness_threshold):
				num_members = 0
				color = 'm'
				counter_undetected += 1
				snr_undetected.append(snr)
			else:
				centroid_coords_index = ind[0][0]
				nearest_centroid = centroid_coords[centroid_coords_index]
				nearest_centroid_x = nearest_centroid[0]
				nearest_cluster = clusters.loc[clusters['centroid_x']==nearest_centroid_x]
				members = nearest_cluster.iloc[0]['members']

				# get number of annotations associated with that centroid 
				num_members = len(members)
				color = 'g'
				counter_detected += 1
				snr_detected.append(snr)

			plt.scatter([num_members],[snr], facecolors = color, alpha = 0.5, s = 20)

		s_1 = str(counter_undetected) + " spots detected by no workers"
		s_2 = str(counter_detected) + " spots detected by at least one worker"

		legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor='g', label=s_2),
							Line2D([0],[0], marker='o', color='w', markerfacecolor='m', label=s_1)]

		plt.title("For each spot, SNR vs. number of clicks in the nearest cluster")
		plt.xlabel("Number of clicks in the nearest cluster")
		plt.ylabel("SNR")
		plt.legend(handles = legend_elements)
		plt.show()

		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		plt.title("Number of spots detected vs. SNR")
		y,x,_ = plt.hist(snr_detected, bins=np.arange(0,max(snr_detected)+2,1)-0.5, color = 'g')
		plt.xticks(np.arange(0,max(snr_detected)+2,step=1))
		if(max(snr_detected)<20):
			y_step = 1
		else:
			y_step = 2
		plt.yticks(np.arange(0,y.max()+1, step=y_step))
		plt.xlabel("SNR")
		plt.ylabel("Number of spots detected")
		plt.show()

		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		if (len(snr_undetected)==0):
			plt.title("No spots undetected")
		else:
			plt.title("Number of spots undetected vs. SNR")
			y,x,_ = plt.hist(snr_undetected, bins=np.arange(max(snr_undetected)+2)-0.5, color = 'm')
			plt.xticks(np.arange(0,max(snr_undetected)+2,step=1))
			plt.yticks(np.arange(0,y.max()+1, step=1))
			plt.xlabel("SNR")
			plt.ylabel("Number of spots undetected")
		plt.show()


	# def get_clusters_cropped(self, df, clustering_params, x_min, x_max, y_min, y_max):

	# 	clustering_alg = clustering_params[0]

	# 	if (clustering_alg not in self.clustering_algs):
	# 		raise ValueError('Invalid clustering algorithm name entered.')

	# 	if (clustering_alg == 'AffinityPropagation'):											# If AffinityPropagation is selected:
	# 		cluster_centroids_list = []																# Initialize a list of cluster centroids

	# 		if(len(clustering_params) != 2):														# Check that there's only one clustering parameter
	# 			raise ValueError('Please enter a list containing the preference parameter.')

	# 		click_properties = self.ba.get_click_properties(df)
	# 		coords = click_properties[:,:2]
	# 		coords_cropped = []															# Get all the coordinates from the annotation dataframe (dissociated from timestamps)
	# 		for coord in coords:
	# 			if (coord[0] > x_min):
	# 				if (coord[0] < x_max):
	# 					if (coord[1] > y_min):
	# 						if (coord[1] < y_max):
	# 							coords_cropped.append([coord[0], coord[1]])

	# 		af = self.get_cluster_object(coords_cropped, clustering_params)

	# 		cluster_centers_indices = af.cluster_centers_indices_									# Get the indices of the cluster centers (list)
	# 		num_clusters = len(cluster_centers_indices)

	# 		labels = af.labels_																		# Each point that was in coords now has a label saying which cluster it belongs to.

	# 		cluster_members_lists = [None]*num_clusters
	# 		for i in range(len(cluster_members_lists)):
	# 			cluster_members_lists[i] = []

	# 		# for j in range(len(click_properties)):
	# 		# 	index = labels[j]
	# 		# 	cluster_members_lists[index].append(click_properties[j])

	# 		for k in range(num_clusters):
	# 			cluster_centers = coords[cluster_centers_indices[k]]	# np array
	# 			cluster_centroids_list.append(cluster_centers)

	# 	return cluster_centroids_list
	"""
	Checks to see whether the cluster object has already been generated
	for the given df and clustering parameters and returns or calculates
	appropriately.
	""" 
	def get_cluster_object(self, coords, clustering_params):

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

	"""
	Inputs:
		string name of clustering alg to use
		df with annotation data (should already be cropped)
		list of clustering params for clustering alg
		csv_filepath (contains reference data)
		img_filename (the cropping)
	Returns:
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
	def anno_and_ref_to_df(self, df, clustering_params, csv_filepath, img_filename):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		clusters = self.get_clusters(anno_one_crop, clustering_params)
		img_height = anno_one_crop['height'].values[0]
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

	"""
	Inputs:
		df in this form: centroid_x | centroid_y | x of nearest ref | y of nearest ref | NN_dist | members (x | y | time_spent | worker_id)
			* the index is the Centroid ID
		int threshold
			for each centroid, if NN_dist <= threshold, centroid is "correct"
	Returns:
		2-column array with a row for each centroid
			column 0 = Centroid ID
			column 1 = True if centroid is "correct", False if centroid is "incorrect"
	"""
	def get_cluster_correctness(self, df, threshold):
		num_centroids = df.shape[0]
		to_return = np.empty([num_centroids, 2])
		for i in range(num_centroids):
			to_return[i] = i
			NN_dist = df['NN_dist'][i]
			if (NN_dist <= threshold):
				to_return[i][1] = True
			else:
				to_return[i][1] = False
		return to_return



	"""
	Inputs:
		df with annotation data
		k-d tree with reference points, aka "ground truth" values
		img_filename to crop to
	Returns:
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
			coords = self.ba.get_click_properties(anno)[:,:2]
			dist, ind = ref_kdt.query(coords, k=1)
			dist_list = dist.tolist()
			values = []
			for i in range(len(dist_list)):
				values.append(dist_list[i][0])
			to_return.append(values)
		return to_return

	"""
	Inputs:
		df with annotation data
		img_filename to crop to
	Returns:
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

	def plot_cluster_membership_in_region(self, clusters, img_height, x_min, x_max, y_min_on_plot, y_max_on_plot, img_name, density):
		y_min = img_height - y_max_on_plot
		y_max = img_height - y_min_on_plot

		# Keep only the clusters in the region of interest
		clusters = clusters[clusters.centroid_x > x_min]
		clusters = clusters[clusters.centroid_x < x_max]
		clusters = clusters[clusters.centroid_y > y_min]
		clusters = clusters[clusters.centroid_y < y_max]

		# Get number of members per cluster in region of interest
		num_members_list = []
		for i in range(len(clusters.index)):
		    row = clusters.iloc[[i]]
		    members = row.iloc[0]['members']
		    worker_list = []
		    for member in members:
		        worker_list.append(member[3])
		    num_members = len(np.unique(worker_list))
		    num_members_list.append(num_members)
		    
		# Plotting
		fig = plt.figure(figsize = (7.5,4))
		plt.hist(num_members_list, bins = np.arange(0,max(num_members_list)+2,step=2)-1)
		plt.xticks(np.arange(0,max(num_members_list),step=2))

		plt.xlabel("Number of unique annotators per cluster")
		plt.ylabel("Number of clusters")
		plt.title(density + " Region: "+ str(len(clusters.index))+ " Clusters")
		plt.show()

		avg_str = "Average number of unique workers in a cluster = " + str(math.floor(np.mean(num_members_list)))
		print(avg_str)
		std_str = "Standard deviation = " + str(math.floor(np.std(num_members_list)))
		print(std_str)

		img = mpimg.imread(img_name)
		fig = plt.figure(figsize = (6,6))
		plt.imshow(img)
		plt.xticks([])
		plt.yticks([])
		plt.show()

	# plot annotations in a certain area
	def plot_annotations_zoom(self, df, x_min, x_max, y_min, y_max, img_height, clustering_params, img_filepath, show_clusters, show_workers, cluster_marker_size, worker_marker_size):
		img = mpimg.imread(img_filepath)

		plt.imshow(img, cmap = 'gray')

		click_properties = self.ba.get_click_properties(df)
		coords = click_properties[:,:2]
		coords_cropped = []															# Get all the coordinates from the annotation dataframe (dissociated from timestamps)
		for coord in coords:
			if (coord[0] > x_min):
				if (coord[0] < x_max):
					if (coord[1] > y_min):
						if (coord[1] < y_max):
							coords_cropped.append([coord[0], coord[1]])
		
		coord_list = []
		for coord in coords_cropped:
			x = coord[0]-x_min
			y = img_height-coord[1]-y_min-130
			coord_list.append([x,y])
			plt.scatter([x], [y], s=4, facecolors='b')

		af = self.get_cluster_object(coord_list, clustering_params)

		cluster_centers_indices = af.cluster_centers_indices_									# Get the indices of the cluster centers (list)
		num_clusters = len(cluster_centers_indices)

		labels = af.labels_																		# Each point that was in coords now has a label saying which cluster it belongs to.

		cluster_centroids_list = []
		for k in range(num_clusters):
			cluster_center = coord_list[cluster_centers_indices[k]]	# np array
			plt.scatter([cluster_center[0]], [cluster_center[1]], s = 20, facecolors = 'none', edgecolors = 'y')

		plt.title('Worker Annotations and Cluster Centroids')
		plt.xticks(np.arange(0,x_max-x_min, step=20))
		plt.yticks(np.arange(0,y_max-y_min, step=20))
		plt.show()


	"""
	Quick visualization of worker annotations, clusters, and/or annotation and cluster "correctness." 

	Inputs:
		pandas df with annotation data
		string img_filename to crop to
		string csv_filepath with reference data
		int size of worker marker
		int size of cluster centroid marker
		bool whether to plot reference annotations
		bool whether to plot workers
		bool whether to plot cluster centroids
		bool whether to color worker markers green/magenta to indicate "correctness"
		bool whether to color centroid markers green/magenta to indicate "correctness"
		bool whether to color an incorrect cluster's nearest neighbor the same color as the incorrect cluster
		int threshold distance from centroid to the nearest reference annotation beyond which the entire cluster is "incorrect"
		string name of clustering algorithm
		list of clustering parameters
		bool whether to use bigger window size (for jupyter notebook)
	Returns:
		none
	"""
	def plot_annotations(self, df, img_filename, img_filepath, csv_filepath, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_params, bigger_window_size):
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)
		if show_clusters or show_correctness_workers:

			if csv_filepath is None:
				clusters = self.get_clusters(df, clustering_params)
			else:
				clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
			
			member_lists = clusters['members'].values	# list of lists

			if correctness_threshold is not None:
				cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)

		img_height = anno_one_crop['height'].values[0]

		if show_workers:

			if show_correctness_workers:
				for i in range(len(member_lists)):			# for every cluster
					members = member_lists[i]					# get the list of annotations (w/ click properties) in that cluster
					if (cluster_correctness[i][1]):
						color = 'g'						
					else:								
						color = 'm'
					for member in members:						# plot each annotation in that cluster
						coords = member[:2]
						plt.scatter([coords[0]], self.ba.flip([coords[1]], img_height), s = worker_marker_size, facecolors = color, alpha = 0.5)

			else:
				handle_list = []
				for worker, color in zip(worker_list, self.colors):			# For each worker, use a different color.
				    anno = self.ba.slice_by_worker(anno_one_crop, worker)		
				    coords = self.ba.get_click_properties(anno)[:,:2]
				    x_coords = coords[:,0]
				    y_coords = coords[:,1]
				    y_coords_flipped = self.ba.flip(y_coords, img_height)
				    handle = plt.scatter(x_coords, y_coords_flipped, s = worker_marker_size, facecolors = color, alpha = 0.5, label = worker)
				    handle_list.append(handle)
				plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.2, 1.015))

			if not show_clusters:
				plt.title('Worker Annotations')

		if show_clusters:

			x_coords = clusters['centroid_x'].values
			y_coords = clusters['centroid_y'].values
			y_coords_flipped = self.ba.flip(y_coords, img_height)

			color_index = 0		

			if show_correctness_clusters:
				for i in range(len(member_lists)):			# for every cluster
					if (cluster_correctness[i][1]):
						color = 'g'								
					else:
						if show_NN_inc:
							color = self.colors[color_index]							
							color_index = (color_index+1)%len(self.colors)
							plt.scatter([clusters['NN_x'].values[i]], [img_height-clusters['NN_y'].values[i]], s = worker_marker_size*2, facecolors = color, edgecolors = color)
						else:
							color = 'm'
					plt.scatter(x_coords[i], y_coords_flipped[i], s = cluster_marker_size, facecolors = 'none', edgecolors = color)					

			else:
				plt.scatter(x_coords, y_coords_flipped, s = cluster_marker_size, facecolors = 'none', edgecolors = '#ffffff')

			if not show_workers:
				plt.title('Cluster Centroids')

		if show_workers and show_clusters:
			plt.title('Worker Annotations and Cluster Centroids')

		if show_ref_points:
			ref_df = pd.read_csv(csv_filepath)							# plot reference points			
			ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()
			for point in ref_points:													
				plt.scatter([point[0]], [point[1]], s = 20, facecolors = 'y')
			legend_list = [Line2D([0],[0], marker='o', color='w', markerfacecolor='y', label='Reference points')]
			if show_workers and not show_correctness_workers:
				legend_list += handle_list
			plt.legend(handles = legend_list, loc = 9, bbox_to_anchor = (1.2, 1.015))
		img = mpimg.imread(img_filepath)
		plt.tick_params(
			axis='both',
			which='both',
			bottom=False,
			top=False,
			left=False,
			right=False)
		plt.imshow(img, cmap = 'gray')

		plt.show()
		if show_clusters or show_correctness_workers:
			return clusters

	def get_clumped_list(self, clusters):
		clumped_list = []
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			workers = []
			for member in members:
				workers.append(member[3])
			unique_workers = np.unique(workers)

			num_instances_list = []
			for unique_worker in unique_workers:
				num_instances_list.append(workers.count(unique_worker))

			singles = num_instances_list.count(1)
			single_fraction = singles/len(unique_workers)
			if (single_fraction < 0.6):
				clumped_list.append(i)
		return clumped_list

	def get_cluster_means(self, clusters):
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

	def plot_clusters(self, clusters, img_filename, img_filepath, img_height, show_ref_points, csv_filepath, worker_marker_size, cluster_marker_size, show_correctness, correctness_threshold, show_possible_clumps, bigger_window_size, plot_title):
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))
		
		member_lists = clusters['members'].values	# list of lists
		color = 'orange'

		if show_ref_points:
			ref_kdt = self.csv_to_kdt(csv_filepath, img_height)

		if show_correctness:
			cluster_correctness = []
			for i in range(len(clusters.index)):
				row = clusters.iloc[[i]]
				centroid_x = row.iloc[0]['centroid_x']
				centroid_y = row.iloc[0]['centroid_y']
				coord = np.asarray([centroid_x, centroid_y]).reshape(1,-1)
				dist, ind = ref_kdt.query(coord, k=1)
				distance = dist[0][0]
				if (distance <= correctness_threshold):
					cluster_correctness.append([i,True])
				else:
					cluster_correctness.append([i,False])

			for i in range(len(member_lists)):			# for every cluster
				members = member_lists[i]					# get the list of annotations (w/ click properties) in that cluster
				if show_correctness:
					if (cluster_correctness[i][1]):
						color = 'g'						
					else:								
						color = 'm'
				else:
					color = 'orange'

			for member in members:						# plot each annotation in that cluster
				coords = member[:2]
				plt.scatter([coords[0]], self.ba.flip([coords[1]], img_height), s = worker_marker_size, facecolors = color, alpha = 0.5)

		# plot cluster centroids
		x_coords = clusters['centroid_x'].values
		y_coords = clusters['centroid_y'].values
		y_coords_flipped = self.ba.flip(y_coords, img_height)
		plt.scatter(x_coords, y_coords_flipped, s = cluster_marker_size, facecolors = 'none', edgecolors = '#ffffff')

		legend_elements = []
		if show_ref_points:
			ref_df = pd.read_csv(csv_filepath)
			ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()
			for point in ref_points:
				plt.scatter([point[0]], [point[1]], s = 20, facecolors = 'c')
			legend_elements.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='c', label='reference spots'))

		legend_elements.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', label='annotations for clusters detected as clumpy'))
		plt.legend(handles = legend_elements, loc = 9, bbox_to_anchor = (1.2, 1.015))

		# plot image
		img = mpimg.imread(img_filepath)
		plt.imshow(img, cmap = 'gray')

		plt.tick_params(
			axis='both',
			which='both',
			bottom=False,
			top=False,
			left=False,
			right=False)

		plt.title(plot_title)

		plt.show()

	"""
	Plots the average time spent per click for all workers 
	in the dataframe.

	Input:
		dataframe with annotation data
		bool whether to use a bigger window size (for jupyter notebook)
	Returns:
		none
	"""
	def plot_avg_time_per_click(self, df, bigger_window_size):
		avg_list = []
		for worker in self.ba.get_workers(df):
			avg_time = self.ba.get_avg_time_per_click(df, worker)
			avg_list.append(avg_time/1000)
		n_bins = 10
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))
		y,x,_ = plt.hist(avg_list, bins = np.arange(0,max(avg_list)+0.5, step=0.25)-0.125)
		plt.title('Average time spent per click')
		plt.xticks(np.arange(0,max(avg_list)+0.5, step=0.25))
		plt.yticks(np.arange(0,y.max()+1, step=1))
		plt.xlabel('Time [s]')
		plt.ylabel('Quantity of workers')
		plt.show()

	"""
	Description:
		For each annotation (each click) in a dataframe, 
		plot nearest neighbor distance (nnd) vs. time spent. 
		Each point represents one annotation (one click). 
		All workers on one plot, colored by worker ID.
		Can color each point by correctness. 
	Implementation notes:
		if show_correctness
			can't use calc_time_per_click and calc_distances, because need
			to look at coordinates one by one and get time_spent on coordinate, 
			NND of associated centroid, and correctness of associated centroid.
		if not show_correctness, it's better to use calc_time_per_click and 
			calc_distances, which do not require clustering.
	Inputs:
		dataframe
		img_filename (the cropping)
		csv_filepath (contains reference data)
		bool whether to color each point by correctness of cluster
		correctness_threshold
		clustering_params
	Returns:
		none
	"""
	def plot_nnd_vs_time_spent(self, df, img_filename, csv_filepath, show_correctness, correctness_threshold, clustering_params):
# heeeere
		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		fig = plt.figure(figsize = (10,7))
		img_height = anno_one_crop['height'].values[0]
		ref_kdt = self.csv_to_kdt(csv_filepath, img_height)

		if show_correctness:

			# Goal: for each coordinate in coords, plot NND vs. time_spent and color with correctness
			# Run Af on all annotation coords (just once) and get labels (a list with a label for each annotation coordinate).
			# For each coordinate in coords:
			#		time_spent: pull from coords_with_times
			#		NND: query using a kdtree.
			#		correctness: index of coordinate is i=index of label. label[i] is index of correctness. correctness[index] is the appropriate correctness.
			#		aaaaand... plot NND vs. time_spent and color with correctness!

			coords = self.ba.get_click_properties(anno_one_crop)[:,:2]
			coords_with_times = self.ba.get_click_properties(anno_one_crop)[:,:3]
			clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)	# clusters -> NND, coordinates
			cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)		# clusters <-> correctness
			af = self.get_cluster_object(coords, clustering_params)
			labels = af.labels_	

			for i in range(len(coords)):
				time_spent = coords_with_times[i][2]
				if(time_spent==0):
					continue
				coordinate = coords[i]		# a single coordinate
				dist, ind = ref_kdt.query([coordinate], k=1)
				NND = dist[0][0]
				index = labels[i]			# label[i] is the index of the cluster affiliated with this coordinate
				if(cluster_correctness[index][1]):
					color = 'g'
					marker_size = 4
					alpha_selection = 0.25
				else:
					color = 'm'
					marker_size = 20
					alpha_selection = 1
				plt.scatter([time_spent], [NND], s = marker_size, facecolors = color, edgecolors = None, alpha = alpha_selection)

		else:
			worker_list = self.ba.get_workers(anno_one_crop)
			time_list = self.calc_time_per_click(anno_one_crop, img_filename)		# list containing one list for each worker
			dist_list = self.calc_distances(anno_one_crop, ref_kdt, img_filename)	# list containing one list for each worker
			handle_list = []
			for i in range(len(worker_list)):			# for each worker
				color = self.colors[i]
				x_coords = time_list[i]
				y_coords = dist_list[i]
				handle = plt.scatter(x_coords, y_coords, s = 8, facecolors = color, alpha = 0.5, label = worker_list[i])
				handle_list.append(handle)
			plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.2, 1.015))
			plt.subplots_adjust(left=0.1, right=0.75)

		plt.title('Nearest Neighbor Distance (NND) vs. Time Spent For Each Click [s]')
		plt.xlabel('Time Spent [ms]')
		plt.ylabel('Nearest Neighbor Distance (NND)')
		plt.show()

	"""
	For each annotation (each click) in a dataframe, 
	plot nearest neighbor distance (nnd) vs. worker index. 
	Each point represents one annotation (one click). 
	Can color each point by correctness. 

	Inputs:
		dataframe
		img_filename (the cropping)
		csv_filepath (contains reference data)
		bool whether to color each point by correctness of cluster
		correctness_threshold
		clustering_params
	Returns:
		none
	"""
	def plot_nnd_vs_worker_index(self, df, img_filename, csv_filepath, show_correctness, correctness_threshold, clustering_params, show_avgs):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)
		img_height = anno_one_crop['height'].values[0]
		ref_kdt = self.csv_to_kdt(csv_filepath, img_height)
		dist_list = self.calc_distances(anno_one_crop, ref_kdt, img_filename)	# list containing one list for each worker

		fig = plt.figure(figsize = (10,7))

		# plot all clicks
		if show_correctness:
			click_properties = self.ba.get_click_properties(anno_one_crop)		
			coords = click_properties[:,:2]

			clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)	# clusters -> NND, coordinates
			cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)		# clusters <-> correctness
			af = self.get_cluster_object(coords, clustering_params)
			labels = af.labels_	
			img_height = anno_one_crop['height'].values[0]
			ref_kdt = self.csv_to_kdt(csv_filepath, img_height)

			for i in range(len(coords)):
				worker_id = click_properties[i][3]
				worker_index = np.where(worker_list == worker_id)

				coordinate = coords[i]
				dist, ind = ref_kdt.query([coordinate], k=1)
				NND = dist[0][0]
				index = labels[i]
				if(cluster_correctness[index][1]):
					color = 'g'
					marker_selection = 'o'
					marker_size = 4
					alpha_selection = 1
				else:
					color = 'm'
					marker_selection = '_'
					marker_size = 40
					alpha_selection = 1
				plt.scatter([worker_index], [NND], s = marker_size, facecolors = color, edgecolors = None, marker = marker_selection, alpha = alpha_selection)

		else:
			for i in range(len(worker_list)):			# for each worker
				x_coords = [i]*len(dist_list[i])
				y_coords = dist_list[i]
				plt.scatter(x_coords, y_coords, s = 4, alpha = 0.5, facecolors = 'c')

		# plot worker average distances
		if show_avgs:
			avg_distances = []
			for i in range(len(worker_list)):
				worker_distances = dist_list[i]
				worker_avg_dist = np.average(worker_distances)
				avg_distances.append(worker_avg_dist) 
			handle = plt.scatter(range(len(worker_list)), avg_distances, s = 60, facecolors = 'b', marker = '_', label = 'Average NND')
			plt.legend(handles = [handle], loc = 9, bbox_to_anchor = (1.15, 0.55))
			plt.subplots_adjust(left=0.1, right=0.8)

		plt.title('Nearest Neighbor Distance (NND) vs. Worker Index For Each Click')
		plt.xlabel('Worker Index')
		plt.ylabel('Nearest Neighbor Distance (NND)')
		plt.xticks(np.arange(0, len(worker_list), step=1))
		plt.show()

	"""
	For each annotation (each click) in a dataframe, 
	plot time spent on the click vs. worker index. 
	Each point represents one annotation (one click). 
	Can color each point by correctness. 

	Inputs:
		dataframe
		img_filename (the cropping)
		csv_filepath
		bool whether to color each point by correctness of cluster
		correctness_threshold
		clustering_params
	Returns:
		none
	"""
	def plot_time_spent_vs_worker_index(self, df, img_filename, csv_filepath, show_correctness, correctness_threshold, clustering_params, show_avgs):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)			# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)
		time_list = self.calc_time_per_click(anno_one_crop, img_filename)	# list containing one list for each worker

		fig = plt.figure(figsize = (10,7))

		# plot all clicks
		if show_correctness:
			click_properties = self.ba.get_click_properties(anno_one_crop)		# coordinates <-> time_spent
			coords = click_properties[:,:2]
			clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)	# clusters -> NND, coordinates
			cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)		# clusters <-> correctness
			af = self.get_cluster_object(coords, clustering_params)
			labels = af.labels_	
			img_height = anno_one_crop['height'].values[0]
			ref_kdt = self.csv_to_kdt(csv_filepath, img_height)

			for i in range(len(coords)):
				time_spent = click_properties[i][2]
				worker_id = click_properties[i][3]
				worker_index = np.where(worker_list == worker_id)

				coordinate = coords[i]
				dist, ind = ref_kdt.query([coordinate], k=1)
				index = labels[i]
				if(cluster_correctness[index][1]):
					color = 'g'
					marker_selection = 'o'
					marker_size = 10
					alpha_selection = 1
				else:
					color = 'm'
					marker_selection = '_'
					marker_size = 40
					alpha_selection = 1
				plt.scatter([worker_index], [time_spent], s = marker_size, facecolors = color, edgecolors = None, marker = marker_selection, alpha = alpha_selection)
		else:	
			for i in range(len(worker_list)):		# for each worker
				x_coords = [i]*len(time_list[i])
				y_coords = time_list[i]
				y_coords.pop(0)						# discard initial fencepost in time_list
				x_coords.pop(0)						# discard corresponding initial entry
				plt.scatter(x_coords, y_coords, s = 4, alpha = 0.5, facecolors = 'c')

		# plot worker average times
		if show_avgs:
			avg_times = []
			for i in range(len(worker_list)):
				worker_times = time_list[i]
				if not worker_times:				# if list of worker times is empty
					avg_times.append(0)
					continue
				worker_times.pop(0)
				worker_avg_time = np.average(worker_times)
				avg_times.append(worker_avg_time/1000) 
			handle = plt.scatter(range(len(worker_list)), avg_times, s = 60, facecolors = 'b', marker = '_', label = 'Average time spent')
			plt.legend(handles = [handle], loc = 9, bbox_to_anchor = (1.15, 0.55))
			plt.subplots_adjust(left=0.1, right=0.8)

		plt.title('Time Spent [s] vs. Worker Index')
		plt.xlabel('Worker Index')
		plt.ylabel('Time Spent [s]')
		plt.xticks(np.arange(0, len(worker_list), step=1))
		plt.show()

	"""
	For each worker, plot total time spent vs. worker index. 
	Each bar represents one worker. 

	Inputs:
		dataframe
		img_filename (the cropping)
	Returns:
		none
	"""
	def plot_total_time_vs_worker_index(self, df, img_filename):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)			# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)

		fig = plt.figure(figsize = (10,7))

		handle_list = []
		for i in range(len(worker_list)):
			total_time = self.ba.get_total_time(anno_one_crop, worker_list[i])
			handle = plt.bar(i, total_time[0]/1000, color = self.colors[i], label = worker_list[i])
			handle_list.append(handle)

		plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.15, 1.015))
		plt.subplots_adjust(left=0.1, right=0.8)
		plt.title('Total Time Spent [s] vs. Worker Index')
		plt.xlabel('Worker Index')
		plt.ylabel('Time Spent [s]')
		plt.xticks(np.arange(0, len(worker_list), step=1))
		plt.show()

	def plot_total_worker_time_hist(self, df, bigger_window_size):
		total_time_list = []
		for worker in self.ba.get_workers(df):
			total_time = self.ba.get_total_time(df, worker)
			total_time_list.append(total_time[0]/1000)
		n_bins = 10
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))
		plt.hist(total_time_list, bins=np.arange(0, max(total_time_list)+20, step=10)-5)
		plt.title('Total Time Spent by Workers [s]')
		plt.xlabel('Time Spent [s]')
		plt.ylabel('Quantity of Workers')
		x_step = 10
		if ((max(total_time_list)) > 100):
			x_step = 20
		if ((max(total_time_list)) > 250):
			x_step = 50
		plt.xticks(np.arange(0, max(total_time_list)+20, step=10))
		plt.show()

	"""
	For one worker in a dataframe,
	plot time spent on click vs. index of that click. 
	Each point represents one annotation (one click).

	Inputs:
		dataframe
		img_filename (the cropping)
		uid (worker ID)
	Returns:
		none
	"""
	def plot_time_spent_vs_click_index(self, df, img_filename, csv_filepath, uid, show_correctness, correctness_threshold, clustering_params):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)
		fig = plt.figure(figsize = (10,7))
		anno_one_worker = self.ba.slice_by_worker(anno_one_crop, uid)

		if show_correctness:
			click_properties = self.ba.get_click_properties(anno_one_worker)		
			coords = click_properties[:,:2]
			clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)	# clusters -> NND, coordinates
			cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)		# clusters <-> correctness
			af = self.get_cluster_object(coords, clustering_params)
			labels = af.labels_
			img_height = anno_one_worker['height'].values[0]
			ref_kdt = self.csv_to_kdt(csv_filepath, img_height)
			num_clicks = len(coords)

			for i in range(num_clicks):
				time_spent = click_properties[i][2]
				click_index = i

				coordinate = coords[i]
				dist, ind = ref_kdt.query([coordinate], k=1)
				index = labels[i]
				if not isinstance(index, int):
					continue
				if(cluster_correctness[index][1]):
					color = 'g'
					marker_selection = 'o'
					marker_size = 10
					alpha_selection = 1
				else:
					color = 'm'
					marker_selection = 'o'
					marker_size = 40
					alpha_selection = 1
				plt.scatter([click_index], [time_spent], s = marker_size, facecolors = color, edgecolors = None, marker = marker_selection, alpha = alpha_selection)

		else:
			index = np.where(worker_list == uid)
			i = index[0][0]		# because np.where() returns a tuple containing an array
			time_list = self.calc_time_per_click(anno_one_crop, img_filename)	# list containing one list for each worker
			worker_time_list = time_list[i]
			num_clicks = len(worker_time_list)
			x_coords = range(num_clicks)
			y_coords = [x / 1000 for x in worker_time_list]
			handle = plt.scatter(x_coords, y_coords, s = 4, facecolors = 'c', label = 'One click')
			plt.legend(handles = [handle], loc = 9, bbox_to_anchor = (1.15, 0.55))
			plt.subplots_adjust(left=0.1, right=0.8)
		
		plt.title('Time Spent [s] vs. Click Index for Worker ' + uid)
		plt.xlabel('Click Index')
		plt.ylabel('Time Spent [s]')
		plt.xticks(np.arange(0, num_clicks, step=10))
		plt.show()


		"""
		[header] plotter methods - to curate
		"""

	"""
	Build curve by varying number of unique workers required for valid cluster.
	"""
	def plot_cluster_membership_threshold_roc(self, df_1, df_2, clustering_params, csv_filepath_1, csv_filepath_2, img_height, correctness_threshold, plot_title, bigger_window_size):
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		tpr_list_1, fpr_list_1 = self.get_tpr_fpr_lists(df_1, clustering_params, csv_filepath_1, img_height, correctness_threshold)
		tpr_list_2, fpr_list_2 = self.get_tpr_fpr_lists(df_2, clustering_params, csv_filepath_2, img_height, correctness_threshold)

		plt.scatter(fpr_list_1, tpr_list_1, facecolors = 'blue', s = 20)
		plt.scatter(fpr_list_2, tpr_list_2, facecolors = 'orange', s = 20)

		leg_elem_1 = Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', label='inverted spot image')
		leg_elem_2 = Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', label='original spot image')
		legend_elements = [leg_elem_1, leg_elem_2]
		plt.legend(handles = legend_elements, bbox_to_anchor=(1.01, 1), loc=2)
		plt.yticks(np.arange(0,1.01,step=0.1))
		plt.title(plot_title)
		plt.xlabel("False positive rate")
		plt.ylabel("True positive rate")
		plt.show()

	def get_tpr_fpr_lists(self, df, clustering_params, csv_filepath, img_height, correctness_threshold):
		worker_list = self.ba.get_workers(df)
		num_workers = len(worker_list)
		all_clusters = self.get_clusters(df, clustering_params)
		tpr_list = []
		fpr_list = []

		for threshold in range(num_workers):
			small_clusters, large_clusters = self.sort_clusters_by_size_input_threshold(all_clusters, threshold)
			anno_and_ref_df = self.anno_and_ref_to_df_input_clusters(large_clusters, csv_filepath, img_height)
			cluster_correctness = self.get_cluster_correctness(anno_and_ref_df, correctness_threshold)

			# Get total number of spots
			ref_kdt = self.csv_to_kdt(csv_filepath, img_height)
			ref_array = np.asarray(ref_kdt.data)
			num_spots_total = len(ref_array)

			# Get number of clusters which are spots and number of clusters which are not spots
			num_true_positives = 0
			num_false_positives = 0
			num_clusters_total = 0
			for i in range(len(anno_and_ref_df.index)):		# sort clusters
				if (cluster_correctness[i][1]):		
					num_true_positives += 1
				else:
					num_false_positives += 1
				num_clusters_total += 1

			tpr = num_true_positives/num_spots_total
			if (tpr > 1):		# no double counting
				tpr = 1
			if (num_clusters_total == 0):
				fpr = 0
			else:
				fpr = num_false_positives/num_clusters_total
			
			tpr_list.append(tpr)
			fpr_list.append(fpr)

		return tpr_list, fpr_list






	"""
	[header] these were moved over from BA class
	"""








	# Returns dataframe with all fast clicks screened
	# Clicks of which time_spent < time_threshold are "fast"
	def screen_clicks_time_spent(self, df, time_threshold):
		to_return = pd.DataFrame()
		occasions = np.unique(df.loc[:, ['time_when_completed']].as_matrix())			# get the list of occasions
		for occasion in occasions:
			one_occasion_df = df[df.time_when_completed == occasion]
			one_occasion_timestamps = one_occasion_df.loc[:, ['timestamp']].as_matrix()
			for i in range(len(one_occasion_timestamps)-1, -1, -1):
				if(i==0):
					one_occasion_df = one_occasion_df.drop([i])
				else:
					time_spent = one_occasion_timestamps[i][0] - one_occasion_timestamps[i-1][0]
					if(time_spent<time_threshold):
						one_occasion_df = one_occasion_df.drop([i])
			to_return = to_return.append(one_occasion_df)
		return to_return

	# Inputs: df, worker ID
	# Returns: float avg time that the worker spent per click 
	def get_avg_time_per_click(self, df, uid):

		worker_timestamps = self.get_timestamps(df, uid)
		time_spent = max(worker_timestamps) - min(worker_timestamps)
		num_clicks = len(worker_timestamps)
		return time_spent[0]/num_clicks

	# Inputs: df, worker ID
	# Returns: time that the worker spent 		
	def get_total_time(self, df, uid):
		worker_timestamps = self.get_timestamps(df, uid)
		return max(worker_timestamps) - min(worker_timestamps)



	"""
	[header] these are probably no longer useful
	"""





	def test_alg(self, df, clustering_params):

		# 1. Cluster workers with good pairwise scores.
		df_good_workers_pairwise = self.slice_workers_by_pairwise_scores(df)
		clusters_good_workers_pairwise = self.get_clusters(df_good_workers_pairwise, clustering_params)

		# 2. Look at clusters from Step 1. Sort clusters with few/many workers annotating (“putatively incorrect/correct”).
		small_clusters, large_clusters = self.sort_clusters_by_size(clusters_good_workers_pairwise)

		# 3. Look at all workers. Sort workers who are in few/many "putatively correct" clusters.
		other_crowd, good_crowd = self.sort_workers_by_membership_in_large_clusters(df, large_clusters)

		# 4. Keep "putatively incorrect" clusters which are mostly comprised of workers who are in many "putatively correct" clusters.
		self.plot_fraction_from_crowd_per_cluster(small_clusters, good_crowd)

		# 5. Keep "putatively correct" clusters which are mostly comprised of workers who are in many "putatively correct" clusters.
		self.plot_fraction_from_crowd_per_cluster(large_clusters, good_crowd)

