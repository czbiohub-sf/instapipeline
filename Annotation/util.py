""" 
This module contains utilities used by the spot annotation analysis pipeline.
"""

# ----- #

import numpy as np

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

# ----- #

"""
Dataframe manipulation
"""

def print_head(df):
	""" Print the first five lines of df 
	"""
	print(df.head(n=5))

def get_workers(df):
	""" Return a numpy array of unique workers in df 
	"""
	uid_list = df.loc[:, ['worker_id']]
	return np.unique(uid_list)

def get_images(df):
	""" Return a numpy array of unique image filenames in df 
	"""
	img_list = df.loc[:, ['image_filename']]
	return np.unique(img_list)

def get_timestamps(df):
	""" Return a list of timestamps in df 
	"""
	matrix = df.loc[:, ['timestamp']].as_matrix()
	return [x[0] for x in matrix]

def slice_by_worker(df, uid):
	""" Return a dataframe with annotations for only one worker

	Parameters
	----------
	df : pandas dataframe
	uid : user ID of worker

	Returns
	-------
	Dataframe with annotations for only that worker 
	"""
	return df[df.worker_id == uid]

# def slice_by_image(df, img_filename):
# 	""" Return a dataframe with annotations for one image

# 	Parameters
# 	----------
# 	df : pandas dataframe
# 	img_filename : string filename of image

# 	Returns
# 	-------
# 	Dataframe with annotations for only that image 
#	
#	No longer useful because each qa object gets data from only one image

# 	"""
# 	return df[df.image_filename == img_filename]

def get_click_properties(df):
	""" Return a numpy array containing properties for all clicks in df

	Parameters
	----------
	df : pandas dataframe

	Returns
	-------
	numpy array
		each row corresponds with one annotation in the dataframe
		columns:
			x coord
			y coord
			time spent (time_spent = 0 indicates first click of an occasion (fencepost case))
			string worker ID
	"""
	occasions = np.unique(df.loc[:, ['time_when_completed']].as_matrix())			# get the list of occasions
	to_return = np.array([]).reshape(0,4)
	for occasion in occasions:
		one_occasion_df = df[df.time_when_completed == occasion]							
		one_occasion_array = one_occasion_df.loc[:, ['x', 'y', 'timestamp', 'worker_id']].as_matrix()
		for i in range(len(one_occasion_array)-1, -1, -1):
			if(i==0):
				time_spent = 0
			else:
				time_spent = one_occasion_array[i][2] - one_occasion_array[i-1][2]
			one_occasion_array[i][2] = time_spent
		to_return = np.vstack([to_return, one_occasion_array])
	return to_return

"""
Other data structure manipulation
"""

def flip(vec, height):
	""" Flip the values of a list about a height
	Useful for flipping y axis to plotting over an image with a flipped coordinate system.

	Parameters
	----------
	vec : list of values to be flipped
	height : height about which to flip values

	Returns
	-------
	flipped list
	"""
	to_return = [None]*len(vec)
	for i in range(len(vec)):
		to_return[i] = height - vec[i]
	return to_return

def csv_to_kdt(csv_filepath, img_height):
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

"""
Pair scores
"""

def get_pair_scores(df):
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

	worker_list = util.get_workers(df)
	pair_scores = pd.DataFrame(index = worker_list, columns = worker_list)
	for worker in worker_list:
		worker_df = util.slice_by_worker(df, worker)
		worker_coords = util.get_click_properties(worker_df)[:,:2]
		worker_kdt = KDTree(worker_coords, leaf_size=2, metric='euclidean')

		for other_worker in worker_list:
			if worker == other_worker:
				pair_scores[worker][other_worker] = 0
				continue

			other_worker_df = util.slice_by_worker(df, other_worker)
			other_worker_coords = util.get_click_properties(other_worker_df)[:,:2]
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

def get_worker_pair_scores(df):
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
	worker_list = util.get_workers(df)
	pair_scores = util.get_pair_scores(df)
	worker_scores = pd.DataFrame(index = worker_list, columns = ["score"])
	for worker in worker_list:
		worker_scores["score"][worker] = sum(pair_scores[worker].values)
	return worker_scores

def get_worker_pair_score_threshold(df):
	""" Calculate a pairwise score threshold for all workers in
	df using Otsu's method. Assumes a bimodal distribution.

	Parameters
	----------
	df : pandas dataframe

	Returns
	-------
	pairwise score threshold value
	"""
	worker_pairwise_scores = util.get_worker_pair_scores(df)	# score workers based on pairwise matching (this step does not use clusters)
	worker_scores_list = worker_pairwise_scores['score'].tolist()	# get IDs of all workers
	return filters.threshold_otsu(np.asarray(worker_scores_list))	# threshold otsu

def slice_by_worker_pair_score(df, threshold):
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

	worker_pair_scores = util.get_worker_pair_scores(df)					# df with all workers. index = worker_ids, values = scores
	high_scores = worker_pair_scores[worker_pair_scores.score > threshold]	# df with only bad workers
	high_scoring_workers = high_scores.index.values
	for worker in high_scoring_workers:
		df = df[df.worker_id != worker]
	return df

"""
Sorting clusters by size and clumpiness
"""

def get_cluster_size_threshold(clusters):
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

def sort_clusters_by_size(clusters, threshold):
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

def get_clumpiness_threshold(clusters, bin_size, cutoff_fraction):
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

def sort_clusters_by_clumpiness(clusters, threshold):
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