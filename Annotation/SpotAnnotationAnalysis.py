""" 
This module contains the SpotAnnotationAnalysis class.
"""

import util
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation

# ------- #

class SpotAnnotationAnalysis():
	""" Tools for annotation analysis

	SpotAnnotationAnalysis takes in a BaseAnnotation 
	object as input and saves it as a property of
	the class.
	"""

	# list of clustering algs handled
	clustering_algs = ['AffinityPropagation']

	def __init__(self, ba_obj):
		"""
		Take in a BaseAnnotation object and save it as 
		a property of the SpotAnnotationAnalysis class
		"""
		self.ba = ba_obj
		self.clusters_done = []
		self.cluster_objects = []

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

