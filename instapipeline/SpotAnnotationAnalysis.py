"""
This module contains the SpotAnnotationAnalysis class.
"""

import numpy as np
import pandas as pd
from instapipeline import util
from sklearn.cluster import AffinityPropagation


class SpotAnnotationAnalysis():
    """
    Tools for annotation analysis
    """
    # list of clustering algs handled
    clustering_algs = ['AffinityPropagation']

    def __init__(self, ba_obj):
        """
        Take in a BaseAnnotation object and save it as
        a property of the SpotAnnotationAnalysis class

        Parameters
        ----------
        ba_obj : BaseAnnotation object to be saved as
            a property of the SpotAnnotationAnalysis class
        """
        self.ba = ba_obj
        self.clusters_done = []
        self.cluster_objects = []

    def get_cluster_object(self, coords, clus_params):
        """
        Check whether the cluster object has already been generated
        for the given df and clustering parameters, and return or calculate
        appropriately.

        Parameters
        ----------
        coords : np_array
            each row is an annotation [int x_coord, int y_coord]
        clus_params : list of clustering parameters
            first element is str name of clustering algorithm
            subsequent elements are additional parameters

        Returns
        -------
        af : sklearn.cluster.AffinityPropagation object
        """
        if (clus_params[0] == 'AffinityPropagation'):
            for i in range(len(self.clusters_done)):
                coords_done = self.clusters_done[i][0]
                clustering_params_done = self.clusters_done[i][1]
                if np.array_equal(coords, coords_done):
                    if clus_params == clustering_params_done:
                        return self.cluster_objects[i]
            af = AffinityPropagation(preference=clus_params[1]).fit(coords)
            self.clusters_done.append([coords, clus_params])
            self.cluster_objects.append(af)
            return af

    def get_clusters(self, df, clus_params):
        """
        Cluster all annotations in df and arrange result as a dataframe.
        Verifies clustering parameters and calls self.get_cluster_object()
        to check whether identical clustering has already been accomplished.

        Parameters
        ----------
        df : pandas dataframe
        clus_params : list of clustering parameters
            first element is str name of clustering algorithm
            subsequent elements are additional parameters

        Returns
        -------
        clusters : pandas dataframe (centroid_x | centroid_y | members)
            centroid_x = int x coord of cluster centroid
            centroid_y = int y coord of cluster centroid
            members = list of annotations belonging to the cluster
                each annotation is a numpy ndarray of properties:
                [int x coord, int y coord, int time spent, str worker ID]
        """
        clustering_alg = clus_params[0]
        if (clustering_alg not in self.clustering_algs):
            raise ValueError('Invalid clustering algorithm name entered.')

        if (clustering_alg == 'AffinityPropagation'):
            cluster_centroids_list = []

            if(len(clus_params) != 2):
                s = 'Please enter a list containing the preference parameter.'
                raise ValueError(s)

            click_properties = util.get_click_properties(df)
            coords = click_properties[:, :2]

            af = self.get_cluster_object(coords, clus_params)

            cluster_centers_indices = af.cluster_centers_indices_
            num_clusters = len(cluster_centers_indices)
            cluster_members_lists = [[] for i in range(num_clusters)]
            labels = af.labels_
            for label_index, click_property in zip(labels, click_properties):
                cluster_members_lists[label_index].append(click_property)
            for cluster_centers_index in cluster_centers_indices:
                cluster_centers = coords[cluster_centers_index]
                cluster_centroids_list.append(cluster_centers)

        centroid_IDs = range(num_clusters)
        column_names = ['centroid_x', 'centroid_y', 'members']
        to_return = pd.DataFrame(index=centroid_IDs, columns=column_names)

        for i in range(num_clusters):
            to_return['centroid_x'][i] = cluster_centroids_list[i][0]
            to_return['centroid_y'][i] = cluster_centroids_list[i][1]
            to_return['members'][i] = cluster_members_lists[i]

        return to_return
