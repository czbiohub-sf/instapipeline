from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from skimage import filters


worker_marker_size = 8
cluster_marker_size = 40
bigger_window_size = False
img_height = 300
show_correctness = True
correctness_threshold = 4
clustering_params = ['AffinityPropagation', -350]

json_filename_list = ['Spot_density.json']
density_list = [0.008]
snr_mu_list = [10]

for json_filename in json_filename_list:
	for snr_mu in snr_mu_list:
		for density in density_list:
			img_name = 'snr_' + str(snr_mu)+ '_0_density_' + str(density)
			if (json_filename == 'Spots_density_no_tissue.json'):
				img_filename = img_name+'_spots.png'
				img_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/density_test/spot_arrays/'+img_filename
			else:
				img_filename = img_name+'_spot_img.png'
				img_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/density_test/spot_images/'+img_filename
			csv_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/density_test/spot_data/'+img_name+'_coord_snr_list.csv'
			json_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/density_test/'+json_filename

			ba = QuantiusAnnotation(json_filepath)
			sa = SpotAnnotationAnalysis(ba)
			anno_all = ba.df()
			anno_one_image = ba.slice_by_image(anno_all, img_filename)
			ref_kdt = sa.csv_to_kdt(csv_filepath, img_height)
			ref_array = np.asarray(ref_kdt.data)

			# Get large clusters
			clusters = sa.get_clusters(anno_one_image, clustering_params)
			threshold = sa.get_cluster_size_threshold(clusters)
			threshold = math.ceil(threshold)
			small_clusters, large_clusters = sa.sort_clusters_by_size_input_threshold(clusters, threshold)

			threshold = 0.8
			clumpy_clusters, nonclumpy_clusters = sa.sort_clusters_by_clumpiness_input_threshold(large_clusters, threshold)

			for i in range(len(clumpy_clusters.index)):
				row = clumpy_clusters.iloc[[i]]
				members = row.iloc[0]['members']
				workers = []
				for member in members:
					workers.append(member[3])
				unique_workers = np.unique(workers)

				print(row.iloc[0]['single_fraction'])
				print('workers = ' + str(workers))
				print('unique_workers = ' + str(unique_workers))


                
            #     num_instances_list = []
            #     for unique_worker in unique_workers:
            #         num_instances_list.append(workers.count(unique_worker))
                    
            #     print(num_instances_list)
                    
            #     singles = num_instances_list.count(1)
            #     single_fraction = singles/len(unique_workers)
                
            #     (n, bins, patches) = plt.hist(num_instances_list, bins = np.arange(0,4,1)-0.5)
            #     plt.title('single_fraction = ' + str(single_fraction))
            #     plt.show()




