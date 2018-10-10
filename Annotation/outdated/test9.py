from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation

json_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/datasets/Density_test_2/Density_test_2.json'
csv_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/datasets/Density_test_2/crop1.csv'
img_filename = 'C2-ISP_293T_TFRC_InSituPrep_20180712_1_MMStack_Pos0_700_crop1.png'
img_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/datasets/Density_test_2/' + img_filename
clustering_params = ['AffinityPropagation', -350]
img_height = 300

ba = QuantiusAnnotation(json_filepath)
anno_all = ba.df()
sa = SpotAnnotationAnalysis(ba)

anno_one_image = ba.slice_by_image(anno_all, img_filename)
clusters = sa.get_clusters(anno_one_image, clustering_params)
centroid_and_ref_df = sa.centroid_and_ref_df(clusters, csv_filepath, img_height)

show_workers = True
show_correctness_workers = True
show_centroids = False
show_correctness_centroids = False
show_ref_points = False
show_NN_inc = False
correctness_threshold = 10
worker_marker_size = 8
cluster_marker_size = 40
bigger_window_size = False

sa.plot_annotations(anno_one_image, show_workers, show_correctness_workers, show_centroids, show_correctness_centroids, show_ref_points, show_NN_inc, centroid_and_ref_df, correctness_threshold, worker_marker_size, cluster_marker_size, img_filepath, csv_filepath, bigger_window_size)



# properties = ba.get_click_properties(anno_one_image)
# coords = properties[:,:2]
# #af = sa.get_cluster_object(coords, clustering_params)
# print(coords)


#clusters = sa.get_clusters(anno_one_image, clustering_params)




# bin_size = 0.1
# cutoff_fraction = 3
# threshold = sa.get_clumpiness_threshold(clusters, bin_size, cutoff_fraction)
# print(threshold)
# clumpy_clusters, nonclumpy_clusters = sa.sort_clusters_by_clumpiness(clusters, threshold)
# print(clumpy_clusters)
# ba.print_head(nonclumpy_clusters)

# threshold = sa.get_cluster_size_threshold(clusters)
# small_clusters, large_clusters = sa.sort_clusters_by_size(clusters, threshold)
# ba.print_head(small_clusters)
# ba.print_head(large_clusters)




# worker_marker_size = 8
# cluster_marker_size = 40
# bigger_window_size = False
# img_height = 300

# json_filename = 'SynthData_cells.json'
# gen_date = '20180719'
# bg_type = 'cells'
# img_name = 'MAX_C3-ISP_300_1_nspots100_spot_sig1.75_snr10_2.5'

# img_filename = img_name+'spot_img.png'
# img_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/gen_'+gen_date+'/spot_images/'+bg_type+'/'+img_name+'spot_img.png'
# csv_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/gen_'+gen_date+'/spot_data/'+bg_type+'/'+img_name+'_coord_snr_list.csv'
# json_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/gen_'+gen_date+'/'+json_filename



# show_ref_points = False
# show_workers = True
# show_clusters = False
# clustering_params = None
# show_correctness_workers = False
# show_correctness_clusters = False
# show_NN_inc = False
# correctness_threshold = None

# sa.plot_annotations(anno_all, img_filename, img_filepath, csv_filepath, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_params, bigger_window_size)