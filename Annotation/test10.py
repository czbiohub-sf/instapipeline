from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation

json_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/datasets/density_test/Spot_density.json'
img_filename = 'snr_5_0_density_0.008_spot_img.png'
img_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/datasets/density_test/spot_images/snr_5_0_density_0.008_spot_img.png'
csv_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/datasets/density_test/spot_data/snr_5_0_density_0.008_coord_snr_list.csv'

ba = QuantiusAnnotation(json_filepath)
sa = SpotAnnotationAnalysis(ba)
anno_all = ba.df()
anno_one_image = ba.slice_by_image(anno_all, img_filename)
img_height = anno_one_image['height'].values[0]

clustering_params = ['AffinityPropagation', -350]
clusters = sa.get_clusters(anno_one_image, clustering_params)

cluster_size_threshold = sa.get_cluster_size_threshold(clusters)
small_clusters, large_clusters = sa.sort_clusters_by_size(clusters, cluster_size_threshold)

worker_marker_size = 8
cluster_marker_size = 40

show_workers = False
show_centroids = True
bigger_window_size = False
plot_title = 'Large Clusters'
sa.visualize_clusters(large_clusters, show_workers, show_centroids, worker_marker_size, cluster_marker_size, img_filepath, img_height, plot_title, bigger_window_size)

clumpiness_threshold = sa.plot_clumpiness_threshold(large_clusters)
print(clumpiness_threshold)