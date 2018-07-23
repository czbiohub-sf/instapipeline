from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation
worker_marker_size = 8
cluster_marker_size = 40
bigger_window_size = True
img_height = 300

json_filename = 'SNR_test.json'
img_filename = 'MAX_ISP_300_1_nspots50_spot_sig1.75_snr5_20_spot_img.png'
img_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/gen_20180713/spot_images/tissue/MAX_ISP_300_1_nspots50_spot_sig1.75_snr5_20_spot_img.png'
csv_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/gen_20180713/spot_data/tissue/MAX_ISP_300_1_nspots50_spot_sig1.75_snr5_20_coord_snr_list.csv'
#csv_filepath = 'MAX_ISP_300_1_nspots50_spot_sig1.75_snr5_20_coord_snr_list.csv'

ba = QuantiusAnnotation(json_filename)
sa = SpotAnnotationAnalysis(ba)
anno_all = ba.df()
anno_one_snr = ba.slice_by_image(anno_all, img_filename)

show_ref_points = False
show_workers = True
show_clusters = False
clustering_params = None
show_correctness_workers = False
show_correctness_clusters = False
show_NN_inc = False
correctness_threshold = None

sa.plot_annotations(anno_all, img_filename, csv_filepath, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_params, bigger_window_size, img_filepath)

show_ref_points = True
show_workers = True
show_clusters = False
clustering_params = None
show_correctness_workers = False
show_correctness_clusters = False
show_NN_inc = False
correctness_threshold = None

sa.plot_annotations(anno_all, img_filename, csv_filepath, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_params, bigger_window_size, img_filepath)