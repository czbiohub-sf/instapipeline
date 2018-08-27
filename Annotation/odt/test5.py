from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation

worker_marker_size = 4
cluster_marker_size = 40
bigger_window_size = True

csv_filepath = None
show_correctness_workers = False
show_correctness_clusters = False
correctness_threshold = None
show_NN_inc = False
show_ref_points = False

img_filename = 'C2-ISP_293T_TFRC_InSituPrep_20180712_1_MMStack_Pos0_700_inv.png'
img_filepath = '/Users/jenny.vo-phamhi/Documents/StarFISH-tools/Annotation/C2-ISP_293T_TFRC_InSituPrep_20180712_1_MMStack_Pos0_700.png'
json_filepath = '/Users/jenny.vo-phamhi/Documents/StarFISH-tools/Annotation/smFISH_cells_inv.json'

ba = QuantiusAnnotation(json_filepath)
sa = SpotAnnotationAnalysis(ba)
anno_all = ba.df()
anno_one_image = ba.slice_by_image(anno_all, img_filename)

show_workers = False
show_clusters = False
clustering_params = ['AffinityPropagation', -350]
sa.plot_annotations(anno_one_image, img_filename, img_filepath, csv_filepath, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_params, bigger_window_size)

show_workers = True
show_clusters = False
clustering_params = ['AffinityPropagation', -350]
sa.plot_annotations(anno_one_image, img_filename, img_filepath, csv_filepath, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_params, bigger_window_size)

show_workers = True
show_clusters = True
clustering_params = ['AffinityPropagation', -350]
sa.plot_annotations(anno_one_image, img_filename, img_filepath, csv_filepath, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_params, bigger_window_size)

show_workers = False
show_clusters = True
clustering_params = ['AffinityPropagation', -350]
sa.plot_annotations(anno_one_image, img_filename, img_filepath, csv_filepath, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_params, bigger_window_size)