
from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation

worker_marker_size = 4
cluster_marker_size = 40
bigger_window_size = False

csv_filepath = None
show_correctness_workers = False
show_correctness_clusters = False
correctness_threshold = None
show_NN_inc = False
show_ref_points = False


img_filename  = 'C2-ISP_293T_TFRC_InSituPrep_20180712_1_MMStack_Pos0_300_inv.png'
img_filepath = '/Users/jenny.vo-phamhi/Documents/StarFISH-tools/Annotation/C2-ISP_293T_TFRC_InSituPrep_20180712_1_MMStack_Pos0_300.png'
json_filepath = '/Users/jenny.vo-phamhi/Documents/StarFISH-tools/Annotation/smFISH_cells_inv.json'

# img_filename = 'C2-ISP_293T_TFRC_InSituPrep_20180712_1_MMStack_Pos0_300_inv.png'
# img_filepath = '/Users/jenny.vo-phamhi/Documents/StarFISH-tools/Annotation/C2-ISP_293T_TFRC_InSituPrep_20180712_1_MMStack_Pos0_300.png'
# json_filepath = '/Users/jenny.vo-phamhi/Documents/StarFISH-tools/Annotation/smFISH_cells_inv.json'


ba = QuantiusAnnotation(json_filepath)
sa = SpotAnnotationAnalysis(ba)
anno_all = ba.df()
anno_one_image = ba.slice_by_image(anno_all, img_filename)

show_workers = True
show_clusters = True
clustering_params = ['AffinityPropagation',-350]

img_height = 450

x_min = 100
x_max = 240
y_min = img_height-360
y_max = img_height-220

#sa.plot_annotations_zoom(anno_one_image, x_min, x_max, y_min, y_max, img_height, clustering_params, img_filepath, show_clusters, show_workers, cluster_marker_size, worker_marker_size)

sa.plot_annotations(anno_one_image, img_filename, img_filepath, csv_filepath, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_params, bigger_window_size)







