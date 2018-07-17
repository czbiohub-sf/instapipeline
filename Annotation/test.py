""" Demonstrating several features of the annotation analysis pipeline.
"""

from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation

img_filename = 'beads_300pxroi.png'
json_filename = 'BeadAnnotation_20180413.json'
csv_filename = 'bead_annotations_20180517_shifted.csv'
worker_marker_size = 8
cluster_marker_size = 40

ba = QuantiusAnnotation(json_filename)	# Load data into an annotation object
sa = SpotAnnotationAnalysis(ba)			# Annotation object is saved as a property of a SpotAnnotationAnalysis object
anno_all = ba.df()						# Get the dataframe from the annotation object

# --- Plot to get an overview of annotations ---

# show_ref_points = False
# show_workers = True
# show_clusters = True
# clustering_alg = 'AffinityPropagation'

# show_correctness_workers = False
# show_correctness_clusters = False
# show_NN_inc = False
# correctness_threshold = None
# bigger_window_size = False

# sa.plot_annotations(anno_all, img_filename, csv_filename, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_alg, clustering_params, bigger_window_size)


# print(ba.slice_by_worker(anno_all, 'A2FZE7CARM54IG'))

# show_correctness = False
# correctness_threshold = None
# clustering_params = None

# show_correctness = False
# correctness_threshold = 5
# clustering_params = [-350]
# sa.plot_nnd_vs_time_spent(anno_all, img_filename, csv_filename, show_correctness, correctness_threshold, clustering_params)


# uid = 'A2FZE7CARM54IG'
# show_correctness = False
# correctness_threshold = None
# clustering_params = None

# sa.plot_time_spent_vs_click_index(anno_all, img_filename, csv_filename, uid, show_correctness, correctness_threshold, clustering_params)


# uid = 'A2FZE7CARM54IG'


# sa.plot_time_spent_vs_click_index(anno_all, img_filename, csv_filename, uid, show_correctness, correctness_threshold, clustering_params)
# #print(ba.get_coords_time_spent_worker_id(anno_all))

bigger_window_size = False

show_ref_points = False
show_workers = True
show_clusters = True
clustering_params = ['AffinityPropagation', -350]
show_correctness_workers = False
show_correctness_clusters = False
show_NN_inc = False
correctness_threshold = None

#sa.plot_annotations(anno_all, img_filename, csv_filename, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_alg, clustering_params, bigger_window_size)

click_properties = ba.get_click_properties(anno_all)
coords = click_properties[:,:2]
print(coords)
coords2 = ba.get_coords(anno_all)
print(coords2)
print(click_properties)



