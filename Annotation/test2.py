
from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation

img_filename = 'beads_300pxroi.png'
json_filename = 'BeadAnnotation_20180413.json'
csv_filename = 'bead_annotations_20180517_shifted.csv'
worker_marker_size = 8
cluster_marker_size = 40
bigger_window_size = True

ba = QuantiusAnnotation(json_filename)	# Load data into an annotation object
sa = SpotAnnotationAnalysis(ba)			# Annotation object is saved as a property of a SpotAnnotationAnalysis object

anno_all = ba.df()						# Get the dataframe from the annotation object

anno_screened = ba.screen_clicks_time_spent(anno_all, 2000)
print(anno_screened)

# show_ref_points = False
# show_workers = True
# show_clusters = True
# clustering_params = ['AffinityPropagation', -350]
# show_correctness_workers = False
# show_correctness_clusters = False
# show_NN_inc = False
# correctness_threshold = None

# sa.plot_annotations(anno_all, img_filename, csv_filename, worker_marker_size, cluster_marker_size, show_ref_points, show_workers, show_clusters, show_correctness_workers, show_correctness_clusters, show_NN_inc, correctness_threshold, clustering_params, bigger_window_size)
