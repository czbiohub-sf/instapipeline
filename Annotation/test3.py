
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

anno_all = ba.df()
anno_one_crop = ba.slice_by_image(anno_all, img_filename)

# worker_scores = sa.get_worker_scores(anno_all)
# print(worker_scores)

sa.plot_worker_scores_hist(anno_one_crop, False)
sa.plot_total_worker_time_hist(anno_one_crop, False)