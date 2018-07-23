
from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation

json_filename = 'SNR_test.json'
img_filename = 'MAX_ISP_300_1_nspots50_spot_sig1.75_snr5_20_spot_img.png'
csv_filename = 'MAX_ISP_300_1_nspots50_spot_sig1.75_snr5_20_coord_snr_list.csv'
worker_marker_size = 8
cluster_marker_size = 40
bigger_window_size = False
clustering_params = ['AffinityPropagation', -350]

img_height = 300
correctness_threshold = 4

ba = QuantiusAnnotation(json_filename)	# Load data into an annotation object
sa = SpotAnnotationAnalysis(ba)			# Annotation object is saved as a property of a SpotAnnotationAnalysis object
anno_all = ba.df()
anno_one_crop = ba.slice_by_image(anno_all, img_filename)



sa.plot_workers_correct_rate(anno_one_crop, clustering_params, correctness_threshold, csv_filename, img_filename, bigger_window_size)
