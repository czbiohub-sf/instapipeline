from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation

worker_marker_size = 8
cluster_marker_size = 40
bigger_window_size = False
img_height = 300
correctness_threshold = 4
clustering_params = ['AffinityPropagation', -350]

json_filename = 'SynthTests_tissue.json'
gen_date = '20180719'
bg_type = 'tissue'
img_name = 'MAX_ISP_300_1_nspots50_spot_sig1.75_snr5_2.5'

img_filename = img_name+'spot_img.png'
img_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/gen_'+gen_date+'/spot_images/'+bg_type+'/'+img_name+'spot_img.png'
csv_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/gen_'+gen_date+'/spot_data/'+bg_type+'/'+img_name+'_coord_snr_list.csv'
json_filepath = '/Users/jenny.vo-phamhi/Documents/FISH-annotation/Annotation/gen_'+gen_date+'/'+json_filename

ba = QuantiusAnnotation(json_filepath)
sa = SpotAnnotationAnalysis(ba)
anno_all = ba.df()
anno_one_snr = ba.slice_by_image(anno_all, img_filename)

plot_title = img_name

sa.plot_snr_sensitivity(anno_one_snr, clustering_params, csv_filepath, img_height, img_filename, correctness_threshold, plot_title, bigger_window_size)

#sa.plot_snr_vs_membership(anno_one_snr, clustering_params, csv_filepath, img_height, img_filename, correctness_threshold, bigger_window_size)
