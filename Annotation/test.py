""" Demonstrating several features of the QuantiusAnnotation class.
"""

from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation


img_filename = 'beads_300pxroi.png'
json_filename = 'BeadAnnotation_20180413.json'
csv_filename = 'bead_annotations_20180517.csv'
worker_marker_size = 8
cluster_marker_size = 40

ba = QuantiusAnnotation(json_filename)	# Load data into an annotation object
sa = SpotAnnotationAnalysis(ba)			# Annotation object is saved as a property of a SpotAnnotationAnalysis object
anno_all = ba.df()						# Get the dataframe from the annotation object

# Plot to get an overview of annotations 

# Uncomment this...
show_workers = True
show_clusters = True
clustering_alg = 'AffinityPropagation'
clustering_params = [-350]

# ...or uncomment this:
# show_workers = True
# show_clusters = False
# clustering_alg = None
# clustering_params = None

# sa.plot_annotations(anno_all, img_filename, worker_marker_size, cluster_marker_size, show_workers, show_clusters, clustering_alg, clustering_params)

# # More investigations
# sa.plot_avg_time_per_click(anno_all)
# sa.plot_nnd_vs_time_spent(anno_all, img_filename, csv_filename)
# sa.plot_nnd_vs_worker_index(anno_all, img_filename, csv_filename)
# sa.plot_time_spent_vs_worker_index(anno_all, img_filename)
# uid = 'A1EFL6UHDB1IZM'
# sa.plot_time_spent_vs_click_index(anno_all, img_filename, uid)
# sa.plot_total_time_vs_worker_index(anno_all, img_filename)



big_df = sa.anno_and_ref_to_df(clustering_alg, anno_all, clustering_params, csv_filename, img_filename)
print(big_df)

# correctness = sa.get_cluster_correctness(big_df, 20)
# print(correctness)


# cluster_df = sa.get_clusters(clustering_alg, anno_all, clustering_params) 

# print(cluster_df)

# print(cluster_df['members'][0])
# print(len(cluster_df['members'][0]))







