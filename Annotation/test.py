""" Demonstrating several features of the QuantiusAnnotation class.
"""

from SpotAnnotationAnalysis import SpotAnnotationAnalysis

img_filename = 'beads_300pxroi.png'
json_filename = 'BeadAnnotation_20180413.json'
csv_filename = 'bead_annotations_20180517.csv'
worker_marker_size = 8
cluster_marker_size = 40

# Load data, get the dataframe 

sa = SpotAnnotationAnalysis(json_filename)
anno_all = sa.df()

# print(sa.get_workers(anno_all))

"""

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

sa.plot_annotations(anno_all, img_filename, worker_marker_size, cluster_marker_size, show_workers, show_clusters, clustering_alg, clustering_params)

# Investigate worker efficiency 
sa.plot_avg_time_per_click(anno_all)

"""

# More investigations
sa.plot_nnd_vs_time_spent(anno_all, img_filename, csv_filename)

sa.plot_nnd_vs_worker_index(anno_all, img_filename, csv_filename)

sa.plot_time_spent_vs_worker_index(anno_all, img_filename)

uid = 'A1EFL6UHDB1IZM'
sa.plot_time_spent_vs_click_index(anno_all, img_filename, uid)
