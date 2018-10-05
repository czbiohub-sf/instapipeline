""" Working with polygon.
"""

from SpotAnnotationAnalysis import SpotAnnotationAnalysis
from BaseAnnotation import BaseAnnotation
from QuantiusAnnotation import QuantiusAnnotation

json_filename = 'circle_spots.json'

ba = QuantiusAnnotation(json_filename)	# Load data into an annotation object
sa = SpotAnnotationAnalysis(ba)			# Annotation object is saved as a property of a SpotAnnotationAnalysis object
anno_all = ba.df()						# Get the dataframe from the annotation object

anno_worker = ba.slice_by_worker(anno_all, 'A2QJP5BZ7B523H')

row = anno_worker.iloc[[0]]
annotations = row.iloc[0]
one_annotation = annotations[0]
print(annotations)
print(type(annotations))
print(one_annotation)
print(type(one_annotation))

