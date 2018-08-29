""" This module contains the QuantiusAnnotation class.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import genfromtxt
import pandas as pd
import scipy
import sklearn as skl
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

from BaseAnnotation import BaseAnnotation

# ----- #

class QuantiusAnnotation(BaseAnnotation):
	"""
	If polygon annotation_type, returns same dataframe as for crosshairs 
	annotation_type except timestamp, x, and y columns are removed and 
	at column index 0, each row is a dataframe containing all x, y, and 
	timestamp data for one set of vertices i.e. one annotation."
	"""
	def _import_annotations(self, json_filepath):
		
		to_return = pd.DataFrame()

		json_string = open(json_filepath).read()
		results = json.loads(json_string)

		for worker in results:

			# Skip the worker if they didn't perform any annotations
			if not worker['raw_data']:
				continue

			# Make a data frame of the coordinates of each annotation
			if (worker['annotation_type'] == 'crosshairs'):
				coords = pd.DataFrame(worker['raw_data'][0])
			elif (worker['annotation_type'] == 'polygon'):
				num_annotations = len(worker['raw_data'])
				annotations = []
				for i in range(num_annotations):
					annotation = worker['raw_data'][i]
					annotation = pd.DataFrame(annotation)
					annotations.append(annotation)
				coords = pd.DataFrame(annotations)

			# Add the worker metadata to all entries in the data frame
			coords['annotation_type'] = worker['annotation_type']
			coords['height'] = worker['height']
			coords['width'] = worker['width']
			coords['image_filename'] = worker['image_filename']
			coords['time_when_completed'] = worker['time_when_completed']
			coords['worker_id'] = worker['worker_id']

			# Append to the total data frame
			to_return = to_return.append(coords)

		return to_return