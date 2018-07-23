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

	def _import_annotations(self, json_filepath):
		
		to_return = pd.DataFrame()

		json_string = open(json_filepath).read()
		results = json.loads(json_string)

		for worker in results:

			# Skip the worker if they didn't perform any annotations
			if not worker['raw_data']:
				continue

			# Make a data frame of the coordinates of each annotation
			coords = pd.DataFrame(worker['raw_data'][0])

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