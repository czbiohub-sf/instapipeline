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

# ----- #

class BaseAnnotation:
	""" The BaseAnnotation class provides tools for
	annotation injestion.
	"""

	"""
	constructor imports annotations to a dataframe and saves the dataframe
	as a property of the BaseAnnotation class
	"""
	def __init__(self, filename):
		self.annotations = self._import_annotations(filename)

	# raise an error if the method has not been overwritten by a child class
	def _import_annotations(self, filename):
		raise NotImplementedError

	# returns the entire pandas dataframe
	def df(self):
		return self.annotations

	# prints first five lines of a dataframe
	def print_head(self, df):
		print(df.head(n=5))

	# returns a numpy array of unique workers in a dataframe
	def get_workers(self, df):
		uid_list = df.loc[:, ['worker_id']]
		return np.unique(uid_list)

	# returns the list of unique image filenames
	def get_images(self, df):
		img_list = df.loc[:, ['image_filename']]
		return img_list.unique()

	# input: df, worker ID
	# output: list with that worker's timestamps 
	def get_timestamps(self, df, uid):
		turker_df = self.slice_by_worker(df, uid)
		turker_timestamps = turker_df.loc[:, ['timestamp']].as_matrix()
		return turker_timestamps

	# returns dataframe with only rows containing crop_filename
	def slice_by_image(self, df, crop_filename):
		return df[df.image_filename == crop_filename]

	# returns dataframe with just that worker's annotations
	def slice_by_worker(self, df, uid):
		return df[df.worker_id == uid]

	# returns np array with all coordinates in the dataframe
	def get_coords(self, df):
		return df.loc[:, ['x', 'y']].as_matrix()

	# flips the values of a list about a height
	# useful for flipping y axis for plotting over an image
	def flip(self, vec, height):
		to_return = [None]*len(vec)
		for i in range(len(vec)):
			to_return[i] = height - vec[i]
		return to_return

	# input: df, worker ID
	# output: float avg time that the worker spent per click 
	def get_avg_time_per_click(self, df, uid):
		turker_timestamps = self.get_timestamps(df, uid)
		time_spent = max(turker_timestamps) - min(turker_timestamps)
		num_clicks = len(turker_timestamps)
		return time_spent[0]/num_clicks




