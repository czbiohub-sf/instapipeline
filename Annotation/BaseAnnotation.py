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
	Constructor imports annotations to a dataframe and saves the dataframe
	as a property of the BaseAnnotation class
	"""
	def __init__(self, filename):
		self.annotations = self._import_annotations(filename)

	# Raise an error if the method has not been overwritten by a child class
	def _import_annotations(self, filename):
		raise NotImplementedError

	# Returns the entire pandas dataframe
	def df(self):
		return self.annotations

	# Prints first five lines of a dataframe
	def print_head(self, df):
		print(df.head(n=5))

	# Returns a numpy array of unique workers in a dataframe
	def get_workers(self, df):
		uid_list = df.loc[:, ['worker_id']]
		return np.unique(uid_list)

	# Returns the list of unique image filenames
	def get_images(self, df):
		img_list = df.loc[:, ['image_filename']]
		return img_list.unique()

	# Inputs: df, worker ID
	# Returns: list with that worker's timestamps 
	def get_timestamps(self, df, uid):
		turker_df = self.slice_by_worker(df, uid)
		turker_timestamps = turker_df.loc[:, ['timestamp']].as_matrix()
		return turker_timestamps

	# Returns dataframe with only rows containing crop_filename
	def slice_by_image(self, df, crop_filename):
		return df[df.image_filename == crop_filename]

	# Returns dataframe with just that worker's annotations
	def slice_by_worker(self, df, uid):
		return df[df.worker_id == uid]

	# Returns np array with all clicks in the dataframe and with associated
	#	coordinates
	#	time spent (time_spent = 0 indicates fencepost case (first click of an occasion))
	#	worker_ID
	def get_click_properties(self, df):
		occasions = np.unique(df.loc[:, ['time_when_completed']].as_matrix())			# get the list of occasions
		to_return = np.array([]).reshape(0,4)
		for occasion in occasions:
			one_occasion_df = df[df.time_when_completed == occasion]							
			one_occasion_array = one_occasion_df.loc[:, ['x', 'y', 'timestamp', 'worker_id']].as_matrix()
			for i in range(len(one_occasion_array)-1, -1, -1):
				if(i==0):
					time_spent = 0
				else:
					time_spent = one_occasion_array[i][2] - one_occasion_array[i-1][2]
				one_occasion_array[i][2] = time_spent
			to_return = np.vstack([to_return, one_occasion_array])
		return to_return

	# Returns dataframe with all fast clicks screened
	# Clicks of which time_spent < time_threshold are "fast"
	def screen_clicks_time_spent(self, df, time_threshold):
		to_return = pd.DataFrame()
		occasions = np.unique(df.loc[:, ['time_when_completed']].as_matrix())			# get the list of occasions
		for occasion in occasions:
			one_occasion_df = df[df.time_when_completed == occasion]
			one_occasion_timestamps = one_occasion_df.loc[:, ['timestamp']].as_matrix()
			for i in range(len(one_occasion_timestamps)-1, -1, -1):
				if(i==0):
					one_occasion_df = one_occasion_df.drop([i])
				else:
					time_spent = one_occasion_timestamps[i][0] - one_occasion_timestamps[i-1][0]
					if(time_spent<time_threshold):
						one_occasion_df = one_occasion_df.drop([i])
			to_return = to_return.append(one_occasion_df)
		return to_return

	# Flips the values of a list about a height
	# Useful for flipping y axis for plotting over an image
	def flip(self, vec, height):
		to_return = [None]*len(vec)
		for i in range(len(vec)):
			to_return[i] = height - vec[i]
		return to_return

	# Inputs: df, worker ID
	# Returns: float avg time that the worker spent per click 
	def get_avg_time_per_click(self, df, uid):
		worker_timestamps = self.get_timestamps(df, uid)
		time_spent = max(worker_timestamps) - min(worker_timestamps)
		num_clicks = len(worker_timestamps)
		return time_spent[0]/num_clicks

	# Inputs: df, worker ID
	# Returns: time that the worker spent 		
	def get_total_time(self, df, uid):
		worker_timestamps = self.get_timestamps(df, uid)
		return max(worker_timestamps) - min(worker_timestamps)



