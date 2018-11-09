""" 
This module contains utilities used by the spot annotation analysis pipeline.
"""

# ----- #

import numpy as np

"""
Dataframe manipulation
"""

def print_head(df):
	""" Print the first five lines of df 
	"""
	print(df.head(n=5))

def get_workers(df):
	""" Return a numpy array of unique workers in df 
	"""
	uid_list = df.loc[:, ['worker_id']]
	return np.unique(uid_list)

def get_images(df):
	""" Return a numpy array of unique image filenames in df 
	"""
	img_list = df.loc[:, ['image_filename']]
	return np.unique(img_list)

def get_timestamps(df):
	""" Return a list of timestamps in df 
	"""
	matrix = df.loc[:, ['timestamp']].as_matrix()
	return [x[0] for x in matrix]

def slice_by_worker(df, uid):
	""" Return a dataframe with annotations for only one worker

	Parameters
	----------
	df : pandas dataframe
	uid : user ID of worker

	Returns
	-------
	Dataframe with annotations for only that worker 
	"""
	return df[df.worker_id == uid]

# def slice_by_image(df, img_filename):
# 	""" Return a dataframe with annotations for one image

# 	Parameters
# 	----------
# 	df : pandas dataframe
# 	img_filename : string filename of image

# 	Returns
# 	-------
# 	Dataframe with annotations for only that image 
#	
#	No longer useful because each qa object gets data from only one image

# 	"""
# 	return df[df.image_filename == img_filename]

def get_click_properties(df):
	""" Return a numpy array containing properties for all clicks in df

	Parameters
	----------
	df : pandas dataframe

	Returns
	-------
	numpy array
		each row corresponds with one annotation in the dataframe
		columns:
			x coord
			y coord
			time spent (time_spent = 0 indicates first click of an occasion (fencepost case))
			string worker ID
	"""
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

"""
Other data structure manipulation
"""

def flip(vec, height):
	""" Flip the values of a list about a height
	Useful for flipping y axis to plotting over an image with a flipped coordinate system.

	Parameters
	----------
	vec : list of values to be flipped
	height : height about which to flip values

	Returns
	-------
	flipped list
	"""
	to_return = [None]*len(vec)
	for i in range(len(vec)):
		to_return[i] = height - vec[i]
	return to_return
