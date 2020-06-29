"""
This module contains functions for:
    - interacting with / manipulating dataframes
    - other data structure manipulation
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


"""

Functions for interacting with / manipulating dataframes

"""


def get_workers(df):
    """
    Return a numpy array of unique workers in df
    """
    uid_list = df.loc[:, ['worker_id']]
    return np.unique(uid_list)


def get_img_filenames(df):
    """
    Returns a numpy array of unique image filenames in df
    """
    img_list = df.loc[:, ['image_filename']]
    return np.unique(img_list)


def get_timestamps(df):
    """
    Returns a list of timestamps in df
    """
    matrix = df.loc[:, ['timestamp']].to_numpy()
    return [x[0] for x in matrix]


def get_click_properties(df):
    """
    Returns a numpy array containing properties for all clicks in df

    Parameters
    ----------
    df : pandas dataframe
        timestamp | x | y | annotation_type | height |
        width image_filename | time_when_completed | worker_id

    Returns
    -------
    numpy array
        each row corresponds with one annotation in the dataframe
        columns:
            x coord
            y coord
            time spent (time_spent = 0 indicates first click of an occasion)
            string worker ID
    """
    occasions = np.unique(df.loc[:, ['time_when_completed']].to_numpy())
    to_return = np.array([]).reshape(0, 4)
    for occasion in occasions:
        one_occasion_df = df[df.time_when_completed == occasion]
        columns = ['x', 'y', 'timestamp', 'worker_id']
        one_occ_arr = one_occasion_df.loc[:, columns].to_numpy()
        for i in range(len(one_occ_arr)-1, -1, -1):
            if(i == 0):
                time_spent = 0
            else:
                time_spent = one_occ_arr[i][2] - one_occ_arr[i-1][2]
            one_occ_arr[i][2] = time_spent
        to_return = np.vstack([to_return, one_occ_arr])
    return to_return


def get_time_per_click(df):
    """
    Get time spent on each annotation.

    Parameters
    ----------
    df : pandas dataframe
        timestamp | x | y | annotation_type | height |
        width image_filename | time_when_completed | worker_id

    Returns
    -------
    time_spent_list : list of the amount of time spent on all clicks in df
        except the first click (fencepost)
        len(time_spent_list) = num rows in df
        time_spent_list[0] = None
        units are miliseconds
    """
    timestamps = get_timestamps(df)
    time_spent_list = [None]*len(timestamps)
    for i in range(1, len(timestamps)):
        x = timestamps[i] - timestamps[i-1]
        time_spent_list[i] = x[0]
    return time_spent_list


def get_avg_time_per_click(df, uid):
    """
    Get the average amount of time that a worker spent on one click.

    Parameters
    ----------
    df : pandas dataframe
        timestamp | x | y | annotation_type | height |
        width image_filename | time_when_completed | worker_id
    uid : string worker ID

    Returns
    -------
    the average time that the worker spent per click
    """

    worker_timestamps = get_timestamps(df, uid)
    time_spent = max(worker_timestamps) - min(worker_timestamps)
    num_clicks = len(worker_timestamps)
    return time_spent[0]/num_clicks


def get_nnd_per_click(df, ref_kdt):
    """
    Get the distance to the nearest neighbor (found in
    the k-d tree of reference points).

    Parameters
    ----------
    df : pandas dataframe
        timestamp | x | y | annotation_type | height |
        width image_filename | time_when_completed | worker_id

    Returns
    -------
    list of distances to the nearest neighbor (found in
        the k-d tree of reference points)
    """
    coords = get_click_properties(df)[:, :2]
    dist, ind = ref_kdt.query(coords, k=1)
    dist_list = dist.tolist()
    return [dist[0] for dist in dist_list]


def slice_by_worker(df, uid):
    """
    Return a dataframe with annotations for only one worker

    Parameters
    ----------
    df : pandas dataframe
    uid : user ID of worker

    Returns
    -------
    Dataframe with annotations for only that worker
    """
    return df[df.worker_id == uid]


def print_head(df):
    """
    Print the first five lines of df
    """
    print(df.head(n=5))


"""

Functions for other data structure manipulation

"""


def csv_to_kdt(csv_filepath, img_height):
    """
    Fit reference spot coordinates to a k-d tree

    Parameters
    ----------
    csv_filepath : string filepath to csv file containing reference points
    img_height : height of image

    Returns
    -------
    ref_kdt : sklearn.neighbors.kd_tree.KDTree containing reference points
                y-coordinates are flipped about img_height
    """
    ref_df = pd.read_csv(csv_filepath)
    ref_points = ref_df.loc[:, ['col', 'row']].to_numpy()

    for i in range(len(ref_points)):
        point = ref_points[i]
        first_elem = point[0]
        second_elem = img_height - point[1]
        point = np.array([first_elem, second_elem])
        ref_points[i] = point

    ref_kdt = KDTree(ref_points, leaf_size=2, metric='euclidean')
    return ref_kdt


def centroid_and_ref_df(clusters, csv_filepath, img_height):
    """
    Assemble a dataframe of centroids found with annotation and
    reference data consolidated.

    Parameters
    ----------
    df : Pandas Dataframe with annotation data (should already be cropped)
    clusters : pandas dataframe (centroid_x | centroid_y | members)
        centroid_x = x coord of cluster centroid
        centroid_y = y coord of cluster centroid
        members = list of annotations belonging to the cluster
            each member is a list of properties of the annotation
            i.e. [x coord, y coord, time spent, worker ID]
    csv_filepath : contains reference data

    Returns
    -------
    this dataframe:
    centroid_x | centroid_y | nearest ref x | nearest ref y | NN_dist | members
        * (the index is the Cluster ID)
        centroid_x = x coord of cluster centroid
        centroid_y = y coord of cluster centroid
        NN_x = x coord of nearest neighbor reference
        NN_y = y coord of nearest neighbor reference
        NN_dist = distance from centroid to nearest neighbor reference
        members = list of annotations belonging to cluster
            each annotation is a list of click properties:
            x_coord | y_coord | time_spent | worker_ID
    """
    ref_kdt = csv_to_kdt(csv_filepath, img_height)
    ref_array = np.asarray(ref_kdt.data)

    centroid_IDs = range(clusters.shape[0])
    column_names = ['centroid_x', 'centroid_y', 'NN_x',
                    'NN_y', 'NN_dist', 'members']
    to_return = pd.DataFrame(index=centroid_IDs, columns=column_names)

    for i in centroid_IDs:

        to_return['centroid_x'][i] = clusters['centroid_x'][i]
        to_return['centroid_y'][i] = clusters['centroid_y'][i]

        coords = [[to_return['centroid_x'][i], to_return['centroid_y'][i]]]

        dist, ind = ref_kdt.query(coords, k=1)
        index = ind[0][0]
        nearest_neighbor = ref_array[index]

        to_return['NN_x'][i] = nearest_neighbor[0]
        to_return['NN_y'][i] = nearest_neighbor[1]
        to_return['NN_dist'][i] = dist[0][0]
        to_return['members'][i] = clusters['members'][i]

    return to_return


def flip(vec, height):
    """
    Flip the values of a list about a height
    Useful for flipping y axis to plotting over an
    image with a flipped coordinate system.

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
