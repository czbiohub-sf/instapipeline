"""
This module contains functions for parameter extraction.
"""

import math
import numpy as np
from skimage.io import imread
from skimage.feature import blob_log
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from scipy import optimize
from instapipeline import util, clus


def get_sigma_list(sample_img_path, ref_coords, margin):
    """
    Get the sigma values of the spots at the ref_coords in the sample
    image at sample_img_path, excluding spots at the edge of the sample image.

    Parameters
    ----------
    sample_img_path : str path to the sample image
    ref_coords : numpy array with locations of the spots on the sample image
        to be analyzed
    margin : int minimum distance from the edge of the sample image
        for a spot to be included and analyzed

    Returns
    -------
    sigma_max_list : list of sigma values of the spots
    """
    im = imread(sample_img_path, as_gray=True)

    sigma_max_list = []

    for x, y in ref_coords:

        x_min = int(x)-margin if int(x)-margin >= 0 else 0
        x_max = int(x)+margin if int(x)+margin < im.shape[1] else im.shape[1]-1
        y_min = int(y)-margin if int(y)-margin >= 0 else 0
        y_max = int(y)+margin if int(y)+margin < im.shape[0] else im.shape[0]-1

        little_crop = im[y_min:y_max, x_min:x_max]

        if np.count_nonzero(little_crop) == 0:
            continue

        params = fitgaussian(little_crop)
        (height, x_param, y_param, width_x, width_y) = params
        q = max(width_x, width_y)/2
        if q < 0:
            continue
        sigma_max = math.sqrt(q)
        sigma_max_list.append(sigma_max)

    return sigma_max_list


def get_best_threshold(sample_coords, sample_img_path, min_sigma,
                       max_sigma, correctness_threshold, thresholds):
    """
    Tries blob detection with various intensity thresholds and
    picks the best one.

    Parameters
    ----------
    sample_coords : numpy array of known locations of spots on the sample image
    sample_img_path : str path to the sample image
    min_sigma : float minimum sigma value to be considered for blob detection
        found using sigma_max_list returned by param.get_sigma_list()
    max_sigma : float maximum sigma value to be considered for blob detection
        found using sigma_max_list returned by param.get_sigma_list()
    correctness_threshold : float user-specified min distance from a reference
        coord for a detected spot to be considered incorrect
    thresholds: list of float intensity threshold values to try

    Returns
    -------
    best_threshold : of the inputted float thresholds, the one with the best
        precision * recall from blob detection using that threshold
    best_rec : float recall from blob detection using best_threshold
    best_prec : float precision from blob detection using best_threshold
    recall_list : float recall values from blob detection using
        inputted thresholds
    precision_list : float precision values from blob detection using
        inputted thresholds
    """
    best_precision_x_recall = 0
    precision_list = []
    recall_list = []

    im = imread(sample_img_path, as_gray=True)
    img_height = len(im)

    for i in range(len(sample_coords)):
        point = sample_coords[i]
        first_elem = point[0]
        second_elem = img_height - point[1]
        point = np.array([first_elem, second_elem])
        sample_coords[i] = point

    sample_kdt = KDTree(sample_coords, leaf_size=2, metric='euclidean')

    best_threshold = 0

    for threshold in thresholds:

        blobs_log = blob_log(im, min_sigma=min_sigma, max_sigma=max_sigma,
                             num_sigma=10, threshold=threshold)
        blobs = []
        for r, c, sigma in blobs_log:
            blobs.append([c, r])
        blobs = np.asarray(blobs)
        if len(blobs) == 0:
            continue
        blobs_kdt = KDTree(blobs, leaf_size=2, metric='euclidean')

        correct_blobs = []
        incorrect_blobs = []
        detected_ref = []
        undetected_ref = []

        # correct vs. incorrect
        for r, c, sigma in blobs_log:
            dist, ind = sample_kdt.query([[c, img_height-r]], k=1)
            if dist[0][0] < correctness_threshold:
                correct_blobs.append((r, c, sigma))
            else:
                incorrect_blobs.append((r, c, sigma))

        # detected vs. undetected
        for x, y in sample_coords:
            dist, ind = blobs_kdt.query([[x, y]], k=1)
            if dist[0][0] < correctness_threshold:
                detected_ref.append([x, y])
            else:
                undetected_ref.append([x, y])

        # calculate precision and recall and see if this is
        # the best precision_x_recall we've found yet
        precision = len(correct_blobs)/(len(blobs_log))
        recall = len(detected_ref)/(len(sample_coords))
        if (precision * recall) > best_precision_x_recall:
            best_precision_x_recall = precision * recall
            best_prec = precision
            best_rec = recall
            best_threshold = threshold
        precision_list.append(precision)
        recall_list.append(recall)

    return best_threshold, best_rec, best_prec, recall_list, precision_list


def sort_clusters_by_correctness(clusters=None, correctness_threshold=4,
                                 csv_filepath=None, img_height=0):
    """
    Sort clusters by correctness.

    Parameters
    ----------
    clusters : pandas dataframe of clusters to sort
        (centroid_x | centroid_y | members)
        centroid_x = int x coord of cluster centroid
        centroid_y = int y coord of cluster centroid
        members = list of annotations belonging to the cluster
            each annotation is a numpy ndarray of properties:
            [int x coord, int y coord, int time spent, str worker ID]
    correctness_threshold : float user-defined min distance from a reference
        coord for a cluster centroid to be considered incorrect
    csv_filepath : str path to csv file with reference spot coordinates
    img_height : int pixel height of annotated image

    Returns
    -------
    correct_list : list of unique member counts of correct clusters
    incorrect_list : list of unique member counts of incorrect clusters
    total_list : list of unique member counts of all clusters
    threshold : threshold unique member count above which a cluster is
        predicted to be correct
    """
    correct_list = []
    incorrect_list = []
    total_list = []

    df = util.centroid_and_ref_df(clusters, csv_filepath, img_height)
    cluster_correctness = clus.get_cluster_correctness(df,
                                                       correctness_threshold)

    for index, row in df.iterrows():
        members = row['members']
        worker_list = [member[3] for member in members]
        num_members = len(np.unique(worker_list))
        if (cluster_correctness[index][1]):     # if cluster is correct
            correct_list.append(num_members)
        else:
            incorrect_list.append(num_members)
        total_list.append(num_members)
    width = max(correct_list)
    if (max(incorrect_list) > width):
        width = max(incorrect_list)

    # threshold kmeans
    total_array = np.asarray(total_list)
    km = KMeans(n_clusters=2).fit(total_array.reshape(-1, 1))
    cluster_centers = km.cluster_centers_
    threshold = (cluster_centers[0][0]+cluster_centers[1][0])/2

    return (correct_list, incorrect_list, total_list, threshold)


def get_precision_recall(test_coords=None, ref_coords=None,
                         correctness_threshold=4):
    """
    Get precision and recall of test_coords based on ref_coords.

    Parameters
    ----------
    test_coords : numpy array of test spot coordinates
    ref_coords : numpy array of known spot coordinates

    Returns
    -------
    float precision
    float recall
    """
    ref_kdt = KDTree(ref_coords, leaf_size=2, metric='euclidean')
    test_kdt = KDTree(test_coords, leaf_size=2, metric='euclidean')

    correct_test, incorrect_test, detected_ref, undetected_ref = [], [], [], []

    for test_coord in test_coords:
        dist, ind = ref_kdt.query([test_coord], k=1)
        if dist[0][0] < correctness_threshold:
            correct_test.append(test_coord)
        else:
            incorrect_test.append(test_coord)

    # detected vs. undetected

    for ref_coord in ref_coords:
        dist, ind = test_kdt.query([ref_coord], k=1)
        if dist[0][0] < correctness_threshold:
            detected_ref.append(ref_coord)
        else:
            undetected_ref.append(ref_coord)

    precision = len(correct_test)/len(test_coords)
    recall = len(detected_ref)/len(ref_coords)

    return precision, recall


def gaussian(height, c_x, c_y, width_x, width_y):
    """
    Return a gaussian function with the given parameters.
    From https://scipy-cookbook.readthedocs.io/items/FittingData.html
    """
    w_x = float(width_x)
    w_y = float(width_y)
    return lambda x, y: height*np.exp(-(((c_x-x)/w_x)**2+((c_y-y)/w_y)**2)/2)


def moments(data):
    """
    Return the gaussian parameters of a 2D distribution by calculating
    its moments.
    From https://scipy-cookbook.readthedocs.io/items/FittingData.html
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total

    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data):
    """
    Return the gaussian parameters of a 2D distribution found by a fit.
    From https://scipy-cookbook.readthedocs.io/items/FittingData.html
    """
    params = moments(data)
    ds = data.shape
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(ds)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p
