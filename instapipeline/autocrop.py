"""
This module contains functions for autocropping.
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from sklearn.neighbors import KDTree
from sklearn.cluster import AffinityPropagation, KMeans
from scipy import ndimage

# list of declumping algs handled
declumping_algs = ['KMeans']

# autocrop parameters
pref_param = -50000
crosshair_ratio = 0.04


def get_nnd(coord, kdt):
    """
    Get the nearest neighbor distance between the given coordinate
    and the contents of the k-D tree.

    Parameters
    ----------
    coord : list or 1D np array of coordinates
    kdt : sklearn.neighbors.kd_tree.KDTree of coordinates to compare coord with

    Returns
    -------
    The nearest neighbor distance between the given coordinate
    and the contents of the k-D tree
    """
    dist, ind = kdt.query([coord], k=2)
    return dist[0][1]


def get_bb_tuples(coords, crosshair_arm_length, max_num_crops):
    """
    Get the list of bounding boxes which will become crops

    Parameters
    ----------
    coords : np array of spot locations
    crosshair_arm_length : int pixel length of one arm of the crosshairs
        symbol used for annotation
    max_num_crops : int user-specified maximum number of crops

    Returns
    -------
    bb_list : list of bounding boxes tuples
        Each tuple: (x_min, x_max, y_min, y_max)
    """
    # 1. Identify crowded spots
    kdt = KDTree(coords, leaf_size=2, metric='euclidean')
    crowded_spots = []
    for coord in coords:
        nnd = get_nnd(coord, kdt)
        if nnd < crosshair_arm_length:
            crowded_spots.append(coord)

    crowd_ratio = len(crowded_spots)/len(coords)

    # 2. Identify regions with many crowded spots

    crowded = np.asarray(crowded_spots)

    # If crowded ratio is small, first try AffinityPropagaion,
    # adjusting the preference parameter
    num_centers = 0
    if crowd_ratio < 0.4:
        num_centers = max_num_crops + 1
        pref_param = -500
        for j in range(3):
            af = AffinityPropagation(preference=pref_param).fit(crowded)
            centers = [crowded[ind] for ind in af.cluster_centers_indices_]
            num_centers = len(centers)
            cluster_members_lists = [[] for i in range(len(centers))]
            for label_index, coord in zip(af.labels_, crowded):
                cluster_members_lists[label_index].append(coord)
            pref_param *= 10
            if num_centers <= max_num_crops:
                break

    # If still too many clusters, or if we didn't try AP,
    # partition using K-means
    if num_centers > max_num_crops or num_centers == 0:
        km = KMeans(n_clusters=max_num_crops).fit(crowded)
        centers = km.cluster_centers_
        cluster_members_lists = [[] for center in centers]
        for label_index, coord in zip(km.labels_, crowded):
            cluster_members_lists[label_index].append(coord)

        # 3. Define bounding box around each region with many crowded spots.
        cluster_members_lists = [[] for center in centers]
        for label_index, coord in zip(km.labels_, crowded):
            cluster_members_lists[label_index].append(coord)

    else:
        cluster_members_lists = [[] for center in centers]
        for label_index, coord in zip(af.labels_, crowded):
            cluster_members_lists[label_index].append(coord)

    bb_list = []
    for cluster_members_list in cluster_members_lists:
        cluster_members_list = np.asarray(cluster_members_list)
        x = cluster_members_list[:, 0]
        y = cluster_members_list[:, 1]
        bb_list.append((min(x), max(x), min(y), max(y)))

    return bb_list


def crop(parent_img_path, bb):
    """
    Crop a parent image based on a bounding box and return the crop

    Parameters
    ----------
    parent_img_path : str path to the parent image to be cropped
    bb : bounding box tuple of ints structured as (x_min, x_max, y_min, y_max)

    Returns
    -------
    the cropped image
    """
    img = imread(parent_img_path+'.png', as_gray=True)
    return img[int(bb[2]): int(bb[3]), int(bb[0]): int(bb[1])]


def blackout(im, bb):
    """
    Set intensity values at all locations on the given image im
    within bounding box bb to 0.

    Parameters
    ----------
    im : ndarray image with region to be blacked out
    bb : bounding box tuple of ints structured as (x_min, x_max, y_min, y_max)

    Returns
    -------
    blacked out : the image with all locations within bounding box bb
        set to 0
    """
    blacked_out = im
    for r in range(blacked_out.shape[0]):
        for c in range(blacked_out.shape[1]):
            if (r >= bb[2]) and (r <= bb[3]):
                if (c >= bb[0]) and (c <= bb[1]):
                    blacked_out[r][c] = 0
    return blacked_out


def get_crop_coords(coords, bb):
    """
    Get all coordinates in coords which are within bounding box bb.

    Parameters
    ----------
    coords : np array of coordinates
    bb : bounding box tuple of ints structured as (x_min, x_max, y_min, y_max)

    Returns
    -------
    crop_coords : all coordinates in coords which are within bb
    """
    crop_coords = []
    for coord in coords:
        if (coord[0] >= bb[0]) and (coord[0] <= bb[1]):
            if (coord[1] >= bb[2]) and (coord[1] <= bb[3]):
                crop_coords.append(coord)
    return crop_coords


def get_crowded_spots(crop_coords, new_crosshair_arm_length):
    """
    Get all coords in crop_coords which are smaller than
        new_crosshair_arm_length

    Parameters
    ----------
    crop_coords : np array of coordinates
    new_crosshair_arm_length : int length of crosshair arm scaled
        within autocrop.autocrop()

    Returns
    -------
    crowded_spots : coords in crop_coords which are smaller than
        new_crosshair_arm_length
    """
    crop_coords = np.asarray(crop_coords)
    crop_kdt = KDTree(crop_coords, leaf_size=2, metric='euclidean')
    crowded_spots = []
    for coord in crop_coords:
        nnd = get_nnd(coord, crop_kdt)
        if nnd < new_crosshair_arm_length:
            crowded_spots.append(coord)
    return crowded_spots


def autocrop(coords, parent_img_name, crosshair_arm_length,
             max_num_crops, max_crowded_ratio, crop_dir, parent_dir):
    """
    Autocrop the parent image (parent_img_name) based
    on the coordinates (coords), recursing with a
    maximum number of crops (max_num_crops) at each
    level until the percentage of spots which are
    crowded as dictated by crosshair_arm_length is
    less than max_crowded_ratio

    Parameters
    ----------
    coords : np array of coordinates
    parent_img_name : str name of parent image
    crosshair_arm_length : int pixel length of one arm of the crosshairs
        symbol used for annotation
    max_num_crops : int user-specified maximum number of crops
    max_crowded_ratio : int threshold ratio of crowded spots to non-crowded
        spots which prompts another level of cropping
    crop_dir : str path to directory where crops will be saved
    parent_dir : str path to directory containing the parent image
    """
    bb_list = get_bb_tuples(coords, crosshair_arm_length, max_num_crops)
    im = imread('./' + parent_dir + '/' + parent_img_name+'.png', as_gray=True)
    parent_width = im.shape[1]

    for i, bb in enumerate(bb_list):

        # black out bb area in parent img
        blacked_out = blackout(im, bb)

        new_img_name = parent_img_name + '_' + str(i)
        parent_img_path = './' + parent_dir + '/' + parent_img_name
        img_array = crop(parent_img_path, bb)
        zoom_factor = float(parent_width)/(bb[1] - bb[0])
        img_array_scaled = ndimage.zoom(img_array, zoom_factor)
        plt.imsave('./%s/%s.png' % (crop_dir, new_img_name),
                   img_array_scaled, cmap='gray')

        to_save = [x for x in bb] + [zoom_factor]
        np.savetxt('./%s/%s.csv' % (crop_dir, new_img_name),
                   to_save, delimiter=",", comments='')

        crop_coords = get_crop_coords(coords, bb)
        new_crosshair_arm_length = (bb[1] - bb[0]) * crosshair_ratio

        crowded_spots = get_crowded_spots(
            crop_coords, new_crosshair_arm_length)

        crowd_ratio = float(len(crowded_spots))/len(coords)
        if crowd_ratio > max_crowded_ratio:
            autocrop(crop_coords, new_img_name, new_crosshair_arm_length,
                     max_num_crops, max_crowded_ratio)

    plt.imsave('./%s/%s_blacked.png' % (crop_dir, parent_img_name),
               blacked_out, cmap='gray')
