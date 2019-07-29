import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.cluster import AffinityPropagation
from scipy import ndimage

##############################################################################

pref_param = -50000
crosshair_ratio = 0.03045       # crosshair arm to image width ratio

def get_nnd(coord, kdt):
    dist, ind = kdt.query([coord], k=2)
    return dist[0][1]

def get_bb_tuples(coords, crosshair_arm_length):

    # 1. Identify crowded spots
    kdt = KDTree(coords, leaf_size=2, metric='euclidean')
    close_distances = []
    crowded_spots = []
    for coord in coords:
        nnd = get_nnd(coord, kdt)
        if nnd < crosshair_arm_length:
            close_distances.append(nnd)
            crowded_spots.append(coord)

    crowd_ratio = len(crowded_spots)/len(coords)

    # 2. Identify regions with many crowded spots
    crowded_coords = np.asarray(crowded_spots)
    af = AffinityPropagation(preference = pref_param).fit(crowded_coords)
    centers = [crowded_coords[index] for index in af.cluster_centers_indices_]

    # 3. Define bounding box around each region with many crowded spots.
    cluster_members_lists = [[] for i in range(len(centers))]
    for label_index, coord in zip(af.labels_, crowded_coords):
        cluster_members_lists[label_index].append(coord)

    bb_list = []
    for l in cluster_members_lists:
        l = np.asarray(l)
        x = l[:,0]
        y = l[:,1]
        bb_list.append((min(x), max(x), min(y), max(y)))

    return bb_list

def crop(parent_img_name, bb):

    img = cv2.imread(parent_img_name+'.png')                  # img is a numpy 2D array
    img_cvt = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_cvt[int(bb[2]) : int(bb[3]), int(bb[0]) : int(bb[1])]

def zoom(coords, parent_img_name, crosshair_arm_length):
    bb_list = get_bb_tuples(coords, crosshair_arm_length)

    img = cv2.imread(parent_img_name+'.png')
    blacked_out = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    parent_width = blacked_out.shape[1]

    for i, bb in enumerate(bb_list):

        # black out bb area in parent img
        for r in range(blacked_out.shape[0]):
            for c in range(blacked_out.shape[1]):
                if (r >= bb[2]) and (r <= bb[3]):
                    if (c >= bb[0]) and (c <= bb[1]):
                        blacked_out[r][c] = 0

        new_img_name = parent_img_name + '_' + str(i)
        img_array = crop(parent_img_name, bb)
        zoom_factor = float(parent_width)/(bb[1] - bb[0])
        img_array_scaled = ndimage.zoom(img_array, zoom_factor)
        plt.imsave(new_img_name + '.png', img_array_scaled, cmap = 'gray')

        to_save = [x for x in bb] + [zoom_factor]
        np.savetxt(new_img_name + '.csv', to_save, delimiter=",", comments='')

        new_crosshair_arm_length = (bb[1] - bb[0]) * crosshair_ratio

        crop_coords = []
        for coord in coords:
            if (coord[0] >= bb[0]) and (coord[0] <= bb[1]):
                if (coord[1] >= bb[2]) and (coord[1] <= bb[3]):
                    crop_coords.append(coord)

        crop_kdt = KDTree(crop_coords, leaf_size=2, metric='euclidean')

        close_distances = []
        crowded_spots = []
        for coord in crop_coords:
            nnd = get_nnd(coord, crop_kdt)
            if nnd < new_crosshair_arm_length:
                close_distances.append(nnd)
                crowded_spots.append(coord)

        crowd_ratio = float(len(crowded_spots))/len(coords)

        if crowd_ratio > 0.1:
            zoom(crop_coords, new_img_name, new_crosshair_arm_length)

    plt.imsave(parent_img_name+'_blacked.png', blacked_out, cmap = 'gray')


"""
Main
"""

image_width = 700
crosshair_arm_length = crosshair_ratio * image_width

parent_img_name = 'smfish'
coords = np.genfromtxt(parent_img_name+'.csv', delimiter=',')
zoom(coords, parent_img_name, crosshair_arm_length)



