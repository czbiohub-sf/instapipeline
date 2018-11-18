import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.cluster import AffinityPropagation

##############################################################################

pref_param = -50000
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
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

def crop(parent_img_path, bb):

        img = cv2.imread(parent_img_path)                  # img is a numpy 2D array
        img_cvt = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img_cvt[bb[2] : bb[3], bb[0] : bb[1]]

def zoom(coords, depth, parent_img_path, crosshair_arm_length):
    bb_list = get_bb_tuples(coords, crosshair_arm_length)
    for i, bb in enumerate(bb_list):

        # black out bb area in parent_img_path

        csv_filename = 'zoom_' + str(depth) + letters[i] + '.csv'
        np.savetxt(csv_filename, bb, delimiter=",", comments='')

        new_image_path = 'zoom_' + str(depth) + letters[i] + '.png'
        img_array = get_crop(parent_img_path, bb)
        plt.imsave(png_filename, img_array, cmap = 'gray')

        crosshair_arm_length = (bb[1] - bb[0]) * crosshair_ratio

        crop_coords = []
        for coord in coords:
            if coord[0] >= xmin and coord[0] <= xmax:
                if coord[1] >= ymin and coord[1] <= ymax:
                    crop_coords.append(coord)

        crop_kdt = KDTree(crop_coords, leaf_size=2, metric='euclidean')

        close_distances = []
        crowded_spots = []
        for coord in crop_coords:
            nnd = get_nnd(coord, crop_kdt)
            if nnd < crosshair_arm_length:
                close_distances.append(nnd)
                crowded_spots.append(coord)

        crowd_ratio = float(len(crowded_spots))/len(coords)

        if crowd_ratio > 0.1:
            zoom(crop_coords, depth + 1, new_image_path)


"""
Main
"""


image_width = 700
crosshair_arm_length = crosshair_ratio * image_width

coords = np.genfromtxt('smfish.csv', delimiter=',')
depth = 1
parent_img_path = 'smfish.png'
zoom(coords, depth, parent_img_path, crosshair_arm_length)



