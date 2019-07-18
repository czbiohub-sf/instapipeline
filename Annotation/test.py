import util
import numpy as np

image_width = 1390
crosshair_arm_length = 0.4 * image_width
max_num_crops = 4
max_crowded_ratio = 0.3

parent_img_name = 'ISS_rnd1_ch1_z0'
coords = np.genfromtxt(parent_img_name+'.csv', delimiter=',')
util.zoom(coords, parent_img_name, crosshair_arm_length, max_num_crops, max_crowded_ratio)