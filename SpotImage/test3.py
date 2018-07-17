from SpotImage import SpotImage
import matplotlib.pyplot as plt

""" 
Parameterizing the spot image. 
"""

bg_img_filename = 'MAX_ISP_300_2.tif'
cmap = 'gray'
img_sz = 300
patch_sz = 11

spot_sigma = 1.75
spot_shape_params = ['2D_Gauss', spot_sigma]

snr_mu = 20
snr_sigma = 20
snr_distr_params = ['Gauss', snr_mu, snr_sigma]
snr_threshold = 3

global_intensity_dial = 0   # raise the threshold found by Otsu's
brightness_bias = False     # bias spots toward higher intensity background pixels within valid coordinate space
brightness_bias_dial = 0
biasing_method = None

"""
Options for visualizing and saving the spots and spot image. 
"""

plot_spots = False
plot_img = True
save_spots = False
save_img = False
spots_filename = None
spot_img_filename = "Spot Image"

# # No bias

num_spots = 300
# brightness_bias = False
# si = SpotImage(bg_img_filename, cmap, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params, snr_threshold, global_intensity_dial, brightness_bias, brightness_bias_dial, biasing_method)
# si.generate_spot_image(plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename)

# Method 2

biasing_method = 1
brightness_bias_dial = 0
si = SpotImage(bg_img_filename, cmap, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params, snr_threshold, global_intensity_dial, brightness_bias, brightness_bias_dial, biasing_method)
si.generate_spot_image(plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename)





