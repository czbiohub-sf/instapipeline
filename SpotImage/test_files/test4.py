from SpotImage import SpotImage

""" 
Instantiate the SpotImage.
"""

bg_img_filename = 'MAX_ISP_300_2.tif'
cmap = 'gray'
img_sz = 300
patch_sz = 11
spot_sigma = 1.75
spot_shape_params = ['2D_Gauss', spot_sigma]

brightness_bias = False     # bias spots toward higher intensity background pixels within valid coordinate space
brightness_bias_dial = 0
biasing_method = None
global_intensity_dial = 0   # raise the threshold found by Otsu's

si = SpotImage(bg_img_filename, cmap, img_sz, patch_sz, spot_shape_params, brightness_bias, brightness_bias_dial, biasing_method, global_intensity_dial)

""" 
Generate a spot image.
"""

plot_spots = False
plot_img = True
save_spots = False
spots_filename = None
save_img = True
spot_img_filename = "SNR_5"
brightness_bias = False

num_spots = 150
snr_mu = 5
snr_sigma = 20
snr_distr_params = ['Gauss', snr_mu, snr_sigma]
snr_threshold = 3

si.generate_spot_image(num_spots, snr_distr_params, snr_threshold, plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename)

""" 
Export data for the most recently generated spot image.
"""

si.get_coord_snr_list_csv("SNR_5.csv")

""" 
Generate another spot image.
"""

plot_spots = False
plot_img = True
save_spots = False
spots_filename = None
save_img = True
spot_img_filename = "SNR_20"
brightness_bias = False

num_spots = 150
snr_mu = 20
snr_sigma = 20
snr_distr_params = ['Gauss', snr_mu, snr_sigma]
snr_threshold = 3

si.generate_spot_image(num_spots, snr_distr_params, snr_threshold, plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename)

""" 
Export data for the most recently generated spot image.
"""

si.get_coord_snr_list_csv("SNR_20.csv")



