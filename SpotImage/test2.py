from SpotImage import SpotImage

""" 
Parameterizing the spot image. 
"""

bg_img_filename = 'MAX_C3-ISP_300_1.tif'
cmap = 'gray'		
img_sz = 300		
patch_sz = 11		

spot_sigma = 1.75		
spot_shape_params = ['2D_Gauss', spot_sigma]

snr_sigma = 20		
snr_threshold = 3

bg_intensity_threshold = 2	# increase realism: raise the threshold found by Otsu's so that spots appear in brighter parts of cells/tissue
brightness_bias = True		# increase realism: bias spots toward higher intensity background pixels within valid coordinate space

"""
Options for visualizing and saving the spots and spot image. 
"""

plot_spots = True
plot_img = True
save_spots = False
save_img = False

"""
Generate images for the experiment
"""

# bg_img_filename_list = ['MAX_C3-ISP_300_1.tif', 'MAX_ISP_300_1.tif']	# one cell image, one tissue image
# num_spots_list = [50, 100, 150]
# snr_mu_list = [5, 10, 20]

bg_img_filename = 'MAX_ISP_300_1.tif'
num_spots = 1000
snr_mu = 20

snr_distr_params = ['Gauss', snr_mu, snr_sigma]

si = SpotImage(bg_img_filename, cmap, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params, snr_threshold, bg_intensity_threshold, brightness_bias) # Load data into a SpotImage object

spots_filename = None
spot_img_filename = "".join(bg_img_filename.rsplit(bg_img_filename[-4:])) + "_nspots" + str(num_spots) + "_spot_sig" + str(spot_sigma) + "_snr" + str(snr_mu) + "_" + str(snr_sigma) + "_coord_snr_list.png"
si.generate_spot_image(plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename)