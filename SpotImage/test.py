from SpotImage import SpotImage

""" 
Parameterizing the spot image. 
"""

bg_img_filename = 'MAX_ISP_300_3.tif'
cmap = 'gray'		# color map	
img_sz = 300		# typical image size
patch_sz = 11		# typical patch size, as sampled from beads_300pxroi.png; using odd number so that center point exists

num_spots = 200
spot_sigma = 1.75		# produces spots that look somewhat typical in size
spot_shape_params = ['2D_Gauss', spot_sigma]

snr_mu = 12				# looks like a typical image
snr_sigma = 20			# looks like a typical image
snr_distr_params = ['Gauss', snr_mu, snr_sigma]

intensity_threshold = 3	# increase realism: raise the threshold found by Otsu's so that spots appear in brighter parts of cells/tissue

si = SpotImage(bg_img_filename, cmap, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params, intensity_threshold) # Load data into a SpotImage object

"""
Visualizing and saving the spots and spot image. 
"""
plot_spots = True
plot_img = True
save_spots = True
save_img = True
spots_filename = "".join(bg_img_filename.rsplit(bg_img_filename[-4:])) + "_nspots" + str(num_spots) + "_spot_sig" + str(spot_sigma) + "_snr" + str(snr_mu) + "_" + str(snr_sigma) + "_spot_array.png"
spot_img_filename = "".join(bg_img_filename.rsplit(bg_img_filename[-4:])) + "_nspots" + str(num_spots) + "_spot_sig" + str(spot_sigma) + "_snr" + str(snr_mu) + "_" + str(snr_sigma) + "_spot_img.png"

si.generate_spot_image(plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename)