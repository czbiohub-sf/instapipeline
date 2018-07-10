from SpotImage import SpotImage

""" 
Parameterizing the spot image. 
"""
bg_img = 'bg.jpg'
cmap = 'gray'		# color map	
img_sz = 300		# typical image size
patch_sz = 11		# typical patch size, as sampled from beads_300pxroi.png; using odd number so that center point exists

num_spots = 3
spot_sigma = 1		# produces spots that look somewhat typical in size
spot_shape_params = ['2D_Gaussian', spot_sigma]

snr_mu = 30			# looks like a typical image
snr_sigma = 1		# chosen to look like a typical image
snr_distr_params = ['Gaussian', snr_mu, snr_sigma]

si = SpotImage(bg_img, cmap, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params) # Load data into a SpotImage object

""" 
Visualizing and saving the spots and spot image. 
"""
plot_spots = True
plot_img = True
save_spots = True
spots_filename = "spot_array.png"
save_img = True
spot_img_filename = "spot_img.png"
show_progress = True

si.generate_spot_image(plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename, show_progress)