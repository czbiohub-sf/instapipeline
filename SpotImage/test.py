from SpotImage import SpotImage

""" 
Parameterizing the spot image. 
"""
bg_img_filename = 'MAX_C2_crop1.tiff'
bg_img_filename = 'MAX_C2-ISP_FixTest_PFA_L-probe_40x_1-3NA_1-6x_20180608_1_MMStack_Pos0.ome.tif'
cmap = 'gray'		# color map	
img_sz = 300		# typical image size
patch_sz = 11		# typical patch size, as sampled from beads_300pxroi.png; using odd number so that center point exists

num_spots = 100
spot_sigma = 1		# produces spots that look somewhat typical in size
spot_shape_params = ['2D_Gaussian', spot_sigma]

snr_mu = 30			# looks like a typical image
snr_sigma = 1		# chosen to look like a typical image
snr_distr_params = ['Gaussian', snr_mu, snr_sigma]

si = SpotImage(bg_img_filename, cmap, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params) # Load data into a SpotImage object

# si.get_valid_coords()

""" 
Visualizing and saving the spots and spot image. 
"""
plot_spots = True
plot_img = True
save_spots = True
spots_filename = "spot_array.png"
save_img = True
spot_img_filename = "spot_img.png"

si.generate_spot_image(plot_spots, plot_img, save_spots, spots_filename, save_img, spot_img_filename)