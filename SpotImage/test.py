from SpotImage import SpotImage
import numpy as np

bg_img = 'bg.jpg'
color_map = 'greyscale'
img_sz = 300
patch_sz = 11		# typical patch size from beads_300pxroi.png; using odd number so that center point exists
num_spots = 4
spot_mu = 5
spot_sigma = 1
spot_shape_params = ['2D_Gaussian', spot_mu, spot_sigma]
snr_mu = 10			
snr_sigma = 1
snr_distr_params = ['Gaussian', snr_mu, snr_sigma]

si = SpotImage(bg_img, color_map, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params) # Load data into a SpotImage object
si.generate_spot_image()

# Testing adding patches to matrix
# patch_sz = 3
# img_sz = 5
# si = SpotImage(bg_img, color_map, img_sz, patch_sz, num_spots, spot_shape_params, snr_distr_params) # Load data into a SpotImage object
# spot_array = np.zeros([si.img_sz, si.img_sz])
# patch = [[1,2,3],[4,5,6],[7,8,9]]
# spot = [2,3,patch]
# spot_array = si.add_spot(spot, spot_array)
# print(patch)
# print(spot_array)





# si.img_to_array()




# array = si.get_spot_image_array()
