from SpotImage import SpotImage
import os

global_intensity_dial = 0   

# plot and save all spot arrays and spot images in this batch.
plot_spots = True
plot_img = True
save_spots = True
save_img = True

bg_img_path = 'MAX_ISP_300_1.tif'	# one cell image, one tissue image
num_spots = 50

snr_mu = 10
snr_sigma = 2.5
snr_threshold = 3

density = None

si = SpotImage(bg_img_path=bg_img_path)
        
snr_distr_params = ['Gauss', snr_mu, snr_sigma]
spots_filename = "".join(bg_img_path.rsplit(bg_img_path[-4:])) + "_nspots" + str(num_spots) + "_snr" + str(snr_mu) + "_" + str(snr_sigma) + "_spot_array.png"
spot_img_filename = "".join(bg_img_path.rsplit(bg_img_path[-4:])) + "_nspots" + str(num_spots) + "_snr" + str(snr_mu) + "_" + str(snr_sigma) + "_spot_img.png"
csv_filename = "".join(bg_img_path.rsplit(bg_img_path[-4:])) + "_nspots" + str(num_spots) + "_snr" + str(snr_mu) + "_" + str(snr_sigma) + "_coord_snr_list.csv"

si.generate_spot_image(num_spots=num_spots, snr_distr_params = snr_distr_params, snr_threshold = snr_threshold, plot_spots=plot_spots, plot_img=True, save_spots=True, save_img=True, spots_filename = spots_filename, spot_img_filename=spot_img_filename, density=density)
si.get_coord_snr_list_csv(csv_filename)

si.plot_spot_nnd()