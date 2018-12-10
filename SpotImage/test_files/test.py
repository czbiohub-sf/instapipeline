from SpotImage import SpotImage

snr_sigma = 2.5
snr_threshold = 3

# no brightness biasing in this batch
brightness_bias = True     
brightness_bias_dial = 0
biasing_method = 3
global_intensity_dial = 0   

# plot and save all spot arrays and spot images in this batch.
plot_spots = True
plot_img = True
save_spots = False
save_img = True

bg_img_filename_list = ['MAX_ISP_300_1.tif']	# one cell image, one tissue image
num_spots_list = [50]
snr_mu_list = [10]

for bg_img_filename in bg_img_filename_list:

    si = SpotImage(bg_img_filename = bg_img_filename, brightness_bias = brightness_bias, brightness_bias_dial = brightness_bias_dial, biasing_method = biasing_method, global_intensity_dial = global_intensity_dial)

    for num_spots in num_spots_list:
        for snr_mu in snr_mu_list:  
            
            snr_distr_params = ['Gauss', snr_mu, snr_sigma]
            spots_filename = "".join(bg_img_filename.rsplit(bg_img_filename[-4:])) + "_nspots" + str(num_spots) + "_snr" + str(snr_mu) + "_" + str(snr_sigma) + "_spot_array.png"
            spot_img_filename = str(biasing_method)+".png"
            # spot_img_filename = "".join(bg_img_filename.rsplit(bg_img_filename[-4:])) + "_nspots" + str(num_spots) + "_snr" + str(snr_mu) + "_" + str(snr_sigma) + "_spot_img.png"
            csv_filename = "".join(bg_img_filename.rsplit(bg_img_filename[-4:])) + "_nspots" + str(num_spots) + "_snr" + str(snr_mu) + "_" + str(snr_sigma) + "_coord_snr_list.csv"

            si.generate_spot_image(num_spots = num_spots, snr_distr_params = snr_distr_params, snr_threshold = snr_threshold, plot_spots=plot_spots, plot_img=True, save_spots=True, save_img=True, spots_filename = spots_filename, spot_img_filename = spot_img_filename, density = density)
            # si.get_coord_snr_list_csv(csv_filename)