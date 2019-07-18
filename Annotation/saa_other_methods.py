
	def plot_clusters(self, clusters, img_filename, img_filepath, img_height, show_ref_points, csv_filepath, worker_marker_size, cluster_marker_size, show_correctness, correctness_threshold, show_possible_clumps, bigger_window_size, plot_title):
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))
		
		member_lists = clusters['members'].values	# list of lists
		color = 'orange'

		if show_ref_points:
			ref_kdt = self.csv_to_kdt(csv_filepath, img_height)

		if show_correctness:
			cluster_correctness = []
			for i in range(len(clusters.index)):
				row = clusters.iloc[[i]]
				centroid_x = row.iloc[0]['centroid_x']
				centroid_y = row.iloc[0]['centroid_y']
				coord = np.asarray([centroid_x, centroid_y]).reshape(1,-1)
				dist, ind = ref_kdt.query(coord, k=1)
				distance = dist[0][0]
				if (distance <= correctness_threshold):
					cluster_correctness.append([i,True])
				else:
					cluster_correctness.append([i,False])

			for i in range(len(member_lists)):			# for every cluster
				members = member_lists[i]					# get the list of annotations (w/ click properties) in that cluster
				if show_correctness:
					if (cluster_correctness[i][1]):
						color = 'g'						
					else:								
						color = 'm'
				else:
					color = 'orange'

			for member in members:						# plot each annotation in that cluster
				coords = member[:2]
				plt.scatter([coords[0]], self.ba.flip([coords[1]], img_height), s = worker_marker_size, facecolors = color, alpha = 0.5)

		# plot cluster centroids
		x_coords = clusters['centroid_x'].values
		y_coords = clusters['centroid_y'].values
		y_coords_flipped = self.ba.flip(y_coords, img_height)
		plt.scatter(x_coords, y_coords_flipped, s = cluster_marker_size, facecolors = 'none', edgecolors = '#ffffff')

		legend_elements = []
		if show_ref_points:
			ref_df = pd.read_csv(csv_filepath)
			ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()
			for point in ref_points:
				plt.scatter([point[0]], [point[1]], s = 20, facecolors = 'c')
			legend_elements.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='c', label='reference spots'))

		legend_elements.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', label='annotations for clusters detected as clumpy'))
		plt.legend(handles = legend_elements, loc = 9, bbox_to_anchor = (1.2, 1.015))

		# plot image
		img = mpimg.imread(img_filepath)
		plt.imshow(img, cmap = 'gray')

		plt.tick_params(
			axis='both',
			which='both',
			bottom=False,
			top=False,
			left=False,
			right=False)

		plt.title(plot_title)

		plt.show()
















		# gap 2














	def plot_annotations_per_cluster(self, df, clustering_params, show_correctness, correctness_threshold, csv_filepath, img_filename, plot_title, bigger_window_size):
		clusters = self.get_clusters(df, clustering_params)
		if not show_correctness:
			hist_list = []
			for i in range(len(clusters.index)):
				row = clusters.iloc[[i]]
				members = row.iloc[0]['members']
				worker_list = []
				for member in members:
					worker_list.append(member[3])
				num_members = len(np.unique(worker_list))
				hist_list.append(num_members)
			plt.title(plot_title)
			y,x,_ = plt.hist(hist_list, bins=np.arange(0,max(hist_list)+4,2)-1)
			width = max(hist_list)
		else:
			correct_list = []
			incorrect_list = []
			total_list = []
			anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
			cluster_correctness = self.get_cluster_correctness(anno_and_ref_df, correctness_threshold)
			for i in range(len(clusters.index)):
				row = clusters.iloc[[i]]
				members = row.iloc[0]['members']
				worker_list = []
				for member in members:
					worker_list.append(member[3])
				num_members = len(np.unique(worker_list))
				if (cluster_correctness[i][1]):		# if cluster is correct
					correct_list.append(num_members)
				else:
					incorrect_list.append(num_members)
				total_list.append(num_members)
			width = max(correct_list)
			if (max(incorrect_list) > width):
				width = max(incorrect_list)

			fig = plt.figure()
			
			y,x,_ = plt.hist([correct_list, incorrect_list], bins = np.arange(0,width+4,2)-1, stacked = True, color = ['g','m'])

			# threshold otsu
			threshold_otsu = filters.threshold_otsu(np.asarray(total_list))

			# treshold kmeans
			total_array = np.asarray(total_list)
			km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
			cluster_centers = km.cluster_centers_
			threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

			plt.axvline(x=threshold_otsu, color='r')
			plt.axvline(x=threshold_kmeans, color='b')

			g_patch = mpatches.Patch(color='g', label='clusters near ref spot')
			m_patch = mpatches.Patch(color='m', label='clusters far from any ref spot')
			otsu_line = Line2D([0],[0], color='r', label='otsu threshold')
			kmeans_line = Line2D([0],[0], color='b', label='k-means threshold')
			plt.legend(handles=[g_patch, m_patch, otsu_line, kmeans_line])

		plt.xlabel("Number of unique workers annotating")
		plt.xticks(np.arange(0,width+2,step=2))
		plt.ylabel("Number of clusters")
		ymin, ymax = plt.ylim()
		if(ymax < 30):
			plt.yticks(np.arange(0,ymax+1,step=3))
		plt.title(plot_title)
		plt.show()

	"""
	The list should contain, for each “putatively incorrect” cluster, 
	the fraction of the cluster’s annotations which are from workers 
	who are in many “putatively correct” clusters.
	"""
	def plot_fraction_from_crowd_per_cluster(self, clusters, crowd, show_correctness, correctness_threshold, csv_filepath, img_height, plot_title, bigger_window_size):
	
		correct_list = []
		incorrect_list = []
		total_list = []
		anno_and_ref_df = self.anno_and_ref_to_df_input_clusters(clusters, csv_filepath, img_height)
		cluster_correctness = self.get_cluster_correctness(anno_and_ref_df, correctness_threshold)
		for i in range(len(clusters.index)):

			# get list of unique members in that cluster
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			worker_list = []
			for member in members:
				worker_list.append(member[3])
			unique_workers = np.unique(worker_list)

			# get fraction of good crowd workers in that list
			numerator = 0
			for worker in unique_workers:
				if(worker in crowd):
					numerator += 1
			denominator = len(unique_workers)
			fract_members = math.floor((numerator/denominator)*100)

			if (cluster_correctness[i][1]):		
				correct_list.append(fract_members)
			else:
				incorrect_list.append(fract_members)
			total_list.append(fract_members)

		width = 100

		fig = plt.figure()
		
		y,x,_ = plt.hist([correct_list, incorrect_list], bins = np.arange(0,width+20,10)-5, stacked = True, color = ['g','m'])

		# # threshold otsu
		# threshold_otsu = filters.threshold_otsu(np.asarray(total_list))

		# treshold kmeans
		total_array = np.asarray(total_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

		# plt.axvline(x=threshold_otsu, color='r')
		plt.axvline(x=threshold_kmeans, color='b')

		g_patch = mpatches.Patch(color='g', label='correct clusters')
		m_patch = mpatches.Patch(color='m', label='incorrect clusters')
		# otsu_line = Line2D([0],[0], color='r', label='otsu threshold')
		kmeans_line = Line2D([0],[0], color='b', label='k-means threshold')
		plt.legend(handles=[g_patch, m_patch, kmeans_line])
		ymin, ymax = plt.ylim()
		y_step = 5
		if ymax<10:
			y_step = 1
		if (ymax>50):
			y_step = 10
		plt.xlabel("Percent of cluster’s annotations from good crowd [%]")
		plt.xticks(np.arange(0,width+10,step=10))
		plt.yticks(np.arange(0,ymax+2, step = y_step))
		plt.ylabel("Number of clusters")
		plt.title(plot_title)
		plt.show()

	def anno_and_ref_to_df_input_clusters(self, clusters, csv_filepath, img_height):

		ref_kdt = self.csv_to_kdt(csv_filepath, img_height)
		ref_array = np.asarray(ref_kdt.data)

		centroid_IDs = range(clusters.shape[0])
		column_names = ['centroid_x', 'centroid_y', 'NN_x', 'NN_y', 'NN_dist', 'members']
		to_return = pd.DataFrame(index = centroid_IDs, columns = column_names)

		for i in centroid_IDs:

			to_return['centroid_x'][i] = clusters['centroid_x'][i]
			to_return['centroid_y'][i] = clusters['centroid_y'][i]

			coords = [[to_return['centroid_x'][i], to_return['centroid_y'][i]]]

			dist, ind = ref_kdt.query(coords, k=1)
			index = ind[0][0]
			nearest_neighbor = ref_array[index]

			to_return['NN_x'][i] = nearest_neighbor[0]
			to_return['NN_y'][i] = nearest_neighbor[1]
			to_return['NN_dist'][i] = dist[0][0]
			to_return['members'][i] = clusters['members'][i]		

		return to_return

	def sort_workers_by_membership_in_large_clusters(self, df, large_clusters):
		other_crowd = []
		good_crowd = []

		worker_list = self.ba.get_workers(df)

		# find threshold (kmeans)
		total_list = []
		for uid in worker_list:
			pc_clusters_found = self.get_pc_clusters_found(large_clusters, uid)
			total_list.append(pc_clusters_found)
		total_array = np.asarray(total_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

		# given threshold, sort all workers
		for uid in worker_list:
			pc_clusters_found = self.get_pc_clusters_found(large_clusters, uid)
			if(pc_clusters_found) > threshold_kmeans:
				good_crowd.append(uid)
			else:
				other_crowd.append(uid)

		return other_crowd, good_crowd

	def plot_workers_pc_yield(self, df, large_clusters, plot_title):
		worker_list = self.ba.get_workers(df)
		hist_list = []
		for uid in worker_list:
			pc_clusters_found = self.get_pc_clusters_found(large_clusters, uid)
			hist_list.append(pc_clusters_found)

		step_size = 5
		if (max(hist_list) > 50):
			step_size = 10
		if (max(hist_list) > 100):
			step_size = 20

		y,x,_ = plt.hist(hist_list, bins=np.arange(0,max(hist_list)+step_size*2, step=step_size)-step_size/2)
		
		# threshold otsu
		threshold_otsu = filters.threshold_otsu(np.asarray(hist_list))

		# threshold kmeans
		total_array = np.asarray(hist_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

		plt.axvline(x=threshold_otsu, color='r')
		plt.axvline(x=threshold_kmeans, color='b')

		otsu_line = Line2D([0],[0], color='r', label='otsu threshold')
		kmeans_line = Line2D([0],[0], color='b', label='k-means threshold')
		plt.legend(handles=[otsu_line, kmeans_line])

		plt.xticks(np.arange(0,max(hist_list)+step_size*2, step=step_size))
		plt.yticks(np.arange(0,y.max()+1))

		plt.title(plot_title)
		plt.xlabel("Quantity of putatively correct clusters found by a worker")
		plt.ylabel("Quantity of workers")
		plt.show()

	def get_pc_clusters_found(self, large_clusters, uid):
		counter = 0
		for i in range(len(large_clusters.index)):
			row = large_clusters.iloc[[i]]
			members = row.iloc[0]['members']
			for member in members:
				worker = member[3]
				if(uid==worker):
					counter+=1
					break
		return counter

	def plot_snr_vs_members(self, df, clustering_params, csv_filepath, img_height, img_filename, correctness_threshold):

		clusters = self.get_clusters(df, clustering_params)			# this dataframe: centroid_x | centroid_y | members
		ref_df = pd.read_csv(csv_filepath)
		ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()	
		snr_val_list = ref_df.loc[:, ['snr']].as_matrix()	

		for i in range(len(ref_points)):			# flip vertical axis
			point = ref_points[i]
			first_elem = point[0]
			second_elem = img_height - point[1]
			point = np.array([first_elem, second_elem])
			ref_points[i] = point

		anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		centroid_coords = anno_and_ref_df.loc[:, ['centroid_x', 'centroid_y']].as_matrix()		
		centroids_kdt = KDTree(centroid_coords, leaf_size=2, metric='euclidean')

		snr_list = []
		num_members_list = []

		# for each spot
		for i in range(len(ref_points)):
			ref_point = ref_points[i]

			# get SNR
			snr = snr_val_list[i][0]	
			# get nearest neighbor centroid
			dist, ind = centroids_kdt.query([ref_point], k=1)
			if (dist[0][0] <= correctness_threshold):			# if the spot is detected
				centroid_coords_index = ind[0][0]
				nearest_centroid = centroid_coords[centroid_coords_index]
				nearest_centroid_x = nearest_centroid[0]
				nearest_cluster = clusters.loc[clusters['centroid_x']==nearest_centroid_x]
				members = nearest_cluster.iloc[0]['members']

				worker_list = []
				for member in members:
					worker_list.append(member[3])
				num_members = len(np.unique(worker_list))
				num_members_list.append(num_members)
				snr_list.append(snr)

		legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor='g', label='one detected spot')]
		plt.legend(handles = legend_elements)
		plt.scatter(num_members_list, snr_list, color = 'g', s = 20)
		plt.title("SNR vs. number of unique workers annotating")
		plt.xlabel("Number of unique workers annotating")
		plt.xticks(np.arange(0,30, step=2))
		plt.yticks(np.arange(min(snr_list)-1,max(snr_list)+1, step=2))
		plt.ylabel("SNR")
		plt.show()

	def plot_annotations_and_snr_per_cluster(self, df, clustering_params, show_correctness, correctness_threshold, csv_filepath, img_filename, img_height, plot_title, bigger_window_size):
		clusters = self.get_clusters(df, clustering_params)			# this dataframe: centroid_x | centroid_y | members
		
		correct_list = []
		incorrect_list = []
		total_list = []
		anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		cluster_correctness = self.get_cluster_correctness(anno_and_ref_df, correctness_threshold)
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			worker_list = []
			for member in members:
				worker_list.append(member[3])
			num_members = len(np.unique(worker_list))
			if (cluster_correctness[i][1]):		# if cluster is correct
				correct_list.append(num_members)
			else:
				incorrect_list.append(num_members)
			total_list.append(num_members)
		width = max(correct_list)
		if (max(incorrect_list) > width):
			width = max(incorrect_list)

		fig, ax1 = plt.subplots(figsize = (10,5))
		
		y,x,_ = ax1.hist([correct_list, incorrect_list], bins = np.arange(0,width+4,2)-1, stacked = True, color = ['g','m'])

		# threshold otsu
		threshold_otsu = filters.threshold_otsu(np.asarray(total_list))

		# treshold kmeans
		total_array = np.asarray(total_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

		ax1.axvline(x=threshold_otsu, color='r')
		ax1.axvline(x=threshold_kmeans, color='b')

		# NEXT PLOT			
		ref_df = pd.read_csv(csv_filepath)
		ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()	
		snr_val_list = ref_df.loc[:, ['snr']].as_matrix()	

		for i in range(len(ref_points)):			# flip vertical axis
			point = ref_points[i]
			first_elem = point[0]
			second_elem = img_height - point[1]
			point = np.array([first_elem, second_elem])
			ref_points[i] = point

		anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		centroid_coords = anno_and_ref_df.loc[:, ['centroid_x', 'centroid_y']].as_matrix()		
		centroids_kdt = KDTree(centroid_coords, leaf_size=2, metric='euclidean')

		snr_list = []
		num_members_list = []

		# for each spot
		for i in range(len(ref_points)):
			ref_point = ref_points[i]

			# get SNR
			snr = snr_val_list[i][0]	
			# get nearest neighbor centroid
			dist, ind = centroids_kdt.query([ref_point], k=1)
			if (dist[0][0] <= correctness_threshold):			# if the spot is detected
				centroid_coords_index = ind[0][0]
				nearest_centroid = centroid_coords[centroid_coords_index]
				nearest_centroid_x = nearest_centroid[0]
				nearest_cluster = clusters.loc[clusters['centroid_x']==nearest_centroid_x]
				members = nearest_cluster.iloc[0]['members']

				worker_list = []
				for member in members:
					worker_list.append(member[3])
				num_members = len(np.unique(worker_list))
				num_members_list.append(num_members)
				snr_list.append(snr)

		ax1.set_xlabel("Number of unique workers annotating")
		ax1.set_ylabel("Number of clusters")

		ax2 = ax1.twinx()
		ax2.scatter(num_members_list, snr_list, color = 'y', s = 20)
		ax2.set_ylabel("SNR")

		g_patch = mpatches.Patch(color='g', label='correct clusters')
		m_patch = mpatches.Patch(color='m', label='incorrect clusters')
		otsu_line = Line2D([0],[0], color='r', label='otsu threshold')
		kmeans_line = Line2D([0],[0], color='b', label='k-means threshold')
		snr_dot = Line2D([0],[0], marker='o', color='w', markerfacecolor='y', label='SNR for one detected spot')
		plt.legend(handles=[g_patch, m_patch, otsu_line, kmeans_line, snr_dot], bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

		plt.title(plot_title)
		fig.tight_layout()
		plt.show()


	def plot_worker_pairwise_scores(self, df):
		worker_scores = self.get_worker_pairwise_scores(df)
		worker_scores = worker_scores["score"].values
		worker_scores_list = []
		for score in worker_scores:
			worker_scores_list.append(score)

		worker_list = self.ba.get_workers(df)

		fig = plt.figure(figsize = (10,7))

		handle_list = []
		for i in range(len(worker_list)):
			score = worker_scores_list[i]
			handle = plt.bar(i, score, color = self.colors[i], label = (str(i) + ". " + worker_list[i]))
			handle_list.append(handle)

		plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.15, 1.015))
		plt.subplots_adjust(left=0.1, right=0.8)
		plt.title('Pairwise Score [s] vs. Worker Index')
		plt.xlabel('Worker Index')
		plt.ylabel('Pairwise Score')
		plt.xticks(np.arange(0, len(worker_list), step=1))
		plt.show()

	def plot_worker_pairwise_scores_hist(self, df, plot_title, bigger_window_size):

		# get worker scores as list
		worker_scores = self.get_worker_pairwise_scores(df)
		worker_scores = worker_scores["score"].values
		worker_scores_list = []
		for score in worker_scores:
			worker_scores_list.append(score)

		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		step_size = 20
		low = math.floor((min(worker_scores_list)-100)/100)*100

		y,x,_ = plt.hist(worker_scores_list, bins=np.arange(low,max(worker_scores_list)+step_size*2, step=step_size)-step_size/2)

		# threshold otsu
		threshold_otsu = filters.threshold_otsu(np.asarray(worker_scores_list))

		# threshold kmeans
		total_array = np.asarray(worker_scores_list)
		km = KMeans(n_clusters = 2).fit(total_array.reshape(-1,1))
		cluster_centers = km.cluster_centers_
		threshold_kmeans = (cluster_centers[0][0]+cluster_centers[1][0])/2

		# threshold 3rd quartile
		threshold_q3 = np.mean(worker_scores_list) + 1.5*np.std(worker_scores_list)

		plt.axvline(x=threshold_otsu, color='r')
		plt.axvline(x=threshold_kmeans, color='b')
#		plt.axvline(x=threshold_q3, color='g')

		otsu_line = Line2D([0],[0], color='r', label='otsu threshold')
		kmeans_line = Line2D([0],[0], color='b', label='k-means threshold')
#		q3_line = Line2D([0],[0], color='g', label='q3 threshold')
		plt.legend(handles=[otsu_line, kmeans_line])

		plt.title(plot_title)
		plt.xlabel('Sum of pairwise NND averages')
		plt.ylabel('Quantity of workers')
		width = max(worker_scores_list) - low
		if(width>2000):
			x_step = 200
		elif (width>1000):
			x_step = 100
		else:
			x_step = 50
		plt.xticks(np.arange(low,max(worker_scores_list)+x_step*2,step=x_step))
		plt.yticks(np.arange(0,y.max()+1))
		plt.show()

	def plot_error_rate_vs_spotted(self, df, clustering_params, correctness_threshold, csv_filepath, img_filename, plot_title, bigger_window_size):
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		worker_list = self.ba.get_workers(df)

		num_good_clusters_list = []
		error_rate_list = []

		for worker in worker_list:
			num_good_clusters_list.append(self.get_worker_num_good_clusters(worker, clusters, correctness_threshold))
			error_rate_list.append(self.get_worker_error_rate(worker, clusters, correctness_threshold) * 100)
		
		plt.scatter(num_good_clusters_list, error_rate_list, facecolors='c', s=20)
		legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor='c', label='one worker')]
		plt.legend(handles = legend_elements)
		plt.title("Error rate vs. number of good clusters found")
		plt.xlabel("Number of good clusters found by the worker")
		plt.ylabel("Worker's error rate [%]")

		if (max(num_good_clusters_list)<=60):
			x_step = 5
		elif(max(num_good_clusters_list)<=110):
			x_step = 10
		else:
			x_step = 20
		plt.xticks(np.arange(0,max(num_good_clusters_list)+2,step=x_step))

		if (max(error_rate_list)<=20):
			y_step = 1
		elif(max(error_rate_list)<=40):
			y_step = 2
		else:
			y_step = 5
		plt.yticks(np.arange(0,101,step=y_step))

		plt.show()

	"""
	For one worker, get the number of good clusters that worker found.
	Inputted df "clusters" is generated by anno_and_ref_to_df()
	"""
	def get_worker_num_good_clusters(self, uid, clusters, correctness_threshold):
		counter = 0
		cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)
		for i in range(len(clusters.index)):
			if not (cluster_correctness[i][1]):
				continue
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			for member in members:
				worker = member[3]
				if(uid==worker):
					counter += 1
					break
		return counter

	"""
	error_rate = (number of bad clusters the worker is a member in)/(number of clusters the worker is a member in)
	"""
	def get_worker_error_rate(self, uid, clusters, correctness_threshold):
		num_bad = num_total = 0
		cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)
		for i in range(len(clusters.index)):
			row = clusters.iloc[[i]]
			members = row.iloc[0]['members']
			for member in members:
				worker = member[3]
				if(uid==worker):
					num_total += 1
					if (cluster_correctness[i][1] == False): # if it's a bad cluster
						num_bad += 1
		return num_bad/num_total

	def get_worker_correct_rate(self, uid, clusters, correctness_threshold):
		return (1 - self.get_worker_error_rate(uid, clusters, correctness_threshold))

	def plot_workers_correct_rate(self, df, clustering_params, correctness_threshold, csv_filepath, img_filename, plot_title, bigger_window_size):
		
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		worker_list = self.ba.get_workers(df)
		
		correct_rates = []
		for worker in worker_list:
			correct_rate = self.get_worker_correct_rate(worker, clusters, correctness_threshold)
			correct_rates.append(correct_rate*100)

		y,x,_ = plt.hist(correct_rates, bins=np.arange(0,105, step=1)-0.5, color = 'g')

		plt.title(plot_title)
		plt.xticks(np.arange(0,105, step=5))
		plt.yticks(np.arange(0,y.max()+1, step=1))
		plt.xlabel("Fraction of the worker's annotations that were in a good cluster [%]")
		plt.ylabel("Quantity of workers")
		plt.show()


	"""
	For a given dataset, take the subset of reference spots with an SNR > n 
	and calculate the fraction of that subset that were detected by the turkers. 
	Build a curve by varying n (3 through SNR_max) to see how high the minimum 
	SNR needs to be for 100% of the spots to be detected.
	"""
	def plot_snr_sensitivity(self, df, clustering_params, csv_filepath, img_height, img_filename, correctness_threshold, plot_title, bigger_window_size):
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		centroid_coords = anno_and_ref_df.loc[:, ['centroid_x', 'centroid_y']].as_matrix()		
		centroids_kdt = KDTree(centroid_coords, leaf_size=2, metric='euclidean')

		ref_df = pd.read_csv(csv_filepath)
		ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()	
		for i in range(len(ref_points)):			# flip vertical axis
			point = ref_points[i]
			first_elem = point[0]
			second_elem = img_height - point[1]
			point = np.array([first_elem, second_elem])
			ref_points[i] = point

		snr_val_list = ref_df.loc[:, ['snr']].as_matrix()	

		snr_min = math.floor(min([snr_val_list[i][0] for i in range(len(snr_val_list))]))
		snr_max = math.ceil(max([snr_val_list[i][0] for i in range(len(snr_val_list))]))
		n_list = range(snr_min,snr_max)
		fraction_list = []
		for n in n_list:
			spots_detected = 0
			spots_total = 0
			# for each spot
			for i in range(len(ref_points)):
				# get SNR
				snr = snr_val_list[i][0]
				if(snr<n):
					continue
				spots_total += 1
				ref_point = ref_points[i]
				# get nearest neighbor centroid
				dist, ind = centroids_kdt.query([ref_point], k=1)
				if (dist[0][0] <= correctness_threshold):
					spots_detected += 1
			if(spots_total == 0):
				fraction_list.append(0)
			else:
				fraction_list.append((spots_detected/spots_total)*100)
			#print ('min_SNR ={0:2d}, spots_detected ={1:3d}, spots_total ={2:3d}'.format(n, spots_detected, spots_total))
		plt.scatter(n_list, fraction_list, facecolors = 'g', s = 20)
		plt.plot(n_list, fraction_list, color = 'green')

		plt.title(plot_title)
		plt.xlabel("Minimum SNR of spots in subset")
		plt.ylabel("Fraction of subset of spots detected by workers [%]")
		plt.show()

	"""
	For each spot, plot SNR vs. number of annotations in the corresponding cluster.
	"""
	def plot_snr_vs_membership(self, df, clustering_params, csv_filepath, img_height, img_filename, correctness_threshold, bigger_window_size):

		clusters = self.get_clusters(df, clustering_params)			# this dataframe: centroid_x | centroid_y | members
		
		ref_df = pd.read_csv(csv_filepath)
		ref_points = ref_df.loc[:, ['col', 'row']].as_matrix()	
		snr_val_list = ref_df.loc[:, ['snr']].as_matrix()	

		for i in range(len(ref_points)):			# flip vertical axis
			point = ref_points[i]
			first_elem = point[0]
			second_elem = img_height - point[1]
			point = np.array([first_elem, second_elem])
			ref_points[i] = point

		anno_and_ref_df = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)
		centroid_coords = anno_and_ref_df.loc[:, ['centroid_x', 'centroid_y']].as_matrix()		
		centroids_kdt = KDTree(centroid_coords, leaf_size=2, metric='euclidean')

		counter_undetected = counter_detected = 0
		snr_undetected = []
		snr_detected = []

		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		# for each spot
		for i in range(len(ref_points)):
			ref_point = ref_points[i]

			# get SNR
			snr = snr_val_list[i][0]	

			# get nearest neighbor centroid
			dist, ind = centroids_kdt.query([ref_point], k=1)
			if (dist[0][0] > correctness_threshold):
				num_members = 0
				color = 'm'
				counter_undetected += 1
				snr_undetected.append(snr)
			else:
				centroid_coords_index = ind[0][0]
				nearest_centroid = centroid_coords[centroid_coords_index]
				nearest_centroid_x = nearest_centroid[0]
				nearest_cluster = clusters.loc[clusters['centroid_x']==nearest_centroid_x]
				members = nearest_cluster.iloc[0]['members']

				# get number of annotations associated with that centroid 
				num_members = len(members)
				color = 'g'
				counter_detected += 1
				snr_detected.append(snr)

			plt.scatter([num_members],[snr], facecolors = color, alpha = 0.5, s = 20)

		s_1 = str(counter_undetected) + " spots detected by no workers"
		s_2 = str(counter_detected) + " spots detected by at least one worker"

		legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor='g', label=s_2),
							Line2D([0],[0], marker='o', color='w', markerfacecolor='m', label=s_1)]

		plt.title("For each spot, SNR vs. number of clicks in the nearest cluster")
		plt.xlabel("Number of clicks in the nearest cluster")
		plt.ylabel("SNR")
		plt.legend(handles = legend_elements)
		plt.show()

		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		plt.title("Number of spots detected vs. SNR")
		y,x,_ = plt.hist(snr_detected, bins=np.arange(0,max(snr_detected)+2,1)-0.5, color = 'g')
		plt.xticks(np.arange(0,max(snr_detected)+2,step=1))
		if(max(snr_detected)<20):
			y_step = 1
		else:
			y_step = 2
		plt.yticks(np.arange(0,y.max()+1, step=y_step))
		plt.xlabel("SNR")
		plt.ylabel("Number of spots detected")
		plt.show()

		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		if (len(snr_undetected)==0):
			plt.title("No spots undetected")
		else:
			plt.title("Number of spots undetected vs. SNR")
			y,x,_ = plt.hist(snr_undetected, bins=np.arange(max(snr_undetected)+2)-0.5, color = 'm')
			plt.xticks(np.arange(0,max(snr_undetected)+2,step=1))
			plt.yticks(np.arange(0,y.max()+1, step=1))
			plt.xlabel("SNR")
			plt.ylabel("Number of spots undetected")
		plt.show()










	"""
	Plots the average time spent per click for all workers 
	in the dataframe.

	Input:
		dataframe with annotation data
		bool whether to use a bigger window size (for jupyter notebook)
	Returns:
		none
	"""
	def plot_avg_time_per_click(self, df, bigger_window_size):
		avg_list = []
		for worker in self.ba.get_workers(df):
			avg_time = self.ba.get_avg_time_per_click(df, worker)
			avg_list.append(avg_time/1000)
		n_bins = 10
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))
		y,x,_ = plt.hist(avg_list, bins = np.arange(0,max(avg_list)+0.5, step=0.25)-0.125)
		plt.title('Average time spent per click')
		plt.xticks(np.arange(0,max(avg_list)+0.5, step=0.25))
		plt.yticks(np.arange(0,y.max()+1, step=1))
		plt.xlabel('Time [s]')
		plt.ylabel('Quantity of workers')
		plt.show()

	"""
	Description:
		For each annotation (each click) in a dataframe, 
		plot nearest neighbor distance (nnd) vs. time spent. 
		Each point represents one annotation (one click). 
		All workers on one plot, colored by worker ID.
		Can color each point by correctness. 
	Implementation notes:
		if show_correctness
			can't use calc_time_per_click and calc_distances, because need
			to look at coordinates one by one and get time_spent on coordinate, 
			NND of associated centroid, and correctness of associated centroid.
		if not show_correctness, it's better to use calc_time_per_click and 
			calc_distances, which do not require clustering.
	Inputs:
		dataframe
		img_filename (the cropping)
		csv_filepath (contains reference data)
		bool whether to color each point by correctness of cluster
		correctness_threshold
		clustering_params
	Returns:
		none
	"""
	def plot_nnd_vs_time_spent(self, df, img_filename, csv_filepath, show_correctness, correctness_threshold, clustering_params):
# heeeere
		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		fig = plt.figure(figsize = (10,7))
		img_height = anno_one_crop['height'].values[0]
		ref_kdt = self.csv_to_kdt(csv_filepath, img_height)

		if show_correctness:

			# Goal: for each coordinate in coords, plot NND vs. time_spent and color with correctness
			# Run Af on all annotation coords (just once) and get labels (a list with a label for each annotation coordinate).
			# For each coordinate in coords:
			#		time_spent: pull from coords_with_times
			#		NND: query using a kdtree.
			#		correctness: index of coordinate is i=index of label. label[i] is index of correctness. correctness[index] is the appropriate correctness.
			#		aaaaand... plot NND vs. time_spent and color with correctness!

			coords = self.ba.get_click_properties(anno_one_crop)[:,:2]
			coords_with_times = self.ba.get_click_properties(anno_one_crop)[:,:3]
			clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)	# clusters -> NND, coordinates
			cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)		# clusters <-> correctness
			af = self.get_cluster_object(coords, clustering_params)
			labels = af.labels_	

			for i in range(len(coords)):
				time_spent = coords_with_times[i][2]
				if(time_spent==0):
					continue
				coordinate = coords[i]		# a single coordinate
				dist, ind = ref_kdt.query([coordinate], k=1)
				NND = dist[0][0]
				index = labels[i]			# label[i] is the index of the cluster affiliated with this coordinate
				if(cluster_correctness[index][1]):
					color = 'g'
					marker_size = 4
					alpha_selection = 0.25
				else:
					color = 'm'
					marker_size = 20
					alpha_selection = 1
				plt.scatter([time_spent], [NND], s = marker_size, facecolors = color, edgecolors = None, alpha = alpha_selection)

		else:
			worker_list = self.ba.get_workers(anno_one_crop)
			time_list = self.calc_time_per_click(anno_one_crop, img_filename)		# list containing one list for each worker
			dist_list = self.calc_distances(anno_one_crop, ref_kdt, img_filename)	# list containing one list for each worker
			handle_list = []
			for i in range(len(worker_list)):			# for each worker
				color = self.colors[i]
				x_coords = time_list[i]
				y_coords = dist_list[i]
				handle = plt.scatter(x_coords, y_coords, s = 8, facecolors = color, alpha = 0.5, label = worker_list[i])
				handle_list.append(handle)
			plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.2, 1.015))
			plt.subplots_adjust(left=0.1, right=0.75)

		plt.title('Nearest Neighbor Distance (NND) vs. Time Spent For Each Click [s]')
		plt.xlabel('Time Spent [ms]')
		plt.ylabel('Nearest Neighbor Distance (NND)')
		plt.show()

	"""
	For each annotation (each click) in a dataframe, 
	plot nearest neighbor distance (nnd) vs. worker index. 
	Each point represents one annotation (one click). 
	Can color each point by correctness. 

	Inputs:
		dataframe
		img_filename (the cropping)
		csv_filepath (contains reference data)
		bool whether to color each point by correctness of cluster
		correctness_threshold
		clustering_params
	Returns:
		none
	"""
	def plot_nnd_vs_worker_index(self, df, img_filename, csv_filepath, show_correctness, correctness_threshold, clustering_params, show_avgs):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)
		img_height = anno_one_crop['height'].values[0]
		ref_kdt = self.csv_to_kdt(csv_filepath, img_height)
		dist_list = self.calc_distances(anno_one_crop, ref_kdt, img_filename)	# list containing one list for each worker

		fig = plt.figure(figsize = (10,7))

		# plot all clicks
		if show_correctness:
			click_properties = self.ba.get_click_properties(anno_one_crop)		
			coords = click_properties[:,:2]

			clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)	# clusters -> NND, coordinates
			cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)		# clusters <-> correctness
			af = self.get_cluster_object(coords, clustering_params)
			labels = af.labels_	
			img_height = anno_one_crop['height'].values[0]
			ref_kdt = self.csv_to_kdt(csv_filepath, img_height)

			for i in range(len(coords)):
				worker_id = click_properties[i][3]
				worker_index = np.where(worker_list == worker_id)

				coordinate = coords[i]
				dist, ind = ref_kdt.query([coordinate], k=1)
				NND = dist[0][0]
				index = labels[i]
				if(cluster_correctness[index][1]):
					color = 'g'
					marker_selection = 'o'
					marker_size = 4
					alpha_selection = 1
				else:
					color = 'm'
					marker_selection = '_'
					marker_size = 40
					alpha_selection = 1
				plt.scatter([worker_index], [NND], s = marker_size, facecolors = color, edgecolors = None, marker = marker_selection, alpha = alpha_selection)

		else:
			for i in range(len(worker_list)):			# for each worker
				x_coords = [i]*len(dist_list[i])
				y_coords = dist_list[i]
				plt.scatter(x_coords, y_coords, s = 4, alpha = 0.5, facecolors = 'c')

		# plot worker average distances
		if show_avgs:
			avg_distances = []
			for i in range(len(worker_list)):
				worker_distances = dist_list[i]
				worker_avg_dist = np.average(worker_distances)
				avg_distances.append(worker_avg_dist) 
			handle = plt.scatter(range(len(worker_list)), avg_distances, s = 60, facecolors = 'b', marker = '_', label = 'Average NND')
			plt.legend(handles = [handle], loc = 9, bbox_to_anchor = (1.15, 0.55))
			plt.subplots_adjust(left=0.1, right=0.8)

		plt.title('Nearest Neighbor Distance (NND) vs. Worker Index For Each Click')
		plt.xlabel('Worker Index')
		plt.ylabel('Nearest Neighbor Distance (NND)')
		plt.xticks(np.arange(0, len(worker_list), step=1))
		plt.show()

	"""
	For each annotation (each click) in a dataframe, 
	plot time spent on the click vs. worker index. 
	Each point represents one annotation (one click). 
	Can color each point by correctness. 

	Inputs:
		dataframe
		img_filename (the cropping)
		csv_filepath
		bool whether to color each point by correctness of cluster
		correctness_threshold
		clustering_params
	Returns:
		none
	"""
	def plot_time_spent_vs_worker_index(self, df, img_filename, csv_filepath, show_correctness, correctness_threshold, clustering_params, show_avgs):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)			# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)
		time_list = self.calc_time_per_click(anno_one_crop, img_filename)	# list containing one list for each worker

		fig = plt.figure(figsize = (10,7))

		# plot all clicks
		if show_correctness:
			click_properties = self.ba.get_click_properties(anno_one_crop)		# coordinates <-> time_spent
			coords = click_properties[:,:2]
			clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)	# clusters -> NND, coordinates
			cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)		# clusters <-> correctness
			af = self.get_cluster_object(coords, clustering_params)
			labels = af.labels_	
			img_height = anno_one_crop['height'].values[0]
			ref_kdt = self.csv_to_kdt(csv_filepath, img_height)

			for i in range(len(coords)):
				time_spent = click_properties[i][2]
				worker_id = click_properties[i][3]
				worker_index = np.where(worker_list == worker_id)

				coordinate = coords[i]
				dist, ind = ref_kdt.query([coordinate], k=1)
				index = labels[i]
				if(cluster_correctness[index][1]):
					color = 'g'
					marker_selection = 'o'
					marker_size = 10
					alpha_selection = 1
				else:
					color = 'm'
					marker_selection = '_'
					marker_size = 40
					alpha_selection = 1
				plt.scatter([worker_index], [time_spent], s = marker_size, facecolors = color, edgecolors = None, marker = marker_selection, alpha = alpha_selection)
		else:	
			for i in range(len(worker_list)):		# for each worker
				x_coords = [i]*len(time_list[i])
				y_coords = time_list[i]
				y_coords.pop(0)						# discard initial fencepost in time_list
				x_coords.pop(0)						# discard corresponding initial entry
				plt.scatter(x_coords, y_coords, s = 4, alpha = 0.5, facecolors = 'c')

		# plot worker average times
		if show_avgs:
			avg_times = []
			for i in range(len(worker_list)):
				worker_times = time_list[i]
				if not worker_times:				# if list of worker times is empty
					avg_times.append(0)
					continue
				worker_times.pop(0)
				worker_avg_time = np.average(worker_times)
				avg_times.append(worker_avg_time/1000) 
			handle = plt.scatter(range(len(worker_list)), avg_times, s = 60, facecolors = 'b', marker = '_', label = 'Average time spent')
			plt.legend(handles = [handle], loc = 9, bbox_to_anchor = (1.15, 0.55))
			plt.subplots_adjust(left=0.1, right=0.8)

		plt.title('Time Spent [s] vs. Worker Index')
		plt.xlabel('Worker Index')
		plt.ylabel('Time Spent [s]')
		plt.xticks(np.arange(0, len(worker_list), step=1))
		plt.show()

	"""
	For each worker, plot total time spent vs. worker index. 
	Each bar represents one worker. 

	Inputs:
		dataframe
		img_filename (the cropping)
	Returns:
		none
	"""
	def plot_total_time_vs_worker_index(self, df, img_filename):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)			# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)

		fig = plt.figure(figsize = (10,7))

		handle_list = []
		for i in range(len(worker_list)):
			total_time = self.ba.get_total_time(anno_one_crop, worker_list[i])
			handle = plt.bar(i, total_time[0]/1000, color = self.colors[i], label = worker_list[i])
			handle_list.append(handle)

		plt.legend(handles = handle_list, loc = 9, bbox_to_anchor = (1.15, 1.015))
		plt.subplots_adjust(left=0.1, right=0.8)
		plt.title('Total Time Spent [s] vs. Worker Index')
		plt.xlabel('Worker Index')
		plt.ylabel('Time Spent [s]')
		plt.xticks(np.arange(0, len(worker_list), step=1))
		plt.show()

	def plot_total_worker_time_hist(self, df, bigger_window_size):
		total_time_list = []
		for worker in self.ba.get_workers(df):
			total_time = self.ba.get_total_time(df, worker)
			total_time_list.append(total_time[0]/1000)
		n_bins = 10
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))
		plt.hist(total_time_list, bins=np.arange(0, max(total_time_list)+20, step=10)-5)
		plt.title('Total Time Spent by Workers [s]')
		plt.xlabel('Time Spent [s]')
		plt.ylabel('Quantity of Workers')
		x_step = 10
		if ((max(total_time_list)) > 100):
			x_step = 20
		if ((max(total_time_list)) > 250):
			x_step = 50
		plt.xticks(np.arange(0, max(total_time_list)+20, step=10))
		plt.show()

	"""
	For one worker in a dataframe,
	plot time spent on click vs. index of that click. 
	Each point represents one annotation (one click).

	Inputs:
		dataframe
		img_filename (the cropping)
		uid (worker ID)
	Returns:
		none
	"""
	def plot_time_spent_vs_click_index(self, df, img_filename, csv_filepath, uid, show_correctness, correctness_threshold, clustering_params):

		anno_one_crop = self.ba.slice_by_image(df, img_filename)	# Remove data from other croppings.
		worker_list = self.ba.get_workers(anno_one_crop)
		fig = plt.figure(figsize = (10,7))
		anno_one_worker = self.ba.slice_by_worker(anno_one_crop, uid)

		if show_correctness:
			click_properties = self.ba.get_click_properties(anno_one_worker)		
			coords = click_properties[:,:2]
			clusters = self.anno_and_ref_to_df(df, clustering_params, csv_filepath, img_filename)	# clusters -> NND, coordinates
			cluster_correctness = self.get_cluster_correctness(clusters, correctness_threshold)		# clusters <-> correctness
			af = self.get_cluster_object(coords, clustering_params)
			labels = af.labels_
			img_height = anno_one_worker['height'].values[0]
			ref_kdt = self.csv_to_kdt(csv_filepath, img_height)
			num_clicks = len(coords)

			for i in range(num_clicks):
				time_spent = click_properties[i][2]
				click_index = i

				coordinate = coords[i]
				dist, ind = ref_kdt.query([coordinate], k=1)
				index = labels[i]
				if not isinstance(index, int):
					continue
				if(cluster_correctness[index][1]):
					color = 'g'
					marker_selection = 'o'
					marker_size = 10
					alpha_selection = 1
				else:
					color = 'm'
					marker_selection = 'o'
					marker_size = 40
					alpha_selection = 1
				plt.scatter([click_index], [time_spent], s = marker_size, facecolors = color, edgecolors = None, marker = marker_selection, alpha = alpha_selection)

		else:
			index = np.where(worker_list == uid)
			i = index[0][0]		# because np.where() returns a tuple containing an array
			time_list = self.calc_time_per_click(anno_one_crop, img_filename)	# list containing one list for each worker
			worker_time_list = time_list[i]
			num_clicks = len(worker_time_list)
			x_coords = range(num_clicks)
			y_coords = [x / 1000 for x in worker_time_list]
			handle = plt.scatter(x_coords, y_coords, s = 4, facecolors = 'c', label = 'One click')
			plt.legend(handles = [handle], loc = 9, bbox_to_anchor = (1.15, 0.55))
			plt.subplots_adjust(left=0.1, right=0.8)
		
		plt.title('Time Spent [s] vs. Click Index for Worker ' + uid)
		plt.xlabel('Click Index')
		plt.ylabel('Time Spent [s]')
		plt.xticks(np.arange(0, num_clicks, step=10))
		plt.show()


		"""
		[header] plotter methods - to curate
		"""

	"""
	Build curve by varying number of unique workers required for valid cluster.
	"""
	def plot_cluster_membership_threshold_roc(self, df_1, df_2, clustering_params, csv_filepath_1, csv_filepath_2, img_height, correctness_threshold, plot_title, bigger_window_size):
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))
		else:
			fig = plt.figure(figsize = (12,7))

		tpr_list_1, fpr_list_1 = self.get_tpr_fpr_lists(df_1, clustering_params, csv_filepath_1, img_height, correctness_threshold)
		tpr_list_2, fpr_list_2 = self.get_tpr_fpr_lists(df_2, clustering_params, csv_filepath_2, img_height, correctness_threshold)

		plt.scatter(fpr_list_1, tpr_list_1, facecolors = 'blue', s = 20)
		plt.scatter(fpr_list_2, tpr_list_2, facecolors = 'orange', s = 20)

		leg_elem_1 = Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', label='inverted spot image')
		leg_elem_2 = Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', label='original spot image')
		legend_elements = [leg_elem_1, leg_elem_2]
		plt.legend(handles = legend_elements, bbox_to_anchor=(1.01, 1), loc=2)
		plt.yticks(np.arange(0,1.01,step=0.1))
		plt.title(plot_title)
		plt.xlabel("False positive rate")
		plt.ylabel("True positive rate")
		plt.show()

	def get_tpr_fpr_lists(self, df, clustering_params, csv_filepath, img_height, correctness_threshold):
		worker_list = self.ba.get_workers(df)
		num_workers = len(worker_list)
		all_clusters = self.get_clusters(df, clustering_params)
		tpr_list = []
		fpr_list = []

		for threshold in range(num_workers):
			small_clusters, large_clusters = self.sort_clusters_by_size_input_threshold(all_clusters, threshold)
			anno_and_ref_df = self.anno_and_ref_to_df_input_clusters(large_clusters, csv_filepath, img_height)
			cluster_correctness = self.get_cluster_correctness(anno_and_ref_df, correctness_threshold)

			# Get total number of spots
			ref_kdt = self.csv_to_kdt(csv_filepath, img_height)
			ref_array = np.asarray(ref_kdt.data)
			num_spots_total = len(ref_array)

			# Get number of clusters which are spots and number of clusters which are not spots
			num_true_positives = 0
			num_false_positives = 0
			num_clusters_total = 0
			for i in range(len(anno_and_ref_df.index)):		# sort clusters
				if (cluster_correctness[i][1]):		
					num_true_positives += 1
				else:
					num_false_positives += 1
				num_clusters_total += 1

			tpr = num_true_positives/num_spots_total
			if (tpr > 1):		# no double counting
				tpr = 1
			if (num_clusters_total == 0):
				fpr = 0
			else:
				fpr = num_false_positives/num_clusters_total
			
			tpr_list.append(tpr)
			fpr_list.append(fpr)

		return tpr_list, fpr_list






	"""
	[header] these were moved over from BA class
	"""








	# Returns dataframe with all fast clicks screened
	# Clicks of which time_spent < time_threshold are "fast"
	def screen_clicks_time_spent(self, df, time_threshold):
		to_return = pd.DataFrame()
		occasions = np.unique(df.loc[:, ['time_when_completed']].as_matrix())			# get the list of occasions
		for occasion in occasions:
			one_occasion_df = df[df.time_when_completed == occasion]
			one_occasion_timestamps = one_occasion_df.loc[:, ['timestamp']].as_matrix()
			for i in range(len(one_occasion_timestamps)-1, -1, -1):
				if(i==0):
					one_occasion_df = one_occasion_df.drop([i])
				else:
					time_spent = one_occasion_timestamps[i][0] - one_occasion_timestamps[i-1][0]
					if(time_spent<time_threshold):
						one_occasion_df = one_occasion_df.drop([i])
			to_return = to_return.append(one_occasion_df)
		return to_return









def calc_fpr_tpr(clusters=None, csv_filepath=None, correctness_threshold=4, plot_tpr=False, plot_fpr=False, img_filepath=None, img_height=300, cluster_marker_size=40, bigger_window_size=True):
	""" Compare the centroids in the clusters dataframe with the reference
	values to calculate the false positive and true positive rates.
	
	Parameters
	----------
	clusters : pandas dataframe (centroid_x | centroid_y | members) ~ output of sort_clusters_by_clumpiness()
		centroid_x = x coord of cluster centroid
		centroid_y = y coord of cluster centroid
		members = list of annotations belonging to the cluster
			each member is a list of properties of the annotation 
			i.e. [x coord, y coord, time spent, worker ID]
	csv_filepath : string filepath to csv file containing reference points
	correctness_threshold : tolerance for correctness in pixels
	plot_tpr : boolean whether to visualize tpr
	plot_fpr : boolean whether to visualize fpr
	img_filepath : string filepath to image file
	img_heit : height of image in pixels
	cluster_marker_size : plotting parameter
	bigger_window_size : bool whether to use bigger window size (for jupyter notebook)
	
	Returns
	-------
	tpr : num spots detected / num spots total
	fpr : num clusters don’t correspond with a spot / num clusters total
	"""

	if plot_tpr or plot_fpr:
		fig = plt.figure(figsize = (12,7))
		if bigger_window_size:
			fig = plt.figure(figsize=(14,12))

	ref_df = pd.read_csv(csv_filepath)
	ref_coords = ref_df.loc[:, ['col', 'row']].as_matrix()
	for i in range(len(ref_coords)):
		point = ref_coords[i]
		first_elem = point[0]
		second_elem = img_height - point[1]
		point = np.array([first_elem, second_elem])
		ref_coords[i] = point
	centroid_coords = clusters.loc[:, ['centroid_x', 'centroid_y']].as_matrix()
	centroid_kdt = KDTree(centroid_coords, leaf_size=2, metric='euclidean')	# kdt is a kd tree with all the reference points

	# calc tpr
	num_spots_detected = 0
	for ref_coord in ref_coords:
		dist, ind = centroid_kdt.query([ref_coord], k=1)
		if dist[0] <= correctness_threshold:
			num_spots_detected += 1
			if plot_tpr:
				plt.scatter([ref_coord[0]], flip([ref_coord[1]], img_height), s=cluster_marker_size, facecolors = 'g')
		else:
			if plot_tpr:
				plt.scatter([ref_coord[0]], flip([ref_coord[1]], img_height), s=cluster_marker_size, facecolors = 'm')
	num_spots_total = len(ref_coords)
	tpr = num_spots_detected/num_spots_total

	# calc fpr
	ref_kdt = csv_to_kdt(csv_filepath, img_height)
	num_centroids_wout_spot = 0
	for centroid_coord in centroid_coords:
		dist, ind = ref_kdt.query([centroid_coord], k=1)
		if dist[0] > correctness_threshold:
			num_centroids_wout_spot += 1
			if plot_fpr:
				plt.scatter([centroid_coord[0]], flip([centroid_coord[1]], img_height), s=cluster_marker_size, edgecolors='m', facecolors='none')
		else:
			if plot_fpr:
				plt.scatter([centroid_coord[0]], flip([centroid_coord[1]], img_height), s=cluster_marker_size, edgecolors='g', facecolors='none')
	num_centroids_total = len(centroid_coords)
	fpr = num_centroids_wout_spot/num_centroids_total

	handle_list = []

	if plot_tpr:
		plt.title("TPR = " + str(round(tpr, 2)))
		handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='g', label='detected spot'))
		handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor='m', label='undetected spot'))

	if plot_fpr:
		plt.title("FPR = " + str(round(fpr, 2)))
		handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=None, markeredgecolor='g', label='correct centroid'))
		handle_list.append(Line2D([0],[0], marker='o', color='w', markerfacecolor=None, markeredgecolor='m', label='incorrect centroid'))

	if plot_tpr or plot_fpr:
		img = mpimg.imread(img_filepath)
		plt.imshow(img, cmap = 'gray')
		plt.legend(handles=handle_list, loc=9, bbox_to_anchor=(1.2, 1.015))
		plt.show()	

	return tpr, fpr


# def slice_by_image(df, img_filename):
# 	""" Return a dataframe with annotations for one image

# 	Parameters
# 	----------
# 	df : pandas dataframe
# 	img_filename : string filename of image

# 	Returns
# 	-------
# 	Dataframe with annotations for only that image 
#	
#	No longer useful because each qa object gets data from only one image

# 	"""
# 	return df[df.image_filename == img_filename]



