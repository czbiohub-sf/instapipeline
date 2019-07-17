"""

main():
	coords = readcsv() to numpy
	depth = 1
	parent_image_path = path

	zoom(coords, depth, parent_image_path, crosshair_arm_length)

zoom(coords, depth, parent_image_path, crosshair_arm_length):

	get bb tuples (xmin, xmax, ymin, ymax)
	
	for i in range len (bb tuples):

		black out bb area in parent_image_path
		write bb tuple to csv named "zoom " +  str(depth) + letter[i]
		write cropped image associated with bb tuple to png named [new_image_path] "zoom " +  str(depth) + letter[i]
		
		update crosshair_arm_length
		get crop_coords in bb

		if crowded ratio > 0.1:

			zoom(crop_coords, depth + 1, new_image_path)


"""