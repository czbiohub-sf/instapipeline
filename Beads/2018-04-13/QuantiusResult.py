import pandas as pd
import json

def importAnnotations(file_path):
	json_string = open(file_path).read()
	results = json.loads(json_string)

	annotations = pd.DataFrame()


	for turker in results:

		# Make a data frame of the coordinates of each annotation
		coords = pd.DataFrame(turker['raw_data'][0])

		# Add the turker metadata to all entries in the data frame
		coords['annotation_type'] = turker['annotation_type']
		coords['height'] = turker['height']
		coords['width'] = turker['width']
		coords['image_filename'] = turker['image_filename']
		coords['time_when_completed'] = turker['time_when_completed']
		coords['worker_id'] = turker['worker_id']

		# Append to the data frame
		annotations = annotations.append(coords)

	return annotations

if __name__ == "__main__":

	anno = importAnnotations('Cy3.json')

	test_plot = anno.plot(x='x', y = 'y', kind= 'scatter')


