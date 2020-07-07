"""
This module contains the QuantiusAnnotation class.
"""

from instapipeline import BaseAnnotation
import json
import pandas as pd


class QuantiusAnnotation(BaseAnnotation):
    """
    Implementation of _import_annotations
    for annotations from Quantius
    """
    def __init__(self, json_filepath, img_filename):
        super().__init__(json_filepath, img_filename)

    def _import_annotations(self, json_filepath, img_filename):
        """
        Import annotations to a dataframe and save the
        dataframe as a property of the QuantiusAnnotation class

        Parameters
        ----------
        json_filepath : str path to json containing annotation data
        img_filename : str name of image file that was annotated
        """
        to_return = pd.DataFrame()
        json_string = open(json_filepath).read()
        results = json.loads(json_string)

        for worker in results:

            # Skip the worker if they didn't perform any annotations
            if not worker['raw_data']:
                continue

            # Make a data frame of the coordinates of each annotation
            if (worker['annotation_type'] == 'crosshairs'):
                coords = pd.DataFrame(worker['raw_data'][0])
            elif (worker['annotation_type'] == 'polygon'):
                num_annotations = len(worker['raw_data'])
                annotations = []
                for i in range(num_annotations):
                    annotation = worker['raw_data'][i]
                    annotation = pd.DataFrame(annotation)
                    annotations.append(annotation)
                coords = pd.DataFrame(annotations)

            # Add the worker metadata to all entries in the data frame
            coords['annotation_type'] = worker['annotation_type']
            coords['height'] = worker['height']
            coords['width'] = worker['width']
            coords['image_filename'] = worker['image_filename']
            coords['time_when_completed'] = worker['time_when_completed']
            coords['worker_id'] = worker['worker_id']

            # Append to the total data frame
            to_return = to_return.append(coords)

        return to_return[to_return.image_filename == img_filename]
