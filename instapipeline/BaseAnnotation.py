"""
This module contains tools for annotation ingestion.
"""


class BaseAnnotation:

    def __init__(self, json_filepath, img_filename):
        """
        Import annotations to a dataframe and save the
        dataframe as a property of the BaseAnnotation class.

        Parameters
        ----------
        json_filepath : str path to json containing annotation data
        img_filename : str name of image file that was annotated
        """
        self.annotations = self._import_annotations(json_filepath,
                                                    img_filename)

    def _import_annotations(self, json_filepath, img_filename):
        """
        Import annotations from a file to a dataframe.
        Raise an error if the method has not been
        overwritten by a child class.

        Parameters
        ----------
        json_filepath : str path to json containing annotation data
        img_filename : str name of image file that was annotated
        """
        raise NotImplementedError

    def df(self):
        """
        Return the dataframe
        """
        return self.annotations
