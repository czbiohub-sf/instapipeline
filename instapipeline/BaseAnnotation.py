"""
This module contains tools for annotation ingestion.
"""


class BaseAnnotation:

    def __init__(self, json_filepath, img_filename):
        """ Import annotations to a dataframe and save the
        dataframe as a property of the BaseAnnotation class
        """
        self.annotations = self._import_annotations(json_filepath,
                                                    img_filename)

    def _import_annotations(self, json_filepath, img_filename):
        """ Import annotations from a file to a dataframe

        Raise an error if the method has not been
        overwritten by a child class
        """
        raise NotImplementedError

    def df(self):
        """ Return the dataframe
        """
        return self.annotations
