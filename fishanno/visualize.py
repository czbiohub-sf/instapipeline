""" 
This module contains functions for visualizing precision and recall with napari.
"""

# ----- #

from napari import Window, Viewer
from skimage import io
from PyQt5.QtWidgets import QApplication
from sklearn.neighbors import KDTree
import numpy as np

# ----- #

def sort_correct_detected(consensus_coords, reference_coords, correctness_threshold):
    """
    
    Input:
    - consensus coords (2D array object)
    - reference coords (2D array objenct)
    
    Returns:
    - all consensus coords
    - correct consensus coords
    - incorrect consensus coords
    - all reference coords
    - correct reference coords
    - incorrect reference coords

    """

    consensus_kdt = KDTree(consensus_coords, leaf_size=2, metric='euclidean')
    reference_kdt = KDTree(reference_coords, leaf_size=2, metric='euclidean')

    correct_consensus = []
    incorrect_consensus = []
    detected_reference = []
    undetected_reference = []

    # correct vs. incorrect

    for consensus_coord in consensus_coords:
        dist, ind = reference_kdt.query([consensus_coord], k=1)
        if dist[0][0] < correctness_threshold:
            correct_consensus.append(consensus_coord)
        else:
            incorrect_consensus.append(consensus_coord)

    # detected vs. undetected

    for reference_coord in reference_coords:
        dist, ind = consensus_kdt.query([reference_coord], k=1)
        if dist[0][0] < correctness_threshold:
            detected_reference.append(reference_coord)
        else:
            undetected_reference.append(reference_coord)

    correct_consensus = np.asarray(correct_consensus)
    incorrect_consensus = np.asarray(incorrect_consensus)
    detected_reference = np.asarray(detected_reference)
    undetected_reference = np.asarray(undetected_reference)
    
    return (consensus_coords, correct_consensus, incorrect_consensus, reference_coords, detected_reference, undetected_reference)

def visualize_precision_recall(coords_sorted, img_filepath):

    consensus_coords, correct_consensus, incorrect_consensus, reference_coords, detected_reference, undetected_reference = coords_sorted

    app = QApplication.instance() or QApplication([])

    # Load the image
    image = io.imread(img_filepath, name='image')

    # initialize the viewer
    viewer = Viewer()
    window = Window(viewer, show=False)
    viewer._window = window

    # Add the image to the viewer
    viewer.add_image(image, name="image")

    # Add annotations to the viewer
    viewer.add_markers(correct_consensus, symbol='ring', face_color='green', name="corr. cons.")
    viewer.add_markers(incorrect_consensus, symbol='ring', face_color='magenta', name="incorr. cons.")
    viewer.add_markers(consensus_coords, symbol='ring', face_color='white', name="all cons.")

    viewer.add_markers(detected_reference, symbol='ring', face_color='cyan', name="det. ref.")
    viewer.add_markers(undetected_reference, symbol='ring', face_color='orange', name="undet. ref.")
    viewer.add_markers(reference_coords, symbol='ring', face_color='yellow', name="all ref.")

    window.show() 
