# FISH-annotation
This repository explores using crowd-sourced annotations for spot detection in RNA in situ transcriptomics images.

The directory `fishanno` contains the codebase
- `BaseAnnotation.py` contains the BaseAnnotation class, which has the tools for annotation injestion
- `QuantiusAnnotation.py` inherits from the BaseAnnotation class and implements annotation injestion for annotations from Quanti.us
- `SpotAnnotationAnalysis.py` contains methods for clustering and keeping track of whether a certain clustering for a given dataset has already been executed (to avoid redundant computations)
- `param.py` contains functions for parameter extraction
- `autocrop.py` contains functions for autocropping input images
- `clus.py` contains functions for
    - sorting clusters by size
    - sorting clusters by clumpiness and declumping
    - other cluster analyses
- `vis.py` contains functions for visualizing annotations and clusters
- `util.py` contains functions for:
    - interacting with / manipulating dataframes
    - other data structure manipulation

## Installation
fishanno supports python 3.6 and above. To install fishanno, first verify that your python version is compatible by running `python -version`.

To install and create a virtualenv:

% python3 -m pip install --user virtualenv
% python3 -m virtualenv venv
% source venv/bin/activate

To install fishanno:

% git clone https://github.com/czbiohub/FISH-annotation.git
% pip install fishanno

To install jupyter (to view test and figure notebooks):

% pip install jupyter