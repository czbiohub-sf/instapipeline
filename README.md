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