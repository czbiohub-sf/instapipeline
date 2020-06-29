# INSTA
The INSTA pipeline uses crowd-sourced annotations for spot detection in RNA _in situ_ transcriptomics images.

The directory `demos` contains notebooks that demonstrate INSTA's preprocessing and postprocessing functionality.

The directory `instapipeline` contains the codebase.
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
`instapipeline` supports python 3.6 and above. To install instapipeline, first verify that your python version is compatible by running `python -version`.

- Create a new anaconda environment: `conda create --name <environment name> python=3.7`
- Enter the new environment: `conda activate <environment name>`
- Install jupyter notebook: `conda install -c conda-forge notebook`
- Clone the instapipeline repo
- Install: `pip install <path to directory with setup.py>`
