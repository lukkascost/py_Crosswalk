# Machine learn tests for Embedded platform to crosswalk detection.


## File: GEN_GLCM.py

This file generate files with features using GLCM extractor.

This file applies Otsu's Algorithm to preprocess images.

```python
MIN_BITS = 2
MAX_BITS = 8

MIN_DECIMATION = 1
MAX_DECIMATION = 10

PATH_TO_IMAGES_FOLDER = '../database-Crosswalk/Original/'
PATH_TO_SAVE_FEATURES = 'GLCM_FILES/EXP_02/'
```

Variables: 
 - MAX and Min bits: means the minimum and maximum values of Co-occurrence matrix.
 - MAX and Min Decimation: means the minimum and maximum values of steps in line and cols of image to generate an smallest image.
 - PATH_TO_IMAGES_FOLDER: Path to folder with images to extract features.
 - PATH_TO_SAVE_FEATURES: Path to folder where will be save result of features extraction.
--- 

## File: GEN_GLCM_PREPROCESS.py

This file generate files with features using GLCM extractor.

In this file does not apply Otsu or any preprocess on the  images.

```python
MIN_BITS = 2
MAX_BITS = 8

MIN_DECIMATION = 1
MAX_DECIMATION = 10

PATH_TO_IMAGES_FOLDER = '../database-Crosswalk/Preprocessed/'
PATH_TO_SAVE_FEATURES = 'GLCM_FILES/EXP_01/'
```

Variables: 
 - MAX and Min bits: means the minimum and maximum values of Co-occurrence matrix.
 - MAX and Min Decimation: means the minimum and maximum values of steps in line and cols of image to generate an smallest image.
 - PATH_TO_IMAGES_FOLDER: Path to folder with images to extract features.
 - PATH_TO_SAVE_FEATURES: Path to folder where will be save result of features extraction.
 
 ---
 
## Authors

* **Lucas Costa** - [lukkascost](https://github.com/lukkascost)

See also the list of [contributors](https://github.com/lukkascost/py_Crosswalk/contributors) who participated in this project.
