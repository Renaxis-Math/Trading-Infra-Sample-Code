# This file contains all helper functions
# to be used in all files in the project.

import importlib
import consts

reload(consts)

def build_feature_map(filename: str, filetype: str = None):
    """Return a feature_name_str -> feature_description_str map,
    given that each line in the input file 'filename' has this format:
    
    'feature_name' - 'feature_description'

    Args:
        filename (str): the filename you want to use. Must specify filetype
        either in 'filename' or 'filetype'
    """
    pass

def explain_all_features():
    """Generate a well-formatted, alphabetically ordered
    descriptions of all features in 'data_description.txt'
    """
    pass

def explain_feature(feature_name: str = None):
    """Print the description of the feature 'feature_name'
    from our built feature map in consts.py

    Args:
        feature_name (str, optional): the feature name you want. Defaults to None.

    Returns:
        str: the description of the input feature
    """
    pass

