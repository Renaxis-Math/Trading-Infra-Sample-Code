# This file contains all helper functions
# to be used in all files in the project.

import consts

def build_feature_map(filename: str, no_dot_filetype: str = None):
    """Return a feature_name_str -> feature_description_str map,
    given that each line in the input file 'filename' has this format:
    
    'feature_name' - 'feature_description'

    Args:
        filename (str): the filename you want to use. Must specify file type
        either in 'filename' or 'no_dot_filetype'
        
        no_dot_filetype (str): a string of file type WITHOUT DOTS.
    """
    
    # Edge cases
    if ('.' not in filename) and (not no_dot_filetype):
        raise Exception(f"Returned Empty. {filename} is not a valid file name, and file type is {no_dot_filetype}")
        return {}
    
    if ('.' not in filename): # 'no_dot_filetype' isn't empty
        if ('.' in no_dot_filetype):
            raise Exception(f"Returned Empty. Please remove the dot in {no_dot_filetype}")
            return {}
        else:
            filename = filename + '.' + no_dot_filetype

    file_path = consts.DATA_PATH + filename
    
    try:
        file = open(file_path, "r")
        lines = file.readlines()
        
        # Build the feature_description_map
        answer = {}
        
        for line in lines:
            line = line.replace("\n", '') # Remove next-line symbol
            
            feature, description = tuple(line.split(" - "))
            feature = feature.strip()
            description = description.strip()
            
            if feature in answer:
                print("Feature already in FEATURE_DESCRIPTION_MAP. Updating {feature} description instead...")
            answer[feature] = description
        
        return answer
    
    except:
        print(f"Returned empty. Can't open file {filename}.")
        return {}
    
    finally:
        file.close()
# Local Constants
FEATURE_DESCRIPTION_MAP = build_feature_map("data_description", "txt")

def explain_all_features():
    """Generate a well-formatted, alphabetically ordered
    descriptions of all features in 'data_description.txt'
    """
    alphabetically_sorted_features = sorted(FEATURE_DESCRIPTION_MAP.keys())
    
    for feature in alphabetically_sorted_features:
        description = FEATURE_DESCRIPTION_MAP[feature]
        print(f"Feature: {feature}\nMeaning: {description}\n")
    
    return

def explain_feature(feature_name: str = None):
    """Print the description of the feature 'feature_name'
    from our built feature map in consts.py

    Args:
        feature_name (str, optional): the feature name you want. Defaults to None.

    Returns:
        str: the description of the input feature
    """
    if not feature_name: explain_all_features()
    elif feature_name not in FEATURE_DESCRIPTION_MAP: raise Exception(f"Feature '{feature_name}' not exists.")
    else: print(FEATURE_DESCRIPTION_MAP[feature_name])
    
    return