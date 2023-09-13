# This file contains all helper functions
# to be used in all files in the project.

import importlib
import re
import consts

importlib.reload(consts)

def build_feature_map(filename: str, filetype: str = None) -> dict[str, str]:
    """Return a feature_name_str -> feature_description_str map,
    given that each line in the input file 'filename' has this STRICT format:
    
    'feature_name' - 'feature_description' 
    [a string (or empty), a space, '-', a space, a string (or empty)]

    Args:
        filename (str): the filename you want to use. Must specify file type
        either in 'filename' or 'filetype'
        
        filetype (str): a string of file type WITHOUT DOTS.
    """
    
    # Edge cases...
    if (not filename) or ((not re.match(r'.+\.+.+', filename)) and (not filetype)):
        print(f"Returned Empty. '{filename}' is not a valid file name")
        return {}
    
    if filetype:
        if filetype.count('.') > 1:
            print(f"Returned Empty. Please remove to < 1 dot in {filetype}")
            return {}
        
        if filetype.find('.') > 0:
            print(f"Return Empty. Please only place the '.' in {filetype}'s 1st position")
            return {}
    # ...Edge cases

    # Build file path...
    if filetype:
        if (filetype[0] == '.'): filename = filename + filetype
        else: filename = filename + '.' + filetype
    file_path = consts.DATA_PATH + filename
    # ...Build file path

    # Read the file...
    lines = []
    try:
        file = open(file_path, "r")
        lines = file.readlines()
        file.close()
    except:
        print(f"Returned empty. Can't open file {filename}.")
        return {}
    # ...Read the file
    
    # Build the feature_description_map...
    answer = {}
    
    validLines_count = 0 # For testing
    for line in lines:
        line = line.strip()
        if len(line) == 0: continue

        validLines_count += 1
        
        line = line.replace("\n", '')  # Remove next-line symbol

        feature, description = tuple(line.split(" - "))
        feature = feature.strip()
        description = description.strip()
        
        if len(feature) > 0:
            if feature in answer:
                print("Feature already in FEATURE_DESCRIPTION_MAP. Updating {feature} description instead...")
            answer[feature] = description
    # ...Build the feature_description_map
    
    assert validLines_count == len(answer), \
    "There exists nonempty lines not included in FEATURE_DESCRIPTION_MAP"

    return answer

def explain_all_features() -> None:
    """Generate a well-formatted, alphabetically ordered
    descriptions of all features in 'data_description.txt'
    """
    alphabetically_sorted_features = sorted(FEATURE_DESCRIPTION_MAP.keys())
    
    for feature in alphabetically_sorted_features:
        description = FEATURE_DESCRIPTION_MAP[feature]
        print(f"Feature: {feature}\nMeaning: {description}\n")
    
    return

def explain_feature(feature_name: str = None) -> None:
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

# Local Constants
FEATURE_DESCRIPTION_MAP = build_feature_map("data_description.txt")