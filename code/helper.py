# This file contains all helper functions
# to be used in all files in the project.
import pandas as pd
import numpy as np
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

def explain_all_features(filename = "data_description.txt") -> None:
    """Generate a well-formatted, alphabetically ordered
    descriptions of all features in 'data_description.txt'
    """
    FEATURE_DESCRIPTION_MAP = build_feature_map(filename)    
    alphabetically_sorted_features = sorted(FEATURE_DESCRIPTION_MAP.keys())
    
    for feature in alphabetically_sorted_features:
        description = FEATURE_DESCRIPTION_MAP[feature]
        print(f"Feature: {feature}\nMeaning: {description}\n")
    
    return

def explain_feature(feature_name: str, filename = "data_description.txt") -> None:
    """Print the description of the feature 'feature_name'
    from "data_description.txt"

    Args:
        feature_name (str, optional): the feature name you want. Defaults to None.

    Returns:
        str: the description of the input feature
    """
    FEATURE_DESCRIPTION_MAP = build_feature_map(filename)
    
    if not feature_name: explain_all_features(filename)
    elif feature_name not in FEATURE_DESCRIPTION_MAP: raise Exception(f"Feature '{feature_name}' not exists.")
    else: print(FEATURE_DESCRIPTION_MAP[feature_name])
    
    return

def shapiro_test(residuals) -> None:
    """Perform a Shapiro test of whether a list of data is normally distributed
    
    Args:
        residuals: a list-like object of similarly typed data
    """
    import scipy.stats as stats
    
    shapiro_test_statistic, shapiro_p_value = stats.shapiro(residuals)
    alpha = 0.05  # Significance level
    if shapiro_p_value > alpha: print(f"Shapiro-Wilk Test: Residuals normally distributed.")
    else: print(f"Shapiro-Wilk Test: Residuals NOT normally distributed.")
    
    return

def white_test(residuals, df) -> None:
    """Perform a White test of heteroskedasticity 
    for a simple linear regression model

    Args:
        df (DataFrame): the testing data (X)
        residuals: a list-like object of similarly typed data
    """
    assert isinstance(df, pd.DataFrame), "2nd argument is not a DataFrame"
    import statsmodels.api as sm

    white_testing_df = sm.add_constant(df)
    white_test = sm.stats.diagnostic.het_white(residuals, white_testing_df)
    white_p_value = white_test[1]

    alpha = 0.05 # Significant level
    if white_p_value < alpha: print(f"White Test: Residuals have constant variance.")
    else: print(f"White Test: Residuals DO NOT have constant variance.")

    return
    
def bp_test(residuals, df) -> None:
    """Perform a Breusch-Pagan test of heteroskedasticity 
    for a multiple linear regression model

    Args:
        df (DataFrame): the testing data (X)
        residuals: a list-like object of similarly typed data
    """
    assert isinstance(df, pd.DataFrame), "2nd argument is not a DataFrame"
    import statsmodels.api as sm    
    bp_testing_df = sm.add_constant(df)
    bp_test = sm.OLS(residuals**2, bp_testing_df).fit()
    bp_p_value = bp_test.pvalues[1]

    alpha = 0.05 # Significant level
    if bp_p_value < alpha: print(f"Breusch-Pagan Test: Residuals have constant variance.")
    else: print(f"Breusch-Pagan Test: Residuals DO NOT have constant variance.")   
    
    return

def vif_test(r_squared: float):
    """Perform a VIF test for multi-collinearity

    Args:
        r_squared (float): the r-squared value of a regression model.
    """
    vif = 1.0 / (1 - r_squared)

    if vif < 1: print(f"VIF = {vif}. This model performs worse than a horizontal line :(")
    elif vif == 1: print(f"VIF = {vif}. All predictors are independent :)")
    elif 1 < vif < 5: print(f"VIF = {vif}. Some dependent predictors exist.")
    else: print(f"VIF = {vif}. Too many dependent predictors!") # vif >= 5

    return

def get_df_with_interaction_terms(df, col_pairs):
    """Return a new DataFrame that has interacting column pairs

    Args:
        df (DataFrame): original training data
        col_pairs (list of list): list of column pairs

    Returns:
        DataFrame: resulting DataFrame
    """
    COLUMN = 1
    ROW = 0
    
    all_columns = set(df.columns)
    for col_pair in col_pairs:
        if col_pair[0] in all_columns and col_pair[1] in all_columns:
            df[f"({col_pair[0]}, {col_pair[1]})"] = df[col_pair[0]] * df[col_pair[1]]
            df = df.drop(col_pair, axis = COLUMN)
    return df