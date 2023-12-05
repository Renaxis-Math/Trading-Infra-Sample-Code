# This file contains all helper functions
# to be used in all files in the project.
import pandas as pd
import numpy as np
import importlib
import re
import consts
import os

importlib.reload(consts)

class Regression():
    def __init__(self, regression_type = 'OLS'):
        #TODO: Whenever you customize this, remember to add your method here!!!
        """
        Always follow this same format if you use default sklearn's model:
        f(train_X, train_y, sample_weight = None, allow_internal_update = True)
        """
        self.availableRegressionName_func_map = {
            'OLS': self.sklearn_ols_regression,
            'LASSO': self.sklearn_LASSO_regression,
            'XGBOOST': self.xgboost_regression
        }
        
        self.betas = None # betas[-1] = intercept
        self.predicted_responses = None
        self.actual_responses = None
        self.model = None
        
        if regression_type in self.availableRegressionName_func_map:
            self.regression_type = regression_type
        elif regression_type.lower() in self.availableRegressionName_func_map:
            self.regression_type = regression_type.lower()
        elif regression_type.upper() in self.availableRegressionName_func_map:
            self.regression_type = regression_type.upper()
        elif regression_type.capitalize() in self.availableRegressionName_func_map:
            self.regression_type = regression_type.capitalize()
        else:
            print(f"{regression_type} does not exist.")
    def __repr__(self) -> str:
        return f"{self.regression_type} Regression Model"
    
    @staticmethod
    def list_all_regression_types():
        available_regressions = Regression().availableRegressionName_func_map.keys()
        for i, regression_type in enumerate(available_regressions):
            print(f"{i+1}: {regression_type}")
        
    ### Sklearn
    def sklearn_ols_regression(self, train_X, train_y, sample_weight = None, allow_internal_update = True):
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X=train_X, y=train_y, sample_weight=sample_weight)

        if allow_internal_update: self.model = model
        return model.coef_
    
    def sklearn_LASSO_regression(self, train_X, train_y, sample_weight = None, allow_internal_update = True):
        from sklearn.linear_model import LassoCV
        
        model = LassoCV(cv=consts.CV, fit_intercept=False) # Higher cv, Lower bias
        model.fit(X=train_X, y=train_y, sample_weight=sample_weight)
        if allow_internal_update: self.model = model
        return model.coef_     

    def xgboost_regression(self, train_X, train_y, sample_weight = None, allow_internal_update = True):
        from xgboost import XGBRegressor 

        model = XGBRegressor("reg:squarederror", booster='gblinear')
        model.fit(X=train_X, y=train_y, sample_weight=sample_weight)
        if allow_internal_update: self.model = model

        return model.get_xgb_params()
    ### \Sklearn

    ### Not being used -- split to train and evaluate methods
    def execute(self, _train_df, _response_col_name,
                    _test_df = None, _sample_weight = None, allow_internal_update = True):
            copied_train_df = append_columnOf_ones(_train_df)
            copied_test_df = append_columnOf_ones(_test_df) if (_test_df is not None) else None
            train_X, train_y, test_X, test_y = self.__train_test_split (
                                                _df = copied_train_df,
                                                _test_df = copied_test_df,
                                                _response_col_name = _response_col_name
                                            )
            actual_responses = test_y.copy()
            regression_function = self.availableRegressionName_func_map[self.regression_type]
            model_attributes = regression_function(train_X, train_y, _sample_weight)
            predicted_responses = self.__predict(test_X)
            if allow_internal_update:
                self.actual_responses = actual_responses
                # self.betas = betas
                self.predicted_responses = predicted_responses
            return model_attributes
    ### \Not being used -- split to train and evaluate methods

    ### Private methods
    def __train_test_split(self, _df, _test_df, _response_col_name): #TODO: customize this!
        train_X = _df.drop(_response_col_name, inplace=False, axis=consts.COL)
        train_y = _df[_response_col_name]
        
        test_X = _test_df.drop(_response_col_name, inplace=False, axis=consts.COL)
        test_y = _test_df[_response_col_name]
        
        return (train_X, train_y, test_X, test_y)        
    
    def __get_betas(self, train_X, train_y, sample_weight = None, allow_internal_update = False):
        REGRESSION_TYPE = 'OLS'
        regression_function = self.availableRegressionName_func_map[REGRESSION_TYPE]
        return regression_function(train_X, train_y, sample_weight, allow_internal_update)

        # Design choice: Make these methods "Public"?
    def __predict(self, test_X):
        return self.model.predict(test_X)
    
    def __get_predictActual_corr(self):
        return np.corrcoef(self.predicted_responses, self.actual_responses)
    
    def __get_weighted_mean_return(self):
        return np.sum((np.abs(self.actual_responses) / len(self.actual_responses)) * \
                (np.sign(self.actual_responses) * np.sign(self.predicted_responses)))
        
    def __get_weighted_scale_factor(self):
        if (self.actual_responses is not None) and \
            (self.predicted_responses is not None):

            train_X = pd.DataFrame(self.predicted_responses)
            train_y = self.actual_responses
            return self.__get_betas(train_X, train_y)

        else: return None
        
    ### \Private methods   
    
    def train(self, _train_df, _response_col_name, 
                _sample_weight = None):

        copied_train_df = append_columnOf_ones(_train_df)
        
        train_X = copied_train_df.drop(_response_col_name, inplace=False, axis=consts.COL)
        train_y = copied_train_df[_response_col_name]
        regression_function = self.availableRegressionName_func_map[self.regression_type] 
        model_attributes = regression_function(train_X = train_X, 
                                               train_y = train_y, sample_weight = _sample_weight) 

        return model_attributes
    
    def train_model(self, train_df:pd.DataFrame, response:str, sample_weights = None):
        """Trains a model with a training data frame
        """
        train_df = append_columnOf_ones(train_df) #???

        #Split into X and y
        train_X = train_df.drop(response, inplace=False, axis=consts.COL)
        train_y = train_df[response]

        regression_function = self.availableRegressionName_func_map[self.regression_type]   
        model_attributes = regression_function(train_X, train_y, sample_weights) 
        return model_attributes

    ### Get metrics
    def evaluate_model(self, test_start:str, test_end:str, x_cols:[str],data_path:str, list_of_interacting_terms:[[str]]):
        """Evaluates the model by taking average metrics between the start and end days. 
        
        Args: 
        test_start (str): Start date of evaluation. YYYYMMDD
        test_end (str): End data of model evaluatino. YYYYMMDD
        x_cols: list of predictors to use. 
        data_path: The path to find the test data
        """
        if data_path == None:
           print("No Path given")
           return 
        filenames = get_file_names(test_start, test_end, data_path)

        wt_corr, wt_mean_ret, wt_sf = [], [], []

        for file in filenames:
            df = pd.read_csv(data_path + file)
            df = get_df_with_interaction_terms(df, list_of_interacting_terms)
            test_X = df[x_cols]
            test_X = append_columnOf_ones(test_X)

            predicted_y = self.__predict(test_X)

            self.predicted_responses = predicted_y
            self.actual_responses = df[consts.RESPONSE_NAME]

            corr,ret,sf = self.get_metric(printMetrics=False)
            wt_corr.append(corr[0][1])
            wt_mean_ret.append(ret)
            wt_sf.append(sf)
        
        avg_wt_corr = np.average(wt_corr) # Could be cleaner with mapping. 
        avg_wt_mean = np.average(wt_mean_ret)
        avg_wt_sf = np.average(wt_sf)
        print(f"Average Weighted Correlation: {avg_wt_corr}")
        print(f"Average Mean Return: {avg_wt_mean}")
        print(f"Average weighted Scale Factor: {avg_wt_sf}")
        return avg_wt_corr,avg_wt_mean,avg_wt_sf

    def get_metric(self, printMetrics = True):
        """Print metrics defind by Scott

        Returns: [weighted_corr, mean_return, scale_factor]
        """
        weighted_corr = self.__get_predictActual_corr()
        weighted_mean_return = self.__get_weighted_mean_return()
        weighted_scale_factor = self.__get_weighted_scale_factor()
        
        if printMetrics:
            print(f"1. Weighted Correlation:\n{weighted_corr}\n")
            print(f"2. Weighted Mean Return:\n{weighted_mean_return}\n")
            print(f"3. Weighted Scale Factor:\n{weighted_scale_factor}\n")
        
        return (weighted_corr, weighted_mean_return, weighted_scale_factor)
        
        

def build_feature_map(filename: str, filetype: str = None): #-> dict[str,str]
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
    file_path = consts.RAW_DATA_PATH + filename
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

def append_columnOf_ones(X):
    """Add a column of 1 to the right-most position of input DataFrame

    Args:
        X (DataFrame): input DataFrame

    Returns:
        DataFrame: a new DataFrame with a column of 1 added to the right
    """
    return X.assign(b0=1)

def get_df_with_interaction_terms(df, listOf_interacting_terms):
    """Return a new DataFrame that has interacting column pairs

    Args:
        df (DataFrame): original training data
        listOf_interacting_terms (list of list): list of column pairs

    Returns:
        DataFrame: resulting DataFrame
    """
    new_df = df.copy()
    all_columns = set(df.columns)
    for interacting_terms in listOf_interacting_terms:
        all_terms_exist = all(interacting_term in all_columns for interacting_term in interacting_terms)
        if all_terms_exist:
            new_col_name = str(tuple(interacting_terms))
            
            new_df[new_col_name] = np.prod(new_df[interacting_terms], axis=1)
            # new_df = new_df.drop(interacting_terms, axis = consts.COL)
        else:
            missing_indices = np.where(~all_terms_exist)
            print(f"{np.take(interacting_terms, missing_indices)} missing or already been grouped!")
            continue

    return new_df

def get_file_names(start, end, data_path)->list:
    """Gets the file names between the start and end. 
    Example parameter: 20150101 is January 1st 2015. yyyymmdd
    
    Args: 
    start (string): Start date of training
    end (string): Past the end date of training. Not included

    Returns: 
        List[string]: List of all training files. 
    """
    files = os.listdir(data_path)
    files = sorted(filter(lambda fname: fname < f"data.{end}" and fname >= f"data.{start}", files))
    return files

def get_df(start:str, end:str,x_cols,data_path, interacting_terms = [])-> pd.DataFrame:
    """Reads data from files to get a df of all days between start and end date
    Args: 
    start: Starting date
    end: Ending date of training
    x_cols: x_columns to train on
    interacting_terms: Columns to multiply together. 
    Returns: 
    Dataframe: training df and testing df. 
    """
    files = get_file_names(start, end, data_path)
    dfs = [pd.read_csv(data_path+ f) for f in files]
    full_df = pd.concat(dfs)
    # call interacting terms fn (needs a little more to ensure cols get saved. )
    saved_cols = x_cols + [consts.RESPONSE_NAME]
    return full_df[saved_cols]
    

def get_train_from_testday(testday):
    """Reads the testday from input and parse the day to get the range of train dates
    Args: 
    testday: a day of the test month, in form of yyyymmdd. (day is irrelevant)
    """
    year = int(testday[:4])
    month_str = testday[4:6]
    month_int = int(month_str)
    day = int(testday[6:])
    if month_str == "01":
        trainMonth = 11
        startYear = year - 2
    elif month_str == "02":
        trainMonth = 12
        startYear = year - 2
    else:
        trainMonth = month_int - 2
        startYear = year - 1   
    zero = "0" * (trainMonth < 10) # checks if zero is needed
    startDay = str(startYear) + zero + str(trainMonth) + "01"

    endDay = str(startYear+1) + zero + str(trainMonth) + "01"

    return [startDay, endDay]
    

    # for the get_file_names method, if the startday does not exist, is that okay
    # do they just return the dates inbetween 2 days even if these dates on the end of ranges does not exist
    # This should be good -- uses the < and > operators on strings. 

def get_weights(df):
    """ return a vector of weights corresponding to each sample of the training df
    Args:
    df: training dataframe, pandas dataframe object
    """
    n = df.shape[0]
    blockSize = n/12
    weights = [i/blockSize+0.5 for i in range(n)]
    return weights
