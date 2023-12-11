import pandas as pd
import numpy as np
import consts
import importlib
import os, re, typing

def get_sorted_file_names(start_date: str, end_date: str, data_path: str) -> list:
    """
    Author: Ryan Butler
    
    Gets the file names between the start and end. 
    Example parameter: 20150101 is January 1st 2015. yyyymmdd
    
    Args: 
    start (string): Start date of training
    end (string): Past the end date of training. Not included

    Returns: 
        List[string]: List of all training files. 
    """
    files = os.listdir(data_path)
    filtered_files = filter(lambda fname: fname < f"data.{end_date}" and fname >= f"data.{start_date}", files)
    filtered_files.sort(reverse=False)
    return filtered_files

def get_df_with_interaction_terms(df: pd.DataFrame, interacting_terms_list: list, will_drop_single_interacting_term = False):
    """
    Author: Hoang Chu
    
    Return a new DataFrame that has interacting column pairs

    Args:
        df (DataFrame): original training data
        interacting_terms_list (list of list): list of column pairs

    Returns:
        DataFrame: resulting DataFrame
    """
    new_df = df.copy()

    for interacting_terms in interacting_terms_list:
        all_terms_exist = np.all(np.isin(np.ravel(interacting_terms), new_df.columns))
        if not all_terms_exist:
            print(f"{interacting_terms} missing! Returned original input DataFrame.")
            return df
                    
        new_col_name = str(tuple(interacting_terms))
        new_df[new_col_name] = np.prod(new_df[interacting_terms], axis=consts.COL)

    if will_drop_single_interacting_term:
        for interacting_terms in interacting_terms_list:
            new_df.drop(interacting_terms, axis = consts.COL, inplace=True)

    return new_df

def filter_df(stayed_col_names: list, interacting_terms_list: Optional[list], *, 
              start: str, end: str, data_path: str, 
              will_drop_single_interacting_term = False) -> pd.DataFrame:
    """
    Author: Ryan Butler

    Args:
        stayed_col_names (list): _description_
        interacting_terms_list (Optional[list]): _description_
        start (str): _description_
        end (str): _description_
        data_path (str): _description_
        will_drop_single_interacting_term (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    file_names = get_sorted_file_names(start, end, data_path)
    dfs = [pd.read_csv(data_path + f) for file_name in file_names]
    
    full_df = pd.concat(dfs)
    if consts.RESPONSE_NAME not in set(stayed_col_names): stayed_col_names.append(consts.RESPONSE_NAME)
    if interacting_terms_list:
        stayed_col_names.extend(interacting_terms_list)
        full_df = get_df_with_interaction_terms(full_df, interacting_terms_list, will_drop_single_interacting_term)
        
    return full_df[stayed_col_names]

class Regression():
    
    def __init__(self, dataframes: list[pd.DataFrame], training_features: list, regression_type: str, *, 
                 train_size: float = .8, test_size: float = .2, **hyperparams):
        
        assert train_size + test_size == 1, print("Train Size + Test Size != 1")
        assert all()
        
        from sklearn import linear_model
        from xgboost import XGBRegressor
        
        self.availableRegressionName_func_map = {
            'OLS': linear_model.LinearRegression(),
            'LASSO': linear_model.LassoCV(),
            'XGBOOST': XGBRegressor() 
        }
        self.model = None
        self.regression_type = regression_type_input
        if self.regression_type is not None: self.model = self.availableRegressionName_func_map[self.regression_type]
        
        # Training variables
        from sklearn.model_selection import train_test_split
        self.train_X, self.test_X, self.train_y, test_y = train_test_split(..., test_size=test_size, train_size=train_size)
        # \Training variables
        
        # Prediction variables
        self.predicted_y = None
        self.actual_y = test_y
        # \Prediction variables
        
        # Hyperparameters
        self.hyperparams = hyperparams
        # \Hyperparameters
        
        if not self.model: print(f"{regression_type} does not exist. Model will be None.")
        else: print(f"Currently using {self.regression_type} Regression Model.\nThe model has following hyper-params: {vars(self.model)}")
        return

    @property
    def regression_type(self) -> None:
        return self._regression_type

    @regression_type.setter
    def regression_type(self, regression_type) -> None:
        if regression_type in self.availableRegressionName_func_map: self._regression_type = regression_type
        elif regression_type.lower() in self.availableRegressionName_func_map: self._regression_type = regression_type.lower()
        elif regression_type.upper() in self.availableRegressionName_func_map: self._regression_type = regression_type.upper()
        elif regression_type.capitalize() in self.availableRegressionName_func_map: self._regression_type = regression_type.capitalize()
        else: self._regression_type = None
        return
    
    @staticmethod
    def list_all_regression_types():
        available_regressions = Regression().availableRegressionName_func_map.keys()
        for i, regression_type in enumerate(available_regressions):
            print(f"{i+1}: {regression_type}")
        
    def _train(self, train_X: Optional[pd.DataFrame] = None, train_y = None) -> None:
        assert any([train_X, self.train_X]), print("Need to feed a train_X.")
        assert any([train_y, self.train_y]), print("Need to feed a train_y.")
        
        input_train_X = train_X if train_X else self.train_X
        input_train_y = train_y if train_y else self.train_y
        
        model.fit(X = input_train_X, y = input_train_y)
        return
    
    def _predict(self, X: Optional[pd.DataFrame] = None):
        assert any([X, self.test_X]), print("Need to feed a test_X")
        input_X = X if X else self.test_X
        predicted_y = self.model.predict(input_X)
        
        self.predicted_y = predicted_y
        return predicted_y
    
    #TODO: Implement get_metric() and helper functions.