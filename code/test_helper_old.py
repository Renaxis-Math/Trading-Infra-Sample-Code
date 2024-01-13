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
                 train_size: float = .8, test_size: float = .2, hyperparams):
        
        assert train_size + test_size == 1, print("Train Size + Test Size != 1")
        
        from sklearn import linear_model
        from xgboost import XGBRegressor
        
        self.availableRegressionName_func_map = {
            'OLS': linear_model.LinearRegression(),
            'LASSO': linear_model.LassoCV(),
            'XGBOOST': XGBRegressor() 
        }
        
        self.hyperparams = hyperparams
        self.regression_type = regression_type
        
        # Train Test Split
        from sklearn.model_selection import train_test_split
        self.train_X_list, self.test_X_list, self.train_y_list, self.actual_y_list = _split(dataframes)
        # \Train Test Split
        
        if not self.regression_type: print(f"{regression_type} does not exist. Model will be None.")
        else: 
            print(f"Currently using {self.regression_type} Regression Model.")
            print(f"The model has following hyper-params: {vars(self.availableRegressionName_func_map[self.regression_type])}")
        
        self.saved_models = []
        return

    @property
    def regression_type(self):
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
        
    def _split(dataframes: list[pd.DataFrame]) -> None:
        train_X_list, test_X_list, train_y_list, test_y_list = [], [], [], []
        
        for i, dataframe in enumerate(dataframes):
            assert set(training_features) <= set(dataframe.columns), print(f"Some input features not exist in the {i + 1} DataFrame")
            
            df = dataframe[[training_features]]
            y = df[consts.RESPONSE_NAME]
            X = df.drop(consts.RESPONSE_NAME, axis = consts.COL, inplace = True)
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=42)
            
            train_X_list.append(train_X)
            test_X_list.append(test_X)
            train_y_list.append(train_y)
            test_y_list.append(test_y)
        
        return (train_X_list, test_X_list, train_y_list, test_y_list)     
    
    def _train(self, *, will_save_model = True, train_X: Optional[pd.DataFrame] = None, train_y = None) -> None:
        assert any([train_X is not None, len(self.train_X_list) > 0]), print("Need to feed a train_X.")
        assert any([train_y is not None, len(self.train_y) > 0]), print("Need to feed a train_y.")
        
        train_X_list = [train_X] if train_X else self.train_X_list
        train_y_list = [train_y] if train_y else self.train_y_list
        
        assert len(train_X_list) == len(train_y_list), print("len(train_X_list) != len(train_y_list)")
        for i in range(len(train_X_list)):
            train_X, train_y = train_X_list[i], train_y_list[i]
            model.fit(X = train_X, y = train_y)
            if will_save_model: self.saved_models.append(model)

        return
    
    def _predict(self, X_list: pd.DataFrame):
        for model in self.saved_models:
            predicted_y = model.predict(input_X)
        
        self.predicted_y = predicted_y
        return predicted_y
    
    #TODO: Implement get_metric() and helper functions.
    def _get_response_corrs(predicted_y, actual_y):
        return np.corrcoef(predicted_y, actual_y)[0, 1]

    def _get_mean_returns(predicted_y, actual_y):
        return np.sum((np.abs(actual_y) / len(predicted_y)) * \
                    (np.sign(actual_y) * np.sign(predicted_y)))

    def _get_scale_factors(predicted_y, actual_y):
        from sklearn import linear_model
        
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(X=pd.DataFrame({"predicted_y": predicted_y}), y=actual_y)
        return model.coef_

    def get_metric(dataframes: list[pd.DataFrame]) -> None:
        """Print metrics defind by Scott

        Returns: [weighted_corr, mean_return, scale_factor]
        """
        
        
        
        assert len(self.predicted_y_list) == len(actual_y_list), print(f"length(predicted_y_list) = {len(predicted_y_list)}, length(actual_y_list) = {len(actual_y_list)}")
        
        weighted_corrs, weighted_mean_returns, weighted_scale_factors = [], [], []
        for i in range(len(actual_y_list)):
            predicted_y, actual_y = predicted_y_list[i], actual_y_list[i]
            
            weighted_corrs.append(get_response_corrs(predicted_y, actual_y))
            weighted_mean_returns.append(get_mean_returns(predicted_y, actual_y))
            weighted_scale_factors.append(get_scale_factors(predicted_y, actual_y))
        
        print(f"1. Weighted Correlation: {np.mean(weighted_corrs)}")
        print(f"2. Weighted Mean Return: {np.mean(weighted_mean_returns)}")
        print(f"3. Weighted Scale Factor: {np.mean(weighted_scale_factors)}")
        
        return    