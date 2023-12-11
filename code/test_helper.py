import pandas as pd
import numpy as np
import consts
import importlib
import os, re, typing
from typing import Optional

class Regression:
    
    def __init__(self, *, regression_input: str, hyperparam_dict: Optional[dict]):
        assert train_size + test_size == 1, print("Train Size + Test Size != 1")
        
        self.regressionName_func_map = {
            'OLS': self._sklearn_ols_regression,
            'LASSO': self._sklearn_LASSO_regression,
            'XGBOOST': self._xgboost_regression }          
                
        self.regression_type = regression_input
        if self.regression_type is None: 
            print(f"Available Regression Inputs: {regressionName_func_map.keys()}\n")
            raise Exception(f"{regression_input} does not exist. Please re-initialize.\n")
               
        self.predicted_y_list = []
        self.actual_y_list = []
        self.saved_models = []
        self.init_hyperparam = hyperparam_dict
        
    @property
    def regression_type(self) -> str:
        return self._regression_type
    @regression_type.setter
    def regression_type(self, regression_input: str) -> None:
        if regression_input in self.regressionName_func_map: self._regression_type = regression_input
        elif regression_input.lower() in self.regressionName_func_map: self._regression_type = regression_input.lower()
        elif regression_input.upper() in self.regressionName_func_map: self._regression_type = regression_input.upper()
        elif regression_input.capitalize() in self.regressionName_func_map: self._regression_type = regression_input.capitalize()
        else: self._regression_type = None
        return

   
    def _sklearn_ols_regression(self, hyperparam_dict: Optional[dict]):
        from sklearn.linear_model import LinearRegression
        if hyperparam_dict is None: return LinearRegression()
        return LinearRegression(**hyperparam_dict)
    def _sklearn_LASSO_regression(self, hyperparam_dict: Optional[dict]):
        from sklearn.linear_model import LassoCV
        if hyperparam_dict is None: return LassoCV()
        return LassoCV(**hyperparam_dict)    
    def _xgboost_regression(self, hyperparam_dict: Optional[dict]):
        from xgboost import XGBRegressor 
        if hyperparam_dict is None: return XGBRegressor()
        return XGBRegressor(**hyperparam_dict)

    def train(self, *, dataframes: Optional[list[pd.DataFrame] | pd.DataFrame],
              feature_col_names = list[str], hyperparam_dict: Optional[dict]):
        
        if dataframes is None: raise Exception("Can't train when nothing is given.\n")
        
        elif isinstance(dataframes, pd.DataFrame):
            train_y = dataframes[consts.RESPONSE_NAME]
            train_X = dataframes[[feature_col_names]]
            
            model = self.regressionName_func_map[self.regression_type](self.init_hyperparam)
            
            if hyperparam_dict is None: model.fit(train_X, train_y)
            else: model.fit(train_X, train_y, **hyperparam_dict)
            self.saved_models.append(model)
            
        else:
            for i, dataframe in enumerate(dataframes):
                train_y = dataframes[consts.RESPONSE_NAME]
                train_X = dataframes[[feature_col_names]]
                
                model = self.regressionName_func_map[self.regression_type](self.init_hyperparam)
                
                if hyperparam_dict is None: model.fit(train_X, train_y)
                else: model.fit(train_X, train_y, **hyperparam_dict)     
                self.saved_models.append(model)
    
    def _predict(self, dataframes: list[pd.DataFrame] | pd.DataFrame) -> list:
        assert len(dataframes) == len(saved_models), print("len(dataframes) != len(saved_models).\n")
        
        predicted_y_list = []

        if isinstance(dataframes, pd.DataFrame):
            if hyperparam_dict is None: predicted_y = self.saved_models[0].predict(dataframes)
            else: predicted_y = self.saved_models[0].predict(dataframes, **hyperparam_dict)
            
            predicted_y_list.append(predicted_y)
        else:
            for i in range(len(self.saved_models)):
                model, dataframe = self.saved_models[i], dataframes[i]

                if hyperparam_dict is None: predicted_y = self.saved_models[i].predict(dataframes)
                else: predicted_y = self.saved_models[i].predict(dataframes, **hyperparam_dict)
                
                predicted_y_list.append(predicted_y)
        
        return predicted_y_list
    
    def _get_response_corr(predicted_y, actual_y):
        return np.corrcoef(predicted_y, actual_y)[0, 1]

    def _get_mean_return(predicted_y, actual_y):
        return np.sum((np.abs(actual_y) / len(predicted_y)) * \
                    (np.sign(actual_y) * np.sign(predicted_y)))

    def _get_scale_factor(predicted_y, actual_y):
        from sklearn import linear_model
        
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(X=pd.DataFrame({"predicted_y": predicted_y}), y=actual_y)
        return model.coef_    
    
    def get_metric(self, dataframes: list[pd.DataFrame] | pd.DataFrame) -> None:
        predicted_y_list = self._predict(dataframes)
        actual_y_list = []
        if isinstance(dataframes, pd.DataFrame): actual_y_list = [dataframes[consts.RESPONSE_NAME]]
        else: actual_y_list = [dataframe[consts.RESPONSE_NAME] for dataframe in dataframes]
        
        assert len(predicted_y_list) == len(actual_y_list), print(f"len(predicted_y_list) != len(actual_y_list)\n")
        
        response_corr = np.mean([_get_response_corr(predicted_y_list[i], actual_y_list[i]) for i in range(len(predicted_y_list))])
        mean_return = np.mean([_get_mean_return(predicted_y_list[i], actual_y_list[i]) for i in range(len(predicted_y_list))])
        scale_factor = np.mean([_get_scale_factor(predicted_y_list[i], actual_y_list[i]) for i in range(len(predicted_y_list))])
        
        print(f"response_corr = {response_corr}")
        print(f"mean_return = {mean_return}")
        print(f"scale factor = {scale_factor}")
        
        return