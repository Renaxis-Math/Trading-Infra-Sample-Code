import pandas as pd
import numpy as np
import consts
import importlib
import os, re, typing
from typing import Optional
from datetime import datetime, timedelta

def binary_search(sorted_items: list, target, elimination_func):
    left_i ,right_i = 0, len(sorted_items) - 1 
    
    while left_i < right_i:
        mid_i = left_i + (right_i - left_i) // 2

        if elimination_func(sorted_items[mid_i], target): left_i = mid_i + 1
        else: right_i = mid_i
        
    return left_i

def reverse_binary_search(sorted_items: list, target, elimination_func):
    left_i ,right_i = 0, len(sorted_items) - 1  
    
    while left_i < right_i:
        mid_i = left_i + (right_i - left_i) // 2 + 1
        if elimination_func(sorted_items[mid_i], target): right_i = mid_i - 1 
        else: left_i = mid_i
        
    return right_i

class Data:
    def __init__(self, data_path: str):
        self.sorted_file_names = self.init_sorted_file_names(data_path)
        self.sorted_file_datetimes = [self._extract_datetime(file_name) \
                                      for file_name in self.sorted_file_names]

        self.train_df = None
        self.test_dfs = []
        
    def init_sorted_file_names(self, data_path: Optional[str]) -> list[str]:
        import os
        if data_path is None: return

        try:
            file_names = os.listdir(data_path)
            data_file_names = list(filter(lambda file_name: consts.DATA_FILTER_KEYWORD in file_name, file_names))
            data_file_names.sort(reverse = False)
            return data_file_names
        except FileNotFoundError: print(f"The directory {data_path} does not exist.")
    
    def _extract_datetime(self, file_name) -> Optional[datetime]:
        import re
        match = re.search(r'\d{8}', file_name)
        
        if match: return datetime.strptime(match.group(), r'%Y%m%d')
        return None   
    
    def _is_leftDate_smallerThan_rightDate(self, left_date: datetime, right_date: datetime) -> bool:
        return left_date < right_date

    def _is_rightDate_smallerThan_leftDate(self, left_date: datetime, right_date: datetime) -> bool:
        return right_date < left_date   
        
    def _filter_file_names(self, start_date: datetime | str, end_date: datetime | str) -> list:
        if isinstance(start_date, str): start_date = datetime.strptime(start_date, r'%Y%m%d')
        if isinstance(end_date, str): end_date = datetime.strptime(end_date, r'%Y%m%d')
        
        leftBound_i = binary_search(self.sorted_file_datetimes, start_date, 
                                    self._is_leftDate_smallerThan_rightDate)
        rightBound_i = reverse_binary_search(self.sorted_file_datetimes, end_date, 
                                             self._is_rightDate_smallerThan_leftDate)

        # Debugging
        print(f"Filtered File Dates: {self.sorted_file_names[leftBound_i : rightBound_i + 1]}\n")
        #\ Debugging
        return self.sorted_file_names[leftBound_i : rightBound_i + 1]
    
    def get_df_between_date(self, *, data_path: str,
                            start_date: datetime | str, end_date: datetime | str) -> list[pd.DataFrame]:
    
        filtered_file_names = self._filter_file_names(start_date, end_date)
        dfs = [pd.read_csv(data_path + file_name) for file_name in filtered_file_names]
        
        self.test_dfs = list(dfs)
        return dfs
    
    def update_and_get_train_df(self, data_path: str, test_start_yyyymmdd: str, 
                                                    movingBack_dayCount: int) -> pd.DataFrame:

        test_start_date = datetime.strptime(test_start_yyyymmdd, r'%Y%m%d')
        train_end_date = test_start_date - timedelta(days = movingBack_dayCount)
        train_start_date = train_end_date - timedelta(days = consts.YEAR_DAY)
        
        filtered_file_names = self._filter_file_names(train_start_date, train_end_date)
        dfs = [pd.read_csv(data_path + file_name) for file_name in filtered_file_names]
        train_df = pd.concat(dfs, axis = consts.ROW)
        
        self.train_df = train_df.copy()
        return train_df

class Regression(Data):
    
    def __init__(self, *, data_path: str, regression_type: str = 'OLS', hyperparam_dict: Optional[dict] = None):
        super().__init__(data_path)
        
        self.regressionName_func_map = {
            'OLS': self._sklearn_ols_regression,
            'LASSO': self._sklearn_LASSO_regression,
            'XGBOOST': self._xgboost_regression }          
                
        self.regression_type = regression_type
        if self.regression_type is None:
            print(f"Available Regression Inputs: {regressionName_func_map.keys()}\n")
            raise Exception(f"{regression_input} does not exist. Please re-initialize.\n")
        else: 
            print(f"You're using: {self.regression_type}")
        
        self.feature_col_names = []
        self.interacting_terms_list = []
        
        self.saved_model = self.regressionName_func_map[self.regression_type](hyperparam_dict)
        self.predicted_y_list = []
        self.actual_y_list = []
        
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

    def _sklearn_ols_regression(self, hyperparam_dict: Optional[dict] = None):
        from sklearn.linear_model import LinearRegression
        
        returning_model = None
        if hyperparam_dict is None: returning_model = LinearRegression()
        else: returning_model = LinearRegression(**hyperparam_dict)
        
        print(f"Available hyperparams: {vars(returning_model)}")
        return returning_model

    def _sklearn_LASSO_regression(self, hyperparam_dict: Optional[dict] = None):
        from sklearn.linear_model import LassoCV
        
        returning_model = None
        if hyperparam_dict is None: returning_model = LassoCV()
        else: returning_model = LassoCV(**hyperparam_dict)
        
        print(f"Available hyperparams: {vars(returning_model)}")
        return returning_model

    def _xgboost_regression(self, hyperparam_dict: Optional[dict] = None):
        from xgboost import XGBRegressor 
        
        returning_model = None
        if hyperparam_dict is None: returning_model = XGBRegressor()
        else: returning_model = XGBRegressor(**hyperparam_dict)
        
        print(f"Available hyperparams: {vars(returning_model)}")
        return returning_model

    def _get_df_with_interaction_terms(self, df: pd.DataFrame, interacting_terms_list: list,
                                       will_drop_single_interacting_term: bool = False) -> pd.DataFrame:

        """Return a new DataFrame that has interacting column pairs

        Args:
            df (DataFrame): original training data
            interacting_terms_list (list of list): list of column pairs

        Returns:
            DataFrame: resulting DataFrame
        """
        new_df = df.copy()
        new_col_names = []
        for interacting_terms in interacting_terms_list:
            all_terms_exist = np.all(np.isin(np.ravel(interacting_terms), df.columns))
            if all_terms_exist:
                new_col_name = str(tuple(interacting_terms))
                new_col_names.append(new_col_name)
                
                new_df[new_col_name] = np.prod(new_df[interacting_terms], axis=consts.COL)
                if will_drop_single_interacting_term: 
                    new_df.drop(interacting_terms, axis = consts.COL, inplace=True)
            else:
                print(f"{interacting_terms} missing!")
                return df
        
        return new_df, new_col_names

    def _predict(self, *, dataframes: list[pd.DataFrame] | pd.DataFrame, 
                hyperparam_dict: Optional[dict] = None) -> list:

        assert self.saved_model is not None, print("No model being trained yet!\n")
        
        predicted_y_list = []

        if isinstance(dataframes, pd.DataFrame):
            if hyperparam_dict is None: predicted_y = self.saved_model.predict(dataframes)
            else: predicted_y = self.saved_model.predict(dataframes, **hyperparam_dict)
            
            predicted_y_list.append(predicted_y)
        else:
            for i in range(len(dataframes)):
                dataframe = dataframes[i]

                if hyperparam_dict is None: predicted_y = self.saved_model.predict(dataframe)
                else: predicted_y = self.saved_model.predict(dataframe, **hyperparam_dict)
                
                predicted_y_list.append(predicted_y)
        
        return predicted_y_list
    
    def _get_response_corr(self, predicted_y, actual_y):
        return np.corrcoef(predicted_y, actual_y)[0, 1]

    def _get_mean_return(self, predicted_y, actual_y):
        return np.mean(np.abs(actual_y) * (np.sign(actual_y) * np.sign(predicted_y)))

    def _get_scale_factor(self, predicted_y, actual_y):
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X=pd.DataFrame({"predicted_y": predicted_y}), y=actual_y)
        return model.coef_    
    
    def train(self, feature_col_names: Optional[list[str]] = None, interacting_terms_list: Optional[list] = None, 
              *, dataframe: Optional[pd.DataFrame] = None, hyperparam_dict: Optional[dict] = None) -> None:

        """Note: this method prefer self.train_df over input 'dataframe'.
        """
        copied_dataframe = dataframe.copy()
        if copied_dataframe is None and self.train_df is None: raise Exception("Can't train when nothing is given.\n")
        training_features = []
        if len(feature_col_names) > 0: training_features.extend(feature_col_names)
        
        training_df = self.train_df if self.train_df is not None else copied_dataframe
        if interacting_terms_list is not None: 
            training_df, new_col_names = self._get_df_with_interaction_terms(training_df, interacting_terms_list)
            training_features.extend(new_col_names)

        train_y = training_df[consts.RESPONSE_NAME]
        if len(training_features) > 0: train_X = training_df[training_features]
        else: train_X = training_df.drop(consts.RESPONSE_NAME, inplace = False, axis = consts.COL)
        
        if hyperparam_dict is None: self.saved_model.fit(train_X, train_y)
        else: self.saved_model.fit(train_X, train_y, **hyperparam_dict)
        
        self.interacting_terms_list = interacting_terms_list
        self.feature_col_names = training_features
        
        print(f"Features being used: {self.feature_col_names}")
        return
        
    def get_metric(self, *, dataframes: Optional[list[pd.DataFrame] | pd.DataFrame] = None,
                   hyperparam_dict: Optional[dict] = None) -> None:
        """Note: this method perfer self.test_dfs over input 'dataframes'

        Args:
            dataframes (list[pd.DataFrame] | pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        if dataframes is None and self.test_dfs == []: raise Exception("Can't test when nothing is given.\n")
        input_dfs = self.test_dfs if len(self.test_dfs) > 0 else dataframes
        
        if isinstance(dataframes, pd.DataFrame):
            input_dfs = dataframes.copy()
            self.actual_y_list = [input_dfs[consts.RESPONSE_NAME]]
            
            if len(self.interacting_terms_list) > 0:
                input_dfs, _ = self._get_df_with_interaction_terms(input_dfs, self.interacting_terms_list)
            
            testing_df = input_dfs[self.feature_col_names]
            self.predicted_y_list = self._predict(dataframes = testing_df, hyperparam_dict = hyperparam_dict)
        else:
            input_dfs = list(dataframes)
            self.actual_y_list = [input_df[consts.RESPONSE_NAME] for input_df in input_dfs]
            
            if len(self.interacting_terms_list) > 0:
                for i in range(len(input_dfs)):
                    input_df = input_dfs[i]
                    assert isinstance(input_df, pd.DataFrame), print(f"input_df {i} not a DataFrame.\n")
                    temp_input_df, _ = self._get_df_with_interaction_terms(input_df, self.interacting_terms_list)
    
                    test_df = temp_input_df[self.feature_col_names]
                    input_dfs[i] = test_df

            self.predicted_y_list = self._predict(dataframes = input_dfs, hyperparam_dict = hyperparam_dict)

        assert len(self.predicted_y_list) == len(self.actual_y_list), \
        print(f"len(predicted_y_list) != len(actual_y_list)\n")
        
        predict_actual_pairs = list(zip(self.predicted_y_list, self.actual_y_list))
        
        response_corr = np.mean([self._get_response_corr(*predict_actual_pair)
                                 for predict_actual_pair in predict_actual_pairs])
        mean_return = np.mean([self._get_mean_return(*predict_actual_pair)
                               for predict_actual_pair in predict_actual_pairs])
        scale_factor = np.mean([self._get_scale_factor(*predict_actual_pair)
                                for predict_actual_pair in predict_actual_pairs])
        
        print(f"response_corr = {response_corr}")
        print(f"mean_return = {mean_return}")
        print(f"scale factor = {scale_factor}")
        
        return
