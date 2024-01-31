import pandas as pd
import numpy as np
import consts
import importlib
import os, re, typing
from typing import Optional
from datetime import datetime, timedelta

# Generalized Methods
def binary_search(sorted_items: list, target, elimination_func):
    """_summary_

    Args:
        sorted_items (list): _description_
        target (_type_): _description_
        elimination_func (_type_): _description_

    Returns:
        _type_: _description_
    """
    if sorted_items is None: return -1
    
    left_i ,right_i = 0, len(sorted_items) - 1 
    
    while left_i < right_i:
        mid_i = left_i + (right_i - left_i) // 2

        if elimination_func(sorted_items[mid_i], target): left_i = mid_i + 1
        else: right_i = mid_i
        
    return left_i

def reverse_binary_search(sorted_items: list, target, elimination_func):
    """_summary_

    Args:
        sorted_items (list): _description_
        target (_type_): _description_
        elimination_func (_type_): _description_

    Returns:
        _type_: _description_
    """
    if sorted_items is None: return -1

    left_i ,right_i = 0, len(sorted_items) - 1  
    
    while left_i < right_i:
        mid_i = left_i + (right_i - left_i) // 2 + 1
        if elimination_func(sorted_items[mid_i], target): right_i = mid_i - 1 
        else: left_i = mid_i
        
    return right_i
# \Generalized Methods

class Data:
    # Data Constructor called with a string data path. 
    # Stores a sorted list of filenames and datetimes
    # Data constructor also has a train_df and list of test_dfs. 
    def __init__(self, data_path: str): 
        """
        Args: 
        data_path (str): The path for the data. 

        Updates: 
        Stores sorted list of filenames and datetimes. Stores train_df and list of test_dfs. 
        """
        self.sorted_file_names = self._init_sorted_file_names(data_path)
        self.sorted_file_datetimes = self.__init__sorted_file_datetimes()
        self.train_df = None
        self.test_dfs = []
        return
    
    def _init_sorted_file_names(self, data_path: Optional[str]) -> list[str]:
        import os
        if data_path is None: return

        try:
            file_names = os.listdir(data_path)
            data_file_names = list(filter(lambda file_name: consts.DATA_FILTER_KEYWORD in file_name, file_names))
            data_file_names.sort(reverse = False)
            return data_file_names
        except FileNotFoundError: print(f"The directory {data_path} does not exist.")
    
    def __init__sorted_file_datetimes(self) -> list[datetime]:
        answers = []
        
        for file_name in self.sorted_file_names:
            file_datetime = self._extract_datetime(file_name)
            if file_datetime is not None: answers.append(file_datetime)
        
        return answers
    
    # Helper Functions
    def _extract_datetime(self, file_name: str) -> Optional[datetime]:
        """_summary_

        Args:
            file_name (str): _description_

        Returns:
            Optional[datetime]: _description_
        """
        import re
        match = re.search(r'\d{8}', file_name)
        
        if match: return datetime.strptime(match.group(), r'%Y%m%d')
        
        print("No YYYYMMDD datetime matched.\n")
        return None   
    
    def _is_leftDate_smallerThan_rightDate(self, left_date: datetime, right_date: datetime) -> bool:
        return left_date < right_date

    def _is_rightDate_smallerThan_leftDate(self, left_date: datetime, right_date: datetime) -> bool:
        return right_date < left_date   
        
    def _filter_file_names(self, *, start_date: datetime | str, end_date: datetime | str) -> list:
        """_summary_

        Args:
            start_date (datetime | str): start date in yyyymmdd form or datetime object
            end_date (datetime | str): end date in yyyymmdd form or datetime object 

        Returns:
            list: Gets all the filenames between the start date and end date. 
        """
        if isinstance(start_date, str): start_date = datetime.strptime(start_date, r'%Y%m%d')
        if isinstance(end_date, str): end_date = datetime.strptime(end_date, r'%Y%m%d')
        
        leftBound_i = binary_search(self.sorted_file_datetimes, start_date, 
                                    self._is_leftDate_smallerThan_rightDate)
        rightBound_i = reverse_binary_search(self.sorted_file_datetimes, end_date, 
                                             self._is_rightDate_smallerThan_leftDate)

        if any([leftBound_i == -1, rightBound_i == -1]): 
            print(f"Filtered File Dates: []")
            return []
        
        # Debugging
        print(f"Filtered File Dates: {self.sorted_file_names[leftBound_i : rightBound_i + 1]}\n")
        #\ Debugging
        return self.sorted_file_names[leftBound_i : rightBound_i + 1]
    # \Helper Functions
    
    # APIs
    def get_df_between_date(self, *, # * means that after this every argument must be specified with arg name
                            data_path: str,
                            start_date: datetime | str,
                            end_date: datetime | str) -> list[pd.DataFrame]:
        """Filters the file names and gets a list of all data frames between the start and end dates
        Args: 
            data_path (str): Path for data
            start_date (datetime | str): Start date for time interval
            end_date (datetime | str): End date for time interval
        Returns:
            list[pd.DataFrame]: list of data frames between all the dates.
        """
        filtered_file_names = self._filter_file_names(start_date = start_date, end_date = end_date)
        dfs = [pd.read_csv(data_path + file_name) for file_name in filtered_file_names] #list of data drames
        
        self.test_dfs = [df for df in dfs] # store dfs as test data frames. 
        return dfs
    
    def update_and_get_train_df(self, data_path: str, test_start_yyyymmdd: str, *,
                                    movingBack_dayCount: int,
                                    years_count: int) -> pd.DataFrame:
        """
        Args: 
            data_path (str): Path to all the data
            test_start_yyyymmdd (str): Start of testing period
            movingBack_dayCount (int): Time between end of training and start of testing (Typically 30 days)
            years_count (int): Number of years used in training. 

        Returns:
            _type_: One training DF. This is a combined data frame of all the training data for the training period. 
        """
        test_start_date = datetime.strptime(test_start_yyyymmdd, r'%Y%m%d')
        train_end_date = test_start_date - timedelta(days = movingBack_dayCount)
        train_start_date = train_end_date - timedelta(days = consts.YEAR_DAY * years_count)
        
        filtered_file_names = self._filter_file_names(start_date = train_start_date, end_date = train_end_date)
        dfs = [pd.read_csv(data_path + file_name) for file_name in filtered_file_names]
        train_df = pd.concat(dfs, axis = consts.ROW)
        
        self.train_df = train_df.copy()
        return train_df
    # \APIs

class Regression(Data):
    def __init__(self, *, 
                 data_path: Optional[str] = None, 
                 regression_type: str = 'OLS', 
                 hyperparam_dict: Optional[dict] = None):

        if data_path is not None: super().__init__(data_path) # Data class is super class
        
        # Given a type of regression, returns the function to train. Defaults to OLS
        self.regressionName_func_map = {
            'OLS': self._sklearn_ols_regression,
            'LASSO': self._sklearn_LASSO_regression,
            'XGBOOST': self._xgboost_regression }
        #\  

        # Build a regression object of appropriate type and give correct hyper params. 
        self.regression_type = regression_type
        if self.regression_type is None: print(f"{regression_input} does not exist. {self._list_all_regression_types}.\n")
        else: print(f"You're using: {self.regression_type}.\n")
        self.saved_model = self.regressionName_func_map[self.regression_type](hyperparam_dict)
        #\
        
        # Initializing data members for predictions and interacting terms. 
        self.feature_col_names = []
        self.interacting_terms_list = []
        self.predicted_y_list = []
        self.actual_y_list = []
        #\
        
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

    # Scikit-learn Region
    # All functions make a version of the regressor given the hyperparam dictionaries. 
    def _sklearn_ols_regression(self, hyperparam_dict: Optional[dict] = None):
        from sklearn.linear_model import LinearRegression
        
        returning_model = None
        if hyperparam_dict is None: returning_model = LinearRegression()
        else: returning_model = LinearRegression(**hyperparam_dict) # ** allows for using keyword args. 
        
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
    # \Scikit-learn Region
    
    # Helper Functions
    def _list_all_regression_types(self) -> None:
        print(f"Available Regression Inputs: {self.regressionName_func_map.keys()}\n")
        return

    def _get_df_with_interaction_terms(self, df: pd.DataFrame, 
                                       interacting_terms_list: list[list[str]],
                                       will_drop_single_interacting_term: bool = False) -> tuple[pd.DataFrame, list]:
        """Gets a new dataframe with interacting terms added. 

        Args:
            interacting_terms_list (list[list[str]]): List of two element terms that get multiplied. 
            will_drop_single_interacting_term (bool): Whether to drop the original column or not. 
        Returns:
            tuple[pd.DataFrame, list[str]]: [new dataframe with interacting terms, new df columns]
        """
        new_df = df.copy()
        new_col_names = []
        
        for interacting_terms in interacting_terms_list:
            all_terms_exist = np.all(np.isin(np.ravel(interacting_terms), df.columns))
            
            if all_terms_exist:
                new_col_name = str(tuple(interacting_terms))
                new_col_names.append(new_col_name)
                
                new_df[new_col_name] = np.prod(new_df[interacting_terms], axis=consts.COL)
                if will_drop_single_interacting_term: new_df.drop(interacting_terms, axis = consts.COL, inplace=True)
            else:
                print(f"{interacting_terms} missing!")
                return df
        
        return new_df, new_col_names

    def _predict(self, dataframes: list[pd.DataFrame] | pd.DataFrame, *, 
                 hyperparam_dict: Optional[dict] = None) -> list:
        """
        Predicts with either a list of dfs or one df for x values. 
        Uses saved_model to predict

        Args:
            dataframes (list[pd.DataFrame] | pd.DataFrame): _description_
            hyperparam_dict (Optional[dict], optional): _description_. Defaults to None.

        Returns:
            list: List of y value predictions given by the model. Does not save predictions. 
        """
        assert self.saved_model is not None, print("No model being trained yet!\n")
        predicted_y_list = []

        # If there is a single data frame. Predict based on the single data frame given hyperparams
        # Add to the prediced_y_list
        if isinstance(dataframes, pd.DataFrame):
            assert len(self.feature_col_names) == len(dataframes.columns)
            
            if hyperparam_dict is None: predicted_y = self.saved_model.predict(dataframes)
            else: predicted_y = self.saved_model.predict(dataframes, **hyperparam_dict)
            
            predicted_y_list.append(predicted_y)
        #\
        
        # Iterate over *all* the data frames. Add to predicted_y_list
        else:
            for i in range(len(dataframes)):
                dataframe = dataframes[i]
                assert len(self.feature_col_names) == len(dataframe.columns)

                if hyperparam_dict is None: predicted_y = self.saved_model.predict(dataframe)
                else: predicted_y = self.saved_model.predict(dataframe, **hyperparam_dict)
                
                predicted_y_list.append(predicted_y)
        #\
        
        return predicted_y_list
    # Helper Functions
    
    # Metric Functions (defined by scott)
    def _get_response_corr(self, predicted_y, actual_y) -> float:
        return np.corrcoef(predicted_y, actual_y)[0, 1]

    def _get_mean_return(self, predicted_y, actual_y) -> float:
        return np.mean(np.abs(actual_y) * (np.sign(actual_y) * np.sign(predicted_y)))

    def _get_scale_factor(self, predicted_y, actual_y):
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X=pd.DataFrame({"predicted_y": predicted_y}), y=actual_y)
        return model.coef_    
    # \Metric Functions

    # APIs
    def train(self, dataframe: Optional[pd.DataFrame] = None, *,
                    feature_col_names: list[str] = [],
                    interacting_terms_list: list[list[str]] = [],
                    hyperparam_dict: Optional[dict] = None) -> None:
        """
        Trains the model.

        Args:
        dataframe (pd.Dataframe): Training Data frame. 
        feature_col_names (list[str]): 
        interacting_terms_list (list[list[str]]):
        hyperparam_dict (Optional[dict]): 

        Raises:
            Exception: Thrown where there is no self.train_df and no dataframe given. 
        """
        copied_dataframe = dataframe.copy()
        if copied_dataframe is None and self.train_df is None: raise Exception("Can't train when nothing is given.\n")
        
        # Establish Training data is there there is no train_df in the regressor and update with interacting terms. 
        training_df = self.train_df if (self.train_df is not None) else copied_dataframe
        training_df, new_col_names = self._get_df_with_interaction_terms(training_df, interacting_terms_list)
        #\

        # Get training y data and drop it from training data
        train_y = training_df[consts.RESPONSE_NAME]
        if consts.RESPONSE_NAME in set(training_df.columns):
            train_X = training_df.drop(consts.RESPONSE_NAME, axis=consts.COL, inplace=False)
        #\        
        
        # Get a list of training features with interacting terms. 
        # If there are none training features becomes the x_columns. 
        training_features = []
        training_features.extend(new_col_names)
        
        if len(feature_col_names) > 0: training_features.extend(feature_col_names)
        else: training_features.extend(train_X.columns)
        train_X = training_df[training_features]
        #\
        
        # Train the model using the train_x, train_y and hyperparams if necessary. 
        if hyperparam_dict is None: 
            self.saved_model.fit(train_X, train_y)
        else: 
            self.saved_model.fit(train_X, train_y) #does not take hyperparam dict
        #\
        
        # Save feature column names and training features. 
        self.interacting_terms_list = interacting_terms_list
        self.feature_col_names = training_features
        #\
        
        print(f"Features being used: {self.feature_col_names}")
        return
        
    def get_metric(self, dataframes: Optional[list[pd.DataFrame] | pd.DataFrame] = None, *, 
                        hyperparam_dict: Optional[dict] = None, printMetrics = True) -> (float, float, float):
        """Get Scotts Metrics and print them out. 

        Args:
            dataframes (Optional[list[pd.DataFrame]  |  pd.DataFrame], optional):Test df or list of test dfs. Defaults to None.
            hyperparam_dict (Optional[dict], optional): _description_. Defaults to None.

        Raises:
            Exception: When there is no dataframe given or test_dfs saved. 

        Returns:
            None
        """
        if dataframes is None and self.test_dfs == []: raise Exception("Can't test when nothing is given.\n")
        input_dfs = self.test_dfs if len(self.test_dfs) > 0 else dataframes
        
        def _get_test_X(input_df: pd.DataFrame) -> pd.DataFrame:
            if consts.RESPONSE_NAME in set(input_df.columns): 
                input_df.drop(consts.RESPONSE_NAME, axis=consts.COL, inplace=True)
                
            test_X, _ = self._get_df_with_interaction_terms(input_df, self.interacting_terms_list)
            if len(self.feature_col_names) > 0: test_X = test_X[self.feature_col_names]
            
            return test_X         
        
        if isinstance(dataframes, pd.DataFrame):
            test_y = dataframes[consts.RESPONSE_NAME]
            test_X = _get_test_X(dataframes.copy())
            
            self.actual_y_list = [test_y]
            self.predicted_y_list = self._predict(test_X, hyperparam_dict = hyperparam_dict)
        else:
            input_dfs = [dataframe.copy() for dataframe in dataframes]
            actual_y_list = [input_df[consts.RESPONSE_NAME] for input_df in input_dfs]
            for i, input_df in enumerate(input_dfs):
                test_X = _get_test_X(input_df)
                input_dfs[i] = test_X

            self.actual_y_list = actual_y_list
            self.predicted_y_list = self._predict(input_dfs, hyperparam_dict = hyperparam_dict)

        assert len(self.predicted_y_list) == len(self.actual_y_list), \
        print(f"len(predicted_y_list) != len(actual_y_list)\n")
        
        # Link the actual and predicted y_s together and call the metric functions
        # Call for all dfs and average them. Use the * operator to unpack tuple. 
        predict_actual_pairs = list(zip(self.predicted_y_list, self.actual_y_list))
        response_corr = np.mean([self._get_response_corr(*predict_actual_pair) # * operator unpacks the tuple into 2 args
                                 for predict_actual_pair in predict_actual_pairs])
        mean_return = np.mean([self._get_mean_return(*predict_actual_pair)
                               for predict_actual_pair in predict_actual_pairs])
        scale_factor = np.mean([self._get_scale_factor(*predict_actual_pair)
                                for predict_actual_pair in predict_actual_pairs])
        #\
        if printMetrics:
            print(f"response_corr = {response_corr}")
            print(f"mean_return = {mean_return}")
            print(f"scale factor = {scale_factor}")
        
        return response_corr, mean_return, scale_factor
    # \APIs

    # Experiment Functions: 
    # TODO: Do something with the betas, toggle output. 
def lasso_exp(data_path, *,
                train_df: pd.DataFrame, 
                test_dfs: [pd.DataFrame],  
                feature_col_names: list[str] = [],
                interacting_terms_list: list[list[str]] = [],
                hyperparam_dict: Optional[dict] = None, 
                num_trials:int = 30) -> None:
    response_corrs, mean_returns,scale_factors = [],[], []
    print(f"Conducting {num_trials} trials on a Lasso model with hyperparams:\n{hyperparam_dict}")
    #Use trial number for seed
    if hyperparam_dict == None: 
        hyperparam_dict = {'random_state': 0}
    elif 'random_state' not in hyperparam_dict:
            hyperparam_dict['random_state'] = 0
    for trial in range(num_trials):
        lasso_model = Regression(regression_type='Lasso', data_path=data_path)
        hyperparam_dict['random_state'] = trial
        lasso_model.train(train_df, 
                            interacting_terms_list= interacting_terms_list,
                        feature_col_names= feature_col_names,
                        hyperparam_dict=hyperparam_dict,
                            )
        rc,mr,sf = lasso_model.get_metric(dataframes=test_dfs, printMetrics=False)
        response_corrs.append(rc)
        mean_returns.append(mr)
        scale_factors.append(sf)

    # Print Average Metrics
    print(f"Average Metrics across {num_trials}:")
    print(f"response_corr = {np.average(response_corrs)}")
    print(f"mean_return = {np.average(mean_returns)}")
    print(f"scale factor = {np.average(scale_factors)}")
    return
        
            