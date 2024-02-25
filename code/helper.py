import pandas as pd
import numpy as np
import consts
import importlib
import os, re, typing
import statsmodels.api as sm
from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
import io, os, pickle

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Generalized Methods
def get_variances(df):
    if consts.RESPONSE_NAME in df.columns: 
        used_df = df.drop(consts.RESPONSE_NAME, inplace = False, axis = consts.COL)
    else: used_df = df
    
    variances = used_df.var()
    return variances

def get_correlations(df):
    if consts.RESPONSE_NAME in df.columns: 
        used_df = df.drop(consts.RESPONSE_NAME, inplace = False, axis = consts.COL)
    else: used_df = df
    
    corr_mat = np.array(used_df.corr())    
    correlations = np.array([corr_mat[r][c] for r in range(len(corr_mat)) for c in range(r)])
    return correlations[~np.isnan(correlations)]

def downloaded_all_data(output_dir: str, file_count: int) -> None:
    if "Credentials stuff":
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', consts.SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

    if "Access":
        service = build('drive', 'v3', credentials=creds)
        results = service.files().list(q="'{}' in parents".format(consts.FOLDER_ID), 
                                       pageSize=file_count, 
                                       fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

    if not items: print('No files found.')
    else:
        print("==> Downloading data from Google Drive...")
        for item in items:
            request = service.files().get_media(fileId=item['id'])
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            
            files = io.FileIO(os.path.join(output_dir, item['name']), 'wb')
            downloader = MediaIoBaseDownload(files, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
    
    print(f"Finish downloading {len(items)} files to ./{output_dir}")
    return

def binary_search(sorted_items: list, target, elimination_func):
    """
    Used in filter_file_name to perform binary search on a sorted list to find the index of a target element.

    Args:
        sorted_items (list): A sorted list of elements.
        target (_type_): The target element to be found in the list.
        elimination_func (_type_): A function that determines whether to eliminate elements during the search.

    Returns:
        int: The index of the target element if found; otherwise, returns -1.
    """
    if sorted_items is None: return -1
    
    left_i ,right_i = 0, len(sorted_items) - 1 
    
    while left_i < right_i:
        mid_i = left_i + (right_i - left_i) // 2

        if elimination_func(sorted_items[mid_i], target): left_i = mid_i + 1
        else: right_i = mid_i
        
    return left_i

def reverse_binary_search(sorted_items: list, target, elimination_func):
    """
    Perform binary search on a sorted list to find the index of a target element.

    Args:
        sorted_items (list): A sorted list of elements.
        target (_type_): The target element to be found in the list.
        elimination_func (_type_): A function that determines whether to eliminate elements during the search.

    Returns:
        int: The index of the target element if found; otherwise, returns -1.
    """
    if sorted_items is None: return -1

    left_i ,right_i = 0, len(sorted_items) - 1  
    
    while left_i < right_i:
        mid_i = left_i + (right_i - left_i) // 2 + 1
        if elimination_func(sorted_items[mid_i], target): right_i = mid_i - 1 
        else: left_i = mid_i
        
    return right_i

def extract_features_to_file(final_features: list[str], filepath: str):
    with open(filepath, 'w') as file:
        for i, feature in enumerate(final_features):
            if i < len(final_features) - 1: feature = feature + '\n'
            file.write(feature)
    file.close()
# \Generalized Methods



# Feature Selection Methods
def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out = 0.05, verbose=True):
    included = list(initial_list)

    def step(direction):
        nonlocal included
        changed = False
        if direction == 'forward':
            excluded = list(set(X.columns) - set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print('Add {:30} with p-value {:.6}'.format(best_feature, best_pval))
        elif direction == 'backward':
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        return changed

    while True:
        changed = step('forward') or step('backward')
        if not changed:
            break

    return included

def LASSO_feature_selection(train_df, features = None):
    df = train_df.copy()
    temp_model = helper.Model('LASSO')
    
    if features == None: temp_model.train(df)
    else: temp_model.train(df, feature_col_names = features)
    temp_model.test(test_df)

    coefficients = temp_model.inner.coef_
    mask = np.not_equal(coefficients, 0)

    filtered_features = np.array(temp_model.feature_col_names)[mask]
    return filtered_features
# \Feature Selection Methods



# Plot Methods
def scatter_lot(df: pd.DataFrame, col: str, rows_count: int = -1):
    plt.scatter(df.index[:rows_count], df[col][:rows_count])
    plt.axhline(0, color='r', linestyle='-')
    plt.title(f'Scatter Plot of {col}')
    plt.xlabel('Observation Count')
    plt.ylabel('Value')
    plt.show()
    
def histogram(data: list, num_bins = 30):    
    plt.hist(data, bins = num_bins, edgecolor='black')
    plt.title('Distribution of Correlation Coefficients')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.show()

def boxplot(data: list, x_axis_label: str):
    # Create a new figure with a specified size (width, height)
    plt.figure(figsize=(10, 6))

    plt.boxplot(data, vert=False)
    plt.title(f'Boxplot of {x_axis_label}')
    plt.xlabel(f'{x_axis_label}')
    plt.show()
    
def validation_plot(data: 'Data', model: 'Model', n_splits: int, 
                    test_start_yyyymmdd: str, backward_dayCount: int = 1, 
                    train_data_count: int = 1, years_count: int = 0, 
                    data_path: str = "", forward_dayCount: int = 0,
                    features: list[str] = []):
    
    from sklearn.model_selection import TimeSeriesSplit
    import random

    df = data.update_and_get_train_df(test_start_yyyymmdd, 
                                      backward_dayCount=backward_dayCount, 
                                      train_data_count=train_data_count, 
                                      years_count=years_count)

    test_dfs = data.update_and_get_test_df(data_path=data_path, 
                                           start_date=test_start_yyyymmdd, 
                                           forward_dayCount=forward_dayCount)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    metrics = {key: [] for key in model.metric.keys()}
    test_metrics = {key: [] for key in model.metric.keys()}

    for train_index, val_index in tscv.split(df):
        train_df, val_df = df.iloc[train_index], df.iloc[val_index]
        model.train(train_df, verbose = False, feature_col_names = features)

        model.test(val_df, verbose = False)
        for metric in metrics.keys():
            metrics[metric].append(model.metric_output[metric])

        test_df = random.choice(test_dfs)
        model.test(test_df, verbose = False)
        for metric in test_metrics.keys():
            test_metrics[metric].append(model.metric_output[metric])

    for metric in model.metric.keys():
        plt.figure(figsize=(10, 6))
        plt.plot(metrics[metric], \
                 label=f'Validation')
        plt.plot(test_metrics[metric], label='Test')
        
        plt.title(f'{metric} over time of {model.model_type} model with {len(features)} features')
        
        plt.xlabel('Time')
        plt.ylabel(metric)

        plt.legend()
        plt.show()
#\ Plot Methods



"""Data Class works with 1 training data and N testing data from a given directory.
"""
class Data:
    """
    Variables:
        sorted_file_names: a sorted list of all file names from the data path 
        sorted_file_datetimes: a sorted list of datetimes from the filenames
    """
    def __init__(self, *, train_data_path: Optional[str] = None, 
                 train_data: Optional[pd.DataFrame] = None,
                 test_data: list[pd.DataFrame] = []):

        self.data_path = train_data_path

        self.sorted_file_names = self._init_sorted_file_names(self.data_path)
        self.sorted_file_datetimes = self._init_sorted_file_datetimes()
        
        self.train_df = train_data
        self.test_dfs = test_data
        
        self.saved_column = defaultdict()
        return
    
    @property
    def data_path(self) -> str:
        return self._data_path
    @data_path.setter
    def data_path(self, train_data_path: Optional[str]) -> None:
        if train_data_path is not None:
            if len(train_data_path) == 0: train_data_path = "./"
            if train_data_path[-1] != "/": train_data_path += "/"
        
        self._data_path = train_data_path
        return
    
    @property
    def test_dfs(self) -> list[pd.DataFrame]:
        return self._test_dfs
    @test_dfs.setter
    def test_dfs(self, test_data) -> None:
        if not isinstance(test_data, list): test_data = [test_data]

        for i, data in enumerate(test_data):
            test_data[i] = self._removed_id(data, consts.ID)
        self._test_dfs = test_data
        return
    
    # APIs
    """Main API
    Extend this!
    """
    def update_and_get_train_df(self, test_start_yyyymmdd: str, *, concat = True,
                                # * is here to indict the start of key-word only arguments,
                                # Keyword-only arguments are parameters that must be passed using their names and cannot be specified positionally
                                    backward_dayCount: int = 1,
                                    train_data_count: int = 1,
                                    years_count: int = 0) -> pd.DataFrame:
        """
        get the training DataFrame using the test_start date and the days between trainning and testing data.

        Args:
            data_path (str): The path to the directory containing the data files.
            test_start_yyyymmdd (str): The start date of the test period in the format '%Y%m%d'.
            backward_dayCount (int): The number of days to move back from the test start date to determine the training end date.
            years_count (int): The number of years of data to include in the training set.

        Returns:
            pd.DataFrame: The training DataFrame.
        """
        if self.data_path is None: print("Please input a data path!"); return pd.DataFrame()

        test_start_date = datetime.strptime(test_start_yyyymmdd, r'%Y%m%d')
        train_end_date = test_start_date - timedelta(days = backward_dayCount)
        train_start_date = train_end_date - timedelta(days = consts.YEAR_DAY * years_count + train_data_count)
        
        filtered_file_names = self._filter_file_names(start_date = train_start_date, end_date = train_end_date)
        dfs = [self._removed_id(self._filtered_duplicatations(
                                pd.read_csv(self.data_path + file_name), 
                                consts.ROW), consts.ID) for file_name in filtered_file_names]
        if len(dfs) > 0: train_df = pd.concat(dfs, axis = consts.ROW)
        else:
            train_df = pd.DataFrame()
            print(f"File w/ end date {train_end_date} does not exist.")
            print(f"Please update 'backward_dayCount' or increase 'train_data_count' (currently {train_data_count}).")
        
        self.train_df = train_df
        return train_df if concat else dfs

    def update_and_get_test_df(self, *,
                            data_path: str,
                            start_date: datetime | str,
                            end_date: datetime | str = "",
                            forward_dayCount: int = 0) -> list[pd.DataFrame]:
        """
        Get test_data from files within a specified date range.

        Args:
            data_path (str): The path to the directory containing the data files.
            start_date (datetime | str): The start date of the range (inclusive).
                                        It can be either a datetime object or a string in the format '%Y%m%d'.
            end_date (datetime | str): The end date of the range (inclusive).
                                    It can be either a datetime object or a string in the format '%Y%m%d'.

        Returns:
            list[pd.DataFrame]: A list of test_data read from files within the specified date range.
        """
        if end_date == "":
            if isinstance(start_date, str): start_date = datetime.strptime(start_date, r'%Y%m%d')
            end_date = start_date + timedelta(days=forward_dayCount)

        filtered_file_names = self._filter_file_names(start_date = start_date, end_date = end_date)
        
        dfs = [pd.read_csv(data_path + file_name) for file_name in filtered_file_names]
        
        self.test_dfs = dfs
        return dfs

    def transform_col(name: str, *, col_name: str, transform_func: 'Function') -> None:
        if self.train_df:
            self.saved_column[name] = pd.Series(self.train_df[col_name])
            self.train_df.loc[:, col_name] = self.train_df[col_name].apply(transform_func)
            
        return
    
    def reverse_transform_col(name: str, *, col_name: str) -> None:
        if name in self.saved_column:
            self.train_df.loc[:, col_name] = self.saved_column[name]
            del self.saved_column[name]
        return

    # \APIs

    # Helper Functions
    
    def _filtered_duplicatations(self, df, axis: int) -> 'DataFrame':
        use_df = df if axis == 0 else df.T
        answer_df = use_df.drop_duplicates()
        return answer_df if axis == 0 else answer_df.T
    
    def _removed_id(self, df: 'DataFrame', id_colname: str) -> None:
        if id_colname not in df.columns: return
        return df.drop(id_colname, inplace=False, axis = consts.COL)
    
    def _init_sorted_file_names(self, data_path: Optional[str]) -> list[str]:
        import os
        if data_path is None: return []

        try:
            file_names = os.listdir(data_path)
            data_file_names = list(filter(lambda file_name: consts.DATA_FILTER_KEYWORD in file_name, file_names))
            data_file_names.sort(reverse = False)
            return data_file_names
        except FileNotFoundError: print(f"The directory {data_path} does not exist.")
    
    def _init_sorted_file_datetimes(self) -> list[datetime]:
        answers = []
        
        for file_name in self.sorted_file_names:
            file_datetime = self._extract_datetime(file_name)
            if file_datetime is not None: answers.append(file_datetime)
        
        return answers    
    
    def _extract_datetime(self, file_name: str) -> Optional[datetime]:  
        """
        Extract the YYYYMMDD datetime from a file name.

        Args:
            file_name (str): The file name containing a YYYYMMDD datetime.

        Returns:
            Optional[datetime]: A datetime object representing the extracted YYYYMMDD datetime.
                Returns None if no valid datetime is found.
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
        """
        Filter out the files within the specified date range from the sorted_file_names list.

        Args:
            start_date (datetime | str): The start date of the range (inclusive).
                                        It can be either a datetime object or a string in the format '%Y%m%d'.
            end_date (datetime | str): The end date of the range (inclusive).
                                    It can be either a datetime object or a string in the format '%Y%m%d'.

        Returns:
            list: A list of file names within the specified date range.
        """
        if isinstance(start_date, str): start_date = datetime.strptime(start_date, r'%Y%m%d')
        if isinstance(end_date, str): end_date = datetime.strptime(end_date, r'%Y%m%d')
        print(f"Getting files from {start_date} to {end_date}, inclusive.")
        
        
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


"""Model Class works with 1 training data and N testing data
"""
class Model(Data):
    def __init__(self, model_type: Optional[str] = None, *, 
                 train_data: Optional[pd.DataFrame] = None,
                 test_data: list[pd.DataFrame] = [],
                 train_data_path: Optional[str] = None,
                 hyperparam_dict: Optional[dict] = None):

        super().__init__()
        
        """ TODO: Step 3
        
        Add a key-value after finishing a new function.
        """
        self.name_model_map = {
            'OLS': self._sklearn_ols_regression,
            'LASSO': self._sklearn_LASSO_regression,
            'XGBOOST': self._xgboost_regression,
            'DecisionTreeClassifier': self._sklearn_tree_classifier,
            'RandomForestClassifier': self._random_forest_classifier
            }
        
        self.metric = {
            'response_corr': self._get_response_corr,
            'mean_return': self._get_mean_return,
            'scale_factor': self._get_scale_factor
        }
        
        self.metric_output = {
            'response_corr': None,
            'mean_return': None,
            'scale_factor': None
        }
        #\  
                
        self.feature_col_names, self.interacting_terms_list = [], []
        self.predicted_y_list, self.actual_y_list = [], []

        # if the type is valid, call the regression function with the hyperparam_dict which return a model
        self.model_type = model_type
        if self.model_type is None:
            print("Model reseted. You must put in a DataFrame for each train/test.")
            self._list_all_types(); return

        # NOTE: This variable is tobe invoked to apply sklearn's built-in functions
        self.inner = self.name_model_map[self.model_type](hyperparam_dict)
        self.print_model(); return
        # \
    
    if "functions for input check":
        @property
        def model_type(self) -> str:
            return self._model_type
        @model_type.setter
        def model_type(self, model_type: str) -> None:
            if model_type is None: self._model_type = model_type
            else:
                if model_type in self.name_model_map: self._model_type = model_type
                elif model_type.lower() in self.name_model_map: self._model_type = model_type.lower()
                elif model_type.upper() in self.name_model_map: self._model_type = model_type.upper()
                elif model_type.capitalize() in self.name_model_map: self._model_type = model_type.capitalize()
                else: self.model_type = None
            return

    if "APIs":
        def train(self, dataframe: Optional[pd.DataFrame] = None, *, verbose = True,
                        feature_col_names: list[str] = [],
                        interacting_terms_list: list[list[str]] = [],
                        hyperparam_dict: Optional[dict] = None) -> None:
            """
            Train the regression model.

            Args:
                dataframe (Optional[pd.DataFrame], optional): Training data. Defaults to None.
                feature_col_names (list[str], optional): List of feature column names. Defaults to an empty list.
                interacting_terms_list (list[list[str]], optional): List of interacting terms. Defaults to an empty list.
                hyperparam_dict (Optional[dict], optional): Hyperparameters for training. Defaults to None.

            Raises:
                Exception: Raised if no training data is provided, and no existing training data is available.
            """
            copied_dataframe = dataframe.copy()
            if copied_dataframe is None and self.train_df is None: raise Exception("Can't train when nothing is given.\n")
            
            # getting the training df (either from class variable or the input of this method)
            training_df = self.train_df if (self.train_df is not None) else copied_dataframe
            training_df, new_col_names = self._get_df_with_interaction_terms(training_df, interacting_terms_list)
            #\

            # getting the response variables and drop it from the training df
            train_y = training_df[consts.RESPONSE_NAME]
            if consts.RESPONSE_NAME in set(training_df.columns):
                train_X = training_df.drop(consts.RESPONSE_NAME, axis=consts.COL, inplace=False)
            #\        
            
            # first, add the interacting term columns. Then, if this method is called with a feature_col_names then used them 
            # as additional training_features, else just use all columns as training features 
            training_features = []
            training_features.extend(new_col_names)
            
            if len(feature_col_names) > 0: training_features.extend(feature_col_names)
            else: training_features.extend(train_X.columns)
            train_X = training_df[training_features]
            #\
            
            # if no hyperparam is used, then fit the model with out hyperparameter 
            if hyperparam_dict is None: self.inner.fit(train_X, train_y)
            else: self.inner.fit(train_X, train_y, **hyperparam_dict)
            #\
            
            # update self.feature_col_names class variables after we get all the training features after line 404
            self.interacting_terms_list = interacting_terms_list
            self.feature_col_names = training_features
            #\
            
            if verbose: print(f"No. features being used: {len(self.feature_col_names)}")
            return
        
        def test(self, test_data: Optional[list[pd.DataFrame] | pd.DataFrame] = None, *, verbose = True,
                        hyperparam_dict: Optional[dict] = None) -> None:
            """
            Get metrics for the regression model using existing test data.

            Args:
                dataframes (Optional[list[pd.DataFrame] | pd.DataFrame], optional): Test data INCLUDE RESPONSE COLUMN. Defaults to None.
                hyperparam_dict (Optional[dict], optional): Hyperparameters for prediction. Defaults to None.

            Raises:
                Exception: Raised if no test data is provided, and no existing test data is available.

            Returns:
                None
            """
            if test_data is None and self.test_dfs == []: raise Exception("Can't test when nothing is given.\n")
            input_dfs = self.test_dfs if len(self.test_dfs) > 0 else test_data
            
            def _get_test_X(input_df: pd.DataFrame) -> pd.DataFrame:
                if consts.RESPONSE_NAME in set(input_df.columns): 
                    input_df.drop(consts.RESPONSE_NAME, axis=consts.COL, inplace=True)
                    
                test_X, _ = self._get_df_with_interaction_terms(input_df, self.interacting_terms_list)
                if len(self.feature_col_names) > 0: test_X = test_X[self.feature_col_names]
                
                return test_X         
            
            if isinstance(test_data, pd.DataFrame):
                input_df = test_data.copy()
                test_y = test_data[consts.RESPONSE_NAME].values
                test_X = _get_test_X(input_df)

                self.actual_y_list = [test_y]
                self.predicted_y_list = self._predict(test_X, hyperparam_dict = hyperparam_dict)
            else:
                input_dfs = [dataframe.copy() for dataframe in test_data]
                actual_y_list = [input_df[consts.RESPONSE_NAME].values for input_df in input_dfs]
                for i, input_df in enumerate(input_dfs):
                    test_X = _get_test_X(input_df)
                    input_dfs[i] = test_X

                self.actual_y_list = actual_y_list
                self.predicted_y_list = self._predict(input_dfs, hyperparam_dict = hyperparam_dict)

            assert len(self.predicted_y_list) == len(self.actual_y_list), \
            print(f"len(predicted_y_list) != len(actual_y_list)\n")
            
            self._reset_metric_output()
            predict_actual_pairs = list(zip(self.predicted_y_list, self.actual_y_list))
            self._update_metric_output(predict_actual_pairs, verbose)
            return
        
        # NOTE: the function must take in tuple of list (predicted_y: list, actual_y: list)
        def add_metric(self, name: str, function: 'Function(predicted_y, actual_y)') -> None:
            """Add a new metric to self.metric

            Args:
                name (str): _description_
                function (Function): the function must take in tuple of list (predicted_y: list, actual_y: list)
            """
            assert name not in self.metric, print(f"{name} existed in metric")
            self.metric[name] = function
            return
        
        def change_metric_name(self, *, old_name: str, new_name: str) -> None:
            if new_name in self.metric: self.metric[new_name] = self.metric[old_name]; del self.metric[old_name]
            return
   
        def print_model(self):
            print(f"You're using: {self.model_type}.\nRemember: Model Class works with 1 training data and N testing data.")
            print(f"Your model's DEFAULT init hyperparams are: {self.name_model_map[self.model_type]().get_params()}")
            
    if "Helper Functions":
        # Sk-learn region
        """TODO: Step 1
        
        Add another function if you want to expand class usage.
        """
        def _sklearn_tree_classifier(self, hyperparam_dict: Optional[dict] = None):
            from sklearn.tree import DecisionTreeClassifier
            
            returning_model = None
            if hyperparam_dict is None: returning_model = DecisionTreeClassifier()
            else: returning_model = DecisionTreeClassifier(**hyperparam_dict)
            
            return returning_model
        
        def _sklearn_ols_regression(self, hyperparam_dict: Optional[dict] = None):
            from sklearn.linear_model import LinearRegression
            
            returning_model = None
            if hyperparam_dict is None: returning_model = LinearRegression()
            else: returning_model = LinearRegression(**hyperparam_dict)
            
            return returning_model

        def _sklearn_LASSO_regression(self, hyperparam_dict: Optional[dict] = None):
            from sklearn.linear_model import LassoCV
            
            returning_model = None
            if hyperparam_dict is None: returning_model = LassoCV()
            else: returning_model = LassoCV(**hyperparam_dict)
            
            return returning_model

        def _xgboost_regression(self, hyperparam_dict: Optional[dict] = None):
            from xgboost import XGBRegressor 
            
            returning_model = None
            if hyperparam_dict is None: returning_model = XGBRegressor()
            else: returning_model = XGBRegressor(**hyperparam_dict)
            
            return returning_model
        
        def _random_forest_classifier(self, hyperparam_dict: Optional[dict] = None):
            from sklearn.ensemble import RandomForestClassifier
            returning_model = None
            
            if hyperparam_dict is None: returning_model = RandomForestClassifier()
            else: returning_model = RandomForestClassifier(**hyperparam_dict)
            
            return returning_model
            
        # \Sk-learn region
        
        def _list_all_types(self) -> None:
            print(f"Available Regression Inputs: {self.name_model_map.keys()}\n")
            return

        def _get_df_with_interaction_terms(self, df: pd.DataFrame, 
                                        interacting_terms_list: list[list[str]],
                                        will_drop_single_interacting_term: bool = False) -> tuple[pd.DataFrame, list]:
            """
            Get DataFrame with interaction terms - a product of two columns.

            Args:
                df (pd.DataFrame): The input DataFrame.
                interacting_terms_list (list[list[str]]): List of lists containing column names for interaction terms.
                will_drop_single_interacting_term (bool, optional): Whether to drop single interacting terms. 

            Returns:
                tuple[pd.DataFrame, list]: A tuple containing the new DataFrame with interaction terms and a list of new column names.
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

        def _predict(self, test_data: list[pd.DataFrame] | pd.DataFrame, *, 
                    hyperparam_dict: Optional[dict] = None) -> list:
            """
            Predict stock prices using the trained model.

            Args:
                test_data (list[pd.DataFrame] | pd.DataFrame): Input DataFrame or list of test_data.
                hyperparam_dict (Optional[dict], optional): Hyperparameters for the regression model. Defaults to None.

            Returns:
                list: List of predicted stock price values.
            """
            assert self.inner is not None, print("No model being trained yet!\n")
            predicted_y_list = []

            # if dataframe is just have one single dataframe 
            if isinstance(test_data, pd.DataFrame):
                if consts.RESPONSE_NAME in test_data.columns: test_data.drop(consts.RESPONSE_NAME, axis = consts.COL, inplace = True)
                assert len(self.feature_col_names) == len(test_data.columns), \
                print(f"In _predict(): len(self.feature_col_names) = {len(self.feature_col_names)}, len(test_data.columns) = {len(test_data.columns)}")
                
                if hyperparam_dict is None: predicted_y = self.inner.predict(test_data)
                else: predicted_y = self.inner.predict(test_data, **hyperparam_dict)
                
                predicted_y_list = [predicted_y]
            #\
            
            # else dataframe is a list of test_data
            else:
                for i in range(len(test_data)):
                    dataframe = test_data[i].copy()
                    if consts.RESPONSE_NAME in dataframe.columns: dataframe.drop(consts.RESPONSE_NAME, axis = consts.COL, inplace = True)
                    assert len(self.feature_col_names) == len(dataframe.columns)

                    if hyperparam_dict is None: predicted_y = self.inner.predict(dataframe)
                    else: predicted_y = self.inner.predict(dataframe, **hyperparam_dict)
                    
                    predicted_y_list.append(predicted_y)
            #\
        
            return predicted_y_list
        
        def _reset_metric_output(self):
            for metric_name in self.metric_output.keys():
                self.metric_output[metric_name] = None
        
        def _update_metric_output(self, predict_actual_pairs: list[list], verbose = True):
            for metric_name, metric_function in self.metric.items():
                metric_outputs = []
                
                for predict_actual_pair in predict_actual_pairs:
                    metric_outputs.append(metric_function(*predict_actual_pair))
                
                self.metric_output[metric_name] = np.mean(metric_outputs)
            
            if verbose: self._print_metrics(predict_actual_pairs)
            return
        
        def _print_metrics(self, predict_actual_pairs: list[list]) -> None:
            for metric_name, metric_function in self.metric.items():
                print(f"{metric_name}: {self.metric_output[metric_name]}")
            return  
        
        """TODO: Step 2
        
        Add another metric if you want to expand class usage.
        """    
        def _get_response_corr(self, predicted_y: list, actual_y: list) -> float:
            return np.corrcoef(predicted_y, actual_y)[0, 1]

        def _get_mean_return(self, predicted_y: list, actual_y: list) -> float:
            return np.mean(np.abs(actual_y) * (np.sign(actual_y) * np.sign(predicted_y)))

        def _get_scale_factor(self, predicted_y: list, actual_y: list):
            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression(fit_intercept=False)
            model.fit(X=pd.DataFrame({"predicted_y": predicted_y}), y=actual_y)
            return model.coef_