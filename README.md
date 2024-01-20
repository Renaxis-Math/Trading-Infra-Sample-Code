# Clinic

Execution direction:

tester.py -> test_helper.py -> data_analysis -> custom_regressions

test.py: contains test for methods in test_helper.py

test_helper.py: contains two classes - data and regression
  data class: reading in files, filtering out files within date range, create DataFrame from these files
  regression class: inherite from data class, train regression model and predict stock prices. Evulate the predictions

data_analysis.ipynb: analysis and select the optimal columns for the prediction, have training and testing DataFrame ready to use

custom_regression.ipynb: perform the regression and predict stock prices
