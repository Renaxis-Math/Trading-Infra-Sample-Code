# Clinic

**Note to Scott:**  This README file will explain what each file does in our repository.  The key takeaways and our conclusions will be in our final report. The code should be able to run if there is access to the data but we realize that in a meeting you mentioned that you wanted to see documented source code so you could see which algorithms we used to experiment in your own research environment.  Some of the files in the repo are just for getting the drive download to work and I will omit the explanation. 

# Explanation of useful file: 

  - `consts.py`: This is a file that contains all the constants used in our project. The raw data paths were used to get the data into the project and were different for each team member. The folder ID and scopes have to do with getting the data from the drive folder with the data in it. 
  
  - `dependencies.txt`: Basic dependency file for installing python libraries
  - `feature_selection.ipynb`: The file is a python notebook that looks at multiple strategies for feature selection. The file has:
  	- OLS with all Feautres (Overfits significantly)
  	- Stepwise Selection (41 Features in `Hoang_stepwise_features.txt`)
  	- Stepwise Selection + LASSO (41 features in `Hoang_lasso_features.txt`)
  	- Genetic Algorithm + Stepwise (18 features in `Hoang_stepwise_genetic_algo_features.txt`)
  	- Genetic Algorithm + LASSO (94 features in `Hoang_lasso_genetic_algo_features.txt`)
  	
  	Concludes that features should be picked from combination of Genetic Algo + Stepwise and Genetic Algo + LASSO and using nonlinear feature selection. 
  - `helper.py`: Contains the helper functions used for the project: 
   Here are the classes and useful functions in the classes and outside we used that are particularly useful for replicating results
  	- Classes:
  		- Data: 
  			- Class to represent model data. Allows for easy integration into the `Model` class
  			- Gets data from the files, stores training and testing dataframes. 
  			- `find_high_corr()`: Finds columns with high correlation to potentially drop
  		- Model:
  			- Class to create, train, evaluate and store models. 
  			- Types of models implemented: (Implemented with the Scikit-learn API)
  				- Ordinary Least Squares
  				- LASSO
  				- XGBoost
  				- DecisionTreeClassifier
  				- DecisionTreeRegressor
  				- Support Vector Regression 
  			- `test()`: Gets Response Correlation, Mean Return and Scale Factor metrics
  			- `train()`: Trains the model
  			
  	- Functions: 
  		- `stepwise_selection()`: Performs Stepwise selection
  		- `hypothesis_test_features()`: Performs hypothesis test on 2 features
  		- `LASSO_feature_selection()`: Feature selection with LASSO regression
  		- `genetic_algorithm()`: Runs a custom genetic algorithm to select features
  		
  - `main_model.ipynb`: File to try nonlinear methods. Methods tried include:
  	- XGBoost
  	- Support Vector Regressions
  		- Incrementing Gamma experiments
  - `pca.ipynb`: Small experiments using PCA
  	- Plot of explaining how much variance contained in the first $n$ eigenvectors
  	- Regressions with PCA using 40 components with OLS, LASSO and XGBoost
  - `tester.py`: Non-rigorous test suite for a few utility functions
  - `tree_model.ipynb`: Experiments using Trees for feature selection 
  	- Trees for Feature Selection 
  	- Random Forest Classification (33 Features in `Hoang_classTree_genetic_features.txt`)
  - `xgb_params.txt/lasso_params.txt`: Copied and pasted from SKlearn api documentation. Mostly there for reference about different hyperparameters that were available. 
  - `Hoang_hyperparams`: Best hyperparameters found. 
