"""
main.py file

Main program which uses functions created in the other files
This programs imports:
1) DataPreparation.py
2) EDA.py
3) MachineLearning.py

As long as all files and datasets ('tasks.xlsx','cost.csv','suppliers.csv') are in the same directory, this main.py file 
can generate all the results


"""


import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge   
from sklearn.svm import SVR 
from DataPreparation import *
from MachineLearning import *
from EDA import *

# Data Preparation:
    
# Read the data
# Reading the CSV files and excel (openpyxl required) files
tasks = pd.read_excel('./data/tasks.xlsx')
costs = pd.read_csv('./data/cost.csv')
suppliers = pd.read_csv('./data/suppliers.csv')
costs.name = 'costs'
tasks.name = 'tasks'
suppliers.name = 'suppliers'

datasets = [tasks, costs, suppliers]


"""
These are the functions imported from Data_Preparation, with a brief description of returned values:

- VerifyDataPrintInfo(costs, tasks, suppliers) -> returns costs, tasks, suppliers after verifying and checking data
- DropNoCostTasks(costs, tasks) -> returns tasks_merged (tasks with no costs are removed)
- GetFeatureVariances(df) -> returns series of variances
- DropLowVariances(df) -> returns df where low vars have been dropped
- ScaleFeatures(df) -> returns a scaled version of the df
- DropExpensiveSuppliers(costs) -> returns costs_final (costs where supplier not in top 20 removed), pd.DataFrame(top_sup)
- GetAbsoluteCorrelations(df) -> returns absolute correlation for a df
- DropFeaturesByCorrelation(df) -> returns nothing, but drops the feats in the df (inplace = True)
"""

for x in datasets:
    print(x.head(), '\n')

costs, tasks, suppliers = VerifyDataPrintInfo(costs, tasks, suppliers)
print('\nDataframes after verifying data:\n')
for x in datasets:
    print('\n', x.name, '\n', x.head(), sep = '')


# Get variances for the features
tasks_var = GetFeatureVariances(tasks)
suppliers_var = GetFeatureVariances(suppliers)
print('Supplier feature variances:\n', suppliers_var.head(), sep = '')

# Check and drop any low variances
tasks = DropLowVariances(tasks)
suppliers = DropLowVariances(suppliers)
print('\nTasks after dropping low variances:')
print(tasks.head())
print('\nSuppliers after dropping low variances:')
print(suppliers.head())

# Scale features
tasks = ScaleFeatures(tasks)
suppliers = ScaleFeatures(suppliers)
print('\nScaled tasks:')
print(tasks.head())
print('\nScaled suppliers:')
print(suppliers.head())

# Drop expensive suppliers
costs, top_suppliers, bad_suppliers = DropExpensiveSuppliers(costs, suppliers)
print('\nCosts after dropping expensive suppliers:')
print(costs.head())
print('\nTop Suppliers:')
print(top_suppliers.head())
print('Suppliers to drop:', bad_suppliers)

# Get abs correlation for tasks
tasks_abs_cors = GetAbsoluteCorrelations(tasks)
print('\nAbsolute correlation between task features:')
print(tasks_abs_cors)

DropFeaturesByCorrelation(tasks)
print('\nTasks after dropping highly correlated features:')
print(tasks.head())


# Exploratory Data Analysis:

"""
EDA.py imported functions:

- FeatureDistributionPerTask -> Displays one figure (using boxplots) that shows the distribution of feature values for each Task ID 
- DistributionErrorsPerSupplier(costs, suppliers) -> Displays one figure (using boxplots) that shows, for each supplier, 
the distribution of Errors if that supplier is selected to perform each task
- HeatmapCostsTasksSuppliers(costs) -> Displays one figure (heatmap plot) that shows the cost values 
as a matrix of tasks (rows) and suppliers (columns)
"""
# Boxplot for the distribution of feature values for each task
FeatureDistributionPerTask(tasks)

# Boxplot for the distribution of errors for each supplier
# Each boxplot annotated with the RMSE of each supplier for all tasks
DistributionErrorsPerSupplier(costs, suppliers)

# cost values as a matrix of tasks and suppliers using heatmap plot
HeatmapCostsTasksSuppliers(costs)

# Machine Learning

"""
Imported functions from Machine Learning file:
    
- MergeData(tasks, suppliers, costs) -> Merges the dataframes to the right format and returns a single dataframe
- SplitToGroupsXandY(fully_merged) -> Splits a fully merged dataframe into Groups, X and y, which are returned
- SelectTestGroup(Groups) -> Uses the Groups df generated before to split to Test and Train Groups, 
returns TestGroup and TrainGroup
- TrainTestSplitting(X_train_merged, X_test_merged) -> Splits into and returns X_train, y_train, X_test, y_test, 
with the help of TestGroup and TrainGroup
- CalculateErrors(predicted, actual) -> Calculates errors as per equation 1 and returns the errors
- GetRMSE(errors) -> Gets the RMSE score of errors (RMSE = sqrt(ΣError(t)^2/abs(T)))
- GetMinCostByGroup(group, costs) -> Gets the minimum cost for a group (i.e. min cost per Task ID). 
Returns a df of costs with Task ID as index
- GetCostOfPredictedSupplier(model, X, X_merge, y) -> Finds the actual cost corresponding to the supplier 
with lowest predicted cost and returns a similar df as above
- OurScoreFunction(y_true, y_pred) -> Equation 1 in a format to be used as scoring function for cross validation
- RunCrossValidation(model, scorer, X, y, Groups, logo) -> Runs cross validation for a model
with given scoring function, returns cross val scores
- RunGridSearch(model, scorer, param_grid, X, y, Groups, logo) -> Runs grid search on model 
with given scoring function and parameters to search, returns the grid search results
- FindLowestRMSEParams(gs_results_df) -> Finds the parameters which give the lowest RMSE for the model 
based on grid search results df
- GetRMSEByParams(params, gs_results_df) -> Returns the RMSE score for a model 
using the supplier parameters within the grid search results df
"""

fully_merged = MergeData(tasks, suppliers, costs)

Groups, X, y = SplitToGroupsXandY(fully_merged)

TestGroup, TrainGroup = SelectTestGroup(Groups)

# These are used to locate the actual corresponding cost later on and splitting by group into Train and Test
X_train_merged = pd.merge(TrainGroup, fully_merged)
X_test_merged = pd.merge(TestGroup, fully_merged)

X_train, y_train, X_test, y_test = TrainTestSplitting(X_train_merged, X_test_merged)


# train the model using X_train, Y_train - score on test set
# Score is not expected to be great due to lack of optimization for the hyper-parameters

ridge = Ridge()
ridge.fit(X_train, y_train)
print('Ridge default score function score: ', ridge.score(X_test, y_test))

# Get min costs and predicted min costs and find out the errors and RMSE of the model
TestGroupMinCosts = GetMinCostByGroup(TestGroup, costs)
PredSup_ActualCost = GetCostOfPredictedSupplier(ridge, X_test, X_test_merged, y_test)

ridge_errors = CalculateErrors(PredSup_ActualCost, TestGroupMinCosts)
print('Errors from EQ1 for ridge model:\n', ridge_errors, sep='')
ridge_RMSE = GetRMSE(ridge_errors)
print('RMSE Score for the ridge model:', ridge_RMSE, '\n')

# Make a scorer to use in cross validation as well as the LeaveOneGroupOut
print('Running Cross Validation for Ridge Model:\n')
my_scorer = make_scorer(OurScoreFunction)
logo = LeaveOneGroupOut()
# Get the cross validation scores for Ridge model
cv_ridge = Ridge()
ridge_cv_scores = RunCrossValidation(cv_ridge, my_scorer, X, y, Groups, logo)

"""
-Define a param_grid to run Grid Search on and then run Grid Search
-First iteration we run with an exponential range of [2^0, 2^15] to see 
what values it favours towards,then this is used in a 2nd iteration
-1 to 2^15 should be sufficient for an initial run as we know as alpha increases a lot it moves 
towards just normal linear regression
-We're using iterative hill search
"""

# Generate fresh models per iteration
gs1_ridge = Ridge()
print('\nRunning Grid Search in 3 iterations for Ridge Model:')
print('\nIteration 1:')
ridge_param_grid1 = dict(alpha = np.exp2(np.arange(0, 15)), solver = ['auto','svd','cholesky','lsqr','sparse_cg','sag','saga'])
ridge_GS_CV_Results1, ridge_grid_best_params1 = RunGridSearch(gs1_ridge, my_scorer, ridge_param_grid1, X, y, Groups, logo)

# See what the RMSE of the best parameters are
ridge_best_params1 = FindLowestRMSEParams(ridge_GS_CV_Results1)
print('Iteration 1 best Ridge parameters:', ridge_best_params1)
ridge_RMSE1 = GetRMSEByParams(ridge_best_params1, ridge_GS_CV_Results1)
print('Iteration 1 RMSE value from Grid Search for Ridge:', ridge_RMSE1.iloc[0])


gs2_ridge = Ridge()
print('\nIteration 2:')
# We see that the initial run gives a best alpha value of 1.0, so it seems to favour low values, and we search a space around 1
ridge_param_grid2 = dict(alpha = np.arange(0.05, 1, 0.05), solver = ['auto','svd','cholesky','lsqr','sparse_cg','sag','saga'])
ridge_GS_CV_Results2, ridge_grid_best_params2 = RunGridSearch(gs2_ridge, my_scorer, ridge_param_grid2, X, y, Groups, logo)

# See what the RMSE of the best parameters are
ridge_best_params2 = FindLowestRMSEParams(ridge_GS_CV_Results2)
print('Iteration 2 best Ridge parameters:', ridge_best_params2)
ridge_RMSE2 = GetRMSEByParams(ridge_best_params2, ridge_GS_CV_Results2)
print('Iteration 2 RMSE value from Grid Search for Ridge:', ridge_RMSE2.iloc[0])


gs3_ridge = Ridge()
print('\nIteration 3:')
# The model seems to favour low alphas and we do one last iteration to get close to an optimum solution
ridge_param_grid3 = dict(alpha = np.arange(0.01, 0.05, 0.001), solver = ['auto','svd','cholesky','lsqr','sparse_cg','sag','saga'])
ridge_GS_CV_Results3, ridge_grid_best_params3 = RunGridSearch(gs3_ridge, my_scorer, ridge_param_grid3, X, y, Groups, logo)

# Check what the RMSE of the best parameters are
ridge_best_params3 = FindLowestRMSEParams(ridge_GS_CV_Results3)
print('Iteration 3 best Ridge parameters:', ridge_best_params3)
ridge_RMSE3 = GetRMSEByParams(ridge_best_params3, ridge_GS_CV_Results3)
print('Iteration 3 RMSE value from Grid Search for Ridge:', ridge_RMSE3.iloc[0])

print('Are the parameters by looking for lowest RMSE the same as by using equation 1?',ridge_best_params3==ridge_grid_best_params3)


# And onwards for SVR: Same procedure as Ridge model

svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)
print('\nSVR default score function score: ', svr.score(X_test, y_test))


svr_PredSup_ActualCost = GetCostOfPredictedSupplier(svr, X_test, X_test_merged, y_test)

svr_errors = CalculateErrors(svr_PredSup_ActualCost, TestGroupMinCosts)
print('Errors from EQ1 for SVR model:\n', svr_errors, sep='')
svr_RMSE = GetRMSE(svr_errors)
print('RMSE Score for the SVR model:', svr_RMSE, '\n')


print('Running Cross Validation for SVR Model:\n')
cv_svr = SVR(kernel = 'rbf')
svr_cv_scores = RunCrossValidation(cv_svr, my_scorer, X, y, Groups, logo)


print('\nRunning Grid Search in 2 iterations for SVR Model:')

gs1_svr = SVR(kernel = 'rbf')
print('\nIteration 1:')
svr_param_grid1 = dict(epsilon = np.float_power(10,np.arange(-3,0)), C = np.float_power(10,np.arange(0,3)))
svr_GS_CV_Results1, svr_grid_best_params1 = RunGridSearch(gs1_svr, my_scorer, svr_param_grid1, X, y, Groups, logo)

svr_best_params1 = FindLowestRMSEParams(svr_GS_CV_Results1)
print('Iteration 1 best SVR parameters:', svr_best_params1)
svr_RMSE1 = GetRMSEByParams(svr_best_params1, svr_GS_CV_Results1)
print('Iteration 1 RMSE value from Grid Search for SVR:', svr_RMSE1.iloc[0])


gs2_svr = SVR(kernel = 'rbf')
print('\nIteration 2:')
# Grid Search takes a while for SVR, so the number of values for epsilon and C are not too big
svr_param_grid2 = dict(epsilon = np.arange(0.0005, 0.001, 0.00025), C = np.arange(0.5, 1, 0.1))
svr_GS_CV_Results2, svr_grid_best_params2 = RunGridSearch(gs2_svr, my_scorer, svr_param_grid2, X, y, Groups, logo)

svr_best_params2 = FindLowestRMSEParams(svr_GS_CV_Results2)
print('Iteration 2 best SVR parameters:', svr_best_params2)
svr_RMSE2 = GetRMSEByParams(svr_best_params2, svr_GS_CV_Results2)
print('Iteration 2 RMSE value from Grid Search for SVR:', svr_RMSE2.iloc[0])

print('Are the parameters by looking for lowest RMSE the same as by using equation 1?', svr_best_params2 == svr_grid_best_params2)

