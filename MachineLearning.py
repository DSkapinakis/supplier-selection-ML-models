

# MachineLearning.py file


# File with ML related functions such as getting the dataset, splitting into test and train and model fitting (Ridge regression)

# Combine feature values in format [Task ID, TF1, ... , TFn, SF1, ... , SF18, cost]
# Each Task ID appears multiple times (for each supplier), and dataset should have same number as rows as costs.csv
import numpy as np
import pandas as pd
from math import sqrt
import time
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def MergeData(tasks, suppliers, costs):

    # First we merge suppliers and costs by Supplier ID
    suppliers.index = suppliers['Supplier ID']
    costs.index = costs.pop('Supplier ID')
    supcosts = suppliers.merge(costs, left_index = True, right_index = True)
    
    # Then we merge tasks with supcosts by Task ID
    supcosts.index = supcosts.pop('Task ID')
    tasks.index = tasks['Task ID']
    fully_merged = tasks.merge(supcosts, left_index = True, right_index = True)
    fully_merged.drop('Supplier ID', axis = 1, inplace = True)
    
    fully_merged.reset_index(drop = True, inplace = True)
    print(fully_merged)
    return fully_merged


def SplitToGroupsXandY(fully_merged_df):

    Groups = fully_merged_df['Task ID']
    X = fully_merged_df.iloc[:,1:len(fully_merged_df.columns)-1]
    y = fully_merged_df['Cost']
    
    return Groups, X, y
    

'''
TestGroup - np.random.default_rng().choice() generates a random sample from a given array
replace = False means that a value cannot be generated twice
--> Train, Test, Split
'''

def SelectTestGroup(Groups):

    rng = np.random.default_rng(42)
    TestGroup = rng.choice(Groups.unique(),size = 20, replace = False)
    TestGroup = pd.DataFrame(TestGroup, columns = ['Task ID'])
    
    TrainGroup = {'Task ID':[]}
    for i,l in enumerate(np.isin(Groups.unique(), TestGroup)):
        if l == False:
            TrainGroup['Task ID'].append(Groups.unique()[i])
    TrainGroup = pd.DataFrame(TrainGroup)
    
    return TestGroup, TrainGroup


# Split the train and test sets into X and y:
def TrainTestSplitting(X_train_merged, X_test_merged):
    
    X_train = X_train_merged.drop(columns = ['Task ID','Cost'])
    y_train = X_train_merged['Cost']
    
    
    X_test = X_test_merged.drop(columns = ['Task ID','Cost'])
    y_test = X_test_merged['Cost']
    
    return X_train, y_train, X_test, y_test
    
'''
Equation 1: Error(t) = min{ c(s,t) | s E S} - c(s_t',t), where 𝑡 is a task, 𝑆 is the set of 64 suppliers, 𝑠𝑡′ 
is the supplier chosen by the ML model for task 𝑡, and 𝑐(𝑠,𝑡) is the cost if task 
𝑡 is performed by supplier 𝑠. That is, the Error is the difference in cost between
the supplier selected by the ML model and the actual best supplier for this task.

'''

# Calculates errors using equation 1
def CalculateErrors(predicted, actual):
    errors = actual - predicted
    return errors

# Calculates RMSE value based on errors generated from equation 1
def GetRMSE(errors):
    groupsize = len(errors)
    RMSE = sqrt((np.sum(errors**2))/groupsize)
    return RMSE

# Returns the minimum cost of a group
def GetMinCostByGroup(group, costs):
    
    minimum_cost_tasks =  costs.groupby('Task ID')['Cost'].min()
    group.index = group.pop('Task ID')
    group['Cost'] = minimum_cost_tasks
    # Minimum actual costs for Group
    Group_MinCosts = group
    
    return Group_MinCosts

# Returns the actual cost for the supplier corresponding to the lowest cost predicted by the model
def GetCostOfPredictedSupplier(model, X, X_merge, y):
    #Actual cost of the predicted supplier
    
    # Use model to predict costs for given feature values
    predicted_costs = model.predict(X)
    predicted_costs_df = pd.DataFrame(predicted_costs,columns = ['Cost'])
    # Allows to group the costs by Task ID
    predicted_costs_df['Task ID'] = X_merge['Task ID']
    # Get the index of the cheapest supplier for a task
    predicted_minindex = predicted_costs_df.groupby('Task ID').idxmin()
    predicted_minindex.rename(columns={'Cost' : 'Min Index'}, inplace = True)
    # Use the index found above to see what the actual cost for a task is when using the supplier which is cheapest
    #according to model prediction
    predicted_supplier_actual_cost =  y[predicted_minindex['Min Index']].to_frame()
    # Merge to get Task ID and cost in the same dataframe, setting the Task ID as index
    predicted_supplier_actual_cost['Task ID'] = X_merge['Task ID']
    predicted_supplier_actual_cost.index = predicted_supplier_actual_cost.pop('Task ID')
    
    return predicted_supplier_actual_cost

# Score function to use in cross validation
def OurScoreFunction(y_true, y_pred):
    
    return y_true.min() - y_true.iloc[y_pred.argmin()]


# Function which runs Leave-One-Group-Out Cross Validation for a model with supplied scoring function, X, y, and groups
def RunCrossValidation(model, scorer, X, y, Groups, logo):
    print('Starting Cross Validation...')
    start_time = time.time() # Timer to see how long it takes
    
     # Set equation 1 to be the score function
     # Get the scores using leave one group out
    scores = cross_val_score(model, X, y, cv = logo, groups = Groups, scoring = scorer, n_jobs = -1) 
    print('Cross Validation Complete!')
    print('Cross validation took', time.time() - start_time, 'seconds')
    print('Cross-Validation Scores:\n', scores)
    print('RMSE of the cross-validation scores:', GetRMSE(scores))
    
    return scores

# Function which runs Grid Search for a model with supplied scoring function, parameter_grid and data
def RunGridSearch(model, scorer, param_grid, X, y, Groups, logo):
    
    print('\nStarting Grid Search...')
    start_time = time.time()
    gs = GridSearchCV(model, param_grid, cv = logo, scoring = scorer, n_jobs = -1)
    gs.fit(X, y, groups = Groups)
    print('Grid Search Complete!')
    print('grid search took', time.time() - start_time, 'seconds')
    print('Best parameters found:', gs.best_params_)
    
    grid_search_CV_results = pd.DataFrame(gs.cv_results_)
    
    return grid_search_CV_results, gs.best_params_
    

def FindLowestRMSEParams(gs_results_df):
    gsRMSEs = gs_results_df.iloc[:, 7:-3].apply(GetRMSE, axis=1)
    best_params_index = gsRMSEs.argmin()
    
    return gs_results_df.loc[best_params_index, 'params']
    
def GetRMSEByParams(params, gs_results_df):
    
    RMSE = gs_results_df[gs_results_df['params'] == params].iloc[:, 7:-3].apply(GetRMSE, axis=1)
    return RMSE
 


