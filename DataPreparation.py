
    
# DataPreparation.py file

import numpy as np
import pandas as pd


# Verify no missing values, check that IDs match, count # of tasks, suppliers, features, cost values
# Identify and remove tasks with no costs associated

# This function prints out information about all the datasets and returns the updated tasks dataset
def VerifyDataPrintInfo(costs, tasks, suppliers):
    # Name the datasets
    
    # Checking for missing or null values 
    for df in [costs, tasks, suppliers]:  
        print('Number of missing or null values in ', df.name,' dataset: ', df.isna().sum().sum(), sep = '')

    # Tasks Dataset:
    print('\nNumber of features in tasks dataset:', tasks.count(axis = 1)[0]-1)
    print('Number of tasks in tasks dataset before removing those without cost:', len(tasks.iloc[:,0].unique()))
    
    # Drop the tasks with no associated costs:
    tasks = DropNoCostTasks(costs, tasks)
    print('Number of tasks in tasks dataset after removing those without cost:', tasks.count()[0])
    # Print a description of the dataset
    print('\nDescription of tasks dataset:\n',tasks.describe(), sep = '')
    print('Variance of tasks:\n', tasks[tasks.select_dtypes('number').columns].var(), sep = '')
    
    # Suppliers:
    print('\n\nNumber of suppliers in suppliers dataset:', len(suppliers.iloc[:,0].unique()))
    print('Number of features in suppliers dataset:', suppliers.count(axis = 1)[0]-1)
    print('\nDescription of suppliers dataset:\n', suppliers.describe(), sep = '')
    print('Variance of suppliers:\n', suppliers[suppliers.select_dtypes('number').columns].var(), sep = '')
    
    # Costs dataset:
    # Checking if there are only numbers in 'Cost' column
    print("\nAre there any non-floats in costs?: ", np.any(isinstance(costs['Cost'],float)),'(True = Yes, False = No)')
    # Checking if all numbers are positive in 'Cost' column
    print("Are there any negative costs?:",np.any(costs['Cost']<=0),'(True = Yes, False = No)')
    # Setting Task ID as a datetime type
    costs['Task ID'] = pd.to_datetime(costs['Task ID'], infer_datetime_format=True)
    costs['Task ID'] = costs['Task ID'].dt.date
    print('Number of suppliers in cost dataset: ', len(costs['Supplier ID'].unique()))
    print('\nDescription of costs dataset:\n',costs.describe(), sep = '')
    print('Variance of costs:\n', costs[costs.select_dtypes('number').columns].var(), sep = '')
     
    return costs, tasks, suppliers

def DropNoCostTasks(costs, tasks):
    cost_pivot = costs.pivot(index = "Task ID", values = "Cost", columns = "Supplier ID")
    # reseting the index
    cost_pivot = cost_pivot.reset_index()
    # Converting the object Task ID into to_datetime format
    cost_pivot["Task ID"] = pd.to_datetime(cost_pivot["Task ID"],format="%d/%m/%Y",infer_datetime_format=True).dt.date
    tasks["Task ID"] = pd.to_datetime(tasks["Task ID"],infer_datetime_format=True).dt.date
    # merging both the dataframes
    tasks_merged = pd.merge(tasks,cost_pivot.iloc[:,0], on="Task ID", how="inner")
    return tasks_merged

# Function which returns the variances of the features
def GetFeatureVariances(df):
    return df.var(numeric_only = True)

# Function which drops the features with variance less than 0.01
def DropLowVariances(df):
    variance = df.iloc[:,1:].var(axis = 0)

    # creating the index for columns with variances < 0.01
    index_def = variance[variance < 0.01].index 
    df_refined = df.drop(index_def, axis = 1)
    df_refined = pd.DataFrame(df_refined)
    return df_refined

"""
Scale the features to [-1, 1] range
Input to function is a dataframe with the "index column" i.e task/supplier ID included
Function returns a scaled version of the dataframe
"""
def ScaleFeatures(df):
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler(feature_range = (-1,1))
    df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:])
    return df

"""
Function to drop all suppliers who don't appear in the top 20 of any tasks
The function returns an updated df costs where suppliers have been dropped, and a df
containing the top 20 suppliers per task ID
"""
def DropExpensiveSuppliers(costs, suppliers):
    
    # Creating a pivot table and converting it into an array to use indirect sorting for each column
    # We use 'Task ID' as rows and 'Supplier ID' as columns 
    pivot_costs= pd.pivot_table(costs, index = 'Task ID',values = 'Cost',columns ='Supplier ID' )
    pivot_array = np.array(pivot_costs)
    
    # Indirect sorting the pivot_array by column(axis = 1) to identify top 20 suppliers regarding minimum cost
    pivot_array_indsort = np.argsort(pivot_array,axis=1)
    pivot_array_20_indsort = pivot_array_indsort[:,:20]
    
    # Top 20 suppliers per Task ID stored in the dictionary
    top_sup = {}
    for task,supplier in zip(pivot_costs.index,pivot_array_20_indsort):
        top_sup[task] = pivot_costs.columns[supplier]
    
    # Remove from the pivot_costs all suppliers that never appear in the top 20 of any task --> pivot_costs_updated
    list_with_insufficient_suppliers = []
    for index_num in range(pivot_array_indsort.shape[1]):
        if index_num not in pivot_array_20_indsort:
            list_with_insufficient_suppliers.append(pivot_costs.columns[index_num])
            pivot_costs_updated = pivot_costs.drop(pivot_costs.columns[index_num],axis =1)
    
    suppliers.index = suppliers['Supplier ID']
    for supplier in list_with_insufficient_suppliers:
        suppliers.drop(supplier, axis = 0, inplace = True)
    suppliers.reset_index(drop = True, inplace = True)
    # pivot_costs_updated is a pivot table without the insufficient suppliers
    # pivot_costs_updated --> melt --> costs_final
    # Melt and ascend by Task ID - costs_final includes only the data with the suppliers who appeared in at least one top 20  
    pivot_costs_updated = pivot_costs_updated.reset_index()
    costs_final = pd.melt(pivot_costs_updated,var_name = 'Supplier ID', value_name = 'Cost', id_vars = 'Task ID')
    costs_final['Task ID'] = pd.to_datetime(costs_final['Task ID'])
    costs_final['Task ID'] = costs_final['Task ID'].dt.date
    costs_final = costs_final.sort_values(by=['Task ID', 'Cost'])

    return costs_final, pd.DataFrame(top_sup), list_with_insufficient_suppliers
# The function returns the final costs dataframe without insufficient suppliers and the top 20 suppliers per task

# Returns the absolute correlations for a dataframe
def GetAbsoluteCorrelations(df):
    return df.corr().abs()

# Function which takes a dataframe and removes the features in the df where abs.cors >= 0.8
def DropFeaturesByCorrelation(df):
    # Get an absolute correlation matrix for the dataframe
    # Get an absolute correlation matrix for the dataframe
    matrix = GetAbsoluteCorrelations(df)
    # make the correlation matrix such that all values >= 0.8 are set to NA
    bool_cors = matrix[(matrix < 0.8) | (matrix == 1)]
    # For every row in the correlation matrix we check if the row has any NA values
    # If there is any NA value in a row, we remove the column for that feature in the tasks dataframe
    for row in matrix:
        # Since the cor matrix contains duplicates we only select the upper triangle of the correlation matrix by selecting
        # The row and the subset of columns from the column and to the right, i.e we choose a smaller amount of columns per loop
        if bool_cors.loc[row, row:].isna().any():
            df.drop(row, axis = 1, inplace = True) # Drop the column feature in the dataset

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            