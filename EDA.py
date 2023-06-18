
# EDA.py file


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

"""
EDA --> FeatureDistributionPerTask(tasks): One figure (using boxplots) that shows the distribution 
of feature values for each Task ID 
"""
def FeatureDistributionPerTask(tasks):
    
    tasks.plot.box()    
    plt.xticks(fontsize = 12, rotation = 90)
    plt.title('Distribution of Feature Values for each Task', fontsize = 30)
    plt.xlabel("Task Features", fontsize = 20)
    
"""  
EDA --> DistributionErrorsPerSupplier(costs, suppliers)
One figure (using boxplots) that shows, for each supplier, the distribution of Errors 
if that supplier is selected to perform each task
"""
def DistributionErrorsPerSupplier(costs, suppliers):
    minimum_cost =  costs.groupby('Task ID')['Cost'].min()
    # create the boxplot 
    costs_order = costs.sort_values(by = ['Supplier ID', 'Task ID']) 
    costs_order['Task ID'] = pd.to_datetime(costs_order['Task ID']).dt.date
    costs_order_pivot = pd.pivot_table(costs_order, index = 'Supplier ID', values = 'Cost', columns = 'Task ID')
    cost_order_pivot = costs_order_pivot.sort_index(key=lambda x: (x.to_series().str[1:].astype(int)), inplace=True)
        
    # errors_array --> Subtract costs_order_pivot from minimum_cost using numpy arrays - broadcasting
    errors_array = np.array(minimum_cost) - np.array(costs_order_pivot)
    
    # Distribution of errors 
    errors_df = pd.DataFrame(errors_array,columns = costs_order_pivot.columns)
    errors_df['Supplier ID'] = costs_order_pivot.index
    errors_df = errors_df.set_index('Supplier ID')
    errors_df_boxplot = errors_df.T
    
    # Calculation of RMSE - Supplier with the minimum RMSE
    total_number_of_suppliers = len(errors_df_boxplot)
    RMSE = round(np.sqrt((np.sum(errors_df_boxplot**2))/total_number_of_suppliers), 4)
    print('The supplier with the minimum RMSE is:',RMSE.idxmin(),',RMSE:',RMSE.min())
    
    # Boxplot of distributions 
    fig, ax1 = plt.subplots()
    ax1.boxplot(errors_df_boxplot, labels = suppliers['Supplier ID'])
    plt.xticks(rotation = 90, ha = 'right')
    plt.xlabel("Supplier ID", fontsize = 18)
    ax2 = ax1.twiny()
    ax2.boxplot(errors_df_boxplot, labels = RMSE)
    plt.xticks(rotation = 90, ha = 'right')
    plt.title("Distribution of Errors per Supplier", fontsize = 25)   
    position_x_axis = np.arange(1, 64, 1)
    for i in range (len(RMSE)):
        plt.annotate(round(RMSE[i], 5), xy = (position_x_axis[i], -(RMSE[i])),size = 9,rotation = 90,color = "blue",ha = "center")
"""
EDA --> HeatmapCostsTasksSuppliers(costs): one figure (heatmap plot) that shows 
the cost values as a matrix of tasks (rows) and suppliers (columns)
"""
def HeatmapCostsTasksSuppliers(costs):
    
    costs_order = costs.sort_values(by=['Supplier ID','Task ID'])
    costs_order['Task ID'] = pd.to_datetime(costs_order['Task ID']).dt.date
    costs_order_pivot = pd.pivot_table(costs_order, index = 'Supplier ID', values = 'Cost', columns = 'Task ID')
    costs_order_pivot.sort_index(key = lambda x: (x.to_series().str[1:].astype(int)), inplace = True)
    plt.figure(figsize = (3, 3))
    plt.title('Costs as a matrix of Tasks and Suppliers', fontsize = 30)
    sns.heatmap(costs_order_pivot.T, linewidth = 0.5, cmap='Blues')
    









