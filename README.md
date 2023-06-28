# Supplier Selection using Ridge Regression and Support Vector Regression with Radial Basis Function ('rbf') Kernel

Developed as a group project for the program "Business Analytics: Operational Research and Risk Analysis" at the Alliance Manchester Business School.


# Project Overview

The objective of this project is the development of a predictive model that selects the most cost efficient supplier for a given task. The main metric that was used to assess the performance of the ML models developed, was the Root Mean Squared Error (RMSE). RMSE (Equation 2) is found after the calculation of the selection errors made by each ML model, as seen in Equation 1.

Equation 1:

![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/128f695f-453b-40e2-8237-e5abf5c0e8cb)

where 𝑡 is a task, 𝑆 is the set of 64 suppliers, 𝑠𝑡′ is the supplier chosen by the ML model for task 𝑡, and 𝑐(𝑠,𝑡) is the cost if task 𝑡 is performed by supplier 𝑠. That is, the Error is the difference in cost between the supplier selected by the ML model and the actual best supplier for this task.

Equation 2:

![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/c1cbb87e-ee53-4340-97fe-993c47e591d4)

where 𝑇 is a set of tasks and Error(t) the error computed from equation 1.


This repository contains a summary of this work, starting with data preparation, followed by the exploratory data analysis and the development of two machine learning models, before choosing the most suitable one. Lastly, the results will be presented, followed by the final conclusions and recommendations. 


# Installation and Setup

## Codes and Resources Used
- **Editor Used:**  Spyder
- **Python Version:** 3.10.9

## Python Packages Used

- **General Purpose:** `time, math`
- **Data Manipulation:** `pandas, numpy`
- **Data Visualization:** `seaborn, matplotlib` 
- **Machine Learning:** `scikit-learn`

# Data


## Source Data

- `tasks.xlsx`: an Excel file that contains one row per task and one column per task feature (TF1, TF2, TF3, …, T116). Each task is uniquely identified by a Task ID (a date, eg.,‘2019 05 30’).
- `suppliers.csv`: A CSV file that contains one row per supplier and one column per supplier feature (SF1, SF2, …, SF18). Each supplier is uniquely identified by a Supplier ID given in the first column of the file (S1, …, S64).
- `costs.csv`: a CSV file that contains data collected and/or estimated by Acme about the cost of a task when performed by each supplier. Each row gives the cost value (in millions of dollars, M$) of one task performed by one supplier. 

## Data Preparation

- Check for missing values and ID match
- Remove incomplete observations
- Remove task features with variances less than 0.01 (34 were removed)
- Scaling of features on a scale of -1 to 1 (MinMaxScaler)
- Remove highly correlated task features with correlation over 0.8 (Multicollinearity issue)

# Code structure

The code is written in 4 `.py` files. `DataPreparation.py`, `EDA.py`, `MachineLearning.py` files contain code related to all the necessary functions that were developed for each part. In the `Main.py` file, those 3 `.py` files are imported so that the functions can be used and the final results can be seen.

## How to run

The `.py` files should be executed in the following order:
1. `DataPreparation.py`
2. `EDA.py`
3. `MachineLearning.py`
4. `Main.py`

As long as all `.py` files and datasets (`tasks.xlsx`,`cost.csv`,`suppliers.csv`) are in the same directory, the `Main.py` file can generate all the results.


# Results and evaluation

## Exploratory Data Analysis
### Figure 1
![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/a61f8b8f-353c-4b50-8ebc-fa99ba59b6de)

Figure 1 shows the distribution of task feature values for all tasks. A significant amount of variability is observed in the boxplots along with the discrepancies in median values for all the task features. It is apparent that tasks are dissimilar in terms of features.
### Figure 2
![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/a64b88f2-ff59-4730-991d-4c3fcf339d88)

Figure 2 shows the distribution of errors for each supplier if selected to perform each task, and the RMSE generated from those errors. If Suppliers with a low RMSE are chosen to perform all tasks, the cost for not choosing the optimal supplier each time will be relatively low. Apparently, the supplier with the lowest cost due to errors is Supplier 56, with an RMSE of 0.025. Supplier 56 can be used as a benchmark for the ML models.
### Figure 3
![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/f4c64f94-1e1b-4fb8-94bb-f02f38cb0acc)

Figure 3 shows the cost values for all suppliers for each task. A consistent pattern of cost values for most of the suppliers can be observed. Tasks between Task ID 2020-03-03 and Task ID 2020-10-30 appear to be a group of expensive Tasks as the darker color in the figure indicates, while those between 2021-03-15 and 2021-05-13 appeared to be a relatively cheap group. Suppliers 1 and 3 appeared to be more expensive compared to the other suppliers. Finally, the heatmap suggests that Task ID 2021-11-05 has the highest cost of execution across the board for all suppliers.

## Machine Learning Models

### Tidy dataset

The ML models were developed using the following final dataframe, where each task appears multiple times – once for each supplier – formulating a group. Each group corresponds to a specific task and includes its feature values, all the possible suppliers (with their features), and their cost to perform the task.

![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/d7b89d32-c819-4ad1-ae97-c128d10e6f84)

### General Methodology

For each model, R-squared scores were calculated using the default scoring function of `scikit-learn` for a random train–test split (20 task groups out of 120 as a test set). Since a random split cannot be indicative of the performance of a model (uncertainty about the information in the train dataset), cross-validation was performed with Equation 1 being the scoring function (Errors) and the RMSE as a metric this time. Cross–validation reduces bias and variance as most of the data is being used for fitting and as a validation set. The `Leave-One-Group-Out` method was used for the cross-validation, leaving one group of tasks out at a time for validation purposes.

The previous procedures were done for the default hyper-parameters of each model. Thus, as the last step, the `Grid Search` method (`scikit-learn`) was used for the hyper-parameter tuning of each model. The same approach regarding the scoring function (Equation 1), the performance metric (Equation 2), and the type of cross-validation (Leave-One-Group-Out) was followed for the hyper-parameter optimization. Regarding the search space of the hyperparameters, an iterative hill climb approach was used. This allowed for efficiently searching over a large range of values and quickly "zooming in" on the optimal values. Doing it in iterations saved time and computational power, where the first iteration was used to see what kind of values were favoured, then in the next iteration “zooming in” around the values chosen in the previous iteration. Because the aim was to reduce the RMSE value, a function was implemented to double-check that the parameters chosen by GridSearch (ranking based on mean error scores) were the same as when chosen based on RMSE. 

Based on the above, the most reliable scores were obtained after hyper-parameter optimization, thus only those are presented in the folowing sections for both models.

### Ridge Regression - Results

For Ridge Regression, RMSE did not change after hyper-parameter tuning and it was the same with the default ones too. The achieved RMSE of 0.0401 is still higher than Supplier 56 (Figure 2 - EDA), thus this model might not be suitable for the problem at hand.

![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/41aa284f-68ab-4ed3-b3a5-68c07b59f182)

![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/774261e1-ab37-45c8-bbcb-47d65f459a7d)


### Support Vector Regression (kernel = 'rbf') - Results
For SVR, after only one iteration of GridSearch, the RMSE improved a lot compared to the model with default hyper-parameters. The difference between iteration 1 and 2 was so little that it did not seem to justify a 3rd iteration. In comparison to the RMSE of the Supplier 56 (0.0256) if chosen to do all tasks, the final RMSE of the model (0.0269) is acceptable.

![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/cb464e52-d263-49f6-a4af-7e920683241b)

![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/23250a14-cfcb-40c9-abe5-3a33d0bea8a5)


# Conclusion

The SVR model outperforms Ridge Regression for choosing the best supplier for a given task, which is clearly shown by a drastically better RMSE value. The cost related to false predictions of the model, is nearly as low as the one if Supplier 56 – cheap for most tasks - is selected every time. Therefore, SVR (kernel = ‘rbf’, C = 0.79, Epsilon = 0.0005) is a sufficient and robust model that given the task features of a task, can predict the least expensive supplier.

![image](https://github.com/DSkapinakis/supplier-selection-ML-models/assets/136902596/f723e561-5b0a-4baf-aedf-61288bd8a069)




