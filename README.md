# Zillow Log Error Prediction Project
--------------
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a>
    <li><a href="#summary">Summary</a></li>
    <li><a href="#preliminary-questions">Questions</a></li>
    <li><a href="#planning">Planning</a></li>
    <li><a href="#data-dictionary">Data Dictionary</a></li>
    <li><a href="#Key-Findings-and-Takeaways">Key Findings and Takeaways</a></li>
    <li><a href="#recommendations">Recommendations</a></li>
    <li><a href="#additional-improvements">Additional Improvements</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#how-to-reproduce">How to Reproduce</a></li>
  </ol>
</details>
    
## About/Goals:
- The goal of this project is to find features or clusters of features to improve Zillow's log error for single family residences in three Southern California counties (Los Angeles, Orange, and Ventura) and to use these features to develop an improved machine learning model.

## Summary:
- After running four models on train and validate, the 2nd degree Polynomial Linear Regression model provided the lowest Root Mean Square Error (RMSE) compared to the baseline.
- We used the square footage of the home, ratio of bedrooms and bathrooms, lot size, age of the home, number of bathrooms, area cluster, and size cluster to predict logerror.. We selected a degree multiplier of 2. The RMSE of the selected model was 0.162 on train, 0.143 on validate, and 0.174 on test.
- The selected features had little impact in improving the overall prediction of log error when compared against baseline. The clusters did not significantly reduce the RMSE, but there was a very small improvement when using the absolute value for the log error. Overall, none of the models significantly outperformed Zillow's current model.

## Preliminary Questions:
1. What is the relationship between square feet and log error? 
2. Do area clusters have a large impact on the overall log error?
3. Does the size of the home affect log error? Can that error be better determined by clustering by size?
4. Does the location have an effect on log error? Where does the most log error occur?

## Initial Hypothesis:
Our initial hypothesis is that the size of the home in square feet, the age of the home, and the location are the main features affecting log error.

# Planning:
![image](https://user-images.githubusercontent.com/98612085/191094469-0c50c67a-d7e1-4711-9eb8-06261a8f10bb.png)

## Data Dictionary:

Zillow Data Features:

<img width="1189" alt="image" src="https://user-images.githubusercontent.com/98612085/191111788-4b487818-535c-43d2-8f16-c3888fc7c706.png">

<img width="1140" alt="image" src="https://user-images.githubusercontent.com/98612085/191111686-21325882-4258-4ec5-9eef-44dec2e681cd.png">

## Key Findings and Takeaways:
- After running four models on our train and validate sets, we decided to use the polynomial linear regression model because it provided the lowest RMSE compared to baseline.

- The RMSE of the selected model was .162 on train, .143 on validate, and .174 on test.

- The selected features improved the overall log error, but not much more than baseline. The clusters did not significantly reduce the RMSE, but there was a very small improvement when using the absolute value for the log error. Overall, none of the models significantly outperformed Zillow's current model.

#### Insights:
- Different combinations of features didn’t seem to have significant improvement on model performance
- All models performed close to baseline

#### Best predictor features, Using visualization and statistical testing:
1. scaled_calculatedfinishedsquarefeet
2. scaled_bathroomcnt
3. scaled_age
4. size_cluster
5. area_cluster

#### Model - The 2nd Degree Polynomial Regression model had the best performance.

- Train: 
  - 0.162 RMSE
  - 0.0174 R-squared value
- Validate: 
  - 0.143 RMSE
  - 0.002 R-squared value
- Test: 
  - 0.174 RMSE
  - 0.0023 R-squared value

#### (For Comparison)
    - Train RMSE (Mean): 0.179
    - Train RMSE (Median): 0.179
  
## Recommendations:
- We discourage the use of these models for predicting log error as they had little efficacy.

- All of the models performed very close to baseline, indicating that the new models with the selected features do not outperform Zillow’s current model. 

- Focus efforts on other methods or model types to improve model performance. 

## Additional Improvements:
- It could be valuable to remove additional outlier log error data and train models to see how much those outliers affected the previous model. We would also see if clustering with log error as a feature can tell us more about why the model cannot accurately predict certain home values.

## Contact:
Dashiell Bringhurst - dashbringhurst@gmail.com

Everett Clark - everett.clark.t@gmail.com

## How to Reproduce:
**In order to reproduce this project, you will need server credentials to the Codeup database or a .csv of the data**

#### Step 1 (Imports):  
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt
- import scipy.stats as stats
- import seaborn as sns
- import os
- from math import sqrt
- import env
- import wrangle
- import model
- import explore
- from sklearn.cluster import KMeans
- from sklearn.preprocessing import MinMaxScaler, RobustScaler
- from sklearn.model_selection import train_test_split
- from sklearn.feature_selection import SelectKBest, f_regression 
- from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
- from sklearn.preprocessing import PolynomialFeatures
- from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
- import warnings
- warnings.filterwarnings('ignore')

#### Step 2 (Acquisition):  
- Acquire the database information from Codeup using a SQL query, which is saved in wrangle.py and has the necessary functions to acquire, prepare, and split the dataset.

#### Step 3 (Preparation):  
- Prepare and split the dataset. The code for these steps can be found in the wrangle.py file within this repository. All random state or seeds used is 217.

#### Step 4 (Exploration):
- Use pandas to explore the dataframe and scipy.stats to conduct statistical testing on selected features.
- Use seaborn or matplotlib.pyplot to create visualizations.
- Conduct a univariate analysis on each feature using barplot for categorical variables and .hist for continuous variables.
- Conduct a bivariate analysis of each feature against logerror and graph each finding.
- Conduct multivariate analyses of the most important features against logerror and graph the results.

#### Step 5 (Modeling):
- Create models (OLS regression, LassoLars, 2nd Degree, Polynomial Regression, TweedieRegressor) with the most important selected features using sklearn.
- Train each model and evaluate its accuracy on both the train and validate sets.
- Select the best performing model and use it on the test set.
- Graph the results of the test using probabilities.
- Document each step of the process and your findings.

[[Back to top](#top)]

