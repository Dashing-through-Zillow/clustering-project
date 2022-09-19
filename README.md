# # Zillow Log Error Prediction Project
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
    
# About/Goals:
- 

# Summary:
- 

# Preliminary Questions:
1.
2.
3.
4.

# Planning:
![image](https://user-images.githubusercontent.com/98612085/191094469-0c50c67a-d7e1-4711-9eb8-06261a8f10bb.png)

# Data Dictionary:

Zillow Data Features:

<img width="1187" alt="image" src="https://user-images.githubusercontent.com/98612085/189556700-b97d7450-bafa-47f8-81a5-10377050600a.png">

<img width="1062" alt="image" src="https://user-images.githubusercontent.com/98612085/189556238-f433cb25-1158-4a29-91bd-4f2b9dc58c55.png">

# Key Findings and Takeaways:

### Insights:
- 

### Best predictor features, Using Recursive Feature Elimination (RFE):
1. 

### Model - The --- model had the best performance.

- Train: 
  - 0.-- RMSE
  - 0.-- R-squared value
- Validate: 
  - 0.-- RMSE
  - 0.-- R-squared value
- Test: 
  - 0.-- RMSE
  - 0.-- R-squared value
  
# Recommendations:
- 

# Additional Improvements:
- 

# Contact:
Dashiell Bringhurst - dashbringhurst@gmail.com

Everett Clark - everett.clark.t@gmail.com

# How to Reproduce:
**In order to reproduce this project, you will need server credentials to the Codeup database or a .csv of the data**

#### Step 1 (Imports):  
    - import pandas as pd
    - import numpy as np
    - import wrangle
    - from math import sqrt
    - import matplotlib.pyplot as plt
    - import seaborn as sns
    - from scipy import stats
    - from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
    - from sklearn.feature_selection import SelectKBest, f_regression 
    - from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
    - from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor

#### Step 2 (Acquisition):  
- Acquire the database information from Codeup using a SQL query, which is saved in wrangle.py and has the necessary functions to acquire, prepare, and split the dataset.

#### Step 3 (Preparation):  
- Prepare and split the dataset. The code for these steps can be found in the wrangle.py file within this repository.

#### Step 4 (Exploration):
- Use pandas to explore the dataframe and scipy.stats to conduct statistical testing on selected features.
- Use seaborn or matplotlib.pyplot to create visualizations.
- Conduct a univariate analysis on each feature using barplot for categorical variables and .hist for continuous variables.
- Conduct a bivariate analysis of each feature against logerror and graph each finding.
- Conduct multivariate analyses of the most important features against logerror and graph the results.

#### Step 5 (Modeling):
- Create models (OLS regression, LassoLars, TweedieRegressor, ---) with the most important selected features using sklearn.
- Train each model and evaluate its accuracy on both the train and validate sets.
- Select the best performing model and use it on the test set.
- Graph the results of the test using probabilities.
- Document each step of the process and your findings.

[[Back to top](#top)]


