import pandas as pd
import numpy as np
import wrangle
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import env
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression 
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

# create a function to print the regression errors for the model
def regression_errors(y, yhat):
    '''This function takes in two arguments, a previously assigned target variable (y) and the model predictions 
    (yhat). The function calculates the sum of squares error, explained sum of squares, total sum of squares,
    mean squared error, and root mean squared error. It prints strings for each value to the first decimal.'''
    # calculate the sum of squares error from the selected variables
    SSE = mean_squared_error(y, yhat)*len([y])
    # calculate the explained sum of squares from the predictions and the baseline
    ESS = sum((yhat - y.mean())**2)
    # calculate the total sum of squares
    TSS = ESS + SSE
    # calculate the mean squared error
    MSE = mean_squared_error(y, yhat)
    # calculate the root mean squared error
    RMSE = sqrt(mean_squared_error(y, yhat))
    # print the calculated values for each
    print(f'Model SSE is: {"{:.1f}".format(SSE)}')
    print(f'Model ESS is: {"{:.1f}".format(ESS)}')
    print(f'Model TSS is: {"{:.1f}".format(TSS)}')
    print(f'Model MSE is: {"{:.1f}".format(MSE)}')
    print(f'Model RMSE is: {"{:.1f}".format(RMSE)}')

# create a function to print the baseline mean errors
def baseline_mean_errors(y, baseline):
    '''This function takes in a single argument, y (the target variable) and prints the sum of squares
    error, mean squared error, and root mean squared error for baseline.'''
    # calculate the baseline sum of squares error
    SSE_baseline = mean_squared_error(y, baseline)*len(y)
    # calculate the baseline mean squared error
    MSE_baseline = mean_squared_error(y, baseline)
    # calculate the baseline root mean squared error
    RMSE_baseline = sqrt(mean_squared_error(y, baseline))
    # print the calculated values for each baseline error
    print(f'SSE baseline: {"{:.1f}".format(SSE_baseline)}')
    print(f'MSE baseline: {"{:.1f}".format(MSE_baseline)}')
    print(f'RMSE baseline: {"{:.1f}".format(RMSE_baseline)}')

# create a function to determine if the model performs better than baseline
def better_than_baseline(y, yhat, baseline):
    '''This function takes in two arguments, y (target variable) and yhat (model predictions) and calculates the 
    model SSE, MSE, and RMSE against the baseline. The function prints three strings, one for each result, with a
    boolean for whether or not the model value is better than baseline value.'''
    SSE = mean_squared_error(y, yhat)*len(y)
    SSE_baseline = mean_squared_error(y, baseline)*len(y)
    MSE = mean_squared_error(y, yhat)
    MSE_baseline = mean_squared_error(y, baseline)
    RMSE = sqrt(mean_squared_error(y, yhat))
    RMSE_baseline = sqrt(mean_squared_error(y, baseline))
    # print whether or not the model performs better than baseline
    print(f'Model SSE is better than SSE baseline: {SSE < SSE_baseline}')
    print(f'Model MSE is better than MSE baseline: {MSE < MSE_baseline}')
    print(f'Model RMSE is better than RMSE baseline: {RMSE < RMSE_baseline}')

def plot_residuals(y, yhat):
    '''This function takes in two arguments, y (target variable) and yhat (model predictions) and returns a 
    scatterplot of the residuals of the target variable.'''
    # calculate residuals and assign to variable
    residuals = y - yhat
    # create a scatterplot of the residuals
    plt.scatterplot(x=y, y=residuals)
    # label the x-axis
    plt.xlabel('Home Value')
    # label the y-axis
    plt.ylabel('Residuals')
    # make a title for the scatterplot
    plt.title('Residuals for Home Value')
    plt.show()

def lasso_lars_model(X_train_scaled, X_validate_scaled, y_train, y_validate, a):
    '''This function takes in six arguments (X_train_scaled, X_validate_scaled, y_train, y_validate, train, and the
    desired alpha. The function initiates a Lasso Lars model object and fits the train dataset to the model. The function
    predicts values for the target variable and outputs the name of the model, the rmse for train, the rmse for validate, 
    the r2 score for the model, whether or not the model is better than baseline, and the difference between the baseline
    RMSE and the model RMSE.'''
    # create the model object
    lars = LassoLars(alpha=a)
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train_scaled, y_train.logerror)
    # predict train
    y_train['yhat'] = lars.predict(X_train_scaled)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.yhat)**(1/2)
    # predict validate
    y_validate['yhat'] = lars.predict(X_validate_scaled)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.yhat)**(1/2)
    # calculate r2 score on train and save to a variable
    r2 = r2_score(y_train.logerror, y_train.yhat)
    # calculate the RMSE for train
    RMSE = sqrt(mean_squared_error(y_train.logerror, y_train.yhat))
    # calculate the baseline RMSE
    RMSE_baseline = sqrt(mean_squared_error(y_train.logerror, X_train_scaled.baseline))
    # determine whether the model performs above baseline (lower RMSE)
    better =  RMSE < RMSE_baseline
    # calculate the difference between the model RMSE and baseline RMSE
    difference = RMSE_baseline - RMSE
    # output saved values as a list to be entered into a dataframe later
    return ['Lasso Lars', rmse_train, rmse_validate, r2, better, difference]

def glm_model(X_train_scaled, X_validate_scaled, y_train, y_validate, p, a):
    '''This function takes in seven arguments (X_train_scaled, X_validate_scaled, y_train, y_validate, train, the desired 
    power, and the desired alpha. The function initiates a Tweedie Regressor model object and fits the train dataset to 
    the model. The function predicts values for the target variable and outputs the name of the model, the rmse for train, 
    the rmse for validate, the r2 score for the model, whether or not the model is better than baseline, and the difference 
    between the baseline RMSE and the model RMSE.'''
    # create the model object
    glm = TweedieRegressor(power=p, alpha=a)
    # fit the model to the training data
    glm.fit(X_train_scaled, y_train.logerror)
    # predict train
    y_train['yhat'] = glm.predict(X_train_scaled)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.yhat)**(1/2)
    # predict validate
    y_validate['yhat'] = glm.predict(X_validate_scaled)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.yhat)**(1/2)
    # calculate the r2 score for the model
    r2 = r2_score(y_train.logerror, y_train.yhat)
    # calculate the RMSE for the model
    RMSE = sqrt(mean_squared_error(y_train.logerror, y_train.yhat))
    # calculate the baseline RMSE
    RMSE_baseline = sqrt(mean_squared_error(y_train.logerror, X_train_scaled.baseline))
    # determine whether the model performs better than baseline for RMSE
    better =  RMSE < RMSE_baseline
    # calculate the difference between the model RMSE and baseline RMSE
    difference = RMSE_baseline - RMSE
    # output saved values as a list to be entered into a dataframe later
    return ['Tweedie Regressor', rmse_train, rmse_validate, r2, better, difference]


def poly_lm(X_train_scaled, X_validate_scaled, y_train, y_validate, d):
    '''This function takes in seven arguments (X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, 
    train, and the desired degree. The function initiates a polynomial features linear regression model object and fits 
    the train dataset to the model. The function predicts values for the target variable and outputs the name of the model, 
    the rmse for train, the rmse for validate, the r2 score for the model, whether or not the model is better than baseline, 
    and the difference between the baseline RMSE and the model RMSE.'''
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=d)
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled)
    # transform X_validate_scaled
    X_validate_degree2 = pf.transform(X_validate_scaled)
    # create the model object
    lm2 = LinearRegression()
    # fit the model to the training data
    lm2.fit(X_train_degree2, y_train.logerror)
    # predict train
    y_train['yhat'] = lm2.predict(X_train_degree2)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.yhat)**(1/2)
    # predict validate
    y_validate['yhat'] = lm2.predict(X_validate_degree2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.yhat)**(1/2)
    # calculate the r2 score for the model
    r2 = r2_score(y_train.logerror, y_train.yhat)
    # calculate the model's RMSE
    RMSE = sqrt(mean_squared_error(y_train.logerror, y_train.yhat))
    # calculate the baseline RMSE
    RMSE_baseline = sqrt(mean_squared_error(y_train.logerror, X_train_scaled.baseline))
    # determine whether the model performs better than baseline
    better =  RMSE < RMSE_baseline
    # calculate the difference between the model RMSE and baseline RMSE
    difference = RMSE_baseline - RMSE
    # output saved values as a list to be entered into a dataframe later
    return ["Poly Linear Regression", rmse_train, rmse_validate, r2, better, difference]

def lrm(X_train_scaled, X_validate_scaled, y_train, y_validate):
    '''This function takes in five arguments (X_train_scaled, X_validate_scaled, y_train, y_validate, and train. The 
    function initiates a linear regression model object and fits the train dataset to the model. The function
    predicts values for the target variable and outputs the name of the model, the rmse for train, the rmse for validate, 
    the r2 score for the model, whether or not the model is better than baseline, and the difference between the baseline
    RMSE and the model RMSE.'''
    # create the model object
    lm = LinearRegression()
    # fit the model to our training data. 
    lm.fit(X_train_scaled, y_train.logerror)
    # predict train
    y_train['yhat'] = lm.predict(X_train_scaled)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.yhat)**(1/2)
    # predict validate
    y_validate['yhat'] = lm.predict(X_validate_scaled)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.yhat)**(1/2)
    # calculate the model's r2 score on the train data
    r2 = r2_score(y_train.logerror, y_train.yhat)
    # calculate the model's RMSE
    RMSE = sqrt(mean_squared_error(y_train.logerror, y_train.yhat))
    # calculate the baseline RMSE
    RMSE_baseline = sqrt(mean_squared_error(y_train.logerror, X_train_scaled.baseline))
    # determine whether the model performs better than baseline
    better =  RMSE < RMSE_baseline
    # calculate the difference between the model RMSE and baseline RMSE
    difference = RMSE_baseline - RMSE
    # output the saved values as a list to be entered into a dataframe later
    return ["OLS", rmse_train, rmse_validate, r2, better, difference]

def model_performance(m1, m2, m3, m4):
    '''This function takes in the four arguments, the list of values from the statistical test functions, and enters
    the values into a dataframe. The function returns a dataframe showing the results for each selected model.'''
    # convert the separate lists into a single dataframe
    df = pd.DataFrame([m1,m2,m3,m4])
    # rename the columns for readability
    df = df.rename(columns={0:'Model', 1:'Train RMSE', 2:'Validate RMSE', 3:'r2 score', 4:'Better than Baseline',
                            5:'RMSE Difference'})
    # return a dataframe of the model results
    return df.round(4)

def test_poly_lm(X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test, d):
    '''This function takes in seven arguments (X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, 
    train, and the desired degree. The function initiates a polynomial features linear regression model object and fits 
    the train dataset to the model. The function predicts values for the target variable and outputs the a graph of the 
    model's performance on the test set, the rmse for test, the r2 score for the model on test.'''
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=d)
    # fit and transform X_train_scaled
    X_train_degree3 = pf.fit_transform(X_train_scaled)
    # transform X_validate_scaled & X_test_scaled
    X_validate_degree3 = pf.transform(X_validate_scaled)
    X_test_degree3 = pf.transform(X_test_scaled)
    # create the model object
    lm2 = LinearRegression()
    # fit the model to our training data
    lm2.fit(X_train_degree3, y_train.logerror)
    # predict train
    y_train['yhat'] = lm2.predict(X_train_degree3)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.yhat)**(1/2)
    # predict validate
    y_validate['yhat'] = lm2.predict(X_validate_degree3)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.yhat)**(1/2)
    # create a new column for the model's predictions on the test set
    y_test['yhat'] = lm2.predict(X_test_degree3)
    # test rmse
    rmse_test = mean_squared_error(y_test.logerror, y_test.yhat)**(1/2)
    # calculate the model r2 score for train
    r2 = r2_score(y_test.logerror, y_test.yhat)
    # calculate the RMSE for the test set
    RMSE = sqrt(mean_squared_error(y_test.logerror, y_test.yhat))
    # calculate the baseline RMSE from the train set
    RMSE_baseline = sqrt(mean_squared_error(y_train.logerror, X_train_scaled.baseline))
    # determine whether the model performs better than baseline on test
    better =  RMSE < RMSE_baseline
    # plot the model's performance
    plt.figure(figsize=(16,8))
    #plt.plot(y_test.logerror, y_test.yhat, alpha=.3, color="grey")
    #plt.annotate("Baseline: Predict Using Mean", (16, 9.5))
    plt.hlines(y=0, xmin=-3, xmax=3, alpha=.5, color="blue", label='Ideal Performance')
    plt.vlines(x=0, ymin=-.02, ymax=.08, alpha=.5, color='blue')
    plt.annotate("The Ideal Line: No Error", (.5, 3.5), rotation=15.5)
    plt.scatter(y_test.logerror, y_test.yhat, 
            alpha=.5, color="red", s=10)
    plt.xlabel("Actual Log Error")
    plt.ylabel("Predicted Log Error")
    plt.title("The Performance of Polynomial Model")
    plt.legend()
    plt.show()
    # output the RMSE for the test set and the r2 score for test
    return rmse_test, r2
