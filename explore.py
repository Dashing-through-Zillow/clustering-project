import pandas as pd
import numpy as np
import wrangle
import os
import model
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn.model_selection import train_test_split
import env
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression 
import warnings
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)

    return df_dummies

def verify_counties(df):
    print("LA County Verified: ", df['fips'][df.fips==6037].count() == df.LA.sum())
    print("Orange County Verified: ", df['fips'][df.fips==6059].count() == df.Orange.sum())
    print("Ventura County Verified: ", df['fips'][df.fips==6111].count() == df.Ventura.sum())

def create_features(df):
    df['age'] = 2017 - df.yearbuilt
    df['age_bin'] = pd.cut(df.age, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
                           labels = [0, .066, .133, .20, .266, .333, .40, .466, .533, 
                                     .60, .666, .733, .8, .866, .933])
    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100
    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560
    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200], 
                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    # square feet bin
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet, 
                            bins = [0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000, 12000],
                            labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                       )
    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet

    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, 
                                             bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 1500],
                                             labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                            )
    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet

    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins = [0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000],
                                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                      )
    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',
                    'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})
    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    return df

def remove_outliers(df):
    '''
    remove outliers in bed, bath, zip, square feet, acres & tax rate
    '''

    return df[((df.bathroomcnt <= 7) & (df.bedroomcnt <= 7) & 
               (df.regionidzip < 100000) & 
               (df.bathroomcnt > 0) & 
               (df.bedroomcnt > 0) & 
               (df.acres < 20) &
               (df.calculatedfinishedsquarefeet < 10000) & 
               (df.taxrate < 30)
              )]

def split(df, target_var):
    '''
    This function takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train, validate, & test.
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state=217)
    # split train_validate into train and validate
    train, validate = train_test_split(train_validate, test_size=.25, random_state=217)

    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]

    return [train, X_train, X_validate, X_test, y_train, y_validate, y_test]

def spearman_test(x,y):
    '''This function takes in two arguments and performs a Spearman's statistical test. It prints whether or not
    we can reject the null hypothesis and returns the coefficient and p-value for the test.'''
    # run the stat test using the two arguments and assign results to variables
    corr, p = stats.spearmanr(x,y)
    # set alpha to .05
    alpha = .05
    # conditional clause that prints whether to accept or reject the null hypothesis
    if p < alpha:
        print('We reject the null hypothesis.')
    else:
        print('We fail to reject the null hypothesis.')
    # output the Spearman coefficient and p-value
    return corr, p

def fit_scale_and_concat(df, X_train):
    # the variables that still need scaling
    scaled_vars = ['latitude', 'longitude', 'bathroomcnt', 'taxrate', 'calculatedfinishedsquarefeet', 'age']
    # create new column names for the scaled variables by adding 'scaled_' to the beginning of each variable name 
    scaled_column_names = ['scaled_' + i for i in scaled_vars]
    # fit the minmaxscaler to X_train
    scaler = MinMaxScaler(copy=True).fit(X_train[scaled_vars])
    # transform scaled_vars and concatenate to df
    scaled_array = scaler.transform(df[scaled_vars])
    scaled_df = pd.DataFrame(scaled_array, columns=scaled_column_names, index=df.index.values)
    return pd.concat((df, scaled_df), axis=1)

def find_k(X_train, cluster_vars, k_range):
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)

        # X[0] is our X_train dataframe..the first dataframe in the list of dataframes stored in X. 
        kmeans.fit(X_train[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
                             sse=sse[0:-1]))

    # plot k with inertia
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
    plt.show()

    return k_comparisons_df

def create_clusters(X_train, k, cluster_vars):
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 217)

    # fit to train and assign cluster ids to observations
    kmeans.fit(X_train[cluster_vars])

    return kmeans

# get the centroids for each distinct cluster...

def get_centroids(kmeans, cluster_vars, cluster_name):
    # get the centroids for each distinct cluster...

    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroid_df = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=centroid_col_names).reset_index().rename(columns={'index': cluster_name})

    return centroid_df

# label cluster for each observation

def assign_clusters(df, kmeans, cluster_vars, cluster_name, centroid_df):
    clusters = pd.DataFrame(kmeans.predict(df[cluster_vars]), 
                            columns=[cluster_name], index=df.index)

    clusters_centroids = clusters.merge(centroid_df, on=cluster_name, copy=False).set_index(clusters.index.values)

    df = pd.concat([df, clusters_centroids], axis=1)
    return df

def plot_age_sqft(X_train):
    plt.figure(figsize=(20,20))
    # plt.scatter(y=X_train.latitude, x=X_train.longitude, c=X_train.area_cluster, alpha=.4)
    plt.subplot(3,2,1)
    plt.scatter(x=X_train[X_train.area_cluster==0].age, y=X_train[X_train.area_cluster==0].calculatedfinishedsquarefeet, alpha=.4,
                c=X_train[X_train.area_cluster==0].fips)
    plt.xlabel('Age of Home')
    plt.ylabel('Square Feet of Home')
    plt.title('Area Cluster 0')
    plt.subplot(3,2,2)
    plt.scatter(x=X_train[X_train.area_cluster==1].age, y=X_train[X_train.area_cluster==1].calculatedfinishedsquarefeet, alpha=.4, 
                c=X_train[X_train.area_cluster==1].fips)
    plt.xlabel('Age of Home')
    plt.ylabel('Square Feet of Home')
    plt.title('Area Cluster 1')
    plt.subplot(3,2,3)
    plt.scatter(x=X_train[X_train.area_cluster==2].age, y=X_train[X_train.area_cluster==2].calculatedfinishedsquarefeet, alpha=.4, 
                c=X_train[X_train.area_cluster==2].fips)
    plt.xlabel('Age of Home')
    plt.ylabel('Square Feet of Home')
    plt.title('Area Cluster 2')
    plt.subplot(3,2,4)
    plt.scatter(x=X_train[X_train.area_cluster==3].age, y=X_train[X_train.area_cluster==3].calculatedfinishedsquarefeet, alpha=.4, 
                c=X_train[X_train.area_cluster==3].fips)
    plt.xlabel('Age of Home')
    plt.ylabel('Square Feet of Home')
    plt.title('Area Cluster 3')
    plt.subplot(3,2,5)
    plt.scatter(x=X_train[X_train.area_cluster==4].age, y=X_train[X_train.area_cluster==4].calculatedfinishedsquarefeet, alpha=.4, 
                c=X_train[X_train.area_cluster==4].fips)
    plt.xlabel('Age of Home')
    plt.ylabel('Square Feet of Home')
    plt.title('Area Cluster 4')

    plt.subplot(3,2,6)
    plt.scatter(x=X_train.age, y=X_train.calculatedfinishedsquarefeet, c=X_train.fips, alpha=.4)
    plt.xlabel('Age of Home')
    plt.ylabel('Square Feet of Home')
    plt.title('All Areas')

    plt.suptitle('Do area clusters reveal differences in age, location, and size?', y=.91)
    plt.show()

def plot_age_error(X_train, y_train):
    plt.figure(figsize=[16,16])
    plt.subplot(3,2,1)
    plt.scatter(y=y_train[X_train.area_cluster==0].logerror, x=X_train[X_train.area_cluster==0].age, alpha=.4, 
                c=X_train[X_train.area_cluster==0].size_cluster)
    plt.xlabel('Age of Property')
    plt.ylabel('Log Error')
    plt.hlines(y=0, xmin=0, xmax=140, color='red')
    plt.title('Cluster 0')

    plt.subplot(3,2,2)
    plt.scatter(y=y_train[X_train.area_cluster==1].logerror, x=X_train[X_train.area_cluster==1].age, alpha=.4, 
                c=X_train[X_train.area_cluster==1].fips)
    plt.xlabel('Age of Property')
    plt.ylabel('Log Error')
    plt.hlines(y=0, xmin=0, xmax=140, color='red')
    plt.title('Cluster 1')

    plt.subplot(3,2,3)
    plt.scatter(y=y_train[X_train.area_cluster==2].logerror, x=X_train[X_train.area_cluster==2].age, alpha=.4, 
                c=X_train[X_train.area_cluster==2].fips)
    plt.xlabel('Age of Property')
    plt.ylabel('Log Error')
    plt.hlines(y=0, xmin=0, xmax=140, color='red')
    plt.title('Cluster 2')

    plt.subplot(3,2,4)
    plt.scatter(y=y_train[X_train.area_cluster==3].logerror, x=X_train[X_train.area_cluster==3].age, alpha=.4, 
                c=X_train[X_train.area_cluster==3].fips)
    plt.xlabel('Age of Property')
    plt.ylabel('Log Error')
    plt.hlines(y=0, xmin=0, xmax=140, color='red')
    plt.title('Cluster 3')

    plt.subplot(3,2,5)
    plt.scatter(y=y_train[X_train.area_cluster==4].logerror, x=X_train[X_train.area_cluster==4].age, alpha=.4, 
                c=X_train[X_train.area_cluster==4].fips)
    plt.xlabel('Age of Property')
    plt.ylabel('Log Error')
    plt.hlines(y=0, xmin=0, xmax=140, color='red')
    plt.title('Cluster 4')

    plt.subplot(3,2,6)
    plt.scatter(y=y_train.logerror, x=X_train.age, alpha=.4, 
                c=X_train.fips)
    plt.xlabel('Age of Property')
    plt.ylabel('Log Error')
    plt.hlines(y=0, xmin=0, xmax=140, color='red')
    plt.title('All Areas')

    plt.suptitle("Do clusters reveal differences in age and error?", y=.91)
    plt.show()

def plot_size_error(X_train, y_train):
    plt.figure(figsize=(20,20))

    plt.subplot(3,2,1)
    plt.scatter(y=y_train[X_train.size_cluster==0].logerror, x=X_train[X_train.size_cluster==0].calculatedfinishedsquarefeet, 
                c=X_train[X_train.size_cluster==0].area_cluster, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Size Cluster 0')

    plt.subplot(3,2,2)
    plt.scatter(y=y_train[X_train.size_cluster==1].logerror, x=X_train[X_train.size_cluster==1].calculatedfinishedsquarefeet, 
                c=X_train[X_train.size_cluster==1].area_cluster, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Size Cluster 1')

    plt.subplot(3,2,3)
    plt.scatter(y=y_train[X_train.size_cluster==2].logerror, x=X_train[X_train.size_cluster==2].calculatedfinishedsquarefeet, 
                c=X_train[X_train.size_cluster==2].area_cluster, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Size Cluster 2')

    plt.subplot(3,2,4)
    plt.scatter(y=y_train[X_train.size_cluster==3].logerror, x=X_train[X_train.size_cluster==3].calculatedfinishedsquarefeet, 
                c=X_train[X_train.size_cluster==3].area_cluster, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Size Cluster 3')

    plt.subplot(3,2,5)
    plt.scatter(y=y_train[X_train.size_cluster==4].logerror, x=X_train[X_train.size_cluster==4].calculatedfinishedsquarefeet, 
                c=X_train[X_train.size_cluster==4].area_cluster, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Size Cluster 4')

    plt.subplot(3,2,6)
    plt.scatter(y=y_train.logerror, x=X_train.calculatedfinishedsquarefeet, alpha=.4, c=X_train.area_cluster)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('All Sizes')

    plt.suptitle('Is there distinction between clusters when visualizing size of the home by the error in zestimate?', y=.91)
    plt.show()

def anova_sqft_fips(X_train):
    F, p = stats.f_oneway(X_train['calculatedfinishedsquarefeet'][X_train['fips'] == 6037],
               X_train['calculatedfinishedsquarefeet'][X_train['fips'] == 6059],
               X_train['calculatedfinishedsquarefeet'][X_train['fips'] == 6111])

    alpha = .05
    if p < .05:
        print('We reject the null hypothesis that there is no difference in square footage of the home between counties.')
    else:
        print('We fail to reject the null hypothesis that there is no difference in square footage of the home between counties.')
    return F, p

def plot_size_county(X_train, y_train):
    plt.figure(figsize=(20,20))

    plt.subplot(3,2,1)
    plt.scatter(y=y_train[X_train.size_cluster==0].logerror, x=X_train[X_train.size_cluster==0].calculatedfinishedsquarefeet, 
                c=X_train[X_train.size_cluster==0].fips, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Size Cluster 0')

    plt.subplot(3,2,2)
    plt.scatter(y=y_train[X_train.size_cluster==1].logerror, x=X_train[X_train.size_cluster==1].calculatedfinishedsquarefeet, 
                c=X_train[X_train.size_cluster==1].fips, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Size Cluster 1')

    plt.subplot(3,2,3)
    plt.scatter(y=y_train[X_train.size_cluster==2].logerror, x=X_train[X_train.size_cluster==2].calculatedfinishedsquarefeet, 
            c=X_train[X_train.size_cluster==2].fips, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Size Cluster 2')

    plt.subplot(3,2,4)
    plt.scatter(y=y_train[X_train.size_cluster==3].logerror, x=X_train[X_train.size_cluster==3].calculatedfinishedsquarefeet, 
                c=X_train[X_train.size_cluster==3].fips, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Size Cluster 3')

    plt.subplot(3,2,5)
    plt.scatter(y=y_train[X_train.size_cluster==4].logerror, x=X_train[X_train.size_cluster==4].calculatedfinishedsquarefeet, 
                c=X_train[X_train.size_cluster==4].fips, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Size Cluster 4')

    plt.subplot(3,2,6)
    plt.scatter(y=y_train.logerror, x=X_train.calculatedfinishedsquarefeet, 
                c=X_train.fips, alpha=.4)
    plt.yscale('symlog')
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('All Sizes')

    plt.suptitle('Is there distinction between clusters when visualizing size of the home by the error in zestimate based on county?', y=.91)
    plt.show()

def plot_size_county_error(X_train, y_train):
    plt.figure(figsize=[20,8])
    plt.subplot(1,3,1)
    plt.scatter(y=y_train[X_train.fips==6037.0].logerror, x=X_train[X_train.fips==6037.0].calculatedfinishedsquarefeet, alpha=.7)
    plt.ylim([-3,3])
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('LA County')

    plt.subplot(1,3,2)
    plt.scatter(y=y_train[X_train.fips==6059.0].logerror, x=X_train[X_train.fips==6059.0].calculatedfinishedsquarefeet, alpha=.7)
    plt.ylim([-3,3])
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Orange County')

    plt.subplot(1,3,3)
    plt.scatter(y=y_train[X_train.fips==6111.0].logerror, x=X_train[X_train.fips==6111.0].calculatedfinishedsquarefeet, alpha=.7)
    plt.ylim([-3,3])
    plt.xlabel('Finished Square Feet')
    plt.ylabel('Log Error')
    plt.title('Ventura County')

    plt.suptitle('Does the size of the home have an effect on log error in each county?')
    plt.show()

