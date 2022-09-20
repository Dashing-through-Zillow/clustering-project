import pandas as pd
import numpy as np
import env
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, RobustScaler

def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''This function uses credentials from an env file to log into a database'''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# def new_zillow_db():
#     '''The function uses the get_connection function to connect to a database and retrieve the zillow dataset'''
    
#     zillow = pd.read_sql('''
#     SELECT p.id, p.parcelid, pd.logerror, pd.transactiondate, p.airconditioningtypeid, ac.airconditioningdesc, 
#     p.architecturalstyletypeid, a.architecturalstyledesc, p.basementsqft, p.bathroomcnt, p.bedroomcnt, 
#     p.buildingclasstypeid, b.buildingclassdesc, p.buildingqualitytypeid, p.calculatedbathnbr, p.decktypeid, 
#     p.finishedfloor1squarefeet, p.calculatedfinishedsquarefeet, p.finishedsquarefeet12, p.finishedsquarefeet13, 
#     p.finishedsquarefeet15, p.finishedsquarefeet50, p.finishedsquarefeet6, p.fips, p.fireplacecnt, p.fullbathcnt, 
#     p.garagecarcnt, p.garagetotalsqft, p.hashottuborspa, p.heatingorsystemtypeid, h.heatingorsystemdesc, p.latitude, 
#     p.longitude, p.lotsizesquarefeet, p.poolcnt, p.poolsizesum, p.pooltypeid10, p.pooltypeid2, p.pooltypeid7, 
#     p.propertycountylandusecode, p.propertylandusetypeid, p.propertyzoningdesc, p.rawcensustractandblock, 
#     p.regionidcity, p.regionidneighborhood, p.regionidzip, p.roomcnt, p.storytypeid, p.threequarterbathnbr, 
#     p.typeconstructiontypeid, p.unitcnt, p.yardbuildingsqft17, p.yardbuildingsqft26, p.yearbuilt, p.numberofstories, 
#     p.fireplaceflag, p.structuretaxvaluedollarcnt, p.taxvaluedollarcnt, p.assessmentyear, p.landtaxvaluedollarcnt, 
#     p.taxamount, p.taxdelinquencyflag, p.taxdelinquencyyear, p.censustractandblock

#     FROM properties_2017 as p
#     INNER JOIN predictions_2017 as pd
#     ON p.id = pd.id
#     LEFT JOIN airconditioningtype as ac
#     ON p.airconditioningtypeid = ac.airconditioningtypeid
#     LEFT JOIN architecturalstyletype as a
#     ON p.architecturalstyletypeid = a.architecturalstyletypeid
#     LEFT JOIN buildingclasstype as b
#     ON p.buildingclasstypeid = b.buildingclasstypeid
#     LEFT JOIN heatingorsystemtype as h
#     ON p.heatingorsystemtypeid = h.heatingorsystemtypeid
#     LEFT JOIN propertylandusetype as l
#     ON p.propertylandusetypeid = l.propertylandusetypeid
#     LEFT JOIN storytype as s
#     ON p.storytypeid = s.storytypeid
#     LEFT JOIN typeconstructiontype as t
#     ON p.typeconstructiontypeid = t.typeconstructiontypeid
#     LEFT JOIN unique_properties as u
#     ON p.parcelid = u.parcelid
#     WHERE p.latitude IS NOT NULL
#     AND p.longitude IS NOT NULL
#     AND p.propertylandusetypeid = 261

#     ;''', get_connection('zillow'))
#     return zillow

def new_zillow_db():
    zillow = pd.read_sql('''select p.parcelid, pred.logerror, p.bathroomcnt, p.bedroomcnt, p.calculatedfinishedsquarefeet, 
    p.fips, p.latitude, p.longitude, p.lotsizesquarefeet, p.regionidcity, p.regionidcounty, p.regionidzip, p.yearbuilt, 
    p.structuretaxvaluedollarcnt, p.taxvaluedollarcnt, p.landtaxvaluedollarcnt, p.taxamount
    from properties_2017 p
    inner join predictions_2017 pred on p.parcelid = pred.parcelid
    where propertylandusetypeid = 261;
    ''', get_connection('zillow'))
    return zillow

def get_zillow_data():
    ''' This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.'''
    if os.path.isfile('zillow.csv'):
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)     
    else:   
        # Read fresh data from db into a DataFrame
        df = new_zillow_db()
        # Cache data
        df.to_csv('zillow.csv')

def wrangle_zillow():
    '''This function acquires the zillow dataset from the Codeup database using a SQL query and returns a cleaned
    dataframe from a csv file. Observations with null values are dropped and column names are changed for
    readability. Values expected as integers are converted to integer types (year, bedrooms, fips).'''
    if os.path.isfile('zillow.csv'):
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)     
    else:   
        # Read fresh data from db into a DataFrame
        df = new_zillow_db()
        # Cache data
        df.to_csv('zillow.csv')
    df = handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75)
    df = df.dropna()
    df = get_counties(df)
    df = create_features(df)
    df = remove_outliers(df)
    
    return df

def split_data(df):
    '''This function takes in a dataframe and returns three dataframes, a training dataframe with 60 percent of the data, 
        a validate dataframe with 20 percent of the data and test dataframe with 20 percent of the data.'''
    # split the dataset into two, with 80 percent of the observations in train and 20 percent in test
    train, test = train_test_split(df, test_size=.2, random_state=217)
    # split the train again into two sets, using a 75/25 percent split
    train, validate = train_test_split(train, test_size=.25, random_state=217)
    # return three datasets, train (60 percent of total), validate (20 percent of total), and test (20 percent of total)
    return train, validate, test

def quantile_scaler_norm(a,b,c):
    '''This function applies the .QuantileTransformer method from sklearn to three arguments, a, b, and c,
    (X_train, X_validate, and X_test) and returns the scaled versions of each variable.'''
    # make the scaler
    scaler = QuantileTransformer(output_distribution='normal')
    # fit and transform the X_train variable
    X_train_quantile = pd.DataFrame(scaler.fit_transform(a))
    # transform the X_validate variable
    X_validate_quantile = pd.DataFrame(scaler.transform(b))
    # transform the X_test variable
    X_test_quantile = pd.DataFrame(scaler.transform(c))
    # return three variables, one for each newly scaled variable
    return X_train_quantile, X_validate_quantile, X_test_quantile

def quantile_scaler(a,b,c):
    '''This function applies the .QuantileTransformer method from sklearn to three arguments, a, b, and c,
    (X_train, X_validate, and X_test) and returns the scaled versions of each variable.'''
    # make the scaler
    scaler = QuantileTransformer()
    # fit and transform the X_train variable
    X_train_quantile = pd.DataFrame(scaler.fit_transform(a))
    # transform the X_validate variable
    X_validate_quantile = pd.DataFrame(scaler.transform(b))
    # transform the X_test variable
    X_test_quantile = pd.DataFrame(scaler.transform(c))
    # return three variables, one for each newly scaled variable
    return X_train_quantile, X_validate_quantile, X_test_quantile

def standard_scaler(a,b,c):
    '''This function applies the .StandardScaler method from sklearn to three arguments, a, b, and c, 
    (X_train, X_validate, and X_test) and returns the scaled versions of each variable.'''
    # make the scaler
    scaler = StandardScaler()
    # fit and transform the X_train data
    X_train_standard = pd.DataFrame(scaler.fit_transform(a))
    # transform the X_validate data
    X_validate_standard = pd.DataFrame(scaler.transform(b))
    # transform the X_test data
    X_test_standard = pd.DataFrame(scaler.transform(c))
    # return the scaled data for each renamed variable
    return X_train_standard, X_validate_standard, X_test_standard

def minmax_scaler(a,b,c):
    '''This function applies the .MinMaxScaler method from sklearn to three arguments, a, b, and c,
    (X_train, X_validate, and X_test) and returns the scaled versions of each variable.'''
    # make the scaler
    scaler = MinMaxScaler()
    # fit and transform the X_train data
    X_train_scaled = pd.DataFrame(scaler.fit_transform(a))
    # transform the X_validate data
    X_validate_scaled = pd.DataFrame(scaler.transform(b))
    # transform the X_test data
    X_test_scaled = pd.DataFrame(scaler.transform(c))
    # return the scaled data for each renamed variable
    return X_train_scaled, X_validate_scaled, X_test_scaled

def robust_scaler(a,b,c):
    '''This function applies the .RobustScaler method from sklearn to three arguments, a, b, and c,
    (X_train, X_validate, and X_test) and returns the scaled versions of each variable.'''
    # make the scaler
    scaler = RobustScaler()
    # fit and transform the X_train data
    X_train_robust = pd.DataFrame(scaler.fit_transform(a))
    # transform the X_validate data
    X_validate_robust = pd.DataFrame(scaler.transform(b))
    # transform the X_test data
    X_test_robust = pd.DataFrame(scaler.transform(c))
    # return the scaled data for each renamed variable
    return X_train_robust, X_validate_robust, X_test_robust

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prnt_miss}).\
    reset_index().groupby(['num_cols_missing', 'percent_cols_missing']).count().\
    reset_index().rename(columns={'customer_id': 'count'})
    return rows_missing

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    prnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prnt_miss}).\
    reset_index().groupby(['num_rows_missing', 'percent_rows_missing']).count().reset_index().\
    rename(columns={'index': 'count'})
    return cols_missing

def summarize(df):
    print('DataFrame head: \n')
    print(df.head())
    print('----------')
    print('DataFrame info: \n')
    print(df.info())
    print('----------')
    print('DataFrame description: \n')
    print(df.describe())
    print('----------')
    print('Null value assessments: \n')
    print('Nulls by column: ', nulls_by_col(df))
    print('----------')
    print('Nulls by row: ', nulls_by_row(df))
    numerical_cols = [col for col in df.columns if df[col].dtypes != 'O']
    cat_cols = [col for col in df.columns if col not in numerical_cols]
    print('----------')
    print('Value counts: \n')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
        print('-----')
    print('----------')
    print('Report Finished')

def get_upper_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k*iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    return df

def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, 
                          prop_required_columns=0.5, 
                          prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df

def split_data_strat(df, column):
    '''This function takes in two arguments, a dataframe and a string. The string argument is the name of the
        column that will be used to stratify the train_test_split. The function returns three dataframes, a 
        training dataframe with 60 percent of the data, a validate dataframe with 20 percent of the data and test
        dataframe with 20 percent of the data.'''
    train, test = train_test_split(df, test_size=.2, random_state=217, stratify=df[column])
    train, validate = train_test_split(train, test_size=.25, random_state=217, stratify=train[column])
    return train, validate, test

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