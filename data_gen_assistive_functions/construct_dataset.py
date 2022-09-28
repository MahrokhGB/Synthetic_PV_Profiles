import numpy as np

import warnings
warnings.filterwarnings('ignore')

def add_lags(data_power,lags,step_ahead):
    max_ind = data_power.p_mp.size
    max_lag = max(lags)
    data_power_reg = data_power.loc[:,:].iloc[max_lag:max_ind-step_ahead+1,:]
    data_power_reg['target'] = data_power.p_mp.iloc[max_lag+step_ahead-1:max_ind].values
    for lag in lags:
        col_name = 'lag ' + str(lag)
        data_power_reg[col_name] = data_power.p_mp.iloc[max_lag-lag:max_ind-lag-step_ahead+1].values

    data_power_reg = data_power_reg.reset_index(drop=True)
    return data_power_reg


def cons_database(data_power, lags, months, filter_night):
    '''
    INPUTS: Data frame obtained from function generate_power_data, the desired lags
    and the month we want to focus on
    OUTPUTS: A normalized data frame including the meteorological data, power output and specified lags
    Note that it filters out data from times when sun is down; dataframe of the normalized features;
    dataframe of the normalized targets; the names of the features used.
    '''

    #Adding the lags
    max_ind = data_power.p_mp.size
    max_lag = max(lags)
    step_ahead=1
    data_power_reg = data_power.loc[:,:].iloc[max_lag:max_ind-step_ahead+1,:]
    data_power_reg['target'] = data_power.p_mp.iloc[max_lag+step_ahead-1:max_ind].values
    for lag in lags:
        col_name = 'lag ' + str(lag)
        data_power_reg[col_name] = data_power.p_mp.iloc[max_lag-lag:max_ind-lag-step_ahead+1].values


    #Filtering by month and times when sun is down
    df_reg_filt = data_power_reg.loc[data_power_reg['month'].isin(months), :]
    del df_reg_filt['p_mp']
    del df_reg_filt['month']
    if filter_night==True:
        df_reg_filt.drop(df_reg_filt[df_reg_filt['H_sun'] <= 0].index , inplace=True)
    #del df_reg_filt['H_sun']



    #extraction
    y = df_reg_filt.target.values
    feat_names = [col for col in df_reg_filt if not col.startswith('target')]
    X = df_reg_filt.loc[:,feat_names]
    feat_names = X.columns.tolist()
    df_reg_filt = df_reg_filt.reset_index(drop=True)


    return df_reg_filt, X, y, feat_names

def cons_database_some_hours(data_power, lags, months, hours, filter_night):
    '''
    INPUTS: Data frame obtained from function generate_power_data, the desired lags
    and the month we want to focus on
    OUTPUTS: A normalized data frame including the meteorological data, power output and specified lags
    Note that it filters out data from times when sun is down; dataframe of the normalized features;
    dataframe of the normalized targets; the names of the features used.
    '''

    #Adding the lags
    max_ind = data_power.p_mp.size
    max_lag = max(lags)
    step_ahead=1
    data_power_reg = data_power.loc[:,:].iloc[max_lag:max_ind-step_ahead+1,:]
    data_power_reg['target'] = data_power.p_mp.iloc[max_lag+step_ahead-1:max_ind].values
    for lag in lags:
        col_name = 'lag ' + str(lag)
        data_power_reg[col_name] = data_power.p_mp.iloc[max_lag-lag:max_ind-lag-step_ahead+1].values


    #Filtering by month and times when sun is down
    df_reg_filt = data_power_reg.loc[data_power_reg['month'].isin(months), :]
    df_reg_filt = df_reg_filt.loc[df_reg_filt['hour_day'].isin(hours), :]
    del df_reg_filt['p_mp']
    del df_reg_filt['month']
    del df_reg_filt['hour_day']
    if filter_night==True:
        df_reg_filt.drop(df_reg_filt[df_reg_filt['H_sun'] <= 0].index , inplace=True)
    #del df_reg_filt['H_sun']



    #Normalization and extraction
    y = df_reg_filt.target.values
    feat_names = [col for col in df_reg_filt if not col.startswith('target')]
    X = df_reg_filt.loc[:,feat_names]
    feat_names = X.columns.tolist()
    #df_reg_filt = df_reg_filt.reset_index(drop=True)


    return df_reg_filt, X, y, feat_names



def cons_database_some_hours_test(data_power, data_train, lags, months, hours, filter_night):
    '''
    INPUTS: Data frame obtained from function generate_power_data, the desired lags
    and the month we want to focus on
    OUTPUTS: A normalized data frame including the meteorological data, power output and specified lags
    Note that it filters out data from times when sun is down; dataframe of the normalized features;
    dataframe of the normalized targets; the names of the features used.
    '''

    #Adding the lags
    max_ind = data_power.p_mp.size
    max_lag = max(lags)
    step_ahead=1
    data_power_reg = data_power.loc[:,:].iloc[max_lag:max_ind-step_ahead+1,:]
    data_power_reg['target'] = data_power.p_mp.iloc[max_lag+step_ahead-1:max_ind].values
    for lag in lags:
        col_name = 'lag ' + str(lag)
        data_power_reg[col_name] = data_power.p_mp.iloc[max_lag-lag:max_ind-lag-step_ahead+1].values


    #Filtering by month and times when sun is down
    df_reg_filt = data_power_reg.loc[data_power_reg['month'].isin(months), :]
    df_reg_filt = df_reg_filt.loc[df_reg_filt['hour_day'].isin(hours), :]
    del df_reg_filt['p_mp']
    del df_reg_filt['month']
    del df_reg_filt['hour_day']
    if filter_night==True:
        df_reg_filt.drop(df_reg_filt[df_reg_filt['H_sun'] <= 0].index , inplace=True)
    #del df_reg_filt['H_sun']

    df_train, X, y, feat_names=cons_database_some_hours(data_train, lags, months, hours, filter_night)

    #Normalization and extraction
    df_reg_filt=normalize_test(df_reg_filt,df_train)
    y = df_reg_filt.target.values
    feat_names = [col for col in df_reg_filt if not col.startswith('target')]
    X = df_reg_filt.loc[:,feat_names]
    feat_names = X.columns.tolist()
    #df_reg_filt = df_reg_filt.reset_index(drop=True)


    return df_reg_filt, X, y, feat_names

def cons_database_hourly(data_power, lags, months, filter_night):
    '''
    INPUTS: Data frame obtained from function generate_power_data, the desired lags
    and the month we want to focus on
    OUTPUTS: A normalized data frame including the meteorological data, power output and specified lags
    Note that it filters out data from times when sun is down; dataframe of the normalized features;
    dataframe of the normalized targets; the names of the features used.
    '''

    #Adding the lags
    max_ind = data_power.p_mp.size
    max_lag = max(lags)
    step_ahead=1
    data_power_reg = data_power.loc[:,:].iloc[max_lag:max_ind-step_ahead+1,:]
    data_power_reg['target'] = data_power.p_mp.iloc[max_lag+step_ahead-1:max_ind].values
    for lag in lags:
        col_name = 'lag ' + str(lag)
        data_power_reg[col_name] = data_power.p_mp.iloc[max_lag-lag:max_ind-lag-step_ahead+1].values


    #Filtering by month and times when sun is down
    df_reg_filt = data_power_reg.loc[data_power_reg['month'].isin(months), :]
    #df_reg_filt = data_power_reg.loc[data_power_reg['week'].isin(weeks), :]
    #df_reg_filt = data_power_reg.loc[data_power_reg['day_year'].isin(days), :]
    del df_reg_filt['p_mp']
    #del df_reg_filt['month']
    #del df_reg_filt['week']


    if filter_night==True:
        df_reg_filt.drop(df_reg_filt[df_reg_filt['H_sun'] <= 0].index , inplace=True)
    #del df_reg_filt['H_sun']

    ################ Determine hourly models needed #####################
    maxi=df_reg_filt.hour_day.max()
    mini=df_reg_filt.hour_day.min()
    hourly_models=np.arange(mini,maxi,1)
    #print(hourly_models)

    dataframes_regression=[]
    Xs=[]
    ys=[]
    feat_names_=[]
    for i in hourly_models:
        df_hourly=df_reg_filt.loc[df_reg_filt['hour_day'].isin([i]),:]
        #df_hourly['hour_day']
        df_hourly=normalize(df_hourly)
        df_hourly = df_hourly.reset_index(drop=True)
        dataframes_regression.append(df_hourly)

        y = df_hourly.target.values
        ys.append(y)

        feat_names = [col for col in df_hourly if not col.startswith('target')]
        X = df_hourly.loc[:,feat_names]
        Xs.append(X)
        feat_names_.append(feat_names)

    ############################################################################################

    #Normalization and extraction
    #df_reg_filt=normalize(df_reg_filt)
    #y = df_reg_filt.target.values
    #feat_names = [col for col in df_reg_filt if not col.startswith('target')]
    #X = df_reg_filt.loc[:,feat_names]
    #feat_names = X.columns.tolist()
    #df_reg_filt = df_reg_filt.reset_index(drop=True)


    return dataframes_regression, Xs, ys, feat_names_, mini, maxi



def normalize(df):
    result = df.copy()
    filt_cols=df[[i for i in list(df.columns) if i not in ['target']]].columns
    for feature_name in filt_cols:
        if df[feature_name].std() != 0:
            result[feature_name] = (df[feature_name].values - df[feature_name].mean()) / df[feature_name].std()
        else:
            del result[feature_name] #= df[feature_name] #=0#df[feature_name]
            #print(df[feature_name])
            #max_value = df[feature_name].max()
            #min_value = df[feature_name].min()
            #result[feature_name] = (df[feature_name].values - min_value) / (max_value - min_value)

    max_value = df['target'].max()
    min_value = df['target'].min()
    result['target'] = (df['target'].values - min_value) / (max_value - min_value)
    return result


def normalize_test(df, df_train):
    result = df.copy()
    filt_cols=df[[i for i in list(df.columns) if i not in ['target']]].columns
    for feature_name in filt_cols:
        if df_train[feature_name].std() != 0:
            result[feature_name] = (df[feature_name].values - df_train[feature_name].mean()) / df_train[feature_name].std()
        else:
            del result[feature_name]
            #= df[feature_name] #=0#df[feature_name]
            #print(df[feature_name])
            #max_value = df[feature_name].max()
            #min_value = df[feature_name].min()
            #result[feature_name] = (df[feature_name].values - min_value) / (max_value - min_value)

    max_value = df_train['target'].max()
    min_value = df_train['target'].min()
    result['target'] = (df['target'].values - min_value) / (max_value - min_value)
    return result

