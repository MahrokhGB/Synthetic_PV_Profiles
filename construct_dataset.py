import warnings
warnings.filterwarnings('ignore')


def cons_database_some_hours(data_power, lags, months, hours, step_ahead=1):
    '''
    INPUTS: Data frame obtained from function generate_power_data, the desired lags
    and the month we want to focus on
    OUTPUTS: A normalized data frame including the meteorological data, power output and specified lags
    Note that it filters out data from times when sun is down; dataframe of the normalized features;
    dataframe of the normalized targets; the names of the features used.
    Notes: at time t, predict target at t+1 givrn all powers up to t and weather prediction for t+1.
           time and weather info about t+1 are used. lag 1 corresponds to power at time t.
           if want to use previous weather feature,s can use step_ahead=2 and add lag 0.
           would make sense to use H_sun at t+1
    '''
    n_rows = data_power.p_mp.size

    #Adding the lags
    if isinstance(lags, list) and len(lags)==0:
        print('[WARNING] AR not used')
        max_lag=0
    else:
        max_lag = max(lags)

    data_power_reg = data_power.loc[:,:].iloc[max_lag:n_rows-step_ahead+1,:]
    del data_power_reg['p_mp']
    targets = data_power.loc[:, 'p_mp'].iloc[max_lag+step_ahead-1:n_rows,].values
    data_power_reg.loc[:, 'target'] = targets
    for lag in lags:
        col_name = 'lag ' + str(lag)
        data_power_reg.loc[:, col_name] = data_power.p_mp.iloc[max_lag-lag:n_rows-lag-step_ahead+1].values


    #Filtering by month and times when sun is down
    df_reg_filt = data_power_reg.loc[data_power_reg['month'].isin(months), :]
    df_reg_filt = df_reg_filt.loc[df_reg_filt['hour_day'].isin(hours), :]

    del df_reg_filt['month']
    del df_reg_filt['hour_day']
    #if filter_night==True:
    #    df_reg_filt.drop(df_reg_filt[df_reg_filt['H_sun'] <= 0].index , inplace=True)
    #del df_reg_filt['H_sun']

    # extraction
    y = df_reg_filt.target.values
    feat_names = [col for col in df_reg_filt if not col.startswith('target')]
    X = df_reg_filt.loc[:,feat_names]
    feat_names = X.columns.tolist()
    #df_reg_filt = df_reg_filt.reset_index(drop=True)


    return df_reg_filt, X, y, feat_names
