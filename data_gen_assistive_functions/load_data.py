#Library importing
import os
import math
import itertools
import pandas as pd
import numpy as np
import datetime as dt

def get_cwd():
    #cwd = '/Users/mahrokh/DECODE/Simulations/London'
    return os.getcwd()


def longestBelowThresh(a, thresh):
    a = a<thresh
    a = a*1
    n = len(a)
    max_len = 0;
    # i decides current ending point
    for i in range(0, n-2):
        if a[i] == 0:
            continue
        l=1
        while a[i+l]>0 :
            l = l+1
            if i+l>=n:
                break
        if l>max_len:
            max_len = l
        
    return max_len;

 
def try_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan
    
    
def find_building(block_nums = np.arange(1,8), verbose=True):
    min_nulls_all = 10000
    block = -1
    best_id_all = -1
    for block_num in block_nums:
        min_nulls = 10000
        house_id = -1
        print("searching block " + str(block_num))
        path = get_cwd()+"/input/halfhourly_dataset/block_"+str(block_num)+".csv"
        data = pd.read_csv(path)
            
        ids = data.LCLid.unique()

        for i, id in enumerate(ids):
            list_1p = data.loc[data.LCLid==id, 'energy(kWh/hh)'].values
            # convert to float
            list_1p = [try_float(v) for v in list_1p]
            list_1p = pd.Series(list_1p).interpolate(method='linear', direction = 'forward')

            if list_1p.isnull().sum().sum()>0:
                print(list_1p[list_1p.isnull()==True])
                print('Data contains '+ str(list_1p.isnull().sum().sum()) +' NaNs')
            # count number of records for this building
            count = list_1p.size
            if count < 365*2*48:
                continue
            nulls = longestBelowThresh(list_1p, 0.2)
            if nulls<min_nulls:
                min_nulls = nulls
                house_id = id
            if min_nulls <= 10:
                break
        print('Best in the block: house id ' + str(house_id) + 
              ' with maximum ' + str(min_nulls) + ' consecutive zeros\n\n')
        if min_nulls<min_nulls_all:
            best_id_all = house_id
            block = block_num
        
    return block, best_id_all
    
    
    
################################# CLEAN ###############################
def clean_data(data, verbose = True):
    # Check for nans
    if data.isnull().sum().sum()>0:
        miss_cols = list(data.columns[data.isna().any()])
        if verbose:
            print('\nData contains '+ str(data.isnull().sum().sum()) +' NaNs in columns')
            print(miss_cols)
            print(data[data.isna().any(axis=1)])
    else:
        return data
    # interpolate
    data = data.interpolate(method='linear', axis=0)
    # check again
    if verbose:
        if data.isnull().sum().sum()==0:
            print('Replaced by interpolation.\n')
        else:
            print('Failed to remove nans')
            miss_cols = list(data.columns[data.isna().any()])
            print(miss_cols)
            print(data[data.isna().any(axis=1)])
    return data


def check_missing(data, h=np.arange(0,24,0.5), insert_row=True, verbose=True):
    dh = h[1]-h[0]
    index, counts = np.unique(data.date,return_counts=True)
    missing_dates = index[counts<len(h)]
    if verbose:
        print('\nData has missing row in')
    for miss_date in missing_dates:
        existing_h = data.loc[data.date==miss_date, 'hourofd']
        miss_hs = np.setdiff1d(h, existing_h)
        if verbose:
            print(miss_date)
            print(miss_hs)
        # insert row
        if insert_row:
            in_miss_dayst = data.date.searchsorted(miss_date, side='left')
            for miss_h in miss_hs:
                in_miss = int(in_miss_dayst + miss_h/dh)
                data_bf = data.iloc[0:in_miss,:]
                data_af = data.iloc[in_miss:,:]
                data_nw = pd.DataFrame(columns=data_bf.columns)
                data_nw.date = [miss_date]
                data_nw.hourofd = [miss_h]
                data = pd.concat([pd.concat([data_bf, data_nw], ignore_index=True), data_af], ignore_index=True)
                data=data.reset_index(drop=True)
    # check again
    if insert_row:
        index, counts = np.unique(data.date,return_counts=True)
        if (counts>=len(h)).all():
            if verbose:
                print('Added row for missing time steps')
        else:
            print('error')
            print(index[counts != 48])
            print(counts[counts !=48])
    return data


def convert_date_time(data, crop_years=True, verbose=True):
    #convert series to datetimes
    date_time = pd.to_datetime(data['tstp'], errors='coerce')
    data = data.drop('tstp', axis=1)
    data['date'] = date_time.dt.date
    if verbose:
        print('Data starts at '), 
        print(data.date.iloc[0]), print('and ends at'), print(data.date.iloc[-1])
    # cut to complete years
    if crop_years:
        dt_st = dt.date(day=1,month=1,year=data.date.iloc[0].year+1)
        dt_en = dt.date(day=1,month=1,year=data.date.iloc[-1].year)-dt.timedelta(days = 1)
    else:
        dt_st = data.date.iloc[0]  + dt.timedelta(days=1)
        dt_en = data.date.iloc[-1] - dt.timedelta(days=1)
    in_st = data.date.searchsorted(dt_st, side='left')
    in_en = data.date.searchsorted(dt_en, side='right')
    data = data.iloc[in_st:in_en,:].reset_index(drop=True)
    if verbose:
        print('\nOnly considering data collected from '), 
        print(dt_st), print(' to '), print(dt_en)
    # hour of day
    minutes = round(date_time[in_st:in_en].dt.minute/60 * 2) / 2
    data['hourofd']= (date_time.iloc[in_st:in_en].dt.hour + minutes).values
    return data


def check_duplicate(data, verbose=True):
    dup = data[data.duplicated(subset=['date','hourofd'])]
    if dup.size==0:
        if verbose:
            print('\nNo duplicated rows')
        return data
    if verbose:
        print('\nDuplicate rows:')
        print(dup)
    # remove
    data = data.drop_duplicates(keep='first', subset=['date','hourofd']).reset_index(drop=True)
    # count
    index, counts = np.unique(data.date,return_counts=True)
    if verbose:
        if (counts != 48).any():
            print('\nRecord count error:')
            print(index[counts != 48])
            print(counts[counts !=48])
        else:
            print('\nAll days have 48 records')
    return data
    


########################### ADD TEMPORAL #########################################
def get_season(date):
    Y = date.year 
    seasons = [('winter', (dt.date(Y,  1,  1),  dt.date(Y,  3, 20))),
               ('spring', (dt.date(Y,  3, 21),  dt.date(Y,  6, 20))),
               ('summer', (dt.date(Y,  6, 21),  dt.date(Y,  9, 22))),
               ('autumn', (dt.date(Y,  9, 23),  dt.date(Y, 12, 20))),
               ('winter', (dt.date(Y, 12, 21),  dt.date(Y, 12, 31)))]
    return next(season for season, (start, end) in seasons
                if start <= date <= end)


def get_day_part(hourofd):
    day_times = [('night',(0,6)), ('morning', (6,10)), ('noon', (10,14)), 
                 ('afternoon',(14,18)), ('evening', (18,22)), ('night',(22,24))]
    return next(day_time for day_time, (start, end) in day_times
                if start <= hourofd < end)

   
def add_uk_holiday(data, verbose=True):
    holiday = ["2012-01-02","2012-03-19","2012-04-06","2012-04-09","2012-05-07",
               "2012-06-04","2012-06-05","2012-07-12","2012-08-06",
               "2012-08-27","2012-11-30","2012-12-25","2012-12-26",
               
               "2013-01-01","2013-03-18","2013-03-29","2013-04-01",
               "2013-05-06","2013-05-27","2013-07-12","2013-08-05",
               "2013-08-26","2013-12-02","2013-12-25","2013-12-26"]
    holiday = [dt.datetime.strptime(date, "%Y-%m-%d").date() for date in holiday]
    data['flg_holiday'] = data['date'].isin(holiday)
    if verbose:
        print('number of holidays')
        print(data['flg_holiday'].values.sum()/48)
    lst1 = data['flag_weekend'].values.astype(int)
    lst2 = data['flg_holiday'].values.astype(int)
    data['day_type'] = lst1+2*lst2
    data['day_type'].replace([0, 1, 2, 3], ["weekday", "weekend", "holiday", "holiday and weekend"], inplace=True)
    return data

    
def calculate_dayofy(datetimes):
    days = datetimes.day.values 
    months = datetimes.month.values 
    years = datetimes.year.values
    dayofy = np.zeros(days.shape)
    for i in range(len(days)):
        diff = dt.date(years[i], months[i], days[i]) - dt.date(years[i], 1, 1)
        dayofy[i] = diff.days
    return dayofy


def add_time_features(data, verbose=True):
    #convert series to datetimes
    date_time = pd.to_datetime(data['date'], errors='coerce')
    # hourofd_x
    data['hourofd_x'] = np.sin(data['hourofd']/24*2*math.pi)
    data['hourofd_y'] = np.cos(data['hourofd']/24*2*math.pi)
    # weekday
    data['weekday'] = date_time.dt.day_name()
    data['flag_weekend'] = data['weekday'].replace(['Saturday', 'Sunday'],True).replace(['Monday','Tuesday','Wednesday','Thursday','Friday'],False)
    # day of year
    data['dayofy'] = calculate_dayofy(date_time.dt)
    data['dayofy_x'] = np.sin(data['dayofy']/365*2*math.pi)
    data['dayofy_y'] = np.cos(data['dayofy']/365*2*math.pi)
    # time of day
    data['daypart'] = [get_day_part(h) for h in data.hourofd] 
    # month
    data['month'] = date_time.dt.month
    # season
    data['season'] = [get_season(date) for date in data.date]
    # holiday
    data = add_uk_holiday(data, verbose=verbose)
    return data


#################################### GET WEATHER ################################
def add_daily_features(df_1p, verbose=False):
    path = get_cwd()+"/input/weather_daily_darksky.csv.xls"
    w_daily_full = pd.read_csv(path)
    if verbose:
        print('\nDaily features:')
        for col in w_daily_full.columns:
            print(col)
    date_time = pd.to_datetime(w_daily_full['temperatureMaxTime'], errors='coerce')
    w_daily = pd.concat([date_time.dt.date, w_daily_full['icon']], axis=1, keys=['date', 'icon_daily'])
    w_daily.sort_values(by='date', inplace=True)
    # missing dates
    d = w_daily.date.values
    date_set = set(d[0] + dt.timedelta(x) for x in range((d[-1] - d[0]).days))
    missing = sorted(date_set - set(d))
    if not missing==[]:
        if verbose:
            print('Missing dates of daily icons:')
            print(missing)

    df_1p['icon_daily'] = df_1p['date'].replace(w_daily['date'].values, w_daily['icon_daily'].values)
    df_1p['icon_daily'] = df_1p['icon_daily'].replace(missing, "missing")
    if verbose:
        print('All daily icons')
        print(df_1p['icon_daily'].value_counts())
    return df_1p
 
    
def add_hourly_weather(df_1p, verbose=False):
    # load data
    path = get_cwd()+"/input/weather_hourly_darksky.csv"
    w_hourly = pd.read_csv(path)  
    if(w_hourly.duplicated().any()):
        if verbose:
            print('\nWarning: weather data has duplicated rows')
    if verbose:
        print('Hourly features:')
        for col in w_hourly.columns:
            print(col)
    w_hourly = w_hourly.loc[:, ['time','temperature','icon']]
    # select dates
    w_hourly['time'] = pd.to_datetime(w_hourly['time'], errors='coerce')
    w_hourly['date'] = w_hourly['time'].dt.date
    w_hourly['hourofd']= w_hourly['time'].dt.hour
    # sort by date
    w_hourly = w_hourly.sort_values(by='time')
    w_hourly = w_hourly.drop(columns=['time'])
    in_st = (w_hourly.date.values == df_1p.date.iloc[0])
    in_st = next((i for i, j in enumerate(in_st) if j), None)
    in_en = (w_hourly.date.values == df_1p.date.iloc[-1])
    in_en = next((i for i, j in reversed(list(enumerate(in_en))) if j), None)
    w_hourly = w_hourly.iloc[in_st:in_en+1, :].reset_index(drop=True)

    # missing dates
    if verbose:
        print('\nCheck for missing dates in the hourly data:')
    w_hourly = check_missing(w_hourly, h=np.arange(0,24), insert_row=True, verbose=verbose)
    
    # find nan
    if verbose:
        print('\nHourly weather contains '+ str(w_hourly.isnull().sum().sum()) +' NaNs in rows:')
        print(w_hourly[w_hourly.isna().any(axis=1)])
    # remove nans
    w_hourly['temperature'] = w_hourly['temperature'].interpolate()
    w_hourly['icon'] = w_hourly['icon'].bfill().ffill().tolist()
    # check for nan
    if verbose:
        if(w_hourly.isnull().sum().sum()):
            print('Hourly weather contains '+ str(w_hourly.isnull().sum().sum()) +' NaNs in rows:')
            print(w_hourly[w_hourly.isna().any(axis=1)])
        else:
            print('Hourly weather nans were interpolated')
        
    # copy all rows
    w_hourly2 = pd.DataFrame(np.repeat(w_hourly.values,2,axis=0))
    w_hourly2.columns = w_hourly.columns
    
    # merge
    if not df_1p.date.size == w_hourly2.date.size:
        print('\nError: data size mismatch')
    df_1p['temperature_hourly'] = w_hourly2['temperature']
    df_1p['icon_hourly'] = w_hourly2['icon']
    
    return df_1p


################################
def get_data_of_a_person(block=5, house_id="MAC004504", crop_years=True, verbose=False):
    path = get_cwd()+"/input/halfhourly_dataset/block_"+str(block)+".csv"
    data = pd.read_csv(path)
    
    if house_id==[]:
        house_id = data.LCLid.iloc[0]
        
    # get data of this house
    data = data.loc[data.LCLid==house_id, :].drop(['LCLid'], axis=1)

    # check
    if(len(data.tstp)<365*48):
        return pd.DataFrame({'date':[]})
    
    data = convert_date_time(data, crop_years=crop_years, verbose=verbose)
    
    # energy to KW
    data.iloc[:,0] = pd.to_numeric(data.iloc[:,0], errors='coerce') *2
    data = data.rename(columns={"energy(kWh/hh)": "energy"})  
       
    # check missing
    data = check_missing(data, verbose=verbose)
    
    # check duplicate
    data = check_duplicate(data, verbose=verbose)
    
    # check
    if(len(data.date)<365*48):
        print('[WARNING] data is too short (less than 1 year)')
        return pd.DataFrame({'date':[]})
    
    # add features
    data = add_time_features(data, verbose=verbose)
    
    data = add_hourly_weather(data, verbose=verbose)
    
    data = add_daily_features(data, verbose=verbose)
    
    # clean
    data = clean_data(data, verbose=verbose)
    # check for NaN again
    if data.isnull().sum().sum()>0:
        print('Error: data contains NaN')
    else:
        if verbose:
            print('\n   ***   DATA IS READY FOR USE   ***\n')
    # return
    data = data.reset_index(drop=True)
    return data


################################
def get_data_of_acorn_group(group="ACORN-L", num_houses=10, crop_years=True, verbose=False):
    path = get_cwd()+"/input/informations_households.csv.xls"
    data = pd.read_csv(path)
    candidates = data.loc[data.Acorn==group]
    # get 1 p
    ind_1p=0
    while True:
        block_name = candidates.file.iloc[ind_1p]
        df = get_data_of_a_person(block=block_name[6:], house_id=candidates.LCLid.iloc[ind_1p], verbose=verbose)
        if len(df.energy)==(365+366)*48:
            break
        ind_1p=ind_1p+1
            
    df['sumg']=df['energy']
    needed=num_houses-1 
    num=0
    while needed>0:
        num=num+1
        path = get_cwd()+"/input/halfhourly_dataset/"+candidates.file.iloc[ind_1p+num]+".csv"
        data = pd.read_csv(path)
        # get data of this house
        data = data.loc[data.LCLid==candidates.LCLid.iloc[num], :].drop(['LCLid'], axis=1)
        print(num)
        # check data
        if len(data.tstp)==0:
            continue
        data = convert_date_time(data, crop_years=crop_years, verbose=verbose)
        # energy to KW
        data.iloc[:,0] = pd.to_numeric(data.iloc[:,0], errors='coerce') *2
        data = data.rename(columns={"energy(kWh/hh)": "energy"})  
        # check missing
        data = check_missing(data, verbose=verbose)
        # check duplicate
        data = check_duplicate(data, verbose=verbose)
        data = clean_data(data, verbose=verbose)
        # add
        if len(data.energy)==len(df.sumg):
            df['sumg'] = df['sumg']+data['energy']
            needed=needed-1
            print('added')
        else:
            print(len(data.energy)/48)
    df.energy = df['sumg'].div(num_houses)
    df = df.drop('sumg', axis=1)
    df = df.reset_index(drop=True)
    return df


################################
def get_data_of_mixed_group(block=20, num_houses=10, crop_years=True, verbose=False):
    path = get_cwd()+"/input/halfhourly_dataset/block_"+str(block)+".csv"
    data = pd.read_csv(path)   
    ids = data.LCLid.unique()
    # get 1 p
    df = get_data_of_a_person(block=block, house_id=ids[0], verbose=verbose)
    df['sumg']=df['energy']    
    
    needed=num_houses-1 
    num=0
    while needed>0:
        num=num+1
        # add a second block if needed
        if num==len(ids):
            print('Adding block ' + str(block+1)+'to search path.')
            path = get_cwd()+"/input/halfhourly_dataset/block_"+str(block+1)+".csv"
            data = pd.read_csv(path)   
            ids = data.LCLid.unique()
            num=0
        # get data of this house
        df1p = data.loc[data.LCLid==ids[num], :].drop(['LCLid'], axis=1)
        df1p = convert_date_time(df1p, crop_years=crop_years, verbose=verbose)
        # energy to KW
        df1p.iloc[:,0] = pd.to_numeric(df1p.iloc[:,0], errors='coerce') *2
        df1p = df1p.rename(columns={"energy(kWh/hh)": "energy"})  
        # check missing
        df1p = check_missing(df1p, verbose=verbose)
        # check duplicate
        df1p = check_duplicate(df1p, verbose=verbose)
        df1p = clean_data(df1p, verbose=verbose)
        # add
        if len(df1p.energy)==len(df.sumg) and df1p.energy.isnull().sum()==0:
            df['sumg'] = df['sumg']+df1p['energy']
            needed=needed-1

    df.energy = df['sumg'].div(num_houses)
    df = df.drop('sumg', axis=1)
    df = df.reset_index(drop=True)
    return df


################################
def get_data_of_acorn_group_2013(group="ACORN-L", num_houses=10, crop_years=True, verbose=False):
    path = get_cwd()+"/input/informations_households.csv.xls"
    data = pd.read_csv(path)
    candidates = data.loc[data.Acorn==group]
    ys=[]
    # get 1 p
    ind_1p=0
    while True:
        block_name = candidates.file.iloc[ind_1p]
        df = get_data_of_a_person(block=block_name[6:], house_id=candidates.LCLid.iloc[ind_1p], verbose=verbose)
        if df.date.iloc[0].year==2012:
            df = df.iloc[48*365, :]
        if df.date.iloc[-1].year==2013 and df.date.iloc[0].year==2013:
            ys.append(df.energy)
            break
        ind_1p=ind_1p+1
            
    df['sumg']=df['energy']
    needed=num_houses-1 
    num=0
    while needed>0:
        num=num+1
        path = get_cwd()+"/input/halfhourly_dataset/"+candidates.file.iloc[ind_1p+num]+".csv"
        data = pd.read_csv(path)
        # get data of this house
        data = data.loc[data.LCLid==candidates.LCLid.iloc[num], :].drop(['LCLid'], axis=1)
        print(num)
        # check data
        if len(data.tstp)==0:
            continue
        data = convert_date_time(data, crop_years=crop_years, verbose=verbose)
        # energy to KW
        data.iloc[:,0] = pd.to_numeric(data.iloc[:,0], errors='coerce') *2
        data = data.rename(columns={"energy(kWh/hh)": "energy"})  
        # check missing
        data = check_missing(data, verbose=verbose)
        # check duplicate
        data = check_duplicate(data, verbose=verbose)
        data = clean_data(data, verbose=verbose)
        # add
        if len(data.energy)==len(df.sumg):
            ys.append(data.energy)
            df['sumg'] = df['sumg']+data['energy']
            needed=needed-1
            print('added')
        else:
            print(len(data.energy)/48)
    df.energy = df['sumg'].div(num_houses)
    df = df.drop('sumg', axis=1)
    df = df.reset_index(drop=True)
    return df, ys