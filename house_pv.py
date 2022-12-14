import numpy as np
import pandas as pd
import os, sys, copy, pvlib, math, datetime, warnings

PV_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, PV_DIR)

from utils_pv import get_city_info
from weather_station import WeatherStation

class HousePV():

    def __init__(self, tilt, azimuth, module_name, inverter_name,
                 latitude, longitude, altitude,
                 weather_station,
                 weather_dev=0, irrad_scale=1,
                 shadows={'st': None, 'en': None, 'peak_red':None},
                 module_family=None, inverter_family=None,
                 random_state=None):
        '''
        Class for simulating PV generation profile for a single household.

        - random_state: fix to enable replicating the results over mutiple runs
        - weather_dev: add noise to temperature and wind speed.
                         value is multiplied by the range of feature
        - inverter_family/ module_family: finds it if not given
        '''
        if random_state == None:
            self.random_state = np.random
        else:
            self.random_state = random_state

        self.tilt, self.azimuth = tilt, azimuth
        self.module_name, self.inverter_name = module_name, inverter_name
        self.latitude, self.longitude, self.altitude = latitude, longitude, altitude
        self.weather_dev, self.shadows, self.irrad_scale = weather_dev, shadows, irrad_scale
        self.shadows=shadows
        self.weather_station=weather_station
        self.module_family, self.inverter_family = module_family, inverter_family

        self._simulate_shadows()
        self._setup_system()



    def _simulate_shadows(self):
        self.shadows_vec = np.ones(24)
        if not (self.shadows['st'] is None or self.shadows['en'] is None or self.shadows['peak_red'] is None):
            assert self.shadows['st']<self.shadows['en']
            assert self.shadows['peak_red']<1
            st, en = int(self.shadows['st']), int(self.shadows['en'])
            mid = math.floor((st+en)/2)
            self.shadows_vec[st:mid+1] = np.linspace(1, self.shadows['peak_red'], mid-st+2)[1:]
            if mid+1<=en:
                if (en-st+1) % 2 == 0:
                    self.shadows_vec[mid+1:en+1] = np.flip(self.shadows_vec[st:mid+1])
                else:
                    self.shadows_vec[mid+1:en+1] = np.flip(self.shadows_vec[st:mid])


    def _setup_system(self):
        with warnings.catch_warnings():
            # NOTE: CSV files for modules contains rows with duplicated names.
            # supppressing duplicated names warnings
            warnings.simplefilter("ignore")
            #Selecting the module database: sandia or CEC
            sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
            cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
            #Selecting the inverter database: CEC or Anton Driesse
            cec_inverters = pvlib.pvsystem.retrieve_sam('CECInverter')
            adr_inverters = pvlib.pvsystem.retrieve_sam('ADRInverter')
        # find family if not specified
        if self.module_family is None:
            from utils_pv import find_module_family # only import if needed
            self.module_family = find_module_family(self.module_name)
        if self.inverter_family is None:
            from utils_pv import find_inverter_family # only import if needed
            self.inverter_family = find_inverter_family(self.inverter_name)
        # set up system
        if self.module_family.lower()=='sandia':
            module = sandia_modules[self.module_name]
        elif self.module_family.lower()=='cec':
            module = cec_modules[self.module_name]
        if self.inverter_family.lower()=='cec':
            inverter = cec_inverters[self.inverter_name]
        elif self.inverter_family.lower()=='adr':
            inverter = adr_inverters[self.inverter_name]

        self.system = {'module':module, 'inverter':inverter,
                       'surface_azimuth':self.azimuth, 'surface_tilt':self.tilt}



    def simulate_pv(self):

        '''
        - delay_irrad: for predicting power at t+1, use irradiation at t or t+1. If False, assumes there exists
                       a perfect model for predicting direct and diffuse irradiation at the weather station.
        OUTPUTS: This function reads the data from the csv downoaled from PVGIS and returns a data frame with
        the power at maximum power point, meteorological and time and date data (Note that some data has been made sinusoidal)
        '''
        data = copy.deepcopy(self.weather_station.weather_data)

        # add noise on T2m, WS10m
        if self.weather_dev>0:
            for col in ['T2m', 'WS10m']:
                mag = data[col].max() - data[col].min()
                data[col] += self.random_state.normal(loc=0.0, scale=mag*self.weather_dev, size=data.shape[0])


        #Using PVLib to obtain a data frame with the DC output of the module/installation
        solpos = pvlib.solarposition.get_solarposition(
            time=data.index,
            latitude=self.latitude, longitude=self.longitude, altitude=self.altitude,
            pressure=pvlib.atmosphere.alt2pres(self.altitude)
        )

        airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
        pressure = pvlib.atmosphere.alt2pres(self.altitude)
        am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
        aoi = pvlib.irradiance.aoi(
            self.system['surface_tilt'],
            self.system['surface_azimuth'],
            solpos["apparent_zenith"],
            solpos["azimuth"],
        )

        total_irradiance = pvlib.irradiance.get_total_irradiance(
            self.system['surface_tilt'],
            self.system['surface_azimuth'],
            solpos['apparent_zenith'],
            solpos['azimuth'],
            data['Gb(i)'],
            data['Gd(i)']+(data['Gb(i)']*np.cos(solpos['apparent_zenith'])),#Global horizontal irradiance (ghi)
            data['Gd(i)'],
            dni_extra=pvlib.irradiance.get_extra_radiation(data.index),
            model='haydavies',
        )

        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass'] #TODO
        cell_temperature = pvlib.temperature.sapm_cell(
            total_irradiance['poa_global'],
            data.T2m,
            data.WS10m,
            **temperature_model_parameters,
        )

        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
            total_irradiance['poa_direct'],
            total_irradiance['poa_diffuse'],
            am_abs,
            aoi,
            self.system['module'],
        )

        for i in np.arange(effective_irradiance.size):
            # scale irrad
            effective_irradiance.values[i] = effective_irradiance.values[i]*self.irrad_scale

            # apply shadows
            if np.std(self.shadows_vec)>0:
                #d = pd.to_datetime(data.index[i], format='%Y-%m-%d %H:%M:%S').dt.hour
                d = datetime.datetime.strptime(str(data.index[i]), '%Y-%m-%d %H:%M:%S').hour
                effective_irradiance.values[i] = effective_irradiance.values[i]*self.shadows_vec[d]
            # negative effective irradiance result in NaNs in the power output
            if effective_irradiance.values[i]<0:
                effective_irradiance.values[i]=0

        #Obtaining DC outputs and creating the data frame data_power with the
        #maximum power output and meteorological data
        dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, self.system['module'])
        dc['p_mp'] = dc['p_mp'].fillna(0) # some modules place Non instead of 0
        power=dc['p_mp']
        data_power = pd.merge(data, power, on='time', how='left')
        data_power=data_power.reset_index()
        data_power['date'] = data_power['time'].dt.date
        data_power['day_year'] = data_power['time'].dt.dayofyear
        data_power['hour_day'] = data_power['time'].dt.hour
        data_power['month'] = data_power['time'].dt.month
        data_power['year'] = data_power['time'].dt.year

        data_power['hourofd_x'] = np.sin(data_power['hour_day']/24*2*math.pi)
        data_power['hourofd_y'] = np.cos(data_power['hour_day']/24*2*math.pi)
        data_power['dayofy_x'] = np.sin(data_power['day_year']/365*2*math.pi)
        data_power['dayofy_y'] = np.cos(data_power['day_year']/365*2*math.pi)
        del data_power['date']
        #del data_power['time']
        del data_power['day_year']
        #del data_power['hour_day']
        del data_power['Gb(i)']
        del data_power['Gd(i)']
        del data_power['Gr(i)']

        # move target to the 1st column
        shiftPos = data_power.pop("p_mp")
        data_power.insert(0, "p_mp", shiftPos)
        data_power.rename(columns={'p_mp': 'target'}, inplace=True)
        data_power.reset_index(drop=False)
        self.data_power = data_power # used for building regression matrices



    def construct_regression_df(
                self,
                use_station_irrad_direct, use_station_irrad_diffuse, delay_irrad,
                lags, hours, months, step_ahead=1):

        self.hours, self.months = hours, months

        # --- add irrad data ---
        if use_station_irrad_direct:
            irrad_direct = copy.deepcopy(self.weather_station.station_irrad_direct)
            if not delay_irrad:
                self.data_power = pd.merge(self.data_power, irrad_direct, on='time')
                self.data_power.rename(columns={'poa_direct': 'station_irrad_direct_cur'}, inplace=True)
            else:
                irrad_direct_vals = irrad_direct.values[:-1]
                irrad_direct_filt = irrad_direct.iloc[1:]
                irrad_direct_filt.iloc[:] = irrad_direct_vals
                self.data_power = pd.merge(self.data_power, irrad_direct_filt, on='time')
                self.data_power.rename(columns={'poa_direct': 'station_irrad_direct_prev'}, inplace=True)

        if use_station_irrad_diffuse:
            irrad_diffuse = copy.deepcopy(self.weather_station.station_irrad_diffuse)
            if not delay_irrad:
                self.data_power = pd.merge(self.data_power, irrad_diffuse, on='time')
                self.data_power.rename(columns={'poa_diffuse': 'station_irrad_diffuse_cur'}, inplace=True)
            else:
                irrad_diffuse_vals = irrad_diffuse.values[:-1]
                irrad_diffuse_filt = irrad_diffuse.iloc[1:]
                irrad_diffuse_filt.iloc[:] = irrad_diffuse_vals
                self.data_power = pd.merge(self.data_power, irrad_diffuse_filt, on='time')
                self.data_power.rename(columns={'poa_diffuse': 'station_irrad_diffuse_prev'}, inplace=True)

        # --- augment lags ---
        self.data_power = _augment_lags(self.data_power, lags=lags, step_ahead=step_ahead)

        # mark last week of each month as validation
        num_days_valid = 7
        # mark all points as not used in train
        self.data_power['is_train'] = [False]*len(self.data_power['target']) # init to False
        # mark points used for validation
        self.data_power['is_valid'] = [False]*len(self.data_power['target']) # init to False
        for month in self.months:
            # No need to specify last year, b.c. peak last num_days_valid
            inds_this_month = self.data_power.index[(self.data_power['month']==month) &
                                             (self.data_power['hour_day'].isin(self.hours))].tolist()

            inds_last_week = inds_this_month[-num_days_valid*len(self.hours):]
            self.data_power.loc[inds_last_week, 'is_valid'] = True


        # feature_names
        cols_to_remove = ['target',     # target is not a feature
                          'month',      # reflects in dayofy_x, dayofy_y
                          'time',       # # reflects in hour_day_x, hour_day_y
                          'hour_day',   # transformed to hour_day_x, hour_day_y
                          'is_valid', 'is_train', 'year']
        self.feature_names = [col for col in self.data_power if not col in cols_to_remove]




    def construct_regression_matrices(self, m_train=None, train_years=None, exclude_last_year=True):
        '''
        convert the regression dataframe into matrices
        validation points were marked in the regression dataframe
        - train_years: only select training samples from these years
        - exclude_last_year: if True, last year of data is not used in training
        - m_train: number of training samples. if None, any sample that is not used
                   for validation and is compatible with exclude_last_year is taken
                   as a training sample
        '''
        # select years that can be used for training
        if train_years is None:
            train_years = self.data_power['year'].unique().tolist()
        last_year = self.data_power.iloc[-1,:].loc['year']
        if exclude_last_year and last_year in train_years:
            train_years.remove(last_year)
        # mark points that can be used for training
        inds_maybe_train = self.data_power.index[
                                (self.data_power['is_valid']==False) &
                                (self.data_power['year'].isin(train_years))&
                                (self.data_power['month'].isin(self.months))&
                                (self.data_power['hour_day'].isin(self.hours))
                                ].tolist()
        # sample some of the potential points for training
        if m_train is None:
            inds_train = inds_maybe_train
            self.random_state.shuffle(inds_train)
        else:
            inds_train = self.random_state.choice(inds_maybe_train, m_train, replace=False)

        # mark training points
        # NOTE: loc uses index, iloc uses row num
        self.data_power['is_train'] = False
        self.data_power.loc[inds_train,'is_train']=[True]*len(inds_train)

        # extraction
        inds_valid = self.data_power.index[self.data_power['is_valid']==True].tolist()
        y_train = self.data_power.target.loc[inds_train].values
        y_valid = self.data_power.target.loc[inds_valid].values

        X_train = self.data_power.loc[inds_train, self.feature_names].values
        X_valid = self.data_power.loc[inds_valid, self.feature_names].values
        y_train = np.reshape(y_train, (-1, 1))
        y_valid = np.reshape(y_valid, (-1, 1))
        return (X_train, y_train, X_valid, y_valid)


def remove_constant_cols(data_power, data_tuple, feature_names):
    '''
    remove all features which were constant in the training set of at least one of the clients
    '''
    data_power = copy.deepcopy(data_power)
    x_train, y_train, x_valid, y_valid = data_tuple
    assert len(feature_names) == x_train.shape[1]

    const_cols = [] # find columns that are constant
    const_feats = []
    for col_num in np.arange(x_train.shape[1]):
        col_std = np.std(x_train[:, col_num])
        col_mean = np.mean(x_train[:, col_num])
        if col_std<=1e-4*col_mean+1e-6:
            const_cols.append(col_num)
            const_feats.append(feature_names[col_num])

    # return if no constant features
    if len(const_cols)==0:
        return data_power, data_tuple, feature_names

    # remove from list of features
    print('[INFO] the following {:2.0f} constnat features were removed: '.format(
                                                    len(const_feats)), *const_feats)
    for const_feat in const_feats:
        feature_names.remove(const_feat)

    # remove constant features in train from both train and valid
    x_train = np.delete(x_train, const_cols, axis=1)
    x_valid = np.delete(x_valid, const_cols, axis=1)
    assert len(feature_names) == x_train.shape[1]
    assert len(feature_names) == x_valid.shape[1]
    # put back
    data_tuple = (x_train, y_train, x_valid, y_valid)
    # remove from time series
    data_power.drop(columns=const_feats, inplace=True)
    assert not any(feat in data_power.columns for feat in const_feats)
    return data_power, data_tuple, feature_names



def _augment_lags(data, lags, step_ahead=1):
    #Adding the lags
    if isinstance(lags, list) and len(lags)==0:
        print('[WARNING] AR not used')
        max_lag=0
    else:
        max_lag = max(lags)

    data_power_reg = copy.deepcopy(data)
    n_rows = data_power_reg.target.size

    data_power_reg = data_power_reg.iloc[max_lag:n_rows-step_ahead+1,:]
    del data_power_reg['target']
    targets = data['target'].values.tolist()[max_lag+step_ahead-1:n_rows]
    #.loc[:, 'target'].iloc[max_lag+step_ahead-1:n_rows,].values
    data_power_reg.loc[:, 'target'] = targets
    # construct matrix with new lags
    new_col_names = []
    new_cols = np.zeros((len(targets), len(lags)))
    for lag_num, lag in enumerate(lags):
        new_col_name = 'lag ' + str(lag)
        if not new_col_name in data.columns:
            new_col_names.append(new_col_name)
            new_cols[:, lag_num] = data['target'].iloc[max_lag-lag:n_rows-lag-step_ahead+1].to_numpy().flatten()
        #data_power_reg[col_name] = data['target'].iloc[max_lag-lag:n_rows-lag-step_ahead+1]
    # add new matrix to the df
    data_power = pd.concat(
                            (data_power_reg,
                            pd.DataFrame(new_cols, index=data_power_reg.index, columns=new_col_names)),
                            axis=1)
    return data_power


def reconstruct_house(env_dict, client_num, scenario_name):
    config=copy.deepcopy(env_dict['clients_config'])
    house = HousePV(
        tilt=config['tilt'][client_num],
        azimuth=config['azimuth'][client_num],
        module_name=config['module_name'][client_num],
        inverter_name=config['inverter_name'][client_num],
        latitude=get_city_info(config['city_name'][client_num])['latitude'],
        longitude=get_city_info(config['city_name'][client_num])['longitude'],
        altitude=config['altitude'][client_num],
        weather_station=WeatherStation(city_name=config['city_name'][client_num]),
        weather_dev=config['weather_dev'][client_num],
        irrad_scale=config['irrad_scale'][client_num],
        shadows=config['shadows'][client_num],
        module_family=config['module_family'][client_num],
        inverter_family=config['inverter_family'][client_num],
        random_state=None)
    # set power data
    house.data_power = env_dict['train_scenarios'][scenario_name]['time_series'][client_num]

    house.hours = copy.deepcopy(env_dict['hours'])
    house.months = copy.deepcopy(env_dict['months'])
    house.feature_names = copy.deepcopy(env_dict['feature_names'])
    return house



if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(PROJECT_DIR)
    from Synthetic_PV_Profiles import WeatherStation

    #Reading the data file: DOWNLOAD DATA FROM PVGIS USING TILT=0 i.e. DATA OF IRRADIANCE ON NORMAL PLANE
    city_name = 'Lausanne'
    weather_station = WeatherStation(city_name)

    house = HousePV(tilt=0, azimuth=180,
                    module_name='Canadian_Solar_CS5P_220M___2009_',
                    inverter_name='ABB__MICRO_0_25_I_OUTD_US_208__208V_',
                    latitude=45.473, longitude=9.185, altitude=133,
                    weather_dev=0.1, irrad_scale=1,
                    weather_station=weather_station,
                    shadows={'st': 3, 'en': 8, 'peak_red':0.8},
                    random_state=None)

    house.simulate_pv()
    house.construct_regression_df(
                    use_station_irrad_direct=True, use_station_irrad_diffuse=True,
                    delay_irrad=True, lags=[1], hours=np.arange(7,18),
                    months=[3,4], step_ahead=1)
    (X_train, y_train, X_valid, y_valid) = house.construct_regression_matrices(
                    m_train=None, train_years=[2018, 2019],
                    exclude_last_year=True)
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, house.feature_names)
    #print(house.shadows_vec)

