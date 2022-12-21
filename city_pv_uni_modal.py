import numpy as np
import os, sys, pickle

PV_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PV_DIR)
sys.path.append(os.path.dirname(PV_DIR))

from Synthetic_PV_Profiles import HousePV, WeatherStation
from utils_pv import tile_in_list
from utils_pv import find_module_family, find_inverter_family, get_city_info
#from assistive_functions.utils import*

class CityPV_UniModal():

    def __init__(self, city_name,
                 tilt_mean, az_mean, tilt_std, az_std,
                 weather_dev, irrad_std, altitude_dev,
                 shadow_peak_red,
                 random_state=None):
        '''
        Class for simulating PV generation profiles for multiple households.
        All households are located at the given city.
        NOTE: currently, we are using 2 years of data from Lausanne.
        NOTE: the first year of data is used for training and the second year for validation
        The tilt and azimuth of PV cells are varried among the households to simulate different installation setups.
        Tilt and azimuth are sampled from a two-dimentional independent Gaussian distribution.
        inputs:
        - city_name: location of households. must be the same city. Bern, Dijon, Lausanne, or Milan
        - tilt_dev:  deviation in the tilt angle of PV cells for different households.
                     multiplied by 90 degrees to get the std of tilts' Gaussian distribution.
                     set to 0 to have the same tilt for all households.
        - az_dev:    deviation in the azimuth angle of PV cells for different clients.
                     multiplied by 360 degrees to get the std of azimuths' Gaussian distribution.
                     set to 0 to have the same azimuth for all households.
        - tilt_mean: mean of the Gaussian distribution over tilt angle for different households.
                     if not specified, latitude of the city is used. tilt is between 0 to 90 degrees.
        - az_mean:   mean of the Gaussian distribution over azimuth angle for different households.
                     if not specified, south-facing cell (180 deg) is used. azimuth is between 0 to 359 degrees.
        - random_state: fix to enable replicating the results over mutiple runs
        - weather_dev: deviation in the temperature and wind speed for different clients.
                       multiplied by the range of temperature or wind speed to get the std of
                       noise added on these weather features.
        - shadow_peak_red:
        '''
        if random_state == None:
            self.random_state = np.random
        else:
            self.random_state = random_state
        self.city_name = city_name
        self.tilt_mean = tilt_mean if not tilt_mean is None else get_city_info(self.city_name)['latitude']
        self.az_mean   = az_mean   if not az_mean   is None else 180
        self.tilt_std, self.az_std = tilt_std, az_std
        self.weather_dev, self.irrad_std, self.altitude_dev = weather_dev, irrad_std, altitude_dev
        self.shadow_peak_red = shadow_peak_red
        self.feature_names = None
        # set up weather station
        self.weather_station = WeatherStation(city_name=self.city_name)



    def simulate_pv(self, num_clients, lags, months, hours,
                    module_name='Canadian_Solar_CS5P_220M___2009_',
                    inverter_name='ABB__MICRO_0_25_I_OUTD_US_208__208V_',
                    use_station_irrad_direct=True, use_station_irrad_diffuse=True,
                    delay_irrad=True):
        '''
        Generate data for a given number of clients over a selected time interval
        NOTE: only diagonal cov has been implemented
        inputs:
        - num_clients: number of households to generate data for
        - module_name: name of the installed PV cell. if is a string, used for all clients.
                       if is a list of length num_clients, specifies the name of PV cell for each client.
        - inverter_name: similar to module_name, indicating the name of the inverter
        - lags (list of ints greater than 0): lags used to construct auto-regressors.
        - months (list of int between 1 and 12, 1 is Jan): list of months to simulate data in
        - hours (list of int between 0 and 23, 0 is midnight): list of hours of day to simulate data

        result:
        - forms self.houses
        '''
        # load auto-regressors saved after pre-processing
        if lags is None:
            lags=[]
        self.lags=lags
        # load saved setup from pre-processing
        if hours is None:
            hours=np.arange(24)
        self.hours = hours
        if months is None:
            months = np.arange(1,13)
        self.months = months

        # initialize clients configuration
        self.houses = [None]*num_clients
        self.clients_time_series = [None]*len(self.houses)
        self._init_clients_config(num_clients=num_clients, module_name=module_name, inverter_name=inverter_name)

        for client_num in range(num_clients):
            house = HousePV(tilt    = self.clients_config['tilt'][client_num],
                            azimuth = self.clients_config['azimuth'][client_num],
                            latitude  = get_city_info(self.city_name)['latitude'],
                            longitude = get_city_info(self.city_name)['longitude'],
                            altitude  = self.clients_config['altitude'][client_num],
                            shadows = self.clients_config['shadows'][client_num],
                            weather_dev   = self.clients_config['weather_dev'][client_num],
                            irrad_scale   = self.clients_config['irrad_scale'][client_num],
                            module_name   = self.clients_config['module_name'][client_num],
                            inverter_name = self.clients_config['inverter_name'][client_num],
                            module_family   = self.clients_config['module_family'][client_num],
                            inverter_family = self.clients_config['inverter_family'][client_num],
                            weather_station=self.weather_station,
                            random_state=self.random_state)
            # simulate PV for this house
            house.simulate_pv()
            # construct the regression data frame
            house.construct_regression_df(use_station_irrad_direct=use_station_irrad_direct,
                                          use_station_irrad_diffuse=use_station_irrad_diffuse,
                                          delay_irrad=delay_irrad,
                                          lags=self.lags, months=self.months,
                                          hours=self.hours, step_ahead=1)
            # add house to the list of houses in the city
            self.houses[client_num]=house
            self.clients_time_series[client_num] = house.data_power
            # check feature names
            if self.feature_names is not None:
                assert self.feature_names == house.feature_names
            else:
                self.feature_names = house.feature_names



    def construct_regression_matrices(self, m_train, train_years=None, exclude_last_year=True,
                                    remove_constant_cols=True):
        # one tuple per client, each tuple is (x_train, y_train, x_valid, y_valid)
        self.clients_data_tuple = [None]*len(self.houses)
        for client_num, house in enumerate(self.houses):
            self.clients_data_tuple[client_num] = house.construct_regression_matrices(m_train=m_train,
                                                        train_years=train_years, exclude_last_year=exclude_last_year)
            # update train inds
            self.clients_time_series[client_num] = house.data_power

        if remove_constant_cols:
            self._remove_constant_cols()



    # initialize clients configuration
    def _init_clients_config(self, num_clients, module_name, inverter_name):
        '''
        initialize config dict: each field is a list containing config info of the clients.
        '''
        self.clients_config={'city_name': tile_in_list(self.city_name, num_clients),
                            'weather_dev':tile_in_list(self.weather_dev, num_clients),
                            'tilt':[None]*num_clients, 'azimuth':[None]*num_clients,
                            'altitude':[None]*num_clients,
                            'shadows':[{'st':None, 'en':None, 'peak_red':None}]*num_clients,
                            'irrad_scale':[None]*num_clients,
                            'module_name': tile_in_list(module_name, num_clients),
                            'inverter_name':tile_in_list(inverter_name, num_clients),
                            'module_family':['sandia']*num_clients, 'inverter_family':['cec']*num_clients}

        # sample tilts, azimuths, and altitutes
        self.clients_config['tilt'] = self.random_state.normal(self.tilt_mean, self.tilt_std, num_clients)   # sample tilts for num_clients households
        self.clients_config['azimuth'] = self.random_state.normal(self.az_mean, self.az_std, num_clients)   # sample azimuths for num_clients households
        alt_mean = get_city_info(self.city_name)['altitude']
        self.clients_config['altitude'] = self.random_state.normal(alt_mean, alt_mean*self.altitude_dev, num_clients)
        self.clients_config['irrad_scale'] = self.random_state.normal(1, self.irrad_std, num_clients)

        # shadows
        if self.shadow_peak_red<1:
            for client_num in np.arange(num_clients):
                st_en = np.sort(self.random_state.choice(self.hours, 2, replace=False))
                self.clients_config['shadows'][client_num] = {'st':st_en[0], 'en':st_en[1],
                                                'peak_red':self.shadow_peak_red}

        # set module and inverter families
        for client_num in np.arange(num_clients):
            family = find_module_family(self.clients_config['module_name'][client_num])
            self.clients_config['module_family'][client_num] = family

            family = find_inverter_family(self.clients_config['inverter_name'][client_num])
            self.clients_config['inverter_family'][client_num] = family


    def _remove_constant_cols(self):
        '''
        remove all features which were constant in the training set of at least one of the clients
        '''
        const_cols = [] # find columns that are constant for at least one client
        const_feats = []
        for client_num, client_data_tuple in enumerate(self.clients_data_tuple):
            x_train = client_data_tuple[0]
            assert len(self.feature_names) == x_train.shape[1]
            for col_num in np.arange(x_train.shape[1]):
                col_std = np.std(x_train[:, col_num])
                col_mean = np.mean(x_train[:, col_num])
                if (col_std<=1e-4*col_mean+1e-6) and (not col_num in const_cols):
                    const_cols.append(col_num)
                    const_feats.append(self.feature_names[col_num])
                    print(self.feature_names[col_num] + ' with feat num {:2.0f} was constant for client {:2.0f}'.format(col_num, client_num))
                    print(x_train[0:20, col_num])
        # remove constant cols from data
        if len(const_cols)==0:
            return
        # remove from list of features
        print('[INFO] the following constnat features were removed: ', *const_feats)
        for const_feat in const_feats:
            self.feature_names.remove(const_feat)

        # remove from data
        for client_num, client_data_tuple in enumerate(self.clients_data_tuple):
            x_train, y_train, x_valid, y_valid = client_data_tuple
            # remove constant features in train from both train and valid
            x_train = np.delete(x_train, const_cols, axis=1)
            x_valid = np.delete(x_valid, const_cols, axis=1)
            assert len(self.feature_names) == x_train.shape[1]
            assert len(self.feature_names) == x_valid.shape[1]
            # put back
            self.clients_data_tuple[client_num] = (x_train, y_train, x_valid, y_valid)
            # remove from time series
            self.clients_time_series[client_num].drop(columns=const_feats, inplace=True)
            assert not any(feat in self.clients_time_series[client_num].columns for feat in const_feats)


    def set_lags(self, lags):
        '''
        function to change lags used for extracting auto-regressors, while
        maintaining the same set of houses.
        generate_clients_data must be called afterwards.
        '''
        self.env_dict['lags'] = lags
        self.feature_names = None
        for client_num, house in enumerate(self.houses):
            # construct the regression data frame
            house.construct_regression_df(lags=self.lags, months=self.months,
                                          hours=self.hours, step_ahead=1)
            # add house to the list of houses in the city
            self.houses[client_num]=house
            # check feature names
            if self.feature_names is not None:
                assert self.feature_names == house.feature_names
            else:
                self.feature_names = house.feature_names


    #def reconstruct_env(self, clients_config, clients_data_tuple)




if __name__ == "__main__":
    # generate data from each mode
    city = CityPV_UniModal(
                city_name='Lausanne',
                tilt_mean=None, az_mean=None, tilt_std=0.1, az_std=0.1,
                weather_dev=0.1, irrad_std=0.1, altitude_dev=0.1,
                shadow_peak_red=1, random_state=None)
    city.simulate_pv(
                num_clients=5, lags=None,
                months=None, hours=None)
    city.construct_regression_matrices(
                m_train=50, exclude_last_year=True,
                train_years=[2018, 2019], remove_constant_cols=True)
    print(city.feature_names)
    #print(city.clients_data_tuple[0][0])


