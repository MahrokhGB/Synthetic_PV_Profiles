import numpy as np
import os, sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'raw_input')

sys.path.append(DATA_DIR)
sys.path.append(PROJECT_DIR)
from data_gen_assistive_functions.simulate_profiles import*
from data_gen_assistive_functions.construct_dataset import*


class PVDataset():

    def __init__(self, mu_t, cov_t, lags,
                 latitude, longitude, altitude,
                 random_state=None):
        '''
        Class for simulating PV generation profiles for multiple households.
        All households are located at the given geographical latitude, longitude, and altitude.
        NOTE: currently, we are using 2 years of data from Lausanne.
        NOTE: the first year of data is used for training and the second year for validation
        The tilt and azimuth of PV cells are varried among the households to simulate different installation setups.
        Tilt and azimuth are sampled from a two-dimentional independent Gaussian distribution.
        inputs:
        - mu_t (list of length 2): mean of the Gaussian distribution over [tilt, azimuth]. # TODO: list or np array?
        - cov_t (np array 2 x 2): covariance of the Gaussian distribution over [tilt, azimuth].
                 A diagonal covariance means that tilt and azimuth are independently selected for each household.
        - lags (list of ints greater than 0): lags used to construct auto-regressors.
        - latitude, longitude, altitude (double): location of the household.
        - random_state: fix to enable replicating the results over mutiple runs
        '''
        if random_state == None:
            self.random_state = np.random
        else:
            self.random_state = random_state
        self.lags = lags
        self.mu_t, self.cov_t = mu_t, cov_t
        self.latitude, self.longitude, self.altitude = latitude, longitude, altitude
        self.feature_names = None


    def generate_clients_data(self, num_clients, months, hours):
        '''
        Generate data for a given number of clients over a selected time interval
        NOTE: only diagonal cov has been implemented
        inputs:
        - num_clients (int): number of clients to simulate
        - months (list of int between 1 and 12, 1 is Jan): list of months to simulate data in
        - hours (list of int between 0 and 23, 0 is midnight): list of hours of day to simulate data
        returns:
        - clients_data_tuple: list of num_clients tuples. each tuple is (x_train, y_train, x_valid, y_valid)
        - clients_train_ts: list of complete (24h * selected_months) generation profile over the first (train) year for each client.
        - clients_valid_ts: list of complete (24h * selected_months) generation profile over the second (validation) year for each client.
        NOTE: for clients_train_ts and clients_valid_ts filtering by hour of day is not performed. These time series are useful for visualization.
        '''
        assert len(self.mu_t) == 2 # tilt and azimuth
        assert self.cov_t.shape == (2,2)

        # sample tilts and azimuths
        mean_tilt, mean_azimuth = self.mu_t
        var_tilt, var_azimuth = np.diagonal(self.cov_t) # assumes tilts and azimuths are independent
        self.tilts = np.random.normal(mean_tilt, var_tilt**0.5, num_clients)            # sample tilts for num_clients households
        self.azimuths = np.random.normal(mean_azimuth, var_azimuth**0.5, num_clients)   # sample azimuths for num_clients households
        self.feature_names = None
        clients_data_tuple = [None]*num_clients # one tuple per client, each tuple is (x_train, y_train, x_valid, y_valid)
        clients_train_ts = [None]*num_clients
        clients_valid_ts = [None]*num_clients
        for client_num in range(num_clients):
            data_power=generate_power_data(latitude=self.latitude,
                                           longitude=self.longitude,
                                           altitude=self.altitude,
                                           tilt=self.tilts[client_num],
                                           azimuth=self.azimuths[client_num],
                                           years=True)
            # H_sun is highly correlated with hour of day (x, y) and day of year x
            del data_power['dayofy_x']
            del data_power['hourofd_x']
            del data_power['hourofd_y']
            data_train=data_power[data_power.year==2018] # year 2018 for train
            data_valid=data_power[data_power.year==2019] # year 2019 for validation
            del data_train['year']
            del data_valid['year']

            # record data frames before filtering to hours
            clients_train_ts[client_num] = data_train[['p_mp', 'hour_day', 'month']]
            clients_valid_ts[client_num]  = data_valid[['p_mp', 'hour_day', 'month']]

            # filter hours and add lags
            _, X_train, y_train, feat_names_train = cons_database_some_hours(data_power=data_train,
                                                                                    lags=self.lags, months=months,
                                                                                    hours=hours, filter_night=True)
            _, X_valid, y_valid, feat_names_valid     = cons_database_some_hours(data_power=data_valid,
                                                                                     lags=self.lags, months=months,
                                                                                     hours=hours, filter_night=True)

            # remove constant columns
            const_cols = [(ind, col) for ind, col in enumerate(X_train.columns) if X_train[col].nunique() == 1]
            for ind, col_name in const_cols:
                del X_train[col_name]
                del X_valid[col_name]
                np.delete(y_train, ind)
                np.delete(y_valid, ind)
                del feat_names_train[ind]
                del feat_names_valid[ind]

            # check feature names
            if self.feature_names is not None:
                assert self.feature_names == feat_names_train
            else:
                self.feature_names = feat_names_train
            assert feat_names_valid == feat_names_train

            clients_data_tuple[client_num] = (X_train.values, y_train, X_valid.values, y_valid)

        return clients_data_tuple, clients_train_ts, clients_valid_ts



class PVDatasetMix(PVDataset):
    def __init__(self, mu_t, cov_t, lags,
                 latitude, longitude, altitude,
                 random_state=None):
        '''
        uses a mixture of Gaussians as the distribution over tilts and azimuths.
        '''
        assert len(mu_t) == len(cov_t)
        super().__init__(mu_t=mu_t, cov_t=cov_t, lags=lags,
                 latitude=latitude, longitude=longitude, altitude=altitude,
                 random_state=random_state)


    def generate_clients_data(self, num_clients, months, hours, weight_modes=[1]):
        '''
        function to generate clients data from the environment.
        '''
        # check if a list is given for weight modes
        try:
            _ = (e for e in weight_modes)
        except TypeError:
            weight_modes = [weight_modes]
        # number of modes, clients per mode
        num_modes = len(weight_modes)
        num_clients_per_mode = [int(weight_modes[i]*num_clients) for i in np.arange(num_modes)]

        # generate data from each mode
        for mode_num in np.arange(num_modes):
            task_environment = PVDataset(mu_t=self.mu_t[mode_num], cov_t=self.cov_t[mode_num], lags=self.lags,
                                         latitude=self.latitude, longitude=self.longitude, altitude=self.altitude,
                                         random_state=self.random_state)

            data_temp, train_ts_temp, valid_ts_temp = task_environment.generate_clients_data(num_clients=num_clients,
                                                               months=months, hours=hours)
            # concatenate modes
            if mode_num == 0:
                clients_data, clients_train_ts, clients_valid_ts = data_temp, train_ts_temp, valid_ts_temp
            else:
                clients_data = [*clients_data, *data_temp]
                clients_train_ts = [*clients_train_ts, *train_ts_temp]
                clients_valid_ts  = [*clients_valid_ts,  *valid_ts_temp]

            # features name
            if self.feature_names is None:
                self.feature_names = task_environment.feature_names
            else:
                assert self.feature_names == task_environment.feature_names
        return clients_data, clients_train_ts, clients_valid_ts

