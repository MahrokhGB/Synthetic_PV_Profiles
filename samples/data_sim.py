from third_party.Synthetic_PV_Profiles import CityPV_MultiModal
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import os, sys, copy


X_LOW = -5
X_HIGH = 5

Y_HIGH = 2.5
Y_LOW = -2.5

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_DIR + '/experiments/inputs/'
PV_DIR = os.path.join(PROJECT_DIR, 'third_party', 'Synthetic_PV_Profiles')
sys.path.insert(1, PROJECT_DIR)
sys.path.insert(1, DATA_DIR)
sys.path.insert(1, PV_DIR)


class MetaDataset():

    def __init__(self, random_state=None):
        if random_state == None:
            self.random_state = np.random
        else:
            self.random_state = random_state
        self.x_mean, self.y_mean = None, None
        self.x_std, self.y_std = None, None


    def generate_meta_train_data(self, n_tasks: int, n_samples: int) -> list:
        raise NotImplementedError


    def generate_meta_test_data(self, n_tasks:int, n_samples_context: int, n_samples_test: int) -> list:
        raise NotImplementedError



# --------------------------------------------
class PVDataset(MetaDataset):
    def __init__(self, env_options, random_state=None):
        super().__init__(random_state)

        # check provided options
        req_fields = [# required for building the task env
                'city_names', 'tilt_std', 'az_std', 'weather_dev',
                'irrad_std' ,'altitude_dev', 'shadow_peak_red',
                # required for simulating pv
                'module_name', 'inverter_name'
                # required for generating data
                'train_scenarios']
        optional_fields = [# optional for building the task env
                'tilt_mean', 'az_mean',
                # optional for simulating pv data
                'lags', 'hours', 'months',
                'num_clients', 'num_clients_per_mode',
                'use_station_irrad_direct', 'use_station_irrad_diffuse', 'delay_irrad',
                # optional for generating datasets
                'remove_constant_cols'
                ]
        #TODO: train_scenarios must have m_train, train_years, and optionally 'exclude_last_year'
        assert [x in env_options.keys() for x in req_fields]
        assert [x in req_fields + optional_fields for x in env_options.keys()]
        assert 'num_clients' in env_options.keys() or 'num_clients_per_mode' in env_options.keys()

        # parse optional options
        for key in optional_fields:
            if not key in env_options.keys():
                if key in ['use_station_irrad_direct', 'use_station_irrad_diffuse', 'delay_irrad', 'remove_constant_cols']:
                    env_options[key] = True
                else:
                    env_options[key] = None
        if env_options['num_clients_per_mode'] is not None:
            if env_options['num_clients'] is None:
                env_options['num_clients'] = np.sum(env_options['num_clients_per_mode'])
            else:
                assert env_options['num_clients'] == np.sum(env_options['num_clients_per_mode'])
        if env_options['num_clients'] is not None and env_options['num_clients_per_mode'] is None:
            env_options['num_clients_per_mode'] = [int(env_options['num_clients']/len(env_options['city_names']))]*len(env_options['city_names'])
            env_options['num_clients'] = np.sum(env_options['num_clients_per_mode'])

        # create task environment
        self.task_environment = CityPV_MultiModal(city_names=env_options['city_names'],
                tilt_mean=env_options['tilt_mean'], az_mean=env_options['az_mean'],
                tilt_std=env_options['tilt_std'], az_std=env_options['az_std'],
                weather_dev=env_options['weather_dev'], irrad_std=env_options['irrad_std'],
                altitude_dev=env_options['altitude_dev'], shadow_peak_red=env_options['shadow_peak_red'],
                random_state=self.random_state)

        # summarize info
        env_options['info'] = '{:2.0f} households at '.format(env_options['num_clients'])+ " ".join(env_options['city_names']) + ' - '
        for key in ['tilt_std', 'az_std', 'weather_dev', 'irrad_std', 'altitude_dev', 'shadow_peak_red']:
            if not isinstance(env_options[key], list): # TODO
                env_options['info'] += key+': {:.1f}, '.format(env_options[key])
        for name_str in ['module_name', 'inverter_name']:
            if len(env_options[name_str])==1:
                env_options['info'] += 'same ' + name_str + ', '
            elif len(env_options[name_str])==env_options['num_clients']:
                env_options['info'] += 'different ' + name_str + ', '
            else:
                env_options['info'] += + name_str + 'not specified, '

        self.env_dict = env_options
        self._simulate_pv()


    def _simulate_pv(self):
        self.task_environment.simulate_pv(num_clients_per_mode=self.env_dict['num_clients_per_mode'],
                                          module_name=self.env_dict['module_name'], inverter_name=self.env_dict['inverter_name'],
                                          lags=self.env_dict['lags'], months=self.env_dict['months'],
                                          hours=self.env_dict['hours'],
                                          use_station_irrad_direct=self.env_dict['use_station_irrad_direct'],
                                          use_station_irrad_diffuse=self.env_dict['use_station_irrad_diffuse'],
                                          delay_irrad=self.env_dict['delay_irrad'])
        # properties of task env that might have changed
        self.env_dict['months'] = self.task_environment.months
        self.env_dict['hours'] = self.task_environment.hours
        self.env_dict['lags'] = self.task_environment.lags

        # new properties
        self.env_dict['clients_config']=self.task_environment.clients_config


    def generate_clients_data(self):
        self.env_dict['feature_names'] = None
        for scenario_name, scenario in self.env_dict['train_scenarios'].items():
            self.task_environment.construct_regression_matrices(
                            m_train=scenario['m_train'], train_years=scenario['train_years'],
                            exclude_last_year=scenario['exclude_last_year'],
                            remove_constant_cols=self.env_dict['remove_constant_cols'])
            self.env_dict['train_scenarios'][scenario_name]['clients_data'] = copy.deepcopy(self.task_environment.clients_data_tuple)
            self.env_dict['train_scenarios'][scenario_name]['time_series'] = copy.deepcopy(self.task_environment.clients_time_series)
            if self.env_dict['feature_names'] is None:
                self.env_dict['feature_names'] = self.task_environment.feature_names
            else:
                assert self.env_dict['feature_names'] == self.task_environment.feature_names
        return self.env_dict

'''
def set_lags(env_dict, lags, random_state):

    # function to change lags used for extracting auto-regressors, while
    # maintaining the same set of houses.
    # generate_clients_data must be called afterwards.

    # create task environment
    task_environment = CityPV_MultiModal(city_names=env_dict['city_names'],
            tilt_mean=env_dict['tilt_mean'], az_mean=env_dict['az_mean'],
            tilt_std=env_dict['tilt_std'], az_std=env_dict['az_std'],
            weather_dev=env_dict['weather_dev'], irrad_std=env_dict['irrad_std'],
            altitude_dev=env_dict['altitude_dev'], shadow_peak_red=env_dict['shadow_peak_red'],
            random_state=random_state)

    self.task_environment.set_lags(lags)
'''


def remove_feature(env_dict, feature_name, in_place=False):
    '''
    removes a specified feature from task environemnt.
    inplace operation
    '''
    if not in_place:
        env_dict = copy.deepcopy(env_dict)
    assert isinstance(feature_name, str)
    assert feature_name in env_dict['feature_names']
    # find index of features except this feature
    feature_index_to_keep = np.delete(
                                np.arange(len(env_dict['feature_names'])),
                                env_dict['feature_names'].index(feature_name)
                                )
    # remove from feature_names
    env_dict['feature_names'].remove(feature_name)
    # no need to remove from time_series
    # remove from X_train and X_test at each scenario and for all clients
    for client_num in np.arange(env_dict['num_clients']):
        for scenario in env_dict['train_scenarios']:
            x_train, y_train, x_valid, y_valid = env_dict['train_scenarios'][scenario]['clients_data'][client_num]
            x_train = x_train[:, feature_index_to_keep]
            x_valid = x_valid[:, feature_index_to_keep]
            env_dict['train_scenarios'][scenario]['clients_data'][client_num] = (x_train, y_train, x_valid, y_valid)
            assert x_train.shape[1] == len(env_dict['feature_names'])
            assert x_valid.shape[1] == len(env_dict['feature_names'])
    if in_place:
        return
    else:
        return env_dict






'''
    def _generate_data_2y(self, env_dict):
        shuffle=False
        clients_data_2y = [None]*env_dict['num_clients']
        if env_dict['train_frac_2y'] is not None:
            for client_num in np.arange(env_dict['num_clients']):
                x_train, y_train, x_valid, y_valid = env_dict['clients_data_full'][client_num]
                x_all = np.concatenate((x_train, x_valid), axis=0)
                y_all = np.concatenate((y_train, y_valid), axis=0)
                if shuffle:
                    train_inds = self.random_state.choice(np.arange(x_all.shape[0]),
                                                    size=int(x_all.shape[0]*env_dict['train_frac_2y']), replace=False)
                else: # use the last portion for validation
                    train_inds = np.arange(int(x_all.shape[0]*env_dict['train_frac_2y']))
                valid_inds = list(set(np.arange(x_all.shape[0])) - set(train_inds))
                # divide again and put back
                clients_data_2y[client_num] = (x_all[train_inds, :], y_all[train_inds, :],
                                            x_all[valid_inds, :], y_all[valid_inds, :])
        env_dict['clients_data_2y']=clients_data_2y
        return env_dict


    def _generate_data_red(self, env_dict):
        # --- data reduced ----
        clients_data_red = [None] * env_dict['num_clients']
        if env_dict['m_train_red'] is not None and env_dict['m_valid_red'] is not None:
            for client_num in np.arange(env_dict['num_clients']):
                data = env_dict['clients_data_full'][client_num]
                # randomly select train_inds and valid_inds
                train_inds = self.random_state.choice(np.arange(data[0].shape[0]),
                                                size=env_dict['m_train_red'], replace=False)
                valid_inds = self.random_state.choice(np.arange(data[3].shape[0]),
                                                size=env_dict['m_valid_red'], replace=False)
                # new train and validation data
                clients_data_red[client_num] = (data[0][train_inds, :], data[1][train_inds],
                                            data[2][valid_inds, :], data[3][valid_inds])
                # mark samples used for train or validation
                is_test_2018 = [True]*len(env_dict['clients_ts_2018'][client_num].index) # mark all as test
                for i in train_inds:
                    is_test_2018[i] = False            # train points are not test
                is_test_2019 = [True]*len(env_dict['clients_ts_2019'][client_num].index) # mark all as test
                for i in valid_inds:
                    is_test_2019[i] = False            # valid points are not test
                env_dict['clients_ts_2018'][client_num]['is_test'] = is_test_2018
                env_dict['clients_ts_2019'][client_num]['is_test'] = is_test_2019
        env_dict['clients_data_red']=clients_data_red
        return env_dict
'''



# -----------------------------------------------------

if __name__ == "__main__":

    random_state = 3

    # geographical characteristics of the location
    latitude=46.520
    longitude=6.632
    city_name='Lausanne'
    altitude=496
    timezone='Etc/GMT-1'

    # clients distribution
    num_modes = 1
    weight_modes = [1/num_modes] * num_modes
    mean_tilt  = latitude
    mean_azimuth = 180
    sigma_tilt = 15
    sigma_azimuth  = 45
    mu_t = [[mean_tilt, mean_azimuth]]
    cov_t = [np.diag([sigma_tilt**2, sigma_azimuth **2])]

    # FL info
    num_clients = 25
    num_clients_per_mode = [int(weight_modes[i]*num_clients) for i in np.arange(num_modes)]*num_modes

    # Configuration w.r.t. data
    generate_normalized_data = True

    # generate data from each mode
    task_environment = PVDatasetMix(mu_t=mu_t, cov_t=cov_t,
                                    city_name=city_name,
                                    random_state=random_state)


    print('[INFO] generating data for {:2.0f} clients'.format(num_clients))
    clients_data, clients_train_ts, clients_test_ts = task_environment.generate_clients_data(num_clients=num_clients,
                                                          weight_modes=weight_modes)
    #print(task_environment.true_data_dist)
    #print(y_obs.shape)



