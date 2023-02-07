import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import os, sys, copy


X_LOW = -5
X_HIGH = 5

Y_HIGH = 2.5
Y_LOW = -2.5

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_DIR + '/raw_inputs/'
sys.path.insert(1, PROJECT_DIR)
sys.path.insert(1, DATA_DIR)

from city_pv_multi_modal import CityPV_MultiModal

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


    def generate_clients_data(self, shuffle=False):
        self.env_dict['feature_names'] = None
        for scenario_name, scenario in self.env_dict['train_scenarios'].items():
            self.task_environment.construct_regression_matrices(
                m_train=scenario['m_train'], train_years=scenario['train_years'],
                valid_years=scenario['valid_years'], shuffle=shuffle,
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


# -----------------------------------------------------
# ---------------------- ICU --------------------------
# -----------------------------------------------------
class ICUDataset(MetaDataset):
    def __init__(self, target_var='GCS', random_state=None, data_dir=None):
        super().__init__(random_state)
        self.data_dir = os.path.realpath(os.path.dirname(__file__))+'/inputs/ICU/set-a' if data_dir is None else data_dir
        self.all_files_names = os.listdir(self.data_dir)
        if '.ipynb_checkpoints' in self.all_files_names: self.all_files_names.remove('.ipynb_checkpoints')
        self.target_var = target_var


    def generate_clients_data(self, num_clients, min_n_obs=1, max_n_obs=1e3,
                              min_n_test=1):
        # check inputs
        assert 0 < min_n_obs <= max_n_obs
        assert min_n_test > 0

        # init
        file_ind = 0
        client_num = 0
        data_clients = [None]*num_clients

        while client_num < num_clients:
            # read data file
            data = pd.read_csv(self.data_dir + '/'+ self.all_files_names[file_ind], sep=",", header=0)

            # only select rows measuring the target var
            data = data[data['Parameter']==self.target_var].drop('Parameter', axis=1).dropna().reset_index()

            # first day is train, the rest is test
            train_rows = [i for i, t in enumerate(data['Time']) if int(t.split(':')[0]) < 24]

            if len(train_rows) >= min_n_obs:
                # discard additional samples
                train_rows = train_rows[0: min(len(train_rows), max_n_obs)]

                # use the rest for test
                test_rows = np.arange(train_rows[-1]+1, len(data['Time']))

                # check number of test samples
                if len(test_rows) >= min_n_test:
                    train_x = np.array([int(t.split(':')[0])*60 + int(t.split(':')[1]) for t in data.iloc[train_rows]['Time']])
                    test_x  = np.array([int(t.split(':')[0])*60 + int(t.split(':')[1]) for t in data.iloc[test_rows]['Time']])
                    train_y = data.iloc[train_rows]['Value'].to_numpy()
                    test_y  = data.iloc[test_rows]['Value'].to_numpy()
                    if train_x.std() > 1e-3 and test_x.std() > 1e-3 :
                        data_clients[client_num] = (train_x, train_y, test_x, test_y)
                        client_num += 1
            # search next file
            file_ind += 1

        return data_clients



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
    task_environment = PVDataset(mu_t=mu_t, cov_t=cov_t,
                                    city_name=city_name,
                                    random_state=random_state)


    print('[INFO] generating data for {:2.0f} clients'.format(num_clients))
    clients_data, clients_train_ts, clients_test_ts = task_environment.generate_clients_data(num_clients=num_clients,
                                                          weight_modes=weight_modes)
