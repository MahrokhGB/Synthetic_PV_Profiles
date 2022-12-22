import numpy as np
import os, sys

PV_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PV_DIR)
sys.path.append(os.path.dirname(PV_DIR))

from city_pv_uni_modal import CityPV_UniModal
from utils_pv import tile_in_list


class CityPV_MultiModal(CityPV_UniModal):

    def __init__(self, city_names, tilt_std, az_std,
                 weather_dev, altitude_dev, irrad_std,
                 shadow_peak_red,
                 tilt_mean=None, az_mean=None,
                 random_state=None):
        '''
        environment of clients located at different cities
        - city_names: list of city names
        - tilt_dev, az_dev, tilt_mean, az_mean: if a single value, used for all cities.
                              if a list, shows dev or mean for households in each city.
                              see PVDataset for more info.
        '''
        if not isinstance(city_names, list):
            city_names=[city_names]
        self.city_names = city_names

        if random_state == None:
            self.random_state = np.random
        else:
            self.random_state = random_state

        # convert all to lists of the same length as city_name
        self.tilt_std  = tile_in_list(tilt_std,  len(city_names))
        self.az_std    = tile_in_list(az_std,    len(city_names))
        self.tilt_mean = tile_in_list(tilt_mean, len(city_names))
        self.az_mean   = tile_in_list(az_mean,   len(city_names))
        self.irrad_std = tile_in_list(irrad_std, len(city_names))
        self.weather_dev = tile_in_list(weather_dev, len(city_names))
        self.altitude_dev = tile_in_list(altitude_dev, len(city_names))
        self.shadow_peak_red = tile_in_list(shadow_peak_red, len(city_names))


    def simulate_pv(
            self, num_clients_per_mode,
            lags, months, hours,
            module_name='Canadian_Solar_CS5P_220M___2009_',
            inverter_name='ABB__MICRO_0_25_I_OUTD_US_208__208V_',
            use_station_irrad_direct=True, use_station_irrad_diffuse=True,
            delay_irrad=True
                    ):

        '''
        function to generate clients data from the environment.


        '''
        self.lags, self.months, self.hours = lags, months, hours
        self.feature_names=None
        # check if a list is given for num_clients_per_mode
        if not isinstance(num_clients_per_mode, list):
            num_clients_per_mode=[num_clients_per_mode]
        # number of modes, clients per mode
        assert len(num_clients_per_mode) == len(self.city_names)
        # tile module and inverter names and weather_dev
        # will be lists of length sum(num_clients_per_mode)
        module_name   = tile_in_list(module_name,   num_clients_per_mode)
        inverter_name = tile_in_list(inverter_name, num_clients_per_mode)

        # generate data from each mode
        self.houses = []
        self.clients_time_series =[]
        client_num=0 # num of the first client in the first mode
        for mode_num, city_name in enumerate(self.city_names):
            print('[INFO] generating data for ' + city_name)
            city = CityPV_UniModal(
                        city_name=self.city_names[mode_num],
                        tilt_mean=self.tilt_mean[mode_num], az_mean=self.az_mean[mode_num],
                        tilt_std=self.tilt_std[mode_num], az_std=self.az_std[mode_num],
                        weather_dev=self.weather_dev[mode_num], irrad_std=self.irrad_std[mode_num],
                        altitude_dev=self.altitude_dev[mode_num],
                        shadow_peak_red=self.shadow_peak_red[mode_num],
                        random_state=self.random_state)

            inds = np.arange(client_num, client_num+num_clients_per_mode[mode_num]) # indices corresponding to this mode
            city.simulate_pv(
                        num_clients=len(inds),
                        lags=self.lags, months=self.months, hours=self.hours,
                        module_name=[module_name[i] for i in inds],
                        inverter_name=[inverter_name[i] for i in inds],
                        use_station_irrad_direct=use_station_irrad_direct, use_station_irrad_diffuse=use_station_irrad_diffuse,
                        delay_irrad=delay_irrad)
            self.houses += city.houses
            self.clients_time_series += city.clients_time_series
            # increase num of first client in this mode
            client_num += num_clients_per_mode[mode_num]

            # concatenate modes
            if mode_num == 0:
                self.lags, self.months, self.hours = city.lags, city.months, city.hours
                self.clients_config = city.clients_config
                self.feature_names = city.feature_names
            else:
                assert self.feature_names == city.feature_names
                for key in self.clients_config.keys():
                    self.clients_config[key] = [*self.clients_config[key], *city.clients_config[key]]



if __name__ == "__main__":
    # generate data from each mode
    city = CityPV_MultiModal(
                city_names='Lausanne',
                tilt_mean=None, az_mean=None, tilt_std=0.1, az_std=0.1,
                weather_dev=0.1, irrad_std=0.1, altitude_dev=0.1,
                shadow_peak_red=1, random_state=None)
    city.simulate_pv(
                num_clients_per_mode=5, lags=None,
                months=None, hours=None)

    city.construct_regression_matrices(
                m_train=5, exclude_last_year=True,
                train_years=[2018, 2019], remove_constant_cols=False)
    clients_data = city.clients_data_tuple
    city._remove_constant_cols()