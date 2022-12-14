import numpy as np
import pandas as pd
import os, sys, pvlib

PV_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, PV_DIR)

from utils_pv import get_city_info

class WeatherStation():

    def __init__(self, city_name, data_file_name=None):
        '''

        '''
        self.azimuth = 180
        self.tilt = get_city_info(city_name)['latitude']
        self.altitude  = get_city_info(city_name)['altitude']
        self.latitude  = get_city_info(city_name)['latitude']
        self.longitude = get_city_info(city_name)['longitude']
        self.city_name = city_name


        # load data
        #Reading the data file: DOWNLOAD DATA FROM PVGIS USING TILT=0 i.e. DATA OF IRRADIANCE ON NORMAL PLANE
        if data_file_name is None:
            data = pd.read_csv(PV_DIR+'/raw_input/weather_'+self.city_name+'.csv',
                            skiprows=8, skipfooter=12, sep ='\,', engine='python')
        else:
            data = pd.read_csv(PV_DIR+'/raw_input/'+data_file_name+'.csv',
                skiprows=8, skipfooter=12, sep ='\,', engine='python')
        del data['Int']
        data['time'] = pd.to_datetime(data['time'], format='%Y%m%d:%H%M')
        data.set_index('time', inplace=True)
        self.weather_data = data

        self._calculate_irrad()



    def _calculate_irrad(self):

        '''

        '''
        data = self.weather_data

        #Using PVLib to obtain a data frame with the DC output of the module/installation
        solpos = pvlib.solarposition.get_solarposition(
            time=data.index,
            latitude=self.latitude, longitude=self.longitude, altitude=self.altitude,
            pressure=pvlib.atmosphere.alt2pres(self.altitude)
        )

        total_irradiance = pvlib.irradiance.get_total_irradiance(
            self.tilt,
            self.azimuth,
            solpos['apparent_zenith'],
            solpos['azimuth'],
            data['Gb(i)'],
            data['Gd(i)']+(data['Gb(i)']*np.cos(solpos['apparent_zenith'])),#Global horizontal irradiance (ghi)
            data['Gd(i)'],
            dni_extra=pvlib.irradiance.get_extra_radiation(data.index),
            model='haydavies',
        )


        #for i in np.arange(total_irradiance.size):
            # negative effective irradiance result in NaNs in the power output
        #    if total_irradiance.values[i]<0:
        #        total_irradiance.values[i]=0

        self.station_irrad_direct = total_irradiance['poa_direct']
        self.station_irrad_diffuse = total_irradiance['poa_diffuse']



if __name__ == "__main__":
    weather_station = WeatherStation('Lausanne')
    print(weather_station.station_irrad_direct[10:20])