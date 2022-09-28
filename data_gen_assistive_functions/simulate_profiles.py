import math, os, pvlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#Setting the geographical characteristics of the location
latitude=46.520
longitude=6.632
name='Lausanne'
altitude=496
timezone='Etc/GMT-1'

#Selecting the modules used as well as the inverter
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

def generate_power_data(latitude, longitude, altitude, tilt, azimuth, years):

    '''
    INPUTS: latitude, longitude and altitude of the location; tilt and azimuth angles of the installation
    (azimuth measured from north, i.e. N is 0 and S is 180). Years is a boolean variable that selects if the output data frame has the year
    or not. This was added later because year filtering was also needed but it can be easily deleted.

    OUTPUTS: This function reads the data from the csv downoaled from PVGIS and returns a data frame with
    the power at maximum power point, meteorological and time and date data (Note that some data has been made sinusoidal)
    '''

    #Reading the data file: DOWNLOAD DATA FROM PVGIS USING TILT=0 i.e. DATA OF IRRADIANCE ON NORMAL PLANE
    data = pd.read_csv(PROJECT_DIR+'/raw_input/Timeseries_46.520_6.632_SA2_0deg_0deg_2005_2020.csv',
                       skiprows=8, skipfooter=12, sep ='\,') #data file
    del data['Int']
    data['time'] = pd.to_datetime(data['time'], format='%Y%m%d:%H%M')
    data.set_index('time', inplace=True)
    system = {'module': module, 'inverter': inverter,
          'surface_azimuth': azimuth}

    #Using PVLib to obtain a data frame with the DC electrical outputs of the module/installation
    system['surface_tilt'] = tilt
    solpos = pvlib.solarposition.get_solarposition(
        time=data.index,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        temperature=data['T2m'],
        pressure=pvlib.atmosphere.alt2pres(altitude),
    )

    dni_extra = pvlib.irradiance.get_extra_radiation(data.index)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = pvlib.atmosphere.alt2pres(altitude)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    aoi = pvlib.irradiance.aoi(
        system['surface_tilt'],
        system['surface_azimuth'],
        solpos["apparent_zenith"],
        solpos["azimuth"],
    )

    total_irradiance = pvlib.irradiance.get_total_irradiance(
        system['surface_tilt'],
        system['surface_azimuth'],
        solpos['apparent_zenith'],
        solpos['azimuth'],
        data['Gb(i)'],
        data['Gd(i)']+(data['Gb(i)']*np.cos(solpos['apparent_zenith'])),#Global horizontal irradiance (ghi)
        data['Gd(i)'],
        dni_extra=dni_extra,
        model='haydavies',
    )

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
        module,
    )

    for i in range(effective_irradiance.size): #To avoid negative effective irradiance that results in NaNs in the power output
        if effective_irradiance[i]<0:
            effective_irradiance[i]=0

    #Obtaining DC outputs and creating the data frame data_power with the
    #maximum power output and meteorological data
    dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module)
    power=dc['p_mp']
    data_power = pd.merge(data, power, on='time', how='left')
    data_power=data_power.reset_index()
    data_power['date'] = data_power['time'].dt.date
    data_power['day_year'] = data_power['time'].dt.dayofyear
    data_power['hour_day'] = data_power['time'].dt.hour
    data_power['month'] = data_power['time'].dt.month
    if years:
        data_power['year'] = data_power['time'].dt.year

    data_power['hourofd_x'] = np.sin(data_power['hour_day']/24*2*math.pi)
    data_power['hourofd_y'] = np.cos(data_power['hour_day']/24*2*math.pi)
    data_power['dayofy_x'] = np.sin(data_power['day_year']/365*2*math.pi)
    data_power['dayofy_y'] = np.cos(data_power['day_year']/365*2*math.pi)
    del data_power['date']
    del data_power['time']
    del data_power['day_year']
    #del data_power['hour_day']
    del data_power['Gb(i)']
    del data_power['Gd(i)']
    del data_power['Gr(i)']

    shiftPos = data_power.pop("p_mp")
    # insert column on the 1st position
    data_power.insert(0, "p_mp", shiftPos)

    return data_power
