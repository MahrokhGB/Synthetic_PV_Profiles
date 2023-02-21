import sys, os

PROJ_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_PATH)

from .house_pv import HousePV
from .weather_station import WeatherStation
from .city_pv_uni_modal import CityPV_UniModal
from .city_pv_multi_modal import CityPV_MultiModal
from .feature_selection import *
from .utils_pv import *

# create folder for saved results
saved_res_path = os.path.join(PROJ_PATH, 'saved_results')
if not os.path.exists(saved_res_path):
    os.makedirs(saved_res_path)