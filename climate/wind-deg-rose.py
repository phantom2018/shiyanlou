mport numpy as np
import pandas as pd
import datetime

df_ferrara = pd.read_csv('WeatherData/ferrara_270615.csv')
df_milano = pd.read_csv('WeatherData/milano_270615.csv')
df_mantova = pd.read_csv('WeatherData/mantova_270615.csv')
df_ravenna = pd.read_csv('WeatherData/ravenna_270615.csv')
df_torino = pd.read_csv('WeatherData/torino_270615.csv')
df_asti = pd.read_csv('WeatherData/asti_270615.csv')
df_bologna = pd.read_csv('WeatherData/bologna_270615.csv')
df_piacenza = pd.read_csv('WeatherData/piacenza_270615.csv')
df_cesena = pd.read_csv('WeatherData/cesena_270615.csv')
df_faenza = pd.read_csv('WeatherData/faenza_270615.csv')

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser



def showRoseWind(values,city_name,max_value):
    N = 8

    # theta = [pi*1/4, pi*2/4, pi*3/4, ..., pi*2]
    theta = np.arange(2 * np.pi / 16, 2 * np.pi, 2 * np.pi / 8)
    radii = np.array(values)
    # ?????????
    plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)

    # ????????????? rgb ??x??????color?????
    colors = [(1-x/max_value, 1-x/max_value, 0.75) for x in radii]

    # ??????
    plt.bar(theta, radii, width=(2*np.pi/N), bottom=0.0, color=colors)

    # ????????
    plt.title(city_name, x=0.2, fontsize=20)

showRoseWind(hist,'Ravenna',max(hist))
hist, bin = np.histogram(df_ferrara['wind_deg'],8,[0,360])
print(hist)
showRoseWind(hist,'Ferrara', max(hist))

