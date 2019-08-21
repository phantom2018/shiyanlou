import numpy as np
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


from sklearn.svm import SVR
dist1 = dist[0:5] #which are close to the sea
dist2 = dist[5:10] #which are far from the sea
dist1 = [[x] for x in dist1]
dist2 = [[x] for x in dist2]
temp_max1 = temp_max[0:5]#each close-to-sea city's max tempe
temp_max2 = temp_max[5:10]
#diaoyong SVR function, given that use xianxingnihe
svr_lin1 = SVR(kernel='linear', C=1e3)
svr_lin2 = SVR(kernel='linear', C=1e3)
#do nihe
svr_lin1.fit(dist1, temp_max1)
svr_lin2.fit(dist2, temp_max2)
xp1 = np.arange(10, 100, 10).reshape((9, 1))
xp2 = np.arange(50, 400, 50).reshape((7, 1))
yp1 = svr_lin1.predict(xp1)
yp2 = svr_lin2.predict(xp2)
#draw
fig, ax = plt.subplots()
ax.set_xlim(0, 400)
ax.plot(xp1, yp1, c='b', label='Strong sea effect')
ax.plot(xp2, yp2, c='g', label='Light sea effect')
ax.plot(dist, temp_max,  'ro')
print(svr_lin1.coef_)
print(svr_lin1.intercept_)
print(svr_lin2.coef_)
print(svr_lin2.intercept_)

from scipy.optimize import fsolve
#define the first nihe line
def line1(x):
    a1 = svr_lin1.coef_[0][0]
    b1 = svr_lin1.intercept_[0]
    return a1*x + b1

def line2(x):
    a2 = svr_lin2.coef_[0][0]
    b2 = svr_lin2.intercept_[0]
    return a2*x + b2

def findIntersection(fun1, fun2, x0):
    return fsolve(lambda x : fun1(x) - fun2(x), x0)

result = findIntersection(line1, line2, 0.0)
print("[x, y] = [ %d, %d ]" % (result, line1(result)))
#x = [0, 10, 20, ..., 300]
x = np.linspace(0, 300, 31)
plt.plot(x, line1(x), x, line2(x), result, line1(result), 'ro')

