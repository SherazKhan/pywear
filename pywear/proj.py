from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.fftpack
from scipy import signal # spectrogram function

import numpy as np;
import h5py;
import json;

from visualization import visualize
from features import build_features
from data.transform import transformations

mean_excel = [1.3060,1.5630,1.3320,1.3450,1.4780,1.2820,1.4050,1.2970,1.5450,1.4500,1.2280,1.2170,1.2280,1.4040,1.2430,1.3380,1.3190,1.2610,1.2660,1.5160]
python_median = [1.2960,1.2960,1.1970,1.2200,1.1680,1.2370,1.2140,1.1740,1.3740,1.3540,1.0780,1.1900,1.2690,1.4750,1.2490,1.4810,1.1600,1.1090,1.0970,1.2720]

visualize.bland_altman_plot(mean_excel,python_median)

print('reading data...\n')
index1 = 1
with h5py.File('data/subj_' + str(index1) + '.h5', "r") as f:
    ts = np.array(f.get('time'))
    fsamp = f.get('sfreq')[()]
    rawAccel = np.array(f.get('acceleration'))

rawAccel.shape
fsamp.shape

print("rawAccel shape = ",rawAccel.shape)

ax = []
ay = []
az = []

for i in range(rawAccel.shape[0]):
    ax.append(float(rawAccel[i][0]))
    ay.append(float(rawAccel[i][1]))
    az.append(float(rawAccel[i][2]))

accel_modulus = []
for i in range(len(ax)):
    accel_modulus.append(np.sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]))

print('finished reading data. ')
print('sampling rate = ',fsamp)

ax = np.array(ax, dtype = np.float32)
ay = np.array(ay, dtype = np.float32)
az = np.array(az, dtype = np.float32)

fig = plt.figure()
plt.plot(ts,ax,'b')
plt.plot(ts,ay,'r')
plt.plot(ts,az,'g')
plt.plot(ts,((np.sqrt(ax * ax + ay * ay + az * az))),'k')
plt.title("blue=accel x, red=accel y, green=accel z, black = modulus")
plt.xlabel('time')
plt.ylabel('N')
plt.show()

#find gravity

gravity_interval = fsamp*2

gravity_total = transformations.calc_gravity(ax,ay,az,gravity_interval)

ax_grav = []
ay_grav = []
az_grav = []

for i in range(len(gravity_total)):
    if(i < (len(gravity_total)/3)):
        ax_grav.append(gravity_total[i])
    if(i >= (len(gravity_total)/3) and i < (2*(len(gravity_total)/3))):
        ay_grav.append(gravity_total[i])
    if(i >= (2*(len(gravity_total)/3))):
        az_grav.append(gravity_total[i])

print("ax grav len = ",len(ax_grav)," , ay grav len = ",len(ay_grav)," , az grav len = ",len(az_grav), " , total = ",len(gravity_total))

fig = plt.figure()
plt.plot(ax_grav,'b')
plt.plot(ay_grav,'r')
plt.plot(az_grav,'k')
plt.title('gravity components: x (blue), y (red), z (black)')
plt.xlabel('datapoints')
plt.ylabel('N')
plt.show()

fig = plt.figure()

plt.plot(ax,'b')
plt.plot(ay,'r')
plt.plot(az,'k')
plt.plot(ax_grav,'b--')
plt.plot(ay_grav,'r--')
plt.plot(az_grav,'k--')
plt.title('components + gravity')
plt.show()

#remove gravity

ax_lin = []
ay_lin = []
az_lin = []

for i in range(len(ax)):
    ax_lin.append(ax[i] - ax_grav[i])
    ay_lin.append(ay[i] - ay_grav[i])
    az_lin.append(az[i] - az_grav[i])

fig = plt.figure()
plt.plot(ax_lin,'b')
plt.plot(ay_lin,'r')
plt.plot(az_lin,'k')
plt.title('linear accelerations: accel x (blue), accel y (red), accel z (black)')
plt.xlabel('datapoints')
plt.ylabel('N')
plt.show()

ax_lin = np.array(ax_lin, dtype = np.float32)
ay_lin = np.array(ay_lin, dtype = np.float32)
az_lin = np.array(az_lin, dtype = np.float32)

accel_global = transformations.convert_to_global_frame(ax,ay,az,ax_grav,ay_grav,az_grav)
ax_global = []
ay_global = []
az_global = []

for i in range(len(accel_global)):
    if(i < (len(accel_global)/3)):
        ax_global.append(accel_global[i])
    if(i >= (len(accel_global)/3) and i < (2*(len(accel_global)/3))):
        ay_global.append(accel_global[i])
    if(i >= (2*(len(accel_global)/3))):
        az_global.append(accel_global[i])

accel_lin_global = transformations.convert_to_global_frame(ax_lin,ay_lin,az_lin,ax_grav,ay_grav,az_grav)
ax_lin_global = []
ay_lin_global = []
az_lin_global = []

for i in range(len(accel_lin_global)):
    if(i < (len(accel_lin_global)/3)):
        ax_lin_global.append(accel_lin_global[i])
    if(i >= (len(accel_lin_global)/3) and i < (2*(len(accel_lin_global)/3))):
        ay_lin_global.append(accel_lin_global[i])
    if(i >= (2*(len(accel_lin_global)/3))):
        az_lin_global.append(accel_lin_global[i])

fig = plt.figure()
plt.plot(ax,'b')
plt.plot(ax_global,'b--')
plt.plot(ay,'r')
plt.plot(ay_global,'r--')
plt.plot(az,'k')
plt.plot(az_global,'k--')
plt.show()

fig = plt.figure()
plt.plot(ax,'b')
plt.plot(((np.sqrt(ax * ax + ay * ay + az * az))) * -1,'k')
plt.plot(ax_global,'b--')
plt.title("blue=accel x, black = -modulus")
plt.xlabel('datapoints')
plt.ylabel('N')
plt.show()

#4th order Butterworth low-pass filter cut-off 5 Hz

b5hz,a5hz = scipy.signal.butter(4, 5/fsamp, btype='low')#this is set for 128 Hz sampling rate
butterworth_signal = signal.filtfilt(b5hz,a5hz, az_lin_global)#global frame

fig = plt.figure()
plt.plot(az_lin,'b')
plt.plot(az_lin_global,'r')
plt.plot(butterworth_signal,'k')
plt.title("blue=lin accel x, red=lin accel x (global)")
plt.xlabel('datapoints')
plt.ylabel('N')
plt.show()

fig = plt.figure()
plt.plot(butterworth_signal,'k')
plt.title("accel x")
plt.xlabel('datapoints')
plt.title('linear accel')
plt.show()

freq_bins, timestamps, spec = signal.spectrogram(ax, fsamp,nperseg=300)#FFT

signal_ratio = build_features.walk_detection(freq_bins, timestamps, spec)

fft_step_flag = []
for i in range(len(ts)):
    fft_step_flag.append(1.0)
fft_signal_ratio_threshold = 0.4
timestamp_window_width = (timestamps[1] - timestamps[0])/2.0
print("timestamp_window_width = ",timestamp_window_width)

for i in range(len(ts)):
    for j in range(len(timestamps)):
        if(ts[i] > (timestamps[j]-timestamp_window_width) and ts[i] < (timestamps[j]+timestamp_window_width) and signal_ratio[j] < fft_signal_ratio_threshold):
            fft_step_flag[i] = 0

fig = plt.figure()
plt.plot(ts,butterworth_signal,'k')
plt.plot(timestamps,signal_ratio,'b')
plt.plot(ts,fft_step_flag,'r')
plt.xlabel('seconds')
plt.title('blue = signal ratio (locomotor band / all bins) \n black = vertical accel (filtered) \n red = walking bout true/false from FFT')
plt.show()

# 3d plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(freq_bins[:, None], timestamps[None, :], spec, cmap=cm.coolwarm)
plt.xlabel('freq. (Hz)')
plt.ylabel('time (arbitrary units)')
plt.show()

