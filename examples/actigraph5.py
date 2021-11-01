
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

#code to analyse IMU data to estimate gait step length and distance

def root_mean_square(data1):
    sq_array = []
    for i in range(len(data1)):
        sq_array.append(data1[i] * data1[i])
    rootmeansq = np.sqrt(np.sum(sq_array) / len(data1))
    return rootmeansq

def median_absolute_deviation(data1):
	#find median
    med = np.median(data1)
	#deviation around median
    deviation_around_median = []
    for i in range(len(data1)):
        deviation_around_median.append(np.abs(data1[i] - np.median(data1)))
	#get median of absolute deviation
    med_abs_dev = np.median(deviation_around_median)
    return med_abs_dev

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    medn = np.median(diff)
    mad = median_absolute_deviation(diff)

    plt.scatter(mean, diff, *args, **kwargs)
	#normal Tukey plot (mean, variances)
    plt.axhline(md,           color='blue', linestyle='-')
    plt.axhline(md + 1.96*sd, color='red', linestyle='--')
    plt.axhline(md - 1.96*sd, color='red', linestyle='--')
	#median + MAD 
    #plt.axhline(medn,           color='blue', linestyle='-')
    #plt.axhline(medn + mad, color='red', linestyle='-')
    #plt.axhline(medn - mad, color='red', linestyle='-')
    #plt.axhline(medn + 2.0*mad, color='red', linestyle='--')
    #plt.axhline(medn - 2.0*mad, color='red', linestyle='--')
    #plt.axhline(medn + 3.0*mad, color='pink', linestyle='--')
    #plt.axhline(medn - 3.0*mad, color='pink', linestyle='--')
    plt.title('Bland-Altman plot')
    #plt.title('Bland-Altman plot. Blue=Median,red=Median±MAD\n red dash=Median±2MAD,pink dash=Median±3MAD')
    plt.xlabel('average of two measures')
    plt.ylabel('difference between two measures')

#algorithm to detect walking periods. Locomotor band is from 0.5-3 Hz
def walk_detection(freq_bins, timestamps, spec):
    print("freq bins = ",freq_bins)
    print("length of timestamps = ",np.shape(timestamps))
    print("length of freq bins = ",np.shape(freq_bins))
    print("size of spectral matrix",np.shape(spec))
    #steps_periods = []
    #threshold = 1.5#signal in locomotor band as function of signal at all frequencies
    signal_ratio = []
    for i in range(len(timestamps)):
        locoband_sum = 0
        all_bins_sum = 0
        for j in range(len(freq_bins)):
            all_bins_sum = all_bins_sum + spec[j][i]
            if(freq_bins[j] > 0.5 and freq_bins[j] < 3.0):
                locoband_sum = locoband_sum + spec[j][i]
        signal_ratio.append(locoband_sum / all_bins_sum)
        #signal_ratio.append(locoband_sum)
    #max_signal_ratio = np.max(signal_ratio)
    #for i in range(len(signal_ratio)):
    #    signal_ratio[i] = signal_ratio[i]
    #fig = plt.figure()
    #plt.plot(timestamps,signal_ratio)
    #plt.title("signal ratio")
    #plt.show()
    return signal_ratio

mean_excel = [1.3060,1.5630,1.3320,1.3450,1.4780,1.2820,1.4050,1.2970,1.5450,1.4500,1.2280,1.2170,1.2280,1.4040,1.2430,1.3380,1.3190,1.2610,1.2660,1.5160]
python_median = [1.2960,1.2960,1.1970,1.2200,1.1680,1.2370,1.2140,1.1740,1.3740,1.3540,1.0780,1.1900,1.2690,1.4750,1.2490,1.4810,1.1600,1.1090,1.0970,1.2720]
python_10perc = [1.3750,1.4440,1.3010,1.2880,1.3090,1.4100,1.3190,1.2480,1.5830,1.4240,1.2020,1.2700,1.3440,1.6110,1.5930,1.5330,1.2370,1.2480,1.1860,1.4560]
python_median_plus_2MAD = [1.4300,1.5000,1.3700,1.3100,1.3500,1.4300,1.3800,1.3100,1.5600,1.4900,1.2700,1.3200,1.3600,1.6400,1.5200,1.5600,1.2800,1.3100,1.2200,1.5700]
python_5perc = [1.3960,1.4980,1.3230,1.3070,1.3480,1.4370,1.3590,1.2740,1.6090,1.4360,1.2310,1.3000,1.3540,1.6330,1.6620,1.5450,1.2590,1.2800,1.2240,1.4850]
for i in range(len(python_median)):
    python_median[i] = python_median[i] * 1.1
bland_altman_plot(mean_excel,python_median)
#bland_altman_plot(mean_excel,python_median_plus_2MAD)

'''
mean_excel = [1.3060,1.5630,1.3320,1.3450,1.4780,1.2820,1.4050,1.2970,1.5450,1.4500,1.2280,1.2170,1.2280,1.4040,1.2430,1.3380,1.3190,1.2610,1.2660,1.5160]
python_median_with_turns = [1.57, 1.64, 1.48, 1.55, 1.59, 1.55, 1.51, 1.41, 1.62, 1.60, 1.34, 1.47, 1.49, 1.68, 1.49, 1.74, 1.44, 1.44, 1.41, 1.53]
for i in range(len(python_median_with_turns)):
    python_median_with_turns[i] = python_median_with_turns[i] * 0.9
bland_altman_plot(mean_excel,python_median_with_turns)
'''
#read in data - using Sheraz dataset

print('reading data...\n')
index1 = 3
with h5py.File('h5_files/subj_' + str(index1) + '.h5', "r") as f:
    ts = np.array(f.get('time'))
    fsamp = f.get('sfreq')[()]
    rawAccel = np.array(f.get('acceleration'))

rawAccel.shape
fsamp.shape

print("rawAccel shape = ",rawAccel.shape)

ax = []
ay = []
az = []

#alternative accel arrangement

for i in range(rawAccel.shape[0]):
    ax.append(float(rawAccel[i][0]))
    #ay.append(float(rawAccel[i][1]) * 9.81)#could try that to stop phase wrap in pitch
    ay.append(float(rawAccel[i][1]))
    az.append(float(rawAccel[i][2]))

accel_modulus = []
for i in range(len(ax)):
    accel_modulus.append(np.sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]))


#alternative accel arrangement - select only subset of "study-Fetisov_subject-b3e4b71f/study-Fetisov_subject-b3e4b71f_utcTimes-1570922101-to-1571009378.h5"
'''
for i in range(rawAccel.shape[0]):
    if(i > 15200 and i < 16200):
        ts.append(float(rawAccel[i][0] - rawAccel[0][0]))
        ay.append(float(rawAccel[i][1]) * 9.81)
    #ay.append(float(rawAccel[i][1]) * 9.81)#could try that to stop phase wrap in pitch
        ax.append(float(rawAccel[i][2]) * -9.81)
        az.append(float(rawAccel[i][3]) * 9.81)
'''
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

#find gravity using low pass filter (moving average)

gravity_interval = 128*2#in datapoints#works quite well
#gravity_interval = 20#for the 128 Hz sampling rate
#gravity_interval = 15

ax_grav = []
ay_grav = []
az_grav = []

for i in range(len(ax)):
    if(i > gravity_interval/2 and i < (len(ax) - gravity_interval/2)):
        ax_grav.append(np.mean(ax[int(i-gravity_interval/2):int(i+gravity_interval/2)]))
        ay_grav.append(np.mean(ay[int(i-gravity_interval/2):int(i+gravity_interval/2)]))
        az_grav.append(np.mean(az[int(i-gravity_interval/2):int(i+gravity_interval/2)]))
    else:
        ax_grav.append(ax[i])
        ay_grav.append(ay[i])
        az_grav.append(az[i])
        #ax_grav.append(float('nan'))
        #ay_grav.append(float('nan'))
        #az_grav.append(float('nan'))

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

#convert to global frame

pitch = []
roll = []
yaw = []
pitch_deg = []
roll_deg = []
ax_global = []
ay_global = []
az_global = []
ax_lin_global = []
ay_lin_global = []
az_lin_global = []

#declare rotation matrix
w, h = 3, 3;
rotmat = [[0 for x in range(w)] for y in range(h)] 
tmp = [0.0,0.0,0.0]
'''
#first rotation
ax_d = []
az_d = []

for i in range(len(ax)):
    pitch_angle = math.atan(az_grav[i]/ax_grav[i])
    az_d.append(az[i] * math.cos(pitch_angle) - ax[i] * math.sin(pitch_angle))
    ax_d.append(az[i] * math.sin(pitch_angle) + ax[i] * math.cos(pitch_angle))

fig = plt.figure()
plt.plot(ax,'b--')
plt.plot(az,'r--')
plt.plot(ax_d,'b')
plt.plot(az_d,'r')
plt.title('first rotation')
plt.show()

#second rotation

'''


#rot_matrix = [cos(theta)*cos(psy) -cos(phi)*sin(psy)+sin(phi)*sin(theta)*cos(psy) sin(phi)*sin(psy)+cos(phi)*sin(theta)*cos(psy);
#                cos(theta)*sin(psy) cos(phi)*cos(psy)+sin(phi)*sin(theta)*sin(psy) -sin(phi)*cos(psy)+cos(phi)*sin(theta)*sin(psy);
#                -sin(theta) sin(phi)*cos(theta) cos(phi)*cos(theta)];

for i in range(len(ax)):
    #pitch.append(math.atan(az_grav[i]/ax_grav[i]))#assuming standard orientation
    #roll.append(math.atan(az_grav[i]/ay_grav[i]))#assuming standard orientation
    #pitch.append(math.atan(az_grav[i]/ax_grav[i]))#assuming real orientation
    #roll.append(0.0*math.atan(ay_grav[i]/ax_grav[i]))#assuming real orientation
    roll.append(math.atan2(ay_grav[i],az_grav[i]))
    pitch.append(math.atan2(-ax_grav[i],np.sqrt(ay_grav[i] * ay_grav[i] + az_grav[i] * az_grav[i])))
    yaw.append(0.0)
    phi = roll[i]
    theta = pitch[i]
    psy = yaw[i]
    pitch_deg.append(pitch[i]*(180/3.1415926))
    roll_deg.append(roll[i]*(180/3.1415926))
    rotmat[0][0] = math.cos(theta) * math.cos(psy)
    rotmat[0][1] = -1.0*math.cos(phi) * math.sin(psy) + math.sin(phi) * math.sin(theta) * math.cos(psy)
    rotmat[0][2] = math.sin(phi) * math.sin(psy) + math.cos(phi) * math.sin(theta) * math.cos(psy)
    rotmat[1][0] = math.cos(theta) * math.sin(psy)
    rotmat[1][1] = math.cos(phi) * math.cos(psy) + math.sin(phi) * math.sin(theta) * math.sin(psy)
    rotmat[1][2] = -1.0*math.sin(phi) * math.cos(psy) + math.cos(phi) * math.sin(theta) * math.sin(psy)
    rotmat[2][0] = -1.0 * math.sin(theta)
    rotmat[2][1] = math.sin(phi) * math.cos(theta)
    rotmat[2][2] = math.cos(phi) * math.cos(theta)
    #rotmat[0][0] = math.cos(yaw[i]) * math.cos(pitch[i])
    #rotmat[0][1] = math.cos(yaw[i]) * math.sin(pitch[i]) * math.sin(roll[i]) - math.sin(yaw[i]) * math.cos(roll[i])
    #rotmat[0][2] = math.cos(yaw[i]) * math.sin(pitch[i]) * math.cos(roll[i]) + math.sin(yaw[i]) * math.sin(roll[i])
    #rotmat[1][0] = math.sin(yaw[i]) * math.sin(pitch[i])
    #rotmat[1][1] = math.sin(yaw[i]) * math.sin(pitch[i]) * math.sin(roll[i]) + math.cos(yaw[i]) * math.cos(roll[i])
    #rotmat[1][2] = math.sin(yaw[i]) * math.sin(pitch[i]) * math.cos(roll[i]) - math.cos(yaw[i]) * math.sin(roll[i])
    #rotmat[2][0] = math.sin(pitch[i])*-1.0
    #rotmat[2][1] = math.cos(pitch[i]) * math.sin(roll[i])
    #rotmat[2][2] = math.cos(pitch[i]) * math.cos(roll[i])
    tmp = np.matmul(rotmat,[ax[i],ay[i],az[i]])
    ax_global.append(tmp[0])
    ay_global.append(tmp[1])
    az_global.append(tmp[2])
    tmp = np.matmul(rotmat,[ax_lin[i],ay_lin[i],az_lin[i]])
    ax_lin_global.append(tmp[0])
    ay_lin_global.append(tmp[1])
    az_lin_global.append(tmp[2])

fig = plt.figure()
plt.plot(pitch_deg,'b')
plt.plot(roll_deg,'r')
plt.title('blue=pitch,red=roll')
plt.show()

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

#4th order Butterworth low-pass filter cut-off 20 Hz

#sos = scipy.signal.butter(4, 0.2, btype='low',output='sos')
#ax_lp = signal.sosfilt(sos, ax)

#filtfilt for zero phase

b20hz,a20hz = scipy.signal.butter(4, 0.2, btype='low')#this is set for 100 Hz sampling rate
ax_lp20hz = signal.filtfilt(b20hz,a20hz, ax)
ax_lin_lp20hz = signal.filtfilt(b20hz,a20hz, ax_lin)

#b5hz,a5hz = scipy.signal.butter(4, 0.05, btype='low')#this is set for 100 Hz sampling rate
#b5hz,a5hz = scipy.signal.butter(4, 0.15625, btype='low')#this is set for 32 Hz sampling rate
b5hz,a5hz = scipy.signal.butter(4, 0.039062, btype='low')#this is set for 128 Hz sampling rate
ax_lp5hz = signal.filtfilt(b5hz,a5hz, ax)
#butterworth_signal = signal.filtfilt(b5hz,a5hz, ax_lin)#local frame
butterworth_signal = signal.filtfilt(b5hz,a5hz, az_lin_global)#global frame

fig = plt.figure()
plt.plot(az_lin,'b')
plt.plot(az_lin_global,'r')
plt.plot(butterworth_signal,'k')
plt.title("blue=lin accel x, red=lin accel x (global)")
plt.xlabel('datapoints')
plt.ylabel('N')
plt.show()
'''
fig = plt.figure()
plt.plot(ax,'b')
#plt.plot(ax_lp20hz,'r')
plt.plot(ax_lp5hz,'k')
#plt.title("accel x")
plt.title('blue=lin accel x, black=4th order Butterworth filtfilt 5 Hz window')
plt.xlabel('datapoints')
plt.show()
'''
fig = plt.figure()
#plt.plot(ax_lin,'b')
#plt.plot(ax_lin_lp20hz,'r')
plt.plot(butterworth_signal,'k')
plt.title("accel x")
plt.xlabel('datapoints')
plt.title('linear accel')
plt.show()

#FFT

#Fs = 100#in Hz - 100 for Luo et al. (2020), 32 Hz for Fetisov dataset, 128 Hz for Shiraz dataset (.h5 files)
Fs = 128

#3D plot
#freq_bins, timestamps, spec = signal.spectrogram(ax, Fs)
freq_bins, timestamps, spec = signal.spectrogram(ax, Fs,nperseg=300)#can set length of window (3 seconds should be good)

signal_ratio = walk_detection(freq_bins, timestamps, spec)

fft_step_flag = []
for i in range(len(ts)):
    fft_step_flag.append(1.0)
fft_signal_ratio_threshold = 0.4
#timestamp_window_width = 2.25#in seconds
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
#ax.plot_surface(freq_bins[:, None], timestamps[None, :], spec, cmap=cm.coolwarm)
#ax.plot_surface(freq_bins[:, None], timestamps[None, :], 10.0*np.log10(spec), cmap=cm.coolwarm)
ax.plot_surface(freq_bins[:, None], timestamps[None, :], spec, cmap=cm.coolwarm)
plt.xlabel('freq. (Hz)')
plt.ylabel('time (arbitrary units)')
#plt.zlabel('log10 power')
plt.show()

#step detection----------------

#flag periods of steps (FFT based)

print("shape freq_bins = ",freq_bins.shape)
print("shape timestamps = ",timestamps.shape)
print("shape spec = ",spec.shape)

print("freq. = ",freq_bins)

#locomotor band is from 0.5 Hz to 3 Hz
locomotor_band_lower_limit = 0.5
locomotor_band_upper_limit = 3.0

locomotor_band_lower_limit_idx = []
locomotor_band_upper_limit_idx = []

for i in range(len(freq_bins)):
    if(freq_bins[i] < locomotor_band_lower_limit):
        locomotor_band_lower_limit_idx = i
    if(freq_bins[i] < locomotor_band_upper_limit):
        locomotor_band_upper_limit_idx = i

print("freq. limits between ",locomotor_band_lower_limit_idx," and ",locomotor_band_upper_limit_idx," , corresponding to ",freq_bins[locomotor_band_lower_limit_idx]," and ",freq_bins[locomotor_band_upper_limit_idx]," Hz.")
'''
threshold = 2.0#threshold for ratio of locomotor band signal vs total signal, before the window is flagged as 'steps'
step_period = []

for i in range(len(timestamps)):
    print("shape spec = ",spec.shape,"i=",i)
    for j in range(locomotor_band_upper_limit_idx - locomotor_band_lower_limit_idx):
        print("locomotor band = ",spec[locomotor_band_lower_limit_idx+j][i])
    print("mean locomotor band = ",np.mean(spec[locomotor_band_lower_limit_idx:locomotor_band_upper_limit_idx][i]))
    print("mean total power = ",np.mean(spec[:][i]))
    print("ratio = ",(np.mean(spec[locomotor_band_lower_limit_idx:locomotor_band_upper_limit_idx][i]) / np.mean(spec[:][i])) )
    if((np.mean(spec[locomotor_band_lower_limit_idx:locomotor_band_upper_limit_idx][i]) / np.mean(spec[:][i])) > threshold):
        print("steps detected at ",timestamps[i])
        step_period.append(timestamps[i])

print("step periods: ",step_period)
'''
#for i in range(timestamps.shape):
#    if(np.sum()):
        

#find peaks and trophs
'''
peaks, _ = scipy.signal.find_peaks(ax_lin, distance=10)
print("peaks = ",peaks)
'''

reverse_butterworth_signal = True

if(reverse_butterworth_signal):
    for i in range(len(butterworth_signal)):
        butterworth_signal[i] = -1.0 * butterworth_signal[i]

#envelope based turn detection algorithm

abs_butterworth_signal = []
low_pass_abs_butterworth_signal = []
steady_stride_flag = []
for i in range(len(butterworth_signal)):
    abs_butterworth_signal.append(np.abs(butterworth_signal[i]))

interval = 150

#low pass filter (moving average)
for i in range(len(abs_butterworth_signal)):
    if(i > (interval/2) and i < (len(abs_butterworth_signal) - (interval/2))):
        low_pass_abs_butterworth_signal.append(np.mean(abs_butterworth_signal[int(i-(interval/2)):int(i+(interval/2))]))
    else:
        low_pass_abs_butterworth_signal.append(0.0)

#find gradient of low-pass filtered signal
d_low_pass_abs_butterworth_signal = []

for i in range(10):
    d_low_pass_abs_butterworth_signal.append(0.0)

for i in range(len(abs_butterworth_signal)):
    if(i > 10):
        d_low_pass_abs_butterworth_signal.append(low_pass_abs_butterworth_signal[i] - low_pass_abs_butterworth_signal[i-10])

low_pass_d_low_pass_abs_butterworth_signal = []
d_interval = 100

for i in range(len(d_low_pass_abs_butterworth_signal)):
    if(i > (d_interval/2) and i < (len(d_low_pass_abs_butterworth_signal) - (d_interval/2))):
        low_pass_d_low_pass_abs_butterworth_signal.append(np.mean(d_low_pass_abs_butterworth_signal[int(i-(d_interval/2)):int(i+(d_interval/2))]))
    else:
        low_pass_d_low_pass_abs_butterworth_signal.append(0.0)

cutoff_threshold = np.quantile(low_pass_abs_butterworth_signal,0.75)
print("cutoff threshold = ",cutoff_threshold)
'''
for i in range(len(butterworth_signal)):
    if(i > (interval/2) and i < (len(butterworth_signal) - (interval/2)) and low_pass_abs_butterworth_signal[i] > cutoff_threshold):
        steady_stride_flag.append(1)
    else:
        steady_stride_flag.append(0)
'''
for i in range(len(butterworth_signal)):
    if(i > (interval/2) and i < (len(butterworth_signal) - (interval/2)) and np.abs(low_pass_d_low_pass_abs_butterworth_signal[i]) > 0.02):
        steady_stride_flag.append(1)
    else:
        steady_stride_flag.append(0)

#remove short periods of steady stride flag

steady_stride_flag_no_short_periods = []
for i in range(len(steady_stride_flag)):
    steady_stride_flag_no_short_periods.append(steady_stride_flag[i])

min_turn_interval = 30

steady_stride_flag_sum = []
steady_stride_flag_sum.append(0)

for i in range(len(steady_stride_flag)):
    if(i > 0):
        if(steady_stride_flag[i]==1):
            steady_stride_flag_sum.append(steady_stride_flag_sum[i-1] + 1)
        else:
            if(steady_stride_flag_sum[len(steady_stride_flag_sum)-1] < min_turn_interval):
                for j in range(steady_stride_flag_sum[len(steady_stride_flag_sum)-1]):
                    steady_stride_flag_no_short_periods[i - (steady_stride_flag_sum[len(steady_stride_flag_sum)-1] - j)] = 0
            steady_stride_flag_sum.append(0)

#detect peaks and trophs

peak_array = []
troph_array = []

for i in range(len(butterworth_signal)):
    if(i > 1 and i < (len(butterworth_signal)-1) and butterworth_signal[i] > butterworth_signal[i-1] and butterworth_signal[i] > butterworth_signal[i+1]):
        peak_array.append(i)
    if(i > 1 and i < (len(butterworth_signal)-1) and butterworth_signal[i] < butterworth_signal[i-1] and butterworth_signal[i] < butterworth_signal[i+1]):
        troph_array.append(i)

print("peaks=",peak_array)
print("trophs=",troph_array)

print("minimum = ",np.minimum(len(peak_array),len(troph_array)))

print("diff between peaks: ",np.diff(peak_array))

#stride detection algorithm (using Figure 6b of Bui et al. 2018 to define a single stride)

stride_begin = []
stride_end = []

for i in range(len(troph_array)-2):
    if(butterworth_signal[troph_array[i]] < butterworth_signal[troph_array[i+1]]):
        stride_begin.append(troph_array[i])
        stride_end.append(troph_array[i+2]-1)

for i in range(len(stride_begin)):
    print(["stride begin = ",stride_begin[i]," , stride end = ",stride_end[i]])

#1. for each entry in peak array, find the trough array values either side of it
#2. calc FWHM using these boundaries

idx_troph_left = []
idx_troph_right = []

for i in range(len(peak_array)):
    delta = []
    idx_troph_left_tmp = -1
    idx_troph_right_tmp = -1
    for j in range(len(troph_array)):
        delta.append(peak_array[i] - troph_array[j])
    delta_min = 99999
    for j in range(len(delta)):
        if(delta[j]<0 and np.abs(delta[j]) < delta_min):
            delta_min = delta[j]
            idx_troph_right_tmp = j
    delta_min = 99999
    for j in range(len(delta)):
        if(delta[j]>0 and np.abs(delta[j]) < delta_min):
            delta_min = delta[j]
            idx_troph_left_tmp = j
    idx_troph_left.append(troph_array[idx_troph_left_tmp])
    idx_troph_right.append(troph_array[idx_troph_right_tmp])
    #print(["troph left idx = ",troph_array[idx_troph_left_tmp]," , peak = ",peak_array[i]," , troph right idx = ",troph_array[idx_troph_right_tmp]])

#get FWHM

fwhm_left_idx = []
fwhm_right_idx = []

for i in range(len(peak_array)):
	#check for sanity
    fwhm_left_idx_tmp = -1
    fwhm_right_idx_tmp = -1
    if(idx_troph_left[i] < peak_array[i] and idx_troph_right[i] > peak_array[i]):
        left_max = butterworth_signal[peak_array[i]] - butterworth_signal[idx_troph_left[i]]
        right_max = butterworth_signal[peak_array[i]] - butterworth_signal[idx_troph_right[i]]
        for j in range(peak_array[i] - idx_troph_left[i]):
            if((butterworth_signal[idx_troph_left[i]+j]-butterworth_signal[idx_troph_left[i]]) < (left_max)/2.0):
                fwhm_left_idx_tmp = idx_troph_left[i]+j
        for j in range(idx_troph_right[i] - peak_array[i]):
            if((butterworth_signal[peak_array[i]+j]-butterworth_signal[idx_troph_right[i]]) > (right_max)/2.0):
                fwhm_right_idx_tmp = peak_array[i]+j
    fwhm_left_idx.append(fwhm_left_idx_tmp)
    fwhm_right_idx.append(fwhm_right_idx_tmp)
    #print(["troph left idx = ",idx_troph_left[i],",left FWHM idx=",fwhm_left_idx_tmp," , peak = ",peak_array[i],",right FWHM idx=",fwhm_right_idx_tmp," , troph right idx = ",idx_troph_right[i]])

#get stride length by calculating difference between the two FWHM points
double_support_length = []
for i in range(len(fwhm_left_idx)):
    if(fwhm_left_idx[i] > 0 and fwhm_right_idx[i] > 0):
        double_support_length.append(fwhm_right_idx[i] - fwhm_left_idx[i])

for i in range(len(double_support_length)):
    double_support_length[i] = double_support_length[i] * (1/128)

print("double support time (sec) = ",double_support_length)

#remove those during turn periods

double_support_lengths_no_turns_indices_to_delete = []

for i in range(len(double_support_length)):
    if(np.sum(steady_stride_flag_no_short_periods[fwhm_left_idx[i]:fwhm_right_idx[i]]) > 0):
        double_support_lengths_no_turns_indices_to_delete.append(i)

print("double support indices to delete = ",double_support_lengths_no_turns_indices_to_delete)

double_support_length_no_turns = []

for i in range(len(double_support_length)):
    double_support_length_no_turns.append(double_support_length[i])

for index in sorted(double_support_lengths_no_turns_indices_to_delete, reverse=True):
    del double_support_length_no_turns[index]

#add in walking bout detection condition from FFT

for i in range(len(steady_stride_flag_no_short_periods)):
    if(fft_step_flag[i] == 0):
        steady_stride_flag_no_short_periods[i] = 0

#plot time series with peaks and trophs

fig = plt.figure()
plt.plot(butterworth_signal,'b')
#plt.plot(abs_butterworth_signal,'r')
plt.plot(low_pass_abs_butterworth_signal,'k')
plt.plot(d_low_pass_abs_butterworth_signal,'m')
plt.plot(low_pass_d_low_pass_abs_butterworth_signal,'m--')
plt.plot(steady_stride_flag,'k--')
plt.plot(steady_stride_flag_no_short_periods,'g--')

plt.plot(fft_step_flag,'r--')

plt.plot(peak_array, butterworth_signal[peak_array], "xr")
plt.plot(troph_array, butterworth_signal[troph_array], "xk")
plt.plot(fwhm_left_idx,butterworth_signal[fwhm_left_idx],"or")
plt.plot(fwhm_right_idx,butterworth_signal[fwhm_right_idx],"ob")
plt.plot()
plt.xlabel('datapoints')
plt.ylabel('N')
plt.title('peaks (red x) and troughs (black x). blue=low-pass filtered lin accel')
plt.show()

#calc step length using Weinberg et al. (2002)

step_lengths_weinberg = []
step_lengths_weinberg_x2 = []
stride_lengths_weinberg = []
stride_lengths_weinberg_no_turns = []
stride_lengths_weinberg_no_turns_indices_to_delete = []
K = 1.0#correction factor

for i in range(np.minimum(len(peak_array),len(troph_array))):
    step_lengths_weinberg.append((np.power((butterworth_signal[peak_array[i]] - butterworth_signal[troph_array[i]]),(1/4))) * K)
    step_lengths_weinberg_x2.append(step_lengths_weinberg[i]*2.0)

#for i in range(int(np.floor((np.minimum(len(peak_array),len(troph_array))) / 2))):
#    stride_lengths_weinberg.append(step_lengths_weinberg[i*2] + step_lengths_weinberg[i*2 + 1])

stride_time = []

for i in range(len(stride_begin)):
    stride_lengths_weinberg.append(np.power(np.max(butterworth_signal[stride_begin[i]:stride_end[i]]) - np.min(butterworth_signal[stride_begin[i]:stride_end[i]]),0.25) * K)
    stride_time.append(stride_end[i] - stride_begin[i])
    if(np.sum(steady_stride_flag_no_short_periods[stride_begin[i]:stride_end[i]]) > 0):
        stride_lengths_weinberg_no_turns_indices_to_delete.append(i)

print("mean stride time = ",np.mean(stride_time),"  datapoints. ")
print("median stride time = ",np.median(stride_time),"  datapoints. ")

print(["deleting indices ", stride_lengths_weinberg_no_turns_indices_to_delete, "..."])

for i in range(len(stride_lengths_weinberg)):
    stride_lengths_weinberg_no_turns.append(stride_lengths_weinberg[i])

for index in sorted(stride_lengths_weinberg_no_turns_indices_to_delete, reverse=True):
    del stride_lengths_weinberg_no_turns[index]

print("total stride count: ",len(stride_lengths_weinberg))
print("total stride count (no turns): ",len(stride_lengths_weinberg_no_turns))

print("Weinberg (2002) method step lengths = ",step_lengths_weinberg)
print("Weinberg (2002) mean step length=",np.mean(step_lengths_weinberg))
print("Weinberg (2002) step length STD=",np.std(step_lengths_weinberg))
print("Weinberg (2002) mean stride length=",np.mean(stride_lengths_weinberg))
print("Weinberg (2002) stride length STD=",np.std(stride_lengths_weinberg))
print("Weinberg (2002) median stride length=",np.median(stride_lengths_weinberg))

print("Weinberg (2002) mean stride length (no turns)=",np.mean(stride_lengths_weinberg_no_turns))
print("Weinberg (2002) stride length STD (no turns)=",np.std(stride_lengths_weinberg_no_turns))
print("Weinberg (2002) median stride length (no turns)=",np.median(stride_lengths_weinberg_no_turns))

print("Weinberg (2002) total distance travelled = ",np.sum(step_lengths_weinberg))
print("Weinberg (2002) total distance travelled (from median) = ",np.median(step_lengths_weinberg) * len(peak_array))

#Kim et al. (2004) step length method - needs accurate identification of when a step begins and ends

ground_truth_dataset1 = [1.33,1.37,1.39,1.35,1.33,1.33,1.3,1.29,1.32,1.32,1.25,1.31,1.34,1.31,1.27,1.29,1.29,1.28,1.3,1.27,1.26,1.3,1.24,1.34,1.31,1.36,1.29,1.26,1.34,1.24]
ground_truth_dataset2 = [1.57,1.54,1.56,1.53,1.58,1.58,1.59,1.48,1.61,1.61,1.6,1.53,1.6,1.6,1.57,1.54,1.64,1.57,1.57,1.59,1.56,1.56,1.56,1.55,1.54,1.48,1.56,1.54,1.53]
ground_truth_dataset3 = [1.4,1.4,1.35,1.36,1.4,1.37,1.34,1.36,1.34,1.29,1.35,1.31,1.34,1.3,1.36,1.34,1.35,1.31,1.32,1.3,1.3,1.28,1.32,1.32,1.32,1.33,1.33,1.3,1.34,1.31,1.35,1.32,1.3,1.32,1.31,1.32,1.35,1.32]

#output step array

output_str = ''

for i in range(len(stride_lengths_weinberg)):
    if(i==0):
        output_str = output_str + str(stride_lengths_weinberg[i])
    else:
        output_str = output_str + "," + str(stride_lengths_weinberg[i])

text_file = open("stride_lengths.txt","w")
text_file.write("%s\n" % output_str)
text_file.close()
output_str = ''

print("ground truth mean (set 1) = ",np.mean(ground_truth_dataset1))
print("ground truth mean (set 2) = ",np.mean(ground_truth_dataset2))
print("ground truth mean (set 2) = ",np.mean(ground_truth_dataset3))

print("top 90% quantile = ",np.quantile(stride_lengths_weinberg, 0.1))
print("median quantile = ",np.quantile(stride_lengths_weinberg, 0.5))
print("top 10% quantile = ",np.quantile(stride_lengths_weinberg, 0.9))
print("top 5% quantile = ",np.quantile(stride_lengths_weinberg, 0.95))

print("median = ",np.median(stride_lengths_weinberg))
print("MAD = ",median_absolute_deviation(stride_lengths_weinberg))
print("2 MAD = ",2.0*median_absolute_deviation(stride_lengths_weinberg))
print("3 MAD = ",3.0*median_absolute_deviation(stride_lengths_weinberg))

print("abs(median + 2 MAD) = ",np.abs(np.median(stride_lengths_weinberg) + 2.0*median_absolute_deviation(stride_lengths_weinberg)))
print("abs(median + 3 MAD) = ",np.abs(np.median(stride_lengths_weinberg) + 3.0*median_absolute_deviation(stride_lengths_weinberg)))

print("Excel median = ",np.median(ground_truth_dataset1))
print("Excel MAD = ",median_absolute_deviation(ground_truth_dataset1))
print("Excel 2 MAD = ",2.0*median_absolute_deviation(ground_truth_dataset1))
print("Excel 3 MAD = ",3.0*median_absolute_deviation(ground_truth_dataset1))

print("RMS linear vertical acceleration = ",root_mean_square(az_lin_global))
print("RMS modulus acceleration = ",root_mean_square(accel_modulus))

fig = plt.figure()
plt.plot(step_lengths_weinberg,'g--')
plt.plot(stride_lengths_weinberg,'b--')
plt.plot(step_lengths_weinberg_x2,'r--')
plt.plot(ground_truth_dataset1,'b')
plt.plot(ground_truth_dataset2,'r')
plt.plot(ground_truth_dataset3,'g')
plt.title('blue=Weinberg(2002) stride,red=strides (truth), \n green=Weinberg(2002) steps')
plt.show()

#histogram

_ = plt.hist(step_lengths_weinberg, bins='auto')
#_ = plt.hist(step_lengths_weinberg, bins=5)
plt.title('Histogram of stride lengths')
plt.ylabel('event counts')
plt.xlabel('stride lengths (m)')
#plt.title("Histogram with 'auto' bins")
plt.show()

#histogram (no turns)

_ = plt.hist(stride_lengths_weinberg_no_turns, bins='auto')
plt.title('Histogram of stride lengths (no turns)')
plt.ylabel('event counts')
plt.xlabel('stride lengths (m)')
plt.show()

#histogram (double support time interval)

_ = plt.hist(double_support_length, bins='auto')
plt.title('Histogram of double support time intervals')
plt.ylabel('event counts')
plt.xlabel('double support time interval (sec)')
plt.show()

#histogram (double support time interval, no turns)

_ = plt.hist(double_support_length_no_turns, bins='auto')
plt.title('Histogram of double support time intervals (no turns)')
plt.ylabel('event counts')
plt.xlabel('double support time interval (sec)')
plt.show()

#rotate to Earth frame (both linear and total accel)

#resample to 50 Hz

#detrend signal with linear regression model

#---walking bout detection---

#downsample to 10 Hz

#FFT - look for rise in PSD in 0.6-2.5 Hz band above threshold

#---stride length estimation---

#locate IC and FC 

#filtering of IC and FC based on loading, stance, and cycle constraint

#vertical height change estimation---

#step length estimation - using inverted pendulum model 2 * sqrt(2*s_h * dh - dh^2)

#---statistical aggregation---

#stride length aggregation and asymmetry computation (diff to previous stride)

#aggregate over all strides and bouts