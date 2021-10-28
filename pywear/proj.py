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
print("MAD = ",visualize.median_absolute_deviation(stride_lengths_weinberg))
print("2 MAD = ",2.0*visualize.median_absolute_deviation(stride_lengths_weinberg))
print("3 MAD = ",3.0*visualize.median_absolute_deviation(stride_lengths_weinberg))

print("abs(median + 2 MAD) = ",np.abs(np.median(stride_lengths_weinberg) + 2.0*visualize.median_absolute_deviation(stride_lengths_weinberg)))
print("abs(median + 3 MAD) = ",np.abs(np.median(stride_lengths_weinberg) + 3.0*visualize.median_absolute_deviation(stride_lengths_weinberg)))

print("Excel median = ",np.median(ground_truth_dataset1))
print("Excel MAD = ",visualize.median_absolute_deviation(ground_truth_dataset1))
print("Excel 2 MAD = ",2.0*visualize.median_absolute_deviation(ground_truth_dataset1))
print("Excel 3 MAD = ",3.0*visualize.median_absolute_deviation(ground_truth_dataset1))

print("RMS linear vertical acceleration = ",visualize.root_mean_square(az_lin_global))
print("RMS modulus acceleration = ",visualize.root_mean_square(accel_modulus))

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