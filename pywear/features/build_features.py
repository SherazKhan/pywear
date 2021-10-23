
import numpy as np

def walk_detection(freq_bins, timestamps, spec):
    print("freq bins = ",freq_bins)
    print("length of timestamps = ",np.shape(timestamps))
    print("length of freq bins = ",np.shape(freq_bins))
    print("size of spectral matrix",np.shape(spec))

    signal_ratio = []
    for i in range(len(timestamps)):
        locoband_sum = 0
        all_bins_sum = 0
        for j in range(len(freq_bins)):
            all_bins_sum = all_bins_sum + spec[j][i]
            if(freq_bins[j] > 0.5 and freq_bins[j] < 3.0):
                locoband_sum = locoband_sum + spec[j][i]
        signal_ratio.append(locoband_sum / all_bins_sum)

    return signal_ratio