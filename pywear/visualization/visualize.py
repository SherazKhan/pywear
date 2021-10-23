import numpy as np
import math
import matplotlib.pyplot as plt

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    medn = np.median(diff)
    
    plt.scatter(mean, diff, *args, **kwargs)
	#normal Tukey plot (mean, variances)
    plt.axhline(md,           color='blue', linestyle='-')
    plt.axhline(md + 1.96*sd, color='red', linestyle='--')
    plt.axhline(md - 1.96*sd, color='red', linestyle='--')

    plt.title('Bland-Altman plot')
    plt.xlabel('average of two measures')
    plt.ylabel('difference between two measures')
    plt.show()

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

