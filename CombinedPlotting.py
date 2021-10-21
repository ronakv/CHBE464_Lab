# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 19:34:49 2021

@author: ronak
"""

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Constants

R = 8.314 # gas constant [Pressure in kPa, volume in L]

# volume of the reactor - water - sand [L]

volume = (325 - 18.3 - 210/1.22)*10**(-3) 

# extracting data from the csv files

expt6_data = genfromtxt('Expt6.csv', delimiter=',')
expt5_data = genfromtxt('Expt5.csv', delimiter=',')
expt4_data = genfromtxt('Expt4.csv', delimiter=',')

# time series of pressure, temp and times at which the former two are sampled

time6 = moving_average(expt6_data[:, 0], 36) # seconds
pressure6 = moving_average(expt6_data[:, 1], 36) # kPa
pressure6 = pressure6 - np.max(pressure6)
temperature6 = moving_average(expt6_data[:, 2] + 273.15, 36) # Kelvin

time5 = moving_average(expt5_data[:, 0], 36) # seconds
pressure5 = moving_average(expt5_data[:, 1], 36) # kPa
pressure5 = pressure5 - np.max(pressure5)
temperature5 = moving_average(expt5_data[:, 2] + 273.15, 36) # Kelvin

time4 = moving_average(expt4_data[:, 0], 36) # seconds
pressure4 = moving_average(expt4_data[:, 1], 36) # kPa
pressure4 = pressure4 - np.max(pressure4)
temperature4 = moving_average(expt4_data[:, 2] + 273.15, 36) # Kelvin

plt.plot(time6, pressure6, label = 'Experiment 6 Pressure Drop (kPa) (3.5% salt)')
plt.plot(time5, pressure5, label = 'Experiment 5 Pressure Drop (kPa) (3.5% salt)')
plt.plot(time4, pressure4, label = 'Experiment 4 Pressure Drop (kPa) (0% salt)')
plt.ylabel('Pressure drop (kPa)')
plt.xlabel('Time (seconds)')
plt.ylim((-200, 0))
plt.legend()
plt.show()
