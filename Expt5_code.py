# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 19:19:17 2021

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

expt_data = genfromtxt('Expt5.csv', delimiter=',')

# time series of pressure, temp and times at which the former two are sampled

time = moving_average(expt_data[:, 0], 36) # seconds
pressure = moving_average(expt_data[:, 1], 36) # kPa
temperature = moving_average(expt_data[:, 2] + 273.15, 36) # Kelvin

# the rate of change of pressure

dPdt = np.diff(pressure)/5
dPdt = np.insert(dPdt, 0, dPdt[0], axis=0)

dpdt_MA = moving_average(dPdt, 36)

d2pdt = np.diff(dPdt)
# the number of sampling points

numPoints = time.size

    
# Since we never reach a steady state. I estimate the pressure drop when the 
# rate of pressure drop is lesser than 0.1 kPa/ 5 seconds (epsilon)

epsilon_Pdrop = 0.44/5

temp_pressure = np.where(np.abs(dPdt) < epsilon_Pdrop, pressure, 0)

# we only start searching for the steady state pressure after 5 * 100 seconds

index = np.arange(1600, numPoints+1)

# creating variables for when the system reaches steady state (ss)

ss_time = -1
ss_temperature = -1
ss_pressure = -1

for i in index:
    if temp_pressure[i] != 0 and temp_pressure[i-1] !=0:
        ss_time = time[i]
        ss_pressure = pressure[i]
        ss_temperature = temperature[i]
        break

# maximum pressure attained
   
max_pressure = np.max(pressure)

# pressure drop = max pressure - steady state pressure

pressureDrop = (max_pressure - ss_pressure) # kPa

# using n = PV/R*T (ideal gas)

molesHydrates_ideal_initial =  max_pressure * volume/(R * ss_temperature)
molesHydrates_ideal_final =  ss_pressure * volume/(R * ss_temperature)

# this function returns the difference between the pressure (steady state) and
# the pressure calculated as a function of the moles of gas
# the goal of this function coupled with fsolve is to find the moles of gas
# at a given P and T

def molesFromSRK_Error(moles, pressure, temperature):

    
    # Select Thermodynamic Parameters of CO2
    
    T_c = 304.35 # K
    P_c = 7380  # kPa
    
    # Antoine Constants
    
    A = 6.81228
    B = 1301.679
    C = -3.494
    
    # Vapor pressure
    
    P_vap_CO2 = np.exp(A - (B/(0.7*T_c - C)))
    
    # accentric factor
    
    omega = -1 - np.log10(P_vap_CO2/P_c)
    
    K = 0.480 + 1.574 * omega - 0.176 * omega**2
    
    # SRK parameter  = a
    
    a = ((0.42748/P_c) * (R * T_c)**2) * (1 + K*(1 - 
        (temperature/T_c)**0.5))**2
    
    # SRK paramater = b
    
    b = 0.08664 * R * T_c/P_c
    
    # returning the difference between the desired pressure (s.s.) and the
    # calculated pressure
    # fsolve solves this function to get the error as close to zero as possible
    
    error = pressure - (R * temperature * moles)/(volume * (1 - 
                       b * (moles/volume))) + (a * moles**2)/((volume**2)*(1 + 
                           b * (moles/volume)))
    
    return error

molesHydrates_SRK = fsolve(molesFromSRK_Error, molesHydrates_ideal_initial, 
                           (max_pressure, 4 + 273.15))[0] - fsolve(molesFromSRK_Error, 
                           molesHydrates_ideal_final, 
                           (ss_pressure, ss_temperature))[0]
fig, ax1 = plt.subplots()

ax1.plot(time, pressure, color = 'red', label = 'Pressure (kPa)')
ax1.set_ylim([1600, 2200])
ax1.set_ylabel('Pressure (kPa)')
ax1.legend(loc = 'upper right')
ax2 = ax1.twinx()
ax2.set_ylabel('Temperature (K)')
ax1.set_xlabel('Time (seconds)')
ax2.plot(time, temperature, color='blue', label = 'Temperature (K)')
ax2.legend(loc = 'upper center')
plt.show()

print('The moles of hydrates formed: ' + str(molesHydrates_SRK) + ' moles')


    
