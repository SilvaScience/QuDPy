# -*- coding: utf-8 -*-
"""
Created on Mon May 12 17:56:13 2025

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.ticker import MaxNLocator

def log_scale(z):
    """
    Simple function for rescaling the 2D input matrix to log scale.
    Note: the negative numbers are downshifted by 1 and the positive numbers are upshifted by 1 to remove numbers
    between -1 and 1.
    """
    x, y = np.shape(z)
    for n in range(x):
        for m in range(y):
            if z[n, m] >= 0:
                z[n, m] = np.log(z[n, m]+1)
            else:
                z[n, m] = -np.log(-z[n, m]+1)
    return z

def plot_2D(spectra_col, spectra_x,spectra_y,color_map, title , diagram ,quadrant = 'All',diagonal=[True,True],scale='linear'
            ,Zoom_val_x=[0,0], Zoom_val_y=[0,0]):
    
    
    # if quadrant == '1':
    #     spectra_col = [x[(len(x)//2 + len(x) % 2):, (len(x)//2 + len(x) % 2):] for x in spectra_col]
    #     spectra_x = [0, scan_range[1], 0, scan_range[3]]
    #     spectra_y = 
    # elif quadrant == '2':
    #     spectra_col = [x[:(len(x)//2 + len(x) % 2), (len(x)//2 + len(x) % 2):] for x in spectra_list]
    #     spectra_x = [scan_range[0], 0, 0, scan_range[3]]
    #     spectra_y = 
    # elif quadrant == '3':
    #     spectra_col = [x[:(len(x)//2 + len(x) % 2), :(len(x)//2 + len(x) % 2)] for x in spectra_list]
    #     spectra_x  = [scan_range[0], 0, scan_range[2], 0]
    #     spectra_y = 
    # elif quadrant == '4':
    #     spectra_col = [x[(len(x)//2 + len(x) % 2):, :(len(x)//2 + len(x) % 2)] for x in spectra_list]
    #     spectra_x = [0, scan_range[1], scan_range[2], 0]
    #     spectra_y = 
    
    if quadrant == 'Zoom':
            for a in range(len(spectra_x[0])):
                if spectra_x[0][a] < Zoom_val_x[0]:
                    val_x_min = a
                if spectra_x[0][a] < Zoom_val_x[1]:
                    val_x_max = a
            first_elements = np.array([a[0] for a in spectra_y])
            for a in range(len(first_elements)):
                if first_elements[a] < Zoom_val_y[0]:
                    val_y_min = a
                if first_elements[a] < Zoom_val_y[1]:
                    val_y_max = a
            spectra_col = [x[val_x_min:val_x_max, val_y_min:val_y_max] for x in spectra_col]
                
            spectra_x = [x[val_x_min:val_x_max] for x in spectra_x]
            s_y = []
            for a in range(val_y_max-val_y_min):
               s_y.append(spectra_y[val_y_min+a]) 
            spectra_y = s_y
    print(np.shape(spectra_y),np.shape(spectra_x),np.shape(spectra_col))
    if scale == 'log':
        spectra_col = np.array([log_scale(s) for s in spectra_col])
    data_real = np.real(spectra_col)
    data_imag = np.imag(spectra_col)
    data_abs = np.abs(spectra_col)
   
    
    data = []
    titles = ['real', 'imag', 'abs']
    for k in range(len(spectra_col)):
        data.append(data_real[k]/np.max(data_abs[k]))
        data.append(data_imag[k]/np.max(data_abs[k]))
        data.append(data_abs[k]/np.max(data_abs[k]))
    c = -1
    diagram.append('Total') 
    for a in range(len(data)):
        if a%3 ==0:
            c=c+1
        fig, ax = plt.subplots()
        subplot_title = (title + ' ' + titles[a % 3] + ' ' + diagram[c])
        ax.set_title(subplot_title)
        if diagonal[0]:
            plt.plot([np.min(spectra_x), np.max(spectra_x)], [np.max(spectra_y), np.min(spectra_y)], '--', color="black", linewidth=0.5)
        if diagonal[1]:
            plt.plot([np.min(spectra_x), np.max(spectra_x)], [np.min(spectra_y), np.max(spectra_y)], '--', color="black", linewidth=0.5)

        levels = MaxNLocator(nbins=100).tick_values(data[a].min(), data[a].max())
        cf = ax.contourf(spectra_x, spectra_y, data[a], levels=levels,cmap=color_map) 
        fig.colorbar(cf, ax=ax)
        plt.show()
        
    return 