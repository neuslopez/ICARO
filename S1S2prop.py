#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:57:50 2017

@author: neus
"""
import numpy as np
from   invisible_cities.core.system_of_units_c import SystemOfUnits
units = SystemOfUnits()
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter

### %%
class S12Prop:

    def __init__(self, S12d):
        self.S12d = S12d 
        self.prop()
    
    def _dict(self):
        return self.S12d

    def length(self):
        return len(self.S12d)
    
    def prop(self):
        self.idxt_       = [] # index array for highest phe signal in an S1/S2
        self.S12SigE_    = [] # highest phe value in peak
        self.S12Sigt_    = [] # time in mus for idxt_
        self.wS12_       = [] # width of signal (mus)
        self.tmean_      = [] # mean time
        self.E_          = [] # total energy
        
        for key,val in self.S12d.items():  # dic of events
            for key2,val2 in val.items():  # dic of peaks
                if(key == 9999): break  #------> PATCH
                if(key2 == 1):   break  #------> select one peak 
                self.idxt_     .append(np.argmax(val2[1]))
                self.S12SigE_  .append(np.amax(val2[1]))
                self.S12Sigt_  .append(val2[0][self.idxt_[-1]] / units.mus)
                self.wS12_     .append((val2[0][-1] - val2[0][0] ) / units.mus)
                self.tmean_    .append(np.mean(val2[0][:])/ units.mus)
                self.E_        .append(np.sum(val2[1][:]))
   
    def S1S2mapd(self, other):
        filt_dict = lambda x, y: dict([ (i , x[i] ) for i in x if i in set(y) ])
      
        keys_S1 = set(self.S12d.keys())
        keys_S2 = set(other.S12d.keys())
        intsect = keys_S1 & keys_S2

        S1map = filt_dict(self.S12d,intsect)
        S2map = filt_dict(other.S12d,intsect)
        return S12Prop(S1map), S12Prop(S2map)
        
#%%
#def multh(x, nbins, xmin, xmax, color="red", title="", xlabel="", ylabel="Entries"):
def multh(x, nbins, color="red", title="", xlabel="", ylabel="Entries"):
    mycolor = color
#    plt.hist(x, nbins, color = mycolor, histtype="step", alpha=0.75)
    #plt.figure(figsize=(7, 5), dpi=100)

    ## no estamos cogiendo xmin y xmax
    plt.hist(x, nbins, color = mycolor, histtype="bar", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #axes.xaxis.set_label_coords(0.95, -0.10)
    #axes.yaxis.set_label_coords(-0.1, 0.95)
    return plt

def h1(x, nbins, color="red", title="", xlabel="", ylabel="Entries",
       legend="hi"):
    mycolor = color
##    plt.hist(x, nbins, color = mycolor, histtype="step", alpha=0.75)
  #  plt.figure(figsize=(7, 5), dpi=100) 
    plt.hist(x, nbins, color = mycolor, histtype="bar", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    majorLocator = MultipleLocator(2)  # subdivisió entre numeros visibles
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(0.5)  # subdivisio entre les ticks petites

    ax = plt.axes()
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(minorLocator)
    ax.tick_params(which='both', direction='in')
    #plt.rc('font', weight='bold')
    #ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_label_coords(0.9, -0.10)
    ax.yaxis.set_label_coords(-0.1, 0.95)

    # add ticks in opposite axis
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    
   # plt.legend(legend)
    return plt






