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
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

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
        self.idxt_       = []  # convert to numpy arrays
        self.S12SigE_    = []
        self.S12Sigt_    = []
        self.wS12_       = []
        self.tmean_      = []
        self.E_      = []
        
        for key,val in self.S12d.items():
            for key2,val2 in val.items():
                if(key == 9999): break  #------> PATCH
                if(key2 == 1):   break  #------> select one peak 
                self.idxt_     .append(np.argmax(val2[1]))
                self.S12SigE_  .append(np.amax(val2[1]))
                self.S12Sigt_  .append(val2[0][self.idxt_[-1]] / units.mus)
                self.wS12_     .append((val2[0][-1] - val2[0][0] ) / units.mus)
                self.tmean_    .append(np.mean(val2[0][:]))
                self.E_    .append(np.sum(val2[1][:]))
   
    def S1S2mapd(self, other):
        filt_dict = lambda x, y: dict([ (i , x[i] ) for i in x if i in set(y) ])
      
        keys_S1 = set(self.S12d.keys())
        keys_S2 = set(other.S12d.keys())
        intsect = keys_S1 & keys_S2

        S1map = filt_dict(self.S12d,intsect)
        S2map = filt_dict(other.S12d,intsect)
        return S12Prop(S1map), S12Prop(S2map)
        
#%%
def myhistos(x, nbins, xmin, xmax, color="red", title="", xlabel="", ylabel="Entries"):
    mycolor = color
#    plt.hist(x, nbins, color = mycolor, histtype="step", alpha=0.75)
    #plt.figure(figsize=(7, 5), dpi=100) 
    plt.hist(x, nbins, color = mycolor, histtype="bar", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #axes = plt.axes()
    #axes.xaxis.set_label_coords(0.95, -0.10)
    #axes.yaxis.set_label_coords(-0.1, 0.95)
    return plt

def h1(x, nbins, xmin, xmax, color="red", title="", xlabel="", ylabel="Entries"):
    mycolor = color
#    plt.hist(x, nbins, color = mycolor, histtype="step", alpha=0.75)
    plt.figure(figsize=(7, 5), dpi=100) 
    plt.hist(x, nbins, color = mycolor, histtype="bar", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
#    axes = plt.axes()
#    axes.xaxis.set_minor_locator(AutoMinorLocator())
#    axes.yaxis.set_minor_locator(AutoMinorLocator())
#    axes.xaxis.set_label_coords(0.95, -0.10)
#    axes.yaxis.set_label_coords(-0.1, 0.95)
    return plt

#%%
def Histo(x, nbins, title="hsimple", xlabel="", ylabel="Frequency"):
    """histograms"""
    plt.hist(x, nbins, histtype="bar", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)





