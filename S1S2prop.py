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
    """
    input: S1L
    returns a S1F object for specific peak
    """        
    
    def __init__(self, S12dict):
        self.S12dict = S12dict
        self.length  = len(self.S12dict)
        self.prop()

    def dict(self):
            return self.S12dict

#    def length(self):
#        return len(self.S12dict)

#test len arrays is == length


    def prop(self):
        self.IDX        = np.zeros(self.length, dtype=np.int) # array index for emax
        self.tIDX       = np.zeros(self.length, dtype=np.double) # time value for IDX
        self.tmin       = np.zeros(self.length, dtype=np.double) # minimum t value
        self.tmax       = np.zeros(self.length, dtype=np.double) # maximum t value
        self.twidth     = np.zeros(self.length, dtype=np.double) # t width of signal (mus) 
        self.tmean      = np.zeros(self.length, dtype=np.double) # mean time
        self.emin       = np.zeros(self.length, dtype=np.double) # min energy the S1/S2 pulse
        self.emax       = np.zeros(self.length, dtype=np.double) # max energy the S1/S2 pulse
        self.etot       = np.zeros(self.length, dtype=np.double) # total energy
        self.emean      = np.zeros(self.length, dtype=np.double) # total energy
        

        # lists to be filled in the loop. Lists name added 'l': lVARIABLE
        lIDX     = []
        ltIDX    = []
        lemax    = []
        ltmin    = []
        ltmax    = []
        ltmean   = []
        ltwidth  = []
        lemin    = []
        lemax    = []
        letot    = []
        lemean   = []

        # 1st loop over dic of events
        # 2nd loop over dic of peaks,(ts,Es)=namedTuple (can also be accessed via value.t, value.E)
        for evtID, evt in self.S12dict.items():  
            for peakID, (ts, Es) in evt.items():  
                if(evtID     == 9999): break  #------> PATCH
                if(peakID        == 1):   break  #------> select one peak 

                lIDX       .append(np.argmax(Es))
                ltIDX      .append(ts[lIDX[-1]]                / units.mus) # use last value of lIDX: [-1]
                ltmin      .append(np.amin  (ts)               / units.mus)
                ltmax      .append(np.amax  (ts)               / units.mus)
                ltwidth    .append((np.amax (ts) - np.amin(ts))/ units.mus)
                ltmean     .append(np.mean  (ts)               / units.mus)
                lemin      .append(np.amin  (Es))
                lemax      .append(np.amax  (Es))
                letot      .append(np.sum   (Es))
                lemean     .append(np.mean  (Es))
                
                
        # convert lists to numpy arrays
        self.IDX      = np.array(lIDX)  
        self.tIDX     = np.array(ltIDX)
        self.tmin     = np.array(ltmin)
        self.tmax     = np.array(ltmax)
        self.twidth   = np.array(ltwidth)
        self.tmean    = np.array(ltmean)
        self.emax     = np.array(lemax)
        self.emin     = np.array(lemin) 
        self.etot     = np.array(letot)
        self.emean    = np.array(lemean)

         
    def S1S2mapd(self, other):
        filt_dict = lambda x, y: dict([ (i , x[i] ) for i in x if i in set(y) ])
      
        keys_S1 = set(self.S12dict.keys())
        keys_S2 = set(other.S12dict.keys())
        intsect = keys_S1 & keys_S2

        S1map = filt_dict(self.S12dict,intsect)
        S2map = filt_dict(other.S12dict,intsect)
        return S12Prop(S1map), S12Prop(S2map)
        


class S12Prop_old:

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
    

    #        self.idxt_       = [] # index array for highest phe signal in an S1/S2
#        self.S12SigE_    = [] # highest phe value in peak
#        self.S12Sigt_    = [] # time in mus for idxt_
#        self.wS12_       = [] # width of signal (mus)
#        self.tmean_      = [] # mean time
#        self.E_          = [] # total energy
