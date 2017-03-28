#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:57:50 2017

@author: neus
"""
import numpy as np
from   invisible_cities.core.system_of_units_c import SystemOfUnits
units = SystemOfUnits()
import invisible_cities.database.load_db as DB
DataPMT = DB.DataPMT()
DataSiPM = DB.DataSiPM()


class S12Prop:
    """
    input: S12dict
    
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
        




class S2SiProp:
    """
    properties
    """        
    
    def __init__(self, S2Sidict):
        self.S2Sidict = S2Sidict
        self.length   = len(self.S2Sidict)
        self.prop()

    def dict(self):
            return self.S2Sidict


    def prop(self):
        self.xsipms        = np.zeros(self.length, dtype=np.int) # 
        self.ysipms        = np.zeros(self.length, dtype=np.int) # 
        self.Qtot          = np.zeros(self.length, dtype=np.double) # Q total en el evento
        self.x             = np.zeros(self.length, dtype=np.double) # x from barycenter
        self.y             = np.zeros(self.length, dtype=np.double) # y from barycenter
        
        
        lxsipms      = []
        lysipms      = []
        lQtot        = []
        lx           = []
        ly           = []
        
    
        for evtID, evt in self.S2Sidict.items():  
           # if evtID != 0: break 
            for peakID, sipms in evt.items(): 
                xsi = []
                ysi = []
                Q   = []   
                if(peakID >=1):   break
                for sipmID, Es in sipms.items(): 
                    lQtot     .append(np.sum(Es))               
                    lxsipms   .append(DataSiPM.X.values[sipmID])
                    lysipms   .append(DataSiPM.Y.values[sipmID])
                    # for the weighted barycenter
                    Q         .append(np.sum(Es))
                    xsi       .append(DataSiPM.X.values[sipmID])  
                    ysi       .append(DataSiPM.Y.values[sipmID])
                # fill weighted average per event   
                print('lenght Q= {}, type = {}' .format(len(Q), type(Q)))
                print('lenght xsi= {}, type = {}'.format(len(xsi), type(xsi)))
                print('lenght ysi= {}, type = {}'.format(len(ysi), type(ysi)))
                print('++++++++')
                lx  .append(np.average(xsi, weights = Q))
                ly  .append(np.average(ysi, weights = Q))
                
       # convert lists to numpy arrays
        self.xsipms      = np.array(lxsipms)  
        self.ysipms      = np.array(lysipms)
        self.Qtot        = np.array(lQtot)
        self.x           = np.array(lx)
        self.y           = np.array(ly)



class Truth_S2SiProp:
    """
    properties
    """        
    
    def __init__(self, Tdict):
        self.Tdict    = Tdict
        self.length   = len(self.Tdict)
        self.prop()

    def dict(self):
            return self.Tdict


    def prop(self):
        self.xposition        = np.zeros(self.length, dtype=np.double) # 
        self.yposition        = np.zeros(self.length, dtype=np.double) # 
        self.energies         = np.zeros(self.length, dtype=np.double) # Particle's energy
        self.hit_energies     = np.zeros(self.length, dtype=np.double) # hit's energy
        self.edepo            = np.zeros(self.length, dtype=np.double) # Sum of energy_hits
        self.xtruth           = np.zeros(self.length, dtype=np.double) # x from barycenter
        self.ytruth           = np.zeros(self.length, dtype=np.double) # y from barycenter
        
        
        lxposition      = []
        lyposition      = []
        lenergies       = []
        lhit_energies   = []
        ledepo          = []
        lx              = []
        ly              = []
    
        energies = 0 
        for evtID, evt in self.Tdict.items(): 
            edepo = 0
            xsi   = []
            ysi   = []
            hitE  = []
            for particle, hits in evt.items(): 
                # fill energy of particles               
                for hitID, sipms in hits.items(): 
                    edepo = edepo + sipms.hit_energies               
                    lxposition    .append(sipms.position[0][0])
                    lyposition    .append(sipms.position[0][1])
                    lhit_energies .append(sipms.hit_energies)
                    ledepo        .append(edepo)
                    energies      = sipms.energies
                    xsi           .append(sipms.position[0][0])  
                    ysi           .append(sipms.position[0][1])
                    hitE          .append(sipms.hit_energies[0])
                    #print('xsi = {}'.format(xsi))
                    #print('ysi = {}'.format(ysi))
                    #print('hitE = {}'.format(hitE))
                    #print('sipms.hit_energies = {}'.format(sipms.hit_energies[0]))
                    #print('sipms.position[0][0] = {}'.format(sipms.position[0][0]))
                    
                
                # fill energy of particles 
                lenergies     .append(energies)  
           
            #fill weighted average per event
            xarray = np.array(xsi)
            yarray = np.array(ysi)
            Earray = np.array(hitE)
           # print('lenght E= {}, type = {}'.format(len(Earray), Earray.shape))
           # print(xarray)
           # print(Earray)
           # print('lenght x= {}, type = {}'.format(len(xarray), xarray.shape))
           # print('lenght y= {}, type = {}'.format(len(yarray), yarray.shape))
           # print('+++++++++++++++++++++++++++++')
            lx  .append(np.average(xarray, weights = Earray))
            ly  .append(np.average(yarray, weights = Earray))
            
            #print('+++++++++++++++++++++++++++++')
                
                
       # convert lists to numpy arrays
        self.xposition      = np.array(lxposition)  
        self.yposition      = np.array(lyposition)
        self.energies       = np.array(lenergies)
        self.hit_energies   = np.array(lhit_energies)
        self.edepo          = np.array(ledepo)
        self.xtruth         = np.array(lx)
        self.ytruth         = np.array(ly)


