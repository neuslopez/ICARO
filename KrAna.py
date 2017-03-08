#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:03:23 2017

@author: neus
"""
from __future__ import print_function

import sys
import os
from glob import glob
from time import time
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2
import matplotlib.pyplot as plt
import pandas as pd
import tables as tb
import numpy as np
import math
#plt.rcParams['figure.figsize'] = 10,8
import datetime

from   invisible_cities.database import load_db
from   invisible_cities.core.system_of_units_c import SystemOfUnits
import invisible_cities.sierpe.blr as blr
import invisible_cities.core.mpl_functions as mpl
import invisible_cities.reco.wfm_functions as wfm
import invisible_cities.reco.tbl_functions as tbl
import invisible_cities.core.peak_functions_c as cpf
import invisible_cities.reco.pmaps_functions as pf
import invisible_cities.core.sensor_functions as sf
from   invisible_cities.core.core_functions import define_window

import invisible_cities.core.pmaps_functions_c as cpm
from   invisible_cities.core.core_functions import lrange

import S1S2prop as p
 
units = SystemOfUnits()
t0 = time()

# Open Krypton MC file
#%%
mydf_file = os.environ['IC_DATA']+'/Kr/pmaps_NEXT_v0_08_09_Kr_ACTIVE_1_0_7bar__10000.root.h5'
print(mydf_file)

mydf = pf.read_pmaps(mydf_file)
#mydf
list(map(type, mydf))
S1df = mydf[0]
S2df = mydf[1]
S2Sidf = mydf[2]
print('S1df entries (tbins x events):',len(S1df))
print('S2df entries (tbins x events):',len(S2df))
print('S2Sidf entries:',len(S2Sidf))
type(S1df)
print('Keys of S1df panda dataframe: {} '.format(S1df.keys()))
print('Keys of S2df panda dataframe: {} '.format(S2df.keys()))
print('Keys of S2Sidf panda dataframe: {} '.format(S2Sidf.keys()))

# Convert S12df object  (an S12 pytable read as a PD dataframe) and return an S12L dictionary (list of dict, first dict)
S1 = pf.s12df_to_s12l(S1df,10000)
S2 = pf.s12df_to_s12l(S2df,10000)

evid_S1min = sorted(S1.keys())[0]
evid_S1max = sorted(S1.keys())[-1]
evid_S2min = sorted(S2.keys())[0]
evid_S2max = sorted(S2.keys())[-1]
print('First/last event ID (first item in sorted S1 dictionary): {}/{}'.format(evid_S1min,evid_S1max))
print('First/last event ID (first item in sorted S2 dictionary): {}/{}'.format(evid_S2min,evid_S2max))
print('Total number of events in S1 = {}'.format(len(S1)))
print('Total number of events in S2 = {}'.format(len(S2)))
len(S1), type(S1), len(S2), type(S2)


#%%
myS1 = p.S12Prop(S1)
myS2 = p.S12Prop(S2)

type(myS1)
myS2.length(), myS1.length()
_S1map = myS1.S1S2mapd(myS2)[0]
_S2map = myS1.S1S2mapd(myS2)[1]
_S1map._dict().keys() == _S2map._dict().keys()


#p.myhistos(_S1map.wS12_, 100, 0.0, 0.5,color="green", title="test_hist","width S1","prova")

p.Histo(_S1map.wS12_,100,"title", "x", "Entries")
p.myhistos(_S1map.wS12_,100,"title", "x", "Entries")
#fig, ax = plt.subplots()

































