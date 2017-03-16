#%%
#def multh(x, nbins, xmin, xmax, color="red", title="", xlabel="", ylabel="Entries"):

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter

def multh(x, nbins, color="red", title="", xlabel="", ylabel="Entries"):
    mycolor = color
#    plt.hist(x, nbins, color = mycolor, histtype="step", alpha=0.75)
    #plt.figure(figsize=(7, 5), dpi=100)

    ## no estamos cogiendo xmin y xmax
    plt.hist(x, nbins, color = mycolor, histtype="bar", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #NOT working here
#    ax = plt.axes()
#    ax.tick_params(which='both', direction='in')
#    ax.xaxis.set_minor_locator(AutoMinorLocator())
#    ax.yaxis.set_minor_locator(AutoMinorLocator())
#    # add ticks in opposite axis
#    ax.xaxis.set_ticks_position('both')
#    ax.yaxis.set_ticks_position('both')
#    # set labels in corners
#    ax.xaxis.set_label_coords(0.9, -0.10)
#    ax.yaxis.set_label_coords(-0.1, 0.95)
    
    
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





def new_h1(x, nbins, color="red", title="", xlabel="", ylabel="Entries",
       label="legend"):
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

    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    ax = plt.axes()
#    ax.xaxis.set_major_locator(majorLocator)
#    ax.xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter
#    ax.xaxis.set_minor_locator(minorLocator)
    ax.tick_params(which='both', direction='in')
    #plt.rc('font', weight='bold')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_label_coords(0.9, -0.10)
    ax.yaxis.set_label_coords(-0.1, 0.95)

    # add ticks in opposite axis
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    
   # plt.legend(legend)
    return plt



def new_h12(x, nbins, ax = None, color="red", title="", xlabel="", ylabel="Entries",
       label="hi"):

    if ax is None:
        fig, ax = plt.subplots()
        
    mycolor = color
    plt.hist(x, nbins, color = mycolor, histtype="bar", alpha=0.75)
    plt.rc('font', weight='bold')
    plt.title(title)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    
   # majorLocator = MultipleLocator(2)  # x * 2 , subdivisió entre numeros visibles, cuants ticks entre els dos consecutius.
    #majorFormatter = FormatStrFormatter('%d')
    #minorLocator = MultipleLocator(0.5)  # x value for small ticks
    #ax.xaxis.set_major_formatter(majorFormatter)
    ax.tick_params(which='both', direction='in')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
   
    ax.xaxis.set_label_coords(0.9, -0.10)
    ax.yaxis.set_label_coords(-0.15, 0.90) # 1st value: distance to the plot, 2nd:  starts to write label from this value up 
    
    # add ticks in opposite axis
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
