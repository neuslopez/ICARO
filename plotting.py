import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter



def h1(x, nbins, color="red", title="", xlabel="", ylabel="Entries",
       label="legend"):
    mycolor = color

    plt.hist(x, nbins, color = mycolor, histtype="bar", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

   # plt.legend(loc='best', fancybox=True, framealpha=0.5)
    ax = plt.axes()

    ax.tick_params(which='both', direction='in')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
   
    ax.xaxis.set_label_coords(0.9, -0.10)
    ax.yaxis.set_label_coords(-0.15, 0.90) # 1st value: distance to the plot, 2nd:  starts to write label from this value up 
    
    # add ticks in opposite axis
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
   
    return plt



def multh1(x, nbins, ax = None, color="red", title="", xlabel="", ylabel="Entries",
       label="hi"):

    if ax is None:
        fig, ax = plt.subplots()
        
    mycolor = color
    plt.hist(x, nbins, color = mycolor, histtype="bar", alpha=0.75)
    plt.rc('font', weight='bold')
    plt.title(title)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
  #  plt.legend(loc='best', fancybox=True, framealpha=0.5)
    
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

##    ax.legend(loc='best', fancybox=True)
    
