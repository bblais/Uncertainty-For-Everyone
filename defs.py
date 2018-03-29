import pandas as pd
import numpy as np
import pylab as pl

from numpy import array,log,exp,std

import matplotlib.pyplot as plt

plt.style.use({
    'axes.titlesize' : 24,
    'axes.labelsize' : 20,
    'lines.linewidth' : 3,
    'lines.markersize' : 5,
    'xtick.labelsize' : 20,
    'ytick.labelsize' : 20,
    'figure.figsize' : (10,8),
    'axes.grid': True,
    'grid.linestyle': '-',
    'grid.color': '0.75',
    'font.size': 20,
    'font.family': 'sans-serif',
    'legend.fontsize': 20,
    'legend.frameon': False,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'lines.solid_capstyle': 'round',
    'text.color': '.15',
    'xtick.color': '.15',
    'ytick.color': '.15',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.axisbelow': True,
    'axes.prop_cycle' : plt.cycler('color', ['#1f77b4','#aec7e8','#ff7f0e','#ffbb78',
                                         '#2ca02c','#98df8a','#d62728','#ff9896',
                                         '#9467bd','#c5b0d5','#8c564b','#c49c94',
                                         '#e377c2','#f7b6d2','#7f7f7f','#c7c7c7',
                                         '#bcbd22','#dbdb8d','#17becf','#9edae5']),
})

class Q(object):
    
    def __init__(self, value,std=None,min=None,max=None):
        self.N=30000
        
        
        if isinstance(value,str):
            
            parts=value.split('+-')
                
            if len(parts)==1:  # look for 456(23) notation
            
                parts=value.split('(')
                self.value=float(parts[0])
                
                parts[1]=parts[1].replace(')','').strip()
                
                parts[0]=parts[0].strip()
                for c in '123456789':
                    parts[0]=parts[0].replace(c,'0')
                    
                parts[0]=parts[0][:-len(parts[1])]+parts[1]
                
                self.std=float(parts[0])
                
            
            elif len(parts)==2:
                self.value=float(parts[0])
            
                if '%' in parts[1]:
                    percent=float(parts[1].replace('%',''))
                    self.std=np.abs(self.value*percent/100.0)
                else:            
                    self.std=float(parts[1])
        
        else:
            if std is None:
                std=0
                
            self.value=float(value)
            self.std=float(std)
            
        if not min is None:  # use uniform
            self.sample=np.random.rand(self.N)*(max-min)+min
            self.std=np.std(self.sample)
        else:
            self.sample=np.random.randn(self.N)*self.std+self.value
        
    @property
    def mean(self):
        return np.mean(self.sample)

    @property
    def median(self):
        return np.median(self.sample)
        
    @property
    def percentile95(self):
        return np.percentile(self.sample,[2.5,97.5])
    
    
    def __add__(self,other):
        if isinstance(other,Q):
            sample=other.sample
        else:
            sample=other
        
        new=Q(5)
        new.sample=self.sample+sample
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new
    def __sub__(self,other):
        if isinstance(other,Q):
            sample=other.sample
        else:
            sample=other
        
        new=Q(5)
        new.sample=self.sample-sample
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new
    
    def __radd__(self,other):
        if isinstance(other,Q):
            sample=other.sample
        else:
            sample=other
        
        new=Q(5)
        new.sample=self.sample+sample
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new
    def __rsub__(self,other):
        if isinstance(other,Q):
            sample=other.sample
        else:
            sample=other
        
        new=Q(5)
        new.sample=sample-self.sample
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new

    def __mul__(self,other):
        if isinstance(other,Q):
            sample=other.sample
        else:
            sample=other
        
        new=Q(5)
        new.sample=self.sample*sample
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new
    
    def __rmul__(self,other):
        if isinstance(other,Q):
            sample=other.sample
        else:
            sample=other
        
        new=Q(5)
        new.sample=self.sample*sample
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new
    def __neg__(self):
        
        new=Q(5)
        new.sample=-self.sample.copy()
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new

    def __abs__(self):
        
        new=Q(5)
        new.sample=np.abs(self.sample)
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new

    def __pos__(self):
        
        new=Q(5)
        new.sample=self.sample.copy()
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new
    
    def __pow__(self,other):
        if isinstance(other,Q):
            sample=other.sample
        else:
            sample=other
        
        new=Q(5)
        new.sample=self.sample**sample
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new
    
    def __rtruediv__(self,other):
        if isinstance(other,Q):
            sample=other.sample
        else:
            sample=other
        
        new=Q(5)
        new.sample=sample/self.sample
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new
    
    def __truediv__(self,other):
        if isinstance(other,Q):
            sample=other.sample
        else:
            sample=other
        
        new=Q(5)
        new.sample=self.sample/sample
        new.std=np.std(new.sample)
        new.value=new.mean
    
        return new
    
    def __iter__(self):
        return self.sample.flat
    
    def __getslice__(self, i, j):
        return self.sample[i:j]
    
    def __getitem__(self,key):
        return self.sample[key]
    
    def __repr__(self):
        
        s=str(self.value)+" +- "+str(self.std)
        s+="\n"
        pp=self.percentile95.ravel()
        s+=str(self.median) + ":: 95% range ["+str(pp[0])+" - "+str(pp[1])+"]"
        return s
    
    
def add_time_periods(bottom=150,top=170):
    import matplotlib.patches as patches
    ax=pl.gca()
    # from https://www.geosociety.org/documents/gsa/timescale/timescl.pdf
    periods={
        'C.':[500,485.4], # Cambrian [542,485.4] changed to [500,485.4] for x limits
        'O':[485.4,443.7], # https://en.wikipedia.org/wiki/Ordovician
        'S':[443.7,416.0],
        'D':[416.0,359.2],
        'C':[359.2,299.0],
        'P':[299.0,251.0],
        'Tr':[251.0,199.6],
        'J':[199.6,145.5],
        'K':[145.5,65.5],
        r'P$\epsilon$':[65.5,23],
        'N':[23,0],  # NEOGENE should be [23,2.6] changed to [23,0] for x limits
    }
    
    for key in periods:
        right,left=periods[key]        

        vcenter=(top+bottom)/2
        hcenter=(left+right)/2
    
    
        p = patches.Rectangle(
            (left, bottom), right-left, top-bottom,
            fill=True, facecolor='w',clip_on=False,edgecolor='k',
            )
        ax.add_patch(p)

        ax.text(hcenter, vcenter, key.replace('.',''), fontsize=14,
                ha='center',va='center')

def calc_K0(T,S):
    A1=-58.0931
    A2=90.5069
    A3=22.2940
    B1=0.027766
    B2=-0.025888
    B3=0.0050578

    T=T+273.15
    try:
        lnK0=A1+A2*(100/T)+A3*log(T/100)+ S*(B1+B2*(T/100)+B3*(T/100)**2)
        K0=exp(lnK0)
    except AttributeError:
        K0=Q(5)
        K0.sample=A1+A2*(100/T.sample)+A3*log(T.sample/100)+ S*(B1+B2*(T.sample/100)+B3*(T.sample/100)**2)
        K0.sample=exp(K0.sample)
        K0.std=std(K0.sample)
        K0.value=K0.mean

    return K0        


def make_plot(x,xl,xu,y,yl,yu):
    x,xl,xu=[array(_) for _ in [x,xl,xu]]
    xerr=array([x-xl,xu-x])
    yerr=array([y-yl,yu-y])

    pl.figure(figsize=(12,8))

    pl.plot(x,y,'o',ms=7,color='#61cfe2',markeredgecolor='k',markeredgewidth=1)

    pl.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='none',lw=1,zorder=0,ecolor='k',capsize=0)
    ax=pl.gca()
    ax.invert_xaxis()
    pl.xlim([470,0])

    pl.xlabel('Age (Ma)')

    ax.set_yscale('log')
    h=pl.ylabel('Reconstructed pCO2 (ppm)',rotation=270,va='bottom', labelpad=20)
    pl.ylim([200,3000])
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_ticks_position("right")
    ax.set_yticks([200,400,1000,2000])
    ax.set_yticklabels([200,400,1000,2000])
    pl.grid('off')

    x2,y2,xerr2,yerr2=x,y,xerr,yerr

    add_time_periods(200,230)


data={}
data['default']=[]
def reset(name=None):
    global data
    
    if name==None:
        data={}
        data['default']=[]
    else:
        data[name]=[]
    
def store(*args,**kwargs):
    global data
    
    if 'name' in kwargs:
        name=kwargs['name']
    else:
        name='default'
    
    if name not in data:
        data[name]=[]
        
    if not args:
        data[name]=[]
    
    if not data[name]:
        for arg in args:
            data[name].append([arg])
            
    else:
        for d,a in zip(data[name],args):
            d.append(a)
    

def recall(name='default'):
    global data
    
    if name not in data:
        data[name]=[]
    
    for i in range(len(data[name])):
        data[name][i]=array(data[name][i])
    
    ret=tuple(data[name])
    if len(ret)==1:
        return ret[0]
    else:
        return ret

    