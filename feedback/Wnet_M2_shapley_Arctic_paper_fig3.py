
import numpy as np

#import pandas as pd
#import netCDF4 as nc
import os
import glob
import pickle
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#import xesmf as xe
import time
#from keras.layers import Input, Dense, Dropout, Reshape
#from keras.layers.advanced_activations import LeakyReLU
#from keras.models import Model
#from keras.models import Sequential

import pandas as pd
#from keras.initializers import HeUniform
#from keras.optimizers import adam
from keras.models import load_model
import shap
#from hostlist import expand_hostlist
#from scikeras.wrappers import KerasRegressor
#from dask_ml.wrappers import Incremental
#from sklearn.preprocessing import StandardScaler
import tensorflow as tf
#from dask_ml.preprocessing import StandardScaler
#from dask_ml.wrappers import ParallelPostFit

#from dask.distributed import Client, LocalCluster 
#import joblib
from dask.distributed import Client, LocalCluster
import copy

#new_p =  [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 675, 650, 
#              625, 600, 575, 550, 525, 500, 475, 450, 425, 400, 375,  350, 325, 300, 
#              275, 250, 225, 200, 175, 150,  125, 100, 75, 50, 25,
#               10, 1, 0.1, 0.02]

#new_p =  new_p[::-1]
res =  []
new_p = np.append([1, 5, 20], np.arange(50, 1005, 5))
fill_value=1.e+15

def int_p(W, p):
    fill_value=1.e+15
    #print('p', p.shape)
    #print('p', p)
    #print('W', W.shape)
    #print('W', W)
    #Wint = log_interpolate_1d(new_p, p, W, fill_value = 1.e+15)
    Wint =  np.interp(new_p, p*0.01, W, left= fill_value, right=fill_value)
    #print('Wint', Wint)    
    return Wint
    
def int_lev2p(w, p):
    Wp= xr.apply_ufunc(int_p, w, p, input_core_dims=[["lev"], ["lev"]], output_core_dims=[["new_p"]], 
        dask_gufunc_kwargs=dict(output_sizes={"new_p":len(new_p)}),
        exclude_dims=set(("lev",)), vectorize=True, dask="parallelized", 
        output_dtypes=[w.dtype]) 
    lat  = Wp["lat"].values
    lon  = Wp["lon"].values
    time  = Wp["time"].values
    Wp =  Wp.assign_coords({"lon": lon, "lat":lat, "time":time, "new_p":new_p})
    Wp =  Wp.rename({'new_p':'p'})
    #Wp =  Wp.to_dataset(name="Wstd")
    return Wp

def dens (ds): 
    d = ds.PL/287.0/ds.T
    ds.PL.data = d
    return ds

def calc_W (ds): 
    o = -ds.OMEGA/9.81/ds.AIRD
    ds.OMEGA.data = o
    return ds 
        
def standardize(ds):
 i=0
 m= [243.9, 0.6, 6.3, 0.013, 0.0002, 5.04, 21.8, 0.002, 9.75e-7, 7.87e-6]  #hardcoded from G5NR
 s = [30.3, 0.42, 16.1, 7.9, 0.05, 20.6, 20.8, 0.0036, 7.09e-6, 2.7e-5]
 for v in  ds.data_vars:
   ds[v] = (ds[v] - m[i])/s[i]
   i = i+1    
 return ds

def get_p (ds): 
    d = ds.AIRD*287.0*ds.T
    ds.AIRD.data = d
    return ds
    
       
def QCT (ds): 
    d = ds.QL + ds.QI    
    ds.QL.data = d
    return ds

#def power_scaler(ds, nexp):
#   func  = lambda x, n: np.power(np.where(x<0, 0, x), n)  
#   return xr.apply_ufunc(func, ds, nexp, dask='parallelized') 

 
def get_data(yr='2005', mo='01', dy='01', lev1 = 40, lev2=72,  chunk_size = 4096,  tit= ''):
     
    levs =  72
    chk  = { "lat": -1, "lon": -1, "lev":  -1, "time": 2} 
    n_features_in = 14
    # Read in input from MERRßA   
    dir_in =  "/gpfsm/dnb05/projects/p53/merra2/data/pub/products/MERRA2_all/Y" + yr + "/M" + mo    
    asm  = dir_in + "/" + "MERRA2.tavg3_3d_asm_Nv." + yr + mo + dy + ".nc4"
    #print(asm)
    dat1 =  xr.open_mfdataset(asm, parallel=True, chunks=chk)[['T', 'PL', 'U', 'V', 'OMEGA', 'QV', 'QI', 'QL' ]] 
    P =  dat1['PL']         

    # calculate density
    da= xr.map_blocks(dens, dat1, template=dat1)
    dat1 = da.rename({"PL":"AIRD"})

    ## Calculate W since we only have omega

    da= xr.map_blocks(calc_W, dat1, template=dat1)
    dat1 = da.rename({"OMEGA":"W"})

    turb  = dir_in + "/" +  "MERRA2.tavg3_3d_trb_Ne." + yr +  mo + dy + ".nc4"
    #print(turb)
    dat2 =  xr.open_mfdataset(turb, parallel=True, chunks=chk)[['KM', 'RI']].sel(lev=slice(1,72))       

    #sort the variables to match training   
    dat_in =  xr.merge([dat1, dat2])
    dat_in =  dat_in.unify_chunks() 
    #print(dat_in)
    vars_in =  ['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL']
    dat_in =  dat_in[vars_in]

    feat_in= xr.map_blocks(standardize, dat_in, template=dat_in) 

    Xall = feat_in.isel(time=[0, 1])
    levs = Xall.coords['lev'].values
    nlev =  len(levs)
    #all_vars = ['T', 'KM', 'RI', 'U', 'V']
    
    surf_vars =  ['AIRD', 'KM', 'RI', 'QV']

    #a neat way to populate 3D with surface variables
    surf_v =  Xall[surf_vars].sel(lev=levs[-2]).squeeze() 
    surf_vs = Xall[surf_vars]*0. + surf_v #this broadcasts to the level dimension
    for var in surf_vs.data_vars:
    	surf_vs = surf_vs.rename({var: var +'_sfc'}) 

   
    Xall =  xr.merge([Xall, surf_vs])	
    Xall =  Xall.unify_chunks()
    Xall = Xall.sel(lev=slice(lev1,lev2))
     
    
    X900   =  Xall.sel(lev=slice(64, 66)).mean('lev').squeeze()
    X500   =  Xall.sel(lev=slice(49, 51)).mean('lev').squeeze()
    X250   =  Xall.sel(lev=slice(42, 44)).mean('lev').squeeze()
    X500arc   =  Xall.sel(lev=slice(49, 51), lat=slice(60, 90)).mean('lev').squeeze()
   
       
  
    X900 = X900.to_array()
    X900 = X900.stack( s = ('time', 'lat', 'lon')) 
    X900 =  X900.rename({"variable":"ft"})                       
    X900 = X900.squeeze()
    X900 = X900.transpose()
    X900 = X900.chunk({"ft":n_features_in, "s": chunk_size}) #chunked this way aligns the blocks/chunks with the samples    
    
    X500 = X500.to_array()
    X500 = X500.stack( s = ('time', 'lat', 'lon')) 
    X500 =  X500.rename({"variable":"ft"})                       
    X500 = X500.squeeze()
    X500 = X500.transpose()
    X500 = X500.chunk({"ft":n_features_in, "s": chunk_size}) #chunked this way aligns the blocks/chunks with the samples  
    
    X250 = X250.to_array()
    X250 = X250.stack( s = ('time', 'lat', 'lon')) 
    X250 =  X250.rename({"variable":"ft"})                       
    X250 = X250.squeeze()
    X250 = X250.transpose()
    X250 = X250.chunk({"ft":n_features_in, "s": chunk_size}) #chunked this way aligns the blocks/chunks with the samples  
    
    x500arc = X500arc.to_array()
    x500arc = x500arc.stack( s = ('time', 'lat', 'lon')) 
    x500arc = x500arc.rename({"variable":"ft"})                       
    x500arc = x500arc.squeeze()
    x500arc = x500arc.transpose()
    x500arc = x500arc.chunk({"ft":n_features_in, "s": chunk_size}) #chunked this way aligns the blocks/chunks with the samples  
    
  
    return x500arc.load(), X900.load(), X500.load(), X250.load()
           

def fix_shap_plot_fonts(small=7, large=9, font='sans-serif'):
    """
    Fix fonts in the current SHAP summary plot.

    Parameters:
    - small (int): Font size for tick labels and feature labels
    - large (int): Font size for axis title (if present)
    - font (str): Font family (e.g., 'sans-serif', 'DejaVu Sans', 'Helvetica')
    """
    fig = plt.gcf()
    ax = plt.gca()

    # Fix tick label fonts
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(small)
        label.set_fontfamily(font)

    # Fix axis labels (SHAP sets these in the plot)
    if ax.xaxis.label:
        ax.xaxis.label.set_fontsize(large)
        ax.xaxis.label.set_fontfamily(font)
    if ax.yaxis.label:
        ax.yaxis.label.set_fontsize(large)
        ax.yaxis.label.set_fontfamily(font)

    # Fix title (SHAP rarely uses it directly)
    if ax.title:
        ax.title.set_fontsize(large)
        ax.title.set_fontfamily(font)

    # Fix tick params
    ax.tick_params(axis='both', which='major', labelsize=small)

      
#=========================================
#=========================================
#=========================================
# ... [rest of your code is unchanged] ...

if __name__ == '__main__':

    #client = Client()             # create local cluster

    t0 = time.time() # begin timer for preprocessing

    Xarc, X900, X500, X250 =  get_data()

    #===================Wnet
    folder = "./"   
    mod_name =  "best_generator"   
    model_best = load_model('/discover/nobackup/dbarahon/ML_param/W_NET//single_level/response_and_final/GAN/best_generator.h5', compile=False)

    print('\n------------------------------------------------------')
    print('Model Summary:')
    print('------------------------------------------------------')
    model_best.summary()

    def f(X):
        y = model_best(X)
        yx = np.where(y < 0., 0., y)
        yx = np.where(yx > 10., 0., yx) 
        return yx[:, 0]

    # === Updated plotting configuration ===
    plt.switch_backend('agg')
    plt.rcParams.update({
   # "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],  # or Arial, if installed in your LaTeX system
    "font.size": 5,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "figure.titlesize": 7,
    "ps.fonttype": 42,  # EPS Type 42 fonts (TrueType)
    "text.latex.preamble": r"\renewcommand{\familydefault}{\sfdefault}",
	})

  

    ns = 1000
    res = [r'$T$', r'$\rho_{a}$', r'$U$', r'$V$', r'$W$', r'$K_{\rm M}$', r'$R_{\rm I}$', r'$Q_{\rm V}$', r'$Q_{\rm ice}$', r'$Q_{\rm liq}$', 
           r'$\rho_{a, sfc}$', r'$K_{\rm M, sfc}$', r'$R_{\rm I,sfc}$', r'$Q_{\rm V, sfc}$']

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5))  # set figure size
    axes = axes.flatten()
    colormap = plt.colormaps['bwr']  # blue-white-red

    plt.sca(axes[3])
    axes[3].set_title('(d) p = 500 hPa, Arctic', loc='left')
    Xt = shap.utils.sample(Xarc.data, nsamples=ns, random_state=0)
    explainer = shap.KernelExplainer(f, Xt, silent=True) 
    shap_values = explainer.shap_values(Xt, nsamples=ns)    
    shap.summary_plot(copy.deepcopy(shap_values), Xt, feature_names=res, cmap=colormap, show=False, color_bar=False)
    fix_shap_plot_fonts()
    axes[3].set_xlabel("")
    axes[3].set_xlim([-.5, .5])

    plt.sca(axes[2])
    axes[2].set_title('(c) p = 900 hPa', loc='left')
    Xt = shap.utils.sample(X900.data, nsamples=ns, random_state=0)
    explainer = shap.KernelExplainer(f, Xt, silent=True) 
    shap_values = explainer.shap_values(Xt, nsamples=ns)    
    shap.summary_plot(copy.deepcopy(shap_values), Xt, feature_names=res, cmap=colormap, show=False, color_bar=False)
    fix_shap_plot_fonts()
    axes[2].set_xlabel("")

    plt.sca(axes[1])
    axes[1].set_title('(b) p = 500 hPa', loc='left')
    Xt = shap.utils.sample(X500.data, nsamples=ns, random_state=0)
    explainer = shap.KernelExplainer(f, Xt, silent=True) 
    shap_values = explainer.shap_values(Xt, nsamples=ns)    
    shap.summary_plot(copy.deepcopy(shap_values), Xt, feature_names=res, cmap=colormap, show=False, color_bar=False)
    fix_shap_plot_fonts()
    axes[1].set_xlabel("")

    plt.sca(axes[0])
    axes[0].set_title('(a) p = 250 hPa', loc='left')
    Xt = shap.utils.sample(X250.data, nsamples=ns, random_state=0)
    explainer = shap.KernelExplainer(f, Xt, silent=True) 
    shap_values = explainer.shap_values(Xt, nsamples=ns)    
    shap.summary_plot(copy.deepcopy(shap_values), Xt, feature_names=res, cmap=colormap, show=False, color_bar=False)
    fix_shap_plot_fonts()
    axes[0].set_xlabel("")

    # Add common X label
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.subplots_adjust(bottom=0.1)
    plt.xlabel("SHAP value (m/s)")

    # Add shared colorbar
    fig.subplots_adjust(right=0.8, wspace=1)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    m = cm.ScalarMappable(cmap=colormap)
    cbar = fig.colorbar(m, cax=cbar_ax, ticks=[0, 1])
    cbar.set_label(label="Input Value", labelpad=-1)
    cbar.ax.set_yticklabels(['Low', 'High'], fontdict={'fontsize': 7})

    fig.set_size_inches(7, 5)  # enforce final size again
    plt.subplots_adjust(hspace=0.3)
    # Save EPS figure at 300 dpi
    fig.savefig('Wnet_shap_M2_Arctic_paper.eps', format='eps', bbox_inches='tight', dpi=300)



 
