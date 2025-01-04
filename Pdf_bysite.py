#######################################################################
# NASA GSFC Global Modeling and Assimilation Office (GMAO), Code 610.1
# code developed by Donifan Barahona and Katherine Breen
# last edited: 06.2023
# purpose: To analyze the W data
######################################################################

########################################################
# IMPORT PACKAGES
########################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import xarray as xr
from keras.models import load_model
from xhistogram.xarray import histogram
from matplotlib.offsetbox import AnchoredText
from scipy.stats import ks_2samp
from xclim.analog import kolmogorov_smirnov

########################################################
# FUNCTIONS
########################################################   
def standardize(ds):
  i = 0
  m= [243.9, 0.6, 6.3, 0.013, 0.0002, 5.04, 21.8, 0.002, 9.75e-7, 7.87e-6]  #hardcoded from G5NR
  s = [30.3, 0.42, 16.1, 7.9, 0.05, 20.6, 20.8, 0.0036, 7.09e-6, 2.7e-5]
  for v in  ds.data_vars:
   ds[v] = (ds[v] - m[i])/s[i]
   i = i+1   
  return ds

def log10_scaler(ds):
   func  = lambda x: np.log10(x)  
   return xr.apply_ufunc(func, ds, dask='parallelized')
   
def outlier(x):
  return abs((x - x.mean(dim='time')) / x.std(dim='time'))

def KS_score(yp, yo):	
    yp =  yp.data.flatten()
    yo =  yo.data.flatten()
    k  = ks_2samp(yo, yp)
    print(k) 
    return k[0]
    
def get_data(site=  '', chunk_size = 512*72):
    path_asr  = ""  # path to obs data
    path_merra = ""  # path to reanalysis data

    asrdata = path_asr  + site + ".nc"   
    print('asr--', asrdata)
    dat_obs =  xr.open_mfdataset(asrdata,  parallel=True)
     
    nam =  "Wstd_" + site
    dat_obs = dat_obs.where(dat_obs != -9999.)
    dat_obs = dat_obs.where(dat_obs < 15.)
    
    if site == 'manus':
    	dat_obs = dat_obs.where((dat_obs["time.year"] < 2005) | (dat_obs["time.year"] > 2007), drop=True)

    if site == 'twp':
   		dat_obs = dat_obs.where((dat_obs["time.year"] < 2014), drop=True)
    
    if site == 'mao':
         d1 =  dat_obs.sel(time=slice('2014-02-01', '2015-01-10'))
         d2 =  dat_obs.sel(time=slice('2015-04-01', '2016-02-01'))
         dat_obs =  xr.concat([d1, d2], dim ='time')
    
    if site == 'ena':
    	dat_obs = dat_obs.where((dat_obs["time.year"] < 2017) | (dat_obs["time.year"] >= 2018), drop=True)    
        
    dat_obs = dat_obs.where(dat_obs != -9999.)   
    kstd =  2.5 # defines outliers beyond 2stdev  
    dat_aux =  dat_obs.where(dat_obs > 0.001) 
    dat_std =  dat_aux.groupby('time.month').map(outlier) # returns abs(anomaly/std)        
    dat_obs = dat_obs.where(dat_std < kstd) # we are filling up with zeros anyway
            
    dat_obs = dat_obs.dropna('time',how='all',thresh=2)  # drop timesteps where all values are nans, keep only timesteps with at least 2 non-nan values (nec for interp)

    #==========MERRA 
    Minp = path_merra + site +  ".nc"
    
    
    dat_merra = xr.open_mfdataset(Minp, parallel=True, chunks={"time": 2560})
           
    # Merra is 3-hourly we have to resample to 10min-half an hour to get enough data    
    dat_merra = dat_merra.resample(time="5min").interpolate("linear")
    # align time steps with obs
    dat_merra, dat_obs = xr.align(dat_merra, dat_obs, exclude = {'height', 'lev'})
    dat  =  dat_merra[['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL']]    
    dat =  dat.fillna(0)
    dat_obs =  dat_obs.fillna(0)
    
    feat_in= xr.map_blocks(standardize, dat, template=dat)
    Xall =  feat_in
    
     #========add_surface_vars======
    levs = Xall.coords['lev'].values
    nlev =  len(levs)
    all_vars =  ['AIRD', 'KM', 'RI', 'QV']

    for v in all_vars:        
        vv =     Xall[v]
        Xs =  vv.sel(lev=[71]).squeeze() #level 1 above surface
        Xsfc =  Xs
        v2 =  v + "_sfc"
        for l in range(nlev-1):         
            Xsfc = xr.concat([Xsfc, Xs], dim ='lev')
        Xsfc =  Xsfc.assign_coords(lev=levs)
        Xall[v2] =  Xsfc
    
    Xall =  Xall.unify_chunks()    
    Xall = Xall.to_array()
    Xall = Xall.stack(s=('time', 'lev'))
    Xall =  Xall.rename({"variable":"ft"})  
    Xall = Xall.squeeze()
    Xall = Xall.transpose()
    Xall = Xall.chunk({"s": 72*1024})
    
    
    yall = dat_obs["W_asr_std"]
    yall = yall.stack(s=('time', 'height'))
    yall = yall.chunk({"s": 72*1024}) 
      
    return Xall.load(), yall.load()

def nb(x, prec=3):
    	return np.format_float_positional(x, precision=prec)
        
def plot_histogram(SWemd, SWtr, SWdo, SWprior, SWwnet, SWobs, ax="", print_legend=False):

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 13
    
    bins = np.linspace(-3.0,1.5, 300)
    xlb =  np.power(10, bins)
    dx = (bins[1:]- bins[0:-1])
    bx = (bins[1:]+ bins[0:-1])/2
     
    W =  log10_scaler(SWobs)
    hs = histogram(W, bins=[bins], block_size=10)
    hobs =  hs.values
    fobs = hobs/dx/np.sum(hobs)
    m = W.mean(skipna=True).values 
    s  = W.std(skipna=True).values
    m =  np.power(10,m) 
    print('obs', ' m = ', m, ' s = ', s )   
    
    lw =  1.5 
    lab  = 'Obs data'  
    obs_l = ax.plot(bx, fobs, color='r', linestyle='-', linewidth=lw, label=lab)   
    curves = obs_l   
     
    W =  log10_scaler(SWprior)
      
    hs = histogram(W, bins=[bins], block_size=10)
    h =  hs.values
    fx = h/dx/np.sum(h) 
    m = W.mean(skipna=True).values 
    s  = W.std(skipna=True).values
    m =  np.power(10,m)
   
    h=  h/np.sum(h)
    k=  KS_score(SWprior, SWobs)    
    lab  = 'Wnet_prior'
     
    print('Wnet_prior', ' m = ', m, ' p = ', k )       
    prior_l = ax.plot(bx, fx, color='b', linestyle='-', linewidth=lw, label=lab)
    curves += prior_l
     
    W =  log10_scaler(SWdo)
      
    hs = histogram(W, bins=[bins], block_size=10)
    h =  hs.values
    fx = h/dx/np.sum(h) 
    m = W.mean(skipna=True).values 
    s  = W.std(skipna=True).values
    m =  np.power(10,m)
    
    h=  h/np.sum(h)
    k=  KS_score(SWdo, SWobs)     
    lab  = 'Obs only'
    
    print('Data', ' m = ', m, ' p = ', k )       
    obsonly_l = ax.plot(bx, fx, color='g', linestyle='-', linewidth=lw, label=lab)
    curves += obsonly_l
    
    W =  log10_scaler(SWtr)
      
    hs = histogram(W, bins=[bins], block_size=10)
    h =  hs.values
    fx = h/dx/np.sum(h) 
    m = W.mean(skipna=True).values 
    s  = W.std(skipna=True).values
    m =  np.power(10,m)
    h=  h/np.sum(h)
    k=  KS_score(SWtr, SWobs)    
    lab  = 'Transfer'
    
    print('Trans', ' m = ', m, ' p = ', k )       
    trans_l = ax.plot(bx, fx, color='c', linestyle='-', linewidth=lw, label=lab)
    curves += trans_l
    
    W =  log10_scaler(SWemd)
      
    hs = histogram(W, bins=[bins], block_size=10)
    h =  hs.values
    fx = h/dx/np.sum(h) 
    m = W.mean(skipna=True).values 
    s  = W.std(skipna=True).values
    m =  np.power(10,m)
    
    h=  h/np.sum(h)
    k=  KS_score(SWemd, SWobs)     
    lab  = 'EMD'
     
    print('EMD', ' m = ', m, ' p = ', k )       
    emd_l = ax.plot(bx, fx, color='m', linestyle='-', linewidth=lw, label=lab)
    curves += emd_l
    
    W =  log10_scaler(SWwnet)
     
    hs = histogram(W, bins=[bins], block_size=10)
    h =  hs.values       
    fx = h/dx/np.sum(h)     
    m = W.mean(skipna=True).values  
    s  = W.std(skipna=True).values
    m =  np.power(10,m)
    
    h=  h/np.sum(h)
    k=  KS_score(SWwnet, SWobs)    
    lab  = 'Wnet'
    
    print('Wnet', ' m = ', m, ' k = ', k ) 
   
    wnet_l = ax.plot(bx, fx, color='k', linestyle='-', linewidth=lw, label=lab)
    curves += wnet_l
    
    if print_legend:
      labels = [c.get_label() for c in curves]
      ax.legend(curves, labels, loc="upper right", frameon=False, framealpha=0., fontsize='small', borderpad=0)
    
    
########################################################
# PLOT
########################################################
if __name__ == '__main__':

  folder = "./"
  print('matplotlib: {}'.format(matplotlib.__version__))
  
  #===========load  models
  
  pth = ""  # path to Wnet model *.h5 file
  mod_name =  ""  # Wnet model name
  model_best=load_model(pth + mod_name + '.h5' , compile=False)
  print('\n------------------------------------------------------')
  print('Wnet Model Summary:')
  print('------------------------------------------------------')
  model_best.summary()
 
  pth = ""  # path to Wnet-prior model *.h5 file
  mod_name =  ""  # Wnet-prior model name
  prior=load_model(pth , compile=False)
  print('\n------------------------------------------------------')
  print('Prior Model Summary:')
  print('------------------------------------------------------')
  prior.summary()
  
  pth = ""  # path to Obs only model *.h5 file
  mod_name =  ""  # Obs only model name
  obs_only=load_model(pth + mod_name + '.h5' , compile=False)
  print('\n------------------------------------------------------')
  print('Obs only Model Summary:')
  print('------------------------------------------------------')
  obs_only.summary()
  
  pth = ""  # path to Transfer model *.h5 file
  mod_name =  ""  # Transfer model name
  trans=load_model(pth + mod_name + '.h5' , compile=False)
  print('\n------------------------------------------------------')
  print('Transfer Model Summary:')
  print('------------------------------------------------------')
  trans.summary()
  
  pth = ""  # path to EMD model *.h5 file
  mod_name =  ""  # EMD model name
  emd=load_model(pth + mod_name + '.h5' , compile=False)
  print('\n------------------------------------------------------')
  print('EMD model Summary:')
  print('------------------------------------------------------')
  emd.summary()
  
  #plotting options
  plt.switch_backend('agg')
  fig, axes = plt.subplots(nrows=4, ncols=3)
  fig.set_size_inches(9,8)  # (w,h) 
  axes =  axes.flatten()
  txt =  "{:.2f}"
  yearsFmt = mdates.DateFormatter('%Y-%m') 
  
  axn =  0 
  nexp =0.75
  SMALL_SIZE = 7
  MEDIUM_SIZE = 8
  BIGGER_SIZE = 11
  mk_space =  6

  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
  
  prior_all = []
  wnet_all = []
  obs_all = []
  do_all =  []
  tr_all = []
  emd_all = []
  n=0  
  xlab = r'$\log_{10}(\sigma_w ~\rm{m~s}^{-1})$'
  ylab = r'$\frac{dP(\sigma_w)}{d \log(\sigma_w )}$' 

  for site in ['sgp_pbl' , 'sgp_cirrus', 'manus', 'lei', 'lim', 'mao', 'nsa' , 'asi',  'twp', 'cor', 'pgh', 'ena']:
    
    ##################################################################
    # GET DATA
    ##################################################################

    X, SWobs = get_data(site = site)
    #===============================
    # predict ====wnet========= 
    SW = model_best.predict(X, batch_size =  32768)
    print('SWnet', SW.shape)
    
    SWpred = SWobs.copy(data=SW[:,0]) #copy all structure 
    SWpred =  SWpred.unstack("s")
    
    # predict ======wnet_prior======= 
    SW = prior.predict(X, batch_size =  32768)
    print('SWprior', SW.shape)
    
    SWprior = SWobs.copy(data=SW[:,0]) #copy all structure 
    SWprior =  SWprior.unstack("s")
    
    # predict ======data_only======= 
    SW = data_only.predict(X, batch_size =  32768)
    print('SWdo', SW.shape)
    
    SWdo = SWobs.copy(data=SW[:,0]) #copy all structure 
    SWdo =  SWdo.unstack("s")
    
    # predict ======transfer learning======= 
    SW = trans.predict(X, batch_size =  32768)
    print('SWtr', SW.shape)
    
    SWtr = SWobs.copy(data=SW[:,0]) #copy all structure 
    SWtr =  SWtr.unstack("s")
    
    # predict ======emd======= 
    SW = emd.predict(X, batch_size =  32768)
    print('SWemd', SW.shape)
    
    SWemd = SWobs.copy(data=SW[:,0]) #copy all structure 
    SWemd =  SWemd.unstack("s")
    
    #obs
    SWobs =  SWobs.unstack("s")
    
    
    tim = SWpred['time'].values
    
    lowW =  0.001
    maxW =  15.0
    SWpred =  SWpred.where(SWobs > lowW)
    SWprior =  SWprior.where(SWobs > lowW)
    SWdo =  SWdo.where(SWobs > lowW)
    SWtr =  SWtr.where(SWobs > lowW)
    SWemd =  SWemd.where(SWobs > lowW)
    
    SWobs =  SWobs.where(SWobs > lowW)
    SWobs =  SWobs.where(SWobs < maxW)
    
    SWpred =  SWpred.where(SWpred > lowW)
    SWpred =  SWpred.where(SWpred < maxW)
    
    SWprior =  SWprior.where(SWprior > lowW)
    SWprior =  SWprior.where(SWprior < maxW)
    
    SWdo =  SWdo.where(SWdo > lowW)
    SWdo =  SWdo.where(SWdo < maxW)
    
    SWtr =  SWtr.where(SWtr > lowW)
    SWtr =  SWtr.where(SWtr < maxW)
    
    SWemd =  SWemd.where(SWemd > lowW)
    SWemd =  SWemd.where(SWemd < maxW)
    
    n = 0
    if n <  1:
        prior_all = SWprior 
        wnet_all = SWpred 
        obs_all = SWobs 
        do_all =  SWdo
        tr_all =  SWtr
        emd_all = SWemd
        
    else:    	
        prior_all = xr.concat([prior_all, SWprior], dim="time", join='override')
        wnet_all = xr.concat([wnet_all, SWpred], dim="time", join='override')
        do_all = xr.concat([do_all, SWdo], dim="time", join='override')
        obs_all = xr.concat([obs_all, SWobs], dim="time", join='override')
        tr_all = xr.concat([tr_all, SWtr], dim="time", join='override')
        emd_all = xr.concat([emd_all, SWemd], dim="time", join='override')

    
    #=================plotting========================
    s =  site
    
    if site in [ 'nsa' , 'asi',  'twp', 'cor', 'pgh', 'ena']:
    	s = s + ' (pbl)'
        
    if site in [ 'manus', 'lei', 'lim']:
    	s = s + ' (cirrus)'
    
    if  site == 'mao':
    	s =  'mao (convective)'        
    if site == 'sgp_pbl':
    	s =  'sgp (pbl)'
    if site  == 'sgp_cirrus':
    	s = 'sgp (cirrus)'
        
    axes[axn].set_title(s)
    
    if axn != 2:
      x = plot_histogram(emd_all, tr_all, do_all, prior_all, wnet_all, obs_all, ax= axes[axn], print_legend=False)
    else:
      print("**************SET THE LEGEND")
      x = plot_histogram(emd_all, tr_all, do_all, prior_all, wnet_all, obs_all, ax= axes[axn], print_legend=True)
   
    axn =  axn+1
    #end of loop
    
  plt.subplots_adjust(hspace=0.5)
  
  fig.text(0.5,0.04, xlab, ha="center", va="center",  fontsize=14)
  fig.text(0.05,0.5, ylab, ha="center", va="center", rotation=90,  fontsize=14)
  fig_title =  mod_name + '_pdf_by_site.png' 
  fig.savefig(fig_title) 
  
  

 
  
 
  
  
  
