#######################################################################
# NASA GSFC Global Modeling and Assimilation Office (GMAO), Code 610.1
# code developed by Donifan Barahona and Katherine Breen
# last edited: 06.2023
# purpose: To analyze the W data (box plots)
######################################################################


########################################################
# IMPORT PACKAGES
########################################################
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import xarray as xr
from keras.models import load_model
from xhistogram.xarray import histogram
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib.offsetbox import AnchoredText
from scipy.stats import ks_2samp
from xclim.analog import kolmogorov_smirnov
import seaborn as sns

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

def outlier(x):
  return abs((x - x.mean(dim='time', skipna=True)) / x.std(dim='time', skipna=True))   
    
def get_data(site=  '', chunk_size = 512*72, test =  False):
	
    path_asr  = ""  # path to obs data    
    path_merra = ""  # path to reanalysis data 

    asrdata = path_asr  + site + ".nc"   
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
    kstd =  2.0 # defines outliers beyond 2stdev  
    dat_aux =  dat_obs.where(dat_obs > 0.01) 
    dat_std =  dat_aux.groupby('time.month').map(outlier) # returns abs(anomaly/std)        
    dat_obs = dat_obs.where(dat_std < kstd) # we are filling up with zeros anyway

    dat_obs = dat_obs.dropna('time',how='all',thresh=2)  # drop timesteps where all values are nans, keep only timesteps with at least 2 non-nan values (nec for interp)

    #==========MERRA 
    Minp = path_merra + site +  ".nc"

    dat_merra = xr.open_mfdataset(Minp, parallel=True, chunks={"time": 2560})#.sel(lev=slice(1,72))

    # Merra is 3-hourly we have to resample to 10min-half an hour to get enough data    
    dat_merra = dat_merra.resample(time="1min").interpolate("linear")
    # align time steps with obs
    dat_merra, dat_obs = xr.align(dat_merra, dat_obs, exclude = {'height', 'lev'})
    dat  =  dat_merra[['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL']]    
    
    dat =  dat.fillna(0)
    dat_obs =  dat_obs.fillna(0)     

    ## get the last 15% of the data for testing
    tim = dat['time']
    tx = int(0.85*len(tim.values))
    test_dates =  tim[tx:]


    dat =  dat.sel (time  = test_dates)      
    dat_obs = dat_obs.sel (time  = test_dates) 

    ## Standardize
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
    
    
    yall = dat_obs["W_asr_std"]
    yall = yall.stack(s=('time', 'height'))
      
    return Xall.load(), yall.load()
 
def SWclean(SWpred, SWobs):     
    lowW =  0.01
    maxW =  15.0
    SWpred =  SWpred.where(SWobs > lowW)
    SWpred =  SWpred.where(SWobs > lowW)
    SWpred =  SWpred.where(SWpred > lowW)
    SWpred =  SWpred.where(SWpred < maxW)
    return SWpred


########################################################
# PLOT
########################################################
if __name__ == '__main__':
 
    print('I am___', sys.argv[0])
    folder = "./"
    #===========load  models
    
    test  =  True

    pth = ""  # path to Wnet model *.h5 file
    mod_name =  ""  # Wnet model name
    model_best=load_model(pth + mod_name + '.h5' , compile=False)
    print('\n------------------------------------------------------')
    print('Wnet Model Summary:')
    print('------------------------------------------------------')
    model_best.summary()

    pth = ""  # path to prior model *.h5 file
    mod_name =  ""  # prior model name
    prior=load_model(pth , compile=False)
    print('\n------------------------------------------------------')
    print('Prior Model Summary:')
    print('------------------------------------------------------')
    prior.summary()

    pth = ""  # path to obs only model *.h5 file
    mod_name =  ""  # obs only model name
    data_only=load_model(pth + mod_name + '.h5' , compile=False)
    print('\n------------------------------------------------------')
    print('Data only Model Summary:')
    print('------------------------------------------------------')
    data_only.summary()

    pth = ""  # path to transfer model *.h5 file
    mod_name =  ""  # transfer model name
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

    n=0 

    #plotting options
    plt.switch_backend('agg')
    fig, axes = plt.subplots(nrows=4, ncols=3, constrained_layout=True) 
    axes =  axes.flatten()
    txt =  "{:.2f}"
    yearsFmt = mdates.DateFormatter('%Y-%m') 
    
    axn =  0 
    SMALL_SIZE = 9
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    mk_space =  6

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    axn =  0
    sns.set_style("whitegrid")
    for site in ['sgp_pbl' , 'sgp_cirrus', 'manus', 'lei', 'lim', 'mao', 'nsa' , 'asi',  'twp', 'cor', 'pgh', 'ena']:

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
            
        ##################################################################
        # GET DATA
        ##################################################################

        X, SWobs = get_data(site = site, test = False)
        #===============================
        # predict ====wnet========= 
        SW = model_best.predict(X, batch_size =  32768)
        SWpred = SWobs.copy(data=SW[:,0])
        SWpred = SWclean(SWpred, SWobs)

        # predict ======wnet_prior=======       
        SW = prior.predict(X, batch_size =  32768)
        SWprior = SWobs.copy(data=SW[:,0])
        SWprior = SWclean(SWprior, SWobs)

        # predict ======data_only======= 
        SW = data_only.predict(X, batch_size =  32768)
        SWdo = SWobs.copy(data=SW[:,0])
        SWdo = SWclean(SWdo, SWobs)

        # predict ======transfer learning======= 
        SW = trans.predict(X, batch_size =  32768)
        SWtr = SWobs.copy(data=SW[:,0])
        SWtr = SWclean(SWtr, SWobs)
 
        # predict ======emd======= 
        SW = emd.predict(X, batch_size =  32768)
        SWemd = SWobs.copy(data=SW[:,0])
        SWemd = SWclean(SWemd, SWobs)
 
        SWobs = SWclean(SWobs, SWobs)

        print('SWpred', SWpred)
  
        ############### ALL data statistics #################

        SWdat =  SWobs
        SWdat = SWdat.to_dataset()
        SWdat =  SWdat.rename({'W_asr_std':'Obs'})
        
        SWdat['Wnet'] = SWpred      
        SWdat['Wpr'] = SWprior 
        SWdat['Oo'] = SWdo 
        SWdat['Tr'] = SWtr 
        SWdat['EMD'] = SWemd 

        
        print('data', SWdat)
        ydat = np.linspace(-100, 100, 100)
        SWdat =  SWdat.to_dataframe()
        print('dataframe', SWdat)
  
        sns.boxplot(data= SWdat, ax=axes[axn], 
        showfliers = False)
        axes[axn].set_ylabel(r'$\sigma_w ~\rm{m ~s}^{-1}$')
        axes[axn].set_title(s)
        axes[axn].locator_params(axis='y', nbins=5)
        axes[axn].yaxis.grid(True)
        axes[axn].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))       

        axn =  axn+1

    tit =  mod_name + '_boxplot_last15percent.png' 
    fig.set_size_inches(9.5, 5.5)
    fig.savefig(tit) 



    

