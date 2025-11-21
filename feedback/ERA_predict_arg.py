
#Predicts Wstd using ERA5 data 
import sys
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import pandas as pd
#import netCDF4 as nc
import os
import glob
import xarray as xr
import time
from dask.distributed import Client, LocalCluster
from datetime import datetime, timedelta
from keras.models import load_model
#from scipy import interpolate
#import keras
import tensorflow as tf
import gc

#'/css/g5nr/Ganymed/7km/c1440_NR/DATA/0.0625_deg/inst/inst30mn_3d_W_Nv/Y2006/M05/D07/c1440_NR.inst30mn_3d_W_Nv.20060507_2230z.nc4'
#'/css/g5nr/Ganymed/7km/c1440_NR/DATA/0.5000_deg/tavg/tavg01hr_3d_T_Cv/Y2006/M05/D07/c1440_NR.tavg01hr_3d_T_Cv.20060507_0830z.nc4'

#for now we base all anomaly correlations on MERRA
#new_p =  [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 
#    600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 40, 30, 20,
#    10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.02]


new_p = np.arange(25, 1005, 25)
new_p = [0.1, 1, 5, 10, new_p]   
#new_p= [1000, 985, 970, 955, 940, 925, 910, 895, 880, 865, 850, 835, 820, 805, 790, 775, 760,
#        745, 730, 715, 700, 685, 670, 655, 640, 625, 610, 595, 580, 565, 550, 535, 520, 505, 490, 475, 
#        460, 445, 430, 415, 400, 385, 370, 355, 340, 325, 310, 295, 280, 265, 250, 235, 220, 205, 190, 175, 
#        160, 145, 130, 115, 100, 85, 70, 55, 40, 30, 20, 10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 
#        0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.02]
#new_p =  new_p[::-1]
fill_value=1.e+15


def corrcoefxy( x, y ):    
    mean_x = np.mean( x )
    mean_y = np.mean( y )
    std_x  = np.std ( x )
    std_y  = np.std ( y )
    n      = len    ( x )
    return (( x - mean_x ) * ( y - mean_y )).sum() / n / ( std_x * std_y )
    
def standardize(ds, s=1, m=0):
  i = 0
  #['T', 'AIRD', 'U', 'V', 'W', 'KM', 'RI', 'QV', 'QI', 'QL'] 
  for v in  ds.data_vars:  
   ds[v] = (ds[v] - m[i])/s[i]
   i = i+1
  return ds

#def power_scaler(ds, nexp):
#   func  = lambda x, n: np.power(np.where(x<0, 0, x), n)  
#   return xr.apply_ufunc(func, ds, nexp, dask='parallelized') 

def linear_scaler(ds, cst):
   func  = lambda x, n: x*n 
   return xr.apply_ufunc(func, ds, cst, dask='parallelized') 
 
def calc_W (ds): 
    o = -ds.w/9.81/ds.AIRD
    ds.w.data = o
    return ds  

def dens (ds): 
    d = ds.PL/287.0/ds.T
    ds.PL.data = d
    return ds 
    
def QCT (ds): 
    d = ds.QL + ds.QI    
    ds.QL.data = d
    return ds    

def int_p(W, p):
    Wint =  np.interp(new_p, p*0.01, W, left= fill_value, right=fill_value)
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
    Wp =  Wp.rename({'new_p':'lev'})
    Wp =  Wp.transpose('time', 'lev', 'lat', 'lon')
    Wp =  Wp.to_dataset(name="Wstd")
    
    return Wp#.load() 


def set_atts(dsx, var="Wstd"):
    ds =  dsx.copy()
    
    # coordinate attributes
    ds["lon"].attrs['long_name'] = 'longitude'
    ds["lon"].attrs['standard_name']     = 'longitude'
    ds["lon"].attrs['units']             = 'degrees_east'
    ds["lon"].attrs['coordinate']        = 'lon'
    #ds["lon"].attrs['_FillValue']        = np.array(fill_value, np.float32) 

    ds["lat"].attrs['long_name']         = 'latitude'
    ds["lat"].attrs['standard_name']     = 'latitude'
    ds["lat"].attrs['units']             = 'degrees_north'
    ds["lat"].attrs['coordinate']        = 'lat'
   # ds["lat"].attrs['_FillValue']        = np.array(fill_value, np.float32)

    ds["lev"].attrs['long_name']         = 'pressure'
    ds["lev"].attrs['standard_name']     = 'middle level pressure'
    ds["lev"].attrs['units']             = 'hPa'
    ds["lev"].attrs['positive']          = 'down'
    ds["lev"].attrs['coordinate']        = 'lev'
    #ds["lev"].attrs['_FillValue']        = np.array(fill_value, np.float32)

    date = ds['time'].values[0] #datetime(2005, 1, 1, 0, 0, 0)
   # date1 = ds['time'].values[1]   
    date  =  pd.to_datetime(date)
    date1  =  0
    
    #print(date, date1)
    #td          = date1-date
    time_increment          = int('{hours}{minutes}{seconds}'.format(hours=int(732), minutes=int(0), seconds=int(0)))  
    begin_date              = int(date.strftime('%Y%m%d'))
    begin_time              = int(date.strftime('%H%M%S'))
    #time_increment =  np.array(6000, dtype=np.float32)

    #ds["time"].attrs['long_name'] = 'time'
    ds["time"].attrs['time_increment'] =np.array(time_increment, dtype=np.int32)
    #ds["time"].attrs['units']            = 'days since {:%Y-%m-%d %H:%M:%S}'.format(date)
    ds["time"].attrs['begin_date']  = np.array(begin_date,     dtype=np.int32)
    ds["time"].attrs['begin_time']  = np.array(begin_time,     dtype=np.int32)

    ds[var].attrs["units"] = "m s-1"
    ds[var].attrs["long_name"] = "M2-derived standard deviation in vertical wind velocity"
    ds[var].attrs["standard_name"] = "STDev in vertical velocity"  
    ds[var].attrs["contact"] = "Donifan Barahona, donifan.o.barahona@nasa.gov" 

    ds[var].attrs["fmissing_value"] = np.array(fill_value, np.float32)
    ds[var].attrs["missing_value"] = np.array(fill_value, np.float32)   
    ds[var].attrs["vmin"]            = np.array(-fill_value, np.float32)
    ds[var].attrs["vmax"]            = np.array(fill_value, np.float32)
    ds[var].attrs["valid_range"]     = np.array((-fill_value, fill_value), np.float32) 

    return ds          


def download_ecmwf_files(year: int, month: int, day: int, max_workers: int = 4):
    """
    Downloads ECMWF files for a given year, month, and day using multi-threaded requests.

    Parameters:
        year (int): Year in YYYY format (e.g., 1980)
        month (int): Month as integer (e.g., 1 for January)
        day (int): Day as integer (e.g., 1 for the 1st)
        max_workers (int): Number of parallel download threads (default: 4)
    """

    base_url = "https://data.rda.ucar.edu/d633000/"
    var_codes = {
        "130": "t",     # temperature
        "131": "u",     # u-wind
        "132": "v",     # v-wind
        "133": "q",     # specific humidity
        "135": "w",     # vertical velocity
        "246": "clwc",  # cloud liquid water content
        "247": "ciwc",  # cloud ice water content
    }

    yyyy = f"{year:04d}"
    mm = f"{month:02d}"
    dd = f"{day:02d}"

    folder = f"e5.oper.an.pl/{yyyy}{mm}/"

    files = [
        f"{folder}e5.oper.an.pl.128_{code}_{var}.ll025{'uv' if code in ['131', '132'] else 'sc'}.{yyyy}{mm}{dd}00_{yyyy}{mm}{dd}23.nc"
        for code, var in var_codes.items()
    ]

    def download(file_path):
        filename = file_path.split("/")[-1]
        url = base_url + file_path
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download, f) for f in files]
        for future in as_completed(futures):
            future.result()




               
if __name__ == '__main__':
    #cluster = LocalCluster(threads_per_worker=8,n_workers =4, processes=False )
    #client = Client(cluster)
    #client = Client()
    #print('---cluster---') 
    #cluster  

    fill_value=1.e+15 
    encx = dict(dtype= 'float32', _FillValue= -9999)
    #print(year)
#===============model==========================
    def  ERA_input(yr='2005', mo='01', dy='01'):

        chk  = {"time": 2} 
        # Read in input from ERA 


        dir_in =  "/gpfsm/dnb34/dbarahon/DATA/ERA5/3Dwnet"     
        asm = dir_in + "/" + "ERA5_3D_" + yr + mo + dy + "*.nc"
        #print(asm)
        dat3D =  xr.open_mfdataset(asm, parallel=True, chunks=chk)[['t', 'u', 'v', 'w', 'q', 'ciwc', 'clwc' ]] 

        dat3D =  dat3D.rename({'valid_time':'time', 'latitude':'lat', 'longitude':'lon', 'pressure_level':'lev' })
        dat3D =  dat3D.reset_coords()
        # calculate density
        T = dat3D.t
        P =  T*0.  + 100.*dat3D.lev #take advantage of broadcasting rules
        AIRDEN = P/287.0/T
        dat3D['AIRD'] =  AIRDEN

        ## Calculate W since we only have omega        
        dat3D= xr.map_blocks(calc_W, dat3D, template=dat3D)
        
        #dat1 = da.rename({"w":"W"})

        #print('======dat3D====', dat3D)
        surf_vars =  ['blh', 'tcwv', 'AIRD_surf']


        dir_in =  "/gpfsm/dnb34/dbarahon/DATA/ERA5/2Dwnet"     
        asm = dir_in + "/" + "ERA5_2D_" + yr + mo + ".nc"
        #print(asm)
        dat2D =  xr.open_mfdataset(asm, parallel=True, chunks=chk)[['blh', 'tcwv', 'sp', 't2m']]
        dat2D =  dat2D.rename({'valid_time':'time' , 'latitude':'lat', 'longitude':'lon'})
        dat2D =  dat2D.sel(time=slice(yr + mo + dy , yr + mo + dy )).reset_coords() 

        sp = dat2D['sp']
        #calculate surface density
        A =dat2D.sp/287.0/dat2D.t2m
        dat2D['AIRD_surf'] =  A

        dat2D = dat2D[surf_vars]

        #print('======dat2D====', dat2D)
        ##normalize (make sure only time isschunked)
        
        dat3D =  dat3D[['t', 'AIRD', 'u', 'v', 'w', 'q', 'ciwc', 'clwc' ]] 

        means= [243.9, 0.6, 6.3, 0.013, 0.0002, 0.002, 9.75e-7, 7.87e-6] #hardcoded from G5NR #hardcoded from G5NR based on 100 time steps
        stds =[30.3, 0.42, 16.1, 7.9, 0.05, 0.0036, 7.09e-6, 2.7e-5]    
        dat3D= xr.map_blocks(standardize, dat3D, kwargs={"m":means, "s": stds}, template=dat3D)


        means= [889., 23.4, 1.] #hardcoded from M2 based on 100 time steps
        stds =[387.,  15.6, 0.1]    
        dat2D= xr.map_blocks(standardize, dat2D, kwargs={"m":means, "s": stds}, template=dat2D)

        #concatenate the 2D vars at each level
        dat3D, dat2D =  xr.align(dat3D, dat2D, exclude = {'lat', 'lon', 'lev'})

        dat =  dat3D
        for v  in surf_vars:
	        dat[v] = T*0. + dat2D[v] #this broadcasts to the level dimension


        dat =  dat[['t', 'AIRD', 'u', 'v', 'w', 'q', 'ciwc', 'clwc', 'blh', 'tcwv', 'AIRD_surf']] 
        #print(dat)
        Xall = dat.unify_chunks()
        yall =  Xall[['t']].rename({"t":"Wstd"})       
        Xall =  Xall.to_array()
        #print('Xall', Xall.data.shape)
        Xall = Xall.stack( s = ('time', 'lat', 'lon', 'lev' )).chunk({"s": 102000}) 
        Xall = Xall.transpose().squeeze()

        yall = yall.stack(s = ('time', 'lat', 'lon', 'lev' ))
        #yall =  yall.squeeze()
        yall =  yall.transpose()   
        yall =  yall.chunk({"s": 102000})

        #print('Xall======', Xall) 
        #print('yall======', yall)


        # print("yall'", yall)

        return Xall.fillna(0.).persist(), yall.fillna(0), P, sp


#=========================================

    year  =  sys.argv[1]
    #model_g=load_model('/discover/nobackup/dbarahon/ML_param/W_NET/final_model_2/GAN_256/generator.h5', compile=False)
    model=load_model('/gpfsm/dnb34/dbarahon/ML_param/W_NET/single_level/ERA5/Wnet_era5/best/Wnet2.hdf5', compile=False)

    #client =  Client()
    
    #=========== 
    yr_st =  int(year)
    mo_st =  1
    dy_st =  1 #
    yr_end=  int(year) + 1
    mo_end = 1
    dy_end =  31
    
    
    
    
    td = np.arange(datetime(yr_st,mo_st,dy_st), datetime(yr_end,mo_end,dy_end), timedelta(days=1)).astype(datetime)
    physical_devices = tf.config.list_physical_devices('GPU')
    print("====Num GPUs:", len(physical_devices))

    for t in td:
        t0 = time.time() 
        y =  str(t.year)
        m =  str(t.month).zfill(2)
        d =  str(t.day).zfill(2)
        print('processing______', y + m + d)
        tit =  "ERA5.inst_3d_Wstd_Np." + y + m + d + ".nc4"

        if os.path.exists(tit): #skip if already calculated (Srun is re-starting the code for some reason)
        	print(tit + '_____already computed')
        	continue

        try:
        	
            download_ecmwf_files(y, str(t.month), str(t.day), max_workers=4)
            X, SWpred, P, sp = ERA_input(yr=y, mo=m, dy= d)
        except:
            print('Error______', y+m+d)
            continue

        y_hat = model.predict(X, batch_size=32768, verbose=0) 
        y_hat =  np.squeeze(y_hat)

        #print ("===yhat==")
        #print(y_hat.shape)
        SWpred.Wstd.data =y_hat

        SWpred = SWpred.unstack("s").set_coords(['time', 'lev', 'lat', 'lon'])
        SWpred = SWpred.transpose('time', 'lev', 'lat', 'lon')

        #=====need to mask places where p>ps
        #print(sp)       
        SWpred =SWpred.where(P<=sp)
        SWpred =SWpred.where(SWpred>=0)
        SWpred =SWpred.where(SWpred<10.)              


        #print(SWpred)
        #=======write it out
        SWpred =  set_atts(SWpred)  
        enc={'Wstd': {'dtype': 'float32', '_FillValue': fill_value}}
        SWpred.to_netcdf(tit, mode = "w", encoding=enc)

        tall = time.time()
        #print('========================')
        print('========total time per month======: ',(tall-t0)/60,' MINUTES')
        #clear memory
        tf.keras.backend.clear_session()
        gc.collect()     




