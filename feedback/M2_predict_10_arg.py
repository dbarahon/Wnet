
# Predicts Wstd using MERRA2
import sys
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
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
#import tensorflow

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
    
def standardize(ds):
  i = 0
  m= [243.9, 0.6, 6.3, 0.013, 0.0002, 5.04, 21.8, 0.002, 9.75e-7, 7.87e-6]  #hardcoded from G5NR
  s = [30.3, 0.42, 16.1, 7.9, 0.05, 20.6, 20.8, 0.0036, 7.09e-6, 2.7e-5]
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
    o = -ds.OMEGA/9.81/ds.AIRD
    ds.OMEGA.data = o
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
               
if __name__ == '__main__':
    #cluster = LocalCluster(threads_per_worker=8,n_workers =4, processes=False )
    #client = Client(cluster)
    #client = Client()
    #print('---cluster---') 
    #cluster  

    year  =  sys.argv[1]
    fill_value=1.e+15 
    encx = dict(dtype= 'float32', _FillValue= -9999)
    #print(year)
#===============model==========================
    def  Merra_input(yr='2005', mo='01', dy='01', lev1 = 24, lev2=71,  chunk_size = 4096, n_features_in_=8,  tit= ''):

        levs =  72
        chk  = { "lat": -1, "lon": -1, "lev":  -1, "time": 2} 
        n_features_in = 14
     # Read in input from MERRA   
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
        #print(feat_in)
        
        surf_vars =  ['AIRD', 'KM', 'RI', 'QV']  
        nfeats = len(vars_in)+len(surf_vars)

        #print('====feat_in', feat_in)
    
   
        Xall = feat_in#.load()       
        levs = Xall.coords['lev'].values
        nlev =  len(levs)
        b = np.zeros((1, nlev, 1, 1))
        

        #this is no longer the bottleneck
        for v in surf_vars:        
            vv = Xall[v]+0.   
            Xs =  vv.sel(lev=[71]).data #level 1 above surface
            v2 =  v + "_sfc"
            Xsfc =  np.tile(Xs,(1, nlev, 1, 1))
            
            #Xsfc =  Xs + b
            vv.data =  Xsfc
            Xall[v2] =  vv
        
        #print('Xall_sfc=======', Xall)
       
        
        #save here to save time

        
        yall =  Xall['T']
       
        Xall =  Xall.to_array().transpose()
        #print('Xall', Xall.data.shape)
        Xall = Xall.stack( s = ('lat', 'lon'))  # cannot stack all variables because it takes ages to unstack        
        Xall = Xall.squeeze()
        Xall = Xall.transpose()
        #print('Xall', Xall.data.shape)
      
        
       # print("yall'", yall)
        
        return Xall.persist(), yall, P


#=========================================



    #model_g=load_model('/discover/nobackup/dbarahon/ML_param/W_NET/final_model_2/GAN_256/generator.h5', compile=False)
    model_g=load_model('/discover/nobackup/dbarahon/ML_param/W_NET//single_level/response_and_final/GAN/best_generator.h5', compile=False)

    print('================year===========', int(year))
    client =  Client()
    #=========== 
    yr_st =  int(year)
    mo_st = 10#1
    dy_st =  1 #
    yr_end=  int(year) + 1 
    mo_end = 1
    dy_end =  1

    
    td = np.arange(datetime(yr_st,mo_st,dy_st), datetime(yr_end,mo_end,dy_end), timedelta(days=1)).astype(datetime) 
    
    #print(td) 
    
    
    #get nature run as default
        
    for t in td:
        t0 = time.time() 
        y =  str(t.year)
        m =  str(t.month).zfill(2)
        d =  str(t.day).zfill(2)
        #print('processing______', t)
        
        tit =  "MERRA2.inst_3d_Wstd_Np." + y + m + d + ".nc4"
        X, SWpred, P = Merra_input(yr=y, mo=m , dy=d, tit=tit)
        
        Ydata = np.squeeze(np.asarray(SWpred)) 
        Yshape = Ydata.shape
      # print('Ydata', Ydata.shape)
        
        #nlevs = len(X.coords['lev'].values)
        #ntimes = len(X.coords['time'].values)
        t1 = time.time()
       # print('read---: ',(t1-t0)/60,' MINUTES') 
                
        for nt in range(Yshape[0]):
            for l  in range(Yshape[1]):   
                Xx= np.asarray(X.isel(time = nt, lev =  l))
#                print('Xx', Xx.shape)
                SW = model_g.predict(Xx, batch_size = 1200)
                #print('SW0', SW.shape) 
                SW =  np.squeeze(SW)  
                #print('SW1', SW.shape)
                SW = np.reshape(SW, Yshape[2:])
                #print('SW2', SW.shape)
                SW =  np.fmax(np.fmin(SW, 10.), 0.0)
                Ydata[nt, l, :, :] =  SW
                #this does not work SWpred["Wstd"].isel(time = nt, lev =  l).data = SW

        t2 = time.time()
      #  print('evaluate---: ',(t2-t1)/60,' MINUTES')
        
        #print('===SWpred2===', SWpred)
        
        SWpred.data =Ydata
       # print('===SWpred===', SWpred)
        SWpred = int_lev2p(SWpred, P) #this returns a numpy array
        SWpred =  set_atts(SWpred)  
                
        enc={'Wstd': {'dtype': 'float32', '_FillValue': fill_value}}
        #print('===SWpred3===', SWpred)
        #SWpred =  Swpred.load()
        SWpred.to_netcdf(tit, mode = "w", encoding=enc)

        tall = time.time()
        #print('========================')
        print(t, '========total time per day======: ',(tall-t0)/60,' MINUTES')     




