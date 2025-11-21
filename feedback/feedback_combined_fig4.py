import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import xarray as xr
import time
from dask.distributed import Client, LocalCluster
from datetime import datetime, timedelta
import xesmf as xe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

plt.switch_backend('agg')

plt.rcParams.update({
    'font.size': 5,
    'font.family': 'sans-serif',
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,
    'figure.titlesize': 6,
})

# Constants
fill_value = 1.e+15
chk = {"time": 1}
np.set_printoptions(precision=3)

def set_atts(dsx, var="Wstd"):
    ds = dsx.copy()
    date, date1 = pd.to_datetime(ds['time'].values[0]), pd.to_datetime(ds['time'].values[1])
    td = date1 - date
    time_increment = int(td.total_seconds())
    begin_date = int(date.strftime('%Y%m%d'))
    begin_time = int(date.strftime('%H%M%S'))

    ds["lon"].attrs.update({'long_name': 'longitude', 'standard_name': 'longitude', 'units': 'degrees_east'})
    ds["lat"].attrs.update({'long_name': 'latitude', 'standard_name': 'latitude', 'units': 'degrees_north'})
    ds["time"].attrs.update({
        'time_increment': np.array(time_increment, dtype=np.int32),
        'begin_date': np.array(begin_date, dtype=np.int32),
        'begin_time': np.array(begin_time, dtype=np.int32)
    })

    ds[var].attrs.update({
        "contact": "Donifan Barahona, donifan.o.barahona@nasa.gov",
        "fmissing_value": np.array(fill_value, np.float32),
        "missing_value": np.array(fill_value, np.float32),
        "vmin": np.array(-fill_value, np.float32),
        "vmax": np.array(fill_value, np.float32),
        "valid_range": np.array([-fill_value, fill_value], np.float32)
    })
    return ds

def nb(x, prec=3):
    return np.format_float_scientific(x, precision=prec)

def amean(ds):
    weights = np.cos(np.deg2rad(ds.lat))
    return ds.weighted(weights).mean(("lon", "lat"), skipna=True)

def corrcoefxy(x, y):
    return np.corrcoef(x, y)[0, 1]

def standardize(ds, m, s):
    for i, v in enumerate(ds.data_vars):
        ds[v] = (ds[v] - m[i]) / s[i]
    return ds

def linear_scaler(ds, cst):
    func = lambda x, n: x * n
    return xr.apply_ufunc(func, ds, cst, dask='parallelized')

def ann_mean(ds):
    month_length = ds.time.dt.days_in_month
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)
    ones = xr.where(ds.isnull(), 0.0, 1.0)
    ds_sum = (ds * wgts).resample(time="YS").sum(dim="time", skipna=True)
    ones_sum = (ones * wgts).resample(time="YS").sum(dim="time", skipna=True)
    return ds_sum / ones_sum

def save_stats(var='MOD_CDNC_T258', pth=''):
    print(pth)
    X = xr.open_mfdataset(pth, parallel=True, chunks=chk)[[var]]
    Xyr = ann_mean(X)
    Xmean = Xyr.mean('time').rename({var: var + '_mean'})
    Xstd = Xyr.std('time').rename({var: var + '_std'})
    Xstats = xr.merge([Xmean, Xstd])
    print('-----stats------', Xstats)
    Xstats.to_netcdf(var + '_stats.nc4', mode="w")
    return Xstats

def open_and_regrid(pth='', refds=None, var="", lv=None):
    ds = xr.open_mfdataset(pth, parallel=True, chunks=chk)[[var]]
    if lv is not None:
        ds = ds.sel(lev=lv)
    regridder = xe.Regridder(ds, refds, 'bilinear', periodic=True)
    return regridder(ds)[var]

if __name__ == '__main__':
    encx = {'dtype': 'float32', '_FillValue': fill_value}
    levels = [975, 950, 925, 900, 875]

    RF_all = []
    dndsw_all = []
    dsw_all = []


  


    paths =  {'MERRA2':'/gpfsm/dnb34/dbarahon/ML_param/W_NET/single_level/full_dataset/stats_2015-2020/SWclim_ann_Np.nc4', 
    'ERA5': '/gpfsm/dnb34/dbarahon/ML_param/W_NET/single_level/ERA5/Wnet_era5/prediction/downd_predict/stats_2015-2020/SWclim_ann_Np.nc4', 
    'Combined': "/gpfsm/dnb34/dbarahon/ML_param/W_NET/single_level/combined_ERA5_M2/3H/stats_2015-2020/SWclim_ann_Np.nc4"}
    
    
    rean =  'Combined'
    
    for lv in levels:
        SWpre = xr.open_mfdataset(
            '/gpfsm/dnb34/dbarahon/ML_param/W_NET/single_level/climatologies/paper_clim/ERA20C_preind/stats/SWclim_ann_Np.nc4',
            parallel=True, chunks=chk)[['SWclim']].sel(lev=lv)

 
        
        SWpd_combined = xr.open_mfdataset(paths[rean],parallel=True, chunks=chk)[['SWclim']].sel(lev=lv)
        
        #print(SWpd_combined)

  
        dsw= SWpd_combined - SWpre
        
        
        dndsw = open_and_regrid(
            pth="/gpfsm/dnb34/dbarahon/ML_param/MAMnet/M2_aero/CDNC_fixAerosol/dlnd_dlnsw.nc",
            refds=SWpre, var='slope', lv=lv
        )


        Cf = open_and_regrid(
        pth="../feedback/stats/Cloud_Retrieval_Fraction_Liquid_Mean_stats.nc4",
        refds=SWpre, var='Cloud_Retrieval_Fraction_Liquid_Mean_mean'
        )


        SW = open_and_regrid(
        pth="../feedback/stats/sfc_sw_down_clr_c_mon_stats.nc4",
        refds=SWpre, var='sfc_sw_down_clr_c_mon_mean'
        )

        Ac = open_and_regrid(pth="../feedback/stats/ISCCPALB_stats.nc4", refds=SWpre, var='ISCCPALB_mean')
        
        RF = -SW * Ac * (1 - Ac) * Cf * dndsw * dsw.SWclim / 3
   
        RF_all.append(RF)
        dsw_all.append(dsw)
        dndsw_all.append(dndsw)
        
 
    # Combine and average across levels
    RF_avg = xr.concat(RF_all, dim='lev').mean(dim='lev')
    dsw_avg = xr.concat(dsw_all, dim='lev').mean(dim='lev')
    dndsw_avg = xr.concat(dndsw_all, dim='lev').mean(dim='lev')
    
   
    print('---mean RF---', amean(RF_avg).load())
   
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(7, 3.5), subplot_kw={'projection': ccrs.Robinson()})
    #plt.subplots_adjust(wspace=0.2, hspace=0.2)

    dsall =   xr.Dataset({'dsw': dsw_avg['SWclim']})
    dsall['dndsw'] = dndsw_avg
    dsall['RF'] = RF_avg
   
    lat, lon = dsall['lat'].values, dsall['lon'].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    vars_to_plot = list(dsall.data_vars)
    lab =[r'$\Delta\sigma_{\rm{W}}$ (m s$^{-1})$', r'$\beta$', r'$\Delta$SW (W m$^{-2}$)']
    cmaps = ['RdBu_r', 'YlGnBu', 'BrBG']
    mins = [-0.3, 0. , -1]
    maxs = [0.3, 1, 1]
    ti=  ['(a)', '(b)', '(c)'] 

    for i, (var, ax) in enumerate(zip(vars_to_plot, axs.flat)):
        ax.set_global()
        ax.stock_img()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        data = dsall[var]
        mean_val = amean(data).load()
        plot = ax.pcolormesh(
            lon2d, lat2d, data,
            transform=ccrs.PlateCarree(),
            cmap=cmaps[i],
            shading='auto',
            vmin=mins[i], vmax=maxs[i],
        )

        ax.set_title(f"mean={mean_val:.3f}", fontsize=7, loc="center")
        ax.set_title(f"               {ti[i]}", fontsize=7, loc="left")

        cb = fig.colorbar(plot, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.6)
        cb.set_label(f"{lab[i]}", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(f'Feedback_{rean}.eps', format='eps', bbox_inches='tight')
