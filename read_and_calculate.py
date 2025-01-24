#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for reading data, universal averaging functions, and computing basin means.

# In[1]:


import numpy as np
import xarray as xr

# modules for plotting datetime data
import matplotlib.dates as mdates
from matplotlib.axis import Axis

# modules for using datetime variables
import datetime
from datetime import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

from xgcm import Grid
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

import cartopy.crs as ccrs
import cmocean

import subprocess as sp

import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from matplotlib.ticker import ScalarFormatter

from xclim import ensembles
from xoverturning import calcmoc
import cmip_basins

import cftime
from pandas.errors import OutOfBoundsDatetime  # Import the specific error


# # Define functions

# ## Read netcdf files and get averages

# In[2]:


def read_data_custom_decode(data_path, debug=False):
    """
    Reads a dataset and ensures time coordinates are compatible with pre-modern dates using cftime.

    Parameters:
    - data_path (str): Path to the NetCDF file.
    - debug (bool): If True, prints debug information.

    Returns:
    - xr.Dataset: Dataset with time converted to cftime-compatible format.
    """
    
    # Open the dataset without decoding times
    dataset = xr.open_dataset(data_path, decode_times=False)
    
    if debug:
        print(f"Initial dataset['time'].dtype: {dataset['time'].dtype}")
        print(f"Initial dataset['time'].attrs: {dataset['time'].attrs}")
    
    # Extract time attributes
    time_units = dataset["time"].attrs.get("units", "days since 1850-01-01")
    calendar = dataset["time"].attrs.get("calendar", "gregorian")
    
    # Validate time units
    if "since" not in time_units:
        time_units = "days since 1850-01-01"  # Fallback
    
    # Convert time using cftime for full compatibility
    try:
        times = cftime.num2date(dataset["time"].values, units=time_units, calendar=calendar)
        dataset["time"] = xr.DataArray(times, dims="time", name="time")
    except Exception as e:
        if debug:
            print(f"Error converting time with cftime: {e}")
        raise

    if debug:
        print(f"Final dataset['time'].dtype: {dataset['time'].dtype}")
        print(f"Final dataset['time'].attrs: {dataset['time'].attrs}")
    
    return dataset


# In[3]:


def get_pp_av_data(exp_name,start_yr,end_yr,chunk_length,pp_type='av-annual',diag_file='ocean_monthly_z',time_decoding=True,month=None,var=None,debug=False):
    """
    Getting post-processed data from the production runs.
        Args:
            exp_name (str)
            start_yr (int)
            end_yr (int)
            chunk_length (int): number of years for av/ts period
            pp_type (str): 'av', 'ts-monthly', 'ts-annual'
            month (int): value between 1 and 12
            var (str): required for reading 'ts' files
            time_decoding (bool): if True, use xr.open_dataset() with use_cftime=True, otherwise use read_data_custom_decode()
        Returns:
            dataset (xr dataset)
    """
    
    static_path = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/{diag_file}.static.nc"
    static_ds = xr.open_dataset(static_path)
    if debug:
        print(f"Done reading static file.")

    if "ocean" in diag_file:
        basin_file = xr.open_dataset("/archive/Kiera.Lowman/basin_AM2_LM3_MOM6i_1deg.nc")
        basin_file = basin_file.rename({'XH': 'xh'})
        basin_file = basin_file.rename({'YH': 'yh'})
        basin_file = basin_file.assign_coords({'xh': static_ds['xh'], 'yh': static_ds['yh']})

    final_start_yr = end_yr - chunk_length + 1
    current_yr = start_yr
    if debug:
        print(f"Final start year: {final_start_yr}")
        print(f"Current year: {current_yr}")

    ### annual averages and time series ###
    if (pp_type == 'av-annual' or pp_type == 'ts-annual') and month == None:
        if pp_type == 'av-annual':
            path_prefix = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/av/annual_{chunk_length}yr"
            data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.ann.nc"
        elif pp_type == 'ts-annual':
            path_prefix = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/ts/annual/{chunk_length}yr"
            data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{var}.nc"

        if time_decoding==True:
            dataset = xr.open_dataset(data_path,use_cftime=True)
        else:
            dataset = read_data_custom_decode(data_path,debug=debug)
        if debug:
            print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")

        while current_yr < final_start_yr:
            current_yr = current_yr + chunk_length
            if pp_type == 'av-annual':
                data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.ann.nc"
            elif pp_type == 'ts-annual':
                data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{var}.nc"
                
            if time_decoding==True:
                chunk_data = xr.open_dataset(data_path,use_cftime=True)
            else:
                chunk_data = read_data_custom_decode(data_path,debug=debug)
                
            dataset = xr.concat([dataset,chunk_data],"time")
            if debug:
                print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")
    
    ### monthly averages for specific month ###
    elif pp_type == 'av-monthly' and month != None:
        path_prefix = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/av/monthly_{chunk_length}yr"
        data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{str(month).zfill(2)}.nc"
        
        if time_decoding==True:
            dataset = xr.open_dataset(data_path,use_cftime=True)
        else:
            dataset = read_data_custom_decode(data_path,debug=debug)
        if debug:
            print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")

        while current_yr < final_start_yr:
            current_yr = current_yr + chunk_length
            data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{str(month).zfill(2)}.nc"
                
            if time_decoding==True:
                chunk_data = xr.open_dataset(data_path,use_cftime=True)
            else:
                chunk_data = read_data_custom_decode(data_path,debug=debug)
                
            dataset = xr.concat([dataset,chunk_data],"time")
            if debug:
                print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")
    
    ### monthly averages for all months ###
    elif pp_type == 'av-monthly' and month == None:
        path_prefix = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/av/monthly_{chunk_length}yr"
        while current_yr <= final_start_yr:
            for month in range(1,13):
                if debug:
                    print(f"Reading month #{month}")
                if month == 1 and current_yr == start_yr:
                    data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{str(month).zfill(2)}.nc"
                    if time_decoding==True:
                        dataset = xr.open_dataset(data_path,use_cftime=True)
                    else:
                        dataset = read_data_custom_decode(data_path,debug=debug)
                else:
                    data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{str(month).zfill(2)}.nc"
                    if time_decoding==True:
                        chunk_data = xr.open_dataset(data_path,use_cftime=True)
                    else:
                        chunk_data = read_data_custom_decode(data_path,debug=debug)
                            
                    dataset = xr.concat([dataset,chunk_data],"time")
    
            if debug:
                print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} month {str(month)} data.")

            current_yr = current_yr + chunk_length

    ### monthly-averaged time series ###
    elif pp_type == 'ts-monthly':
        path_prefix = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/ts/monthly/{chunk_length}yr"
        data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}01-{str(current_yr+chunk_length-1).zfill(4)}12.{var}.nc"

        if time_decoding==True:
            dataset = xr.open_dataset(data_path,use_cftime=True)
        else:
            dataset = read_data_custom_decode(data_path,debug=debug)
        if debug:
            print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")

        while current_yr < final_start_yr:
            current_yr = current_yr + chunk_length
            data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}01-{str(current_yr+chunk_length-1).zfill(4)}12.{var}.nc"
                
            if time_decoding==True:
                chunk_data = xr.open_dataset(data_path,use_cftime=True)
            else:
                chunk_data = read_data_custom_decode(data_path,debug=debug)
                
            dataset = xr.concat([dataset,chunk_data],"time")
            if debug:
                print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")

    
    if "ocean" in diag_file and "scalar" not in diag_file:
        dataset['basin'] = basin_file['basin']
        dataset['dxt'] = static_ds['dxt']
        dataset['dyt'] = static_ds['dyt']
        dataset['areacello'] = static_ds['areacello']
        dataset['wet'] = static_ds['wet']
        dataset['geolon'] = static_ds['geolon']
        dataset['geolat'] = static_ds['geolat']
        dataset = dataset.assign_coords({'geolon': static_ds['geolon'], 'geolat': static_ds['geolat']})
        
    if "geolon_u" in dataset:
        dataset = dataset.drop_vars("geolon_u")
    if "geolon_v" in dataset:
        dataset = dataset.drop_vars("geolon_v")

    return dataset


# In[4]:


# modified version of select_basins function from xoverturning

def selecting_basins(
    ds,
    basin="global",
    lon="geolon",
    lat="geolat",
    mask="wet",
    verbose=True,
):
    """generate a mask for selected basin

    Args:
        ds (xarray.Dataset): dataset contaning model grid
        basin (str or list, optional): global/atl-arc/indopac/atl/pac/arc/antarc or list of codes. Defaults to "global".
        lon (str, optional): name of geographical lon in dataset. Defaults to "geolon".
        lat (str, optional): name of geographical lat in dataset. Defaults to "geolat".
        mask (str, optional): name of land/sea mask in dataset. Defaults to "wet".
        verbose (bool, optional): Verbose output. Defaults to True.

    Returns:
        xarray.DataArray: mask for selected basin
        # xarray.DataArray: mask for MOC streamfunction
    """

    # read or recalculate basin codes
    if "basin" in ds:
        basincodes = ds["basin"]
    else:
        if verbose:
            print("generating basin codes")
        basincodes = cmip_basins.generate_basin_codes(ds, lon=lon, lat=lat, mask=mask)

    # expand land sea mask to remove other basins
    if isinstance(basin, str):
        if basin == "global":
            maxcode = basincodes.max()
            assert not np.isnan(maxcode)
            selected_codes = np.arange(1, maxcode + 1)
            maskbin = ds[mask].where(basincodes.isin(selected_codes))
        elif basin == "glob-no-marg":
            selected_codes = [1, 2, 3, 4, 5] # getting weird AMOC results with inclusion of Med and marginal seas
            maskbin = ds[mask].where(basincodes.isin(selected_codes))
        elif basin == "atl-arc":
            selected_codes = [2, 4, 6, 7, 8, 9]
            maskbin = ds[mask].where(basincodes.isin(selected_codes))
        elif basin == "atl-arc-no-marg":
            selected_codes = [2, 4] # getting weird AMOC results with inclusion of Med and marginal seas
            maskbin = ds[mask].where(basincodes.isin(selected_codes))
        elif basin == "indopac":
            selected_codes = [3, 5, 10, 11]
            maskbin = ds[mask].where(basincodes.isin(selected_codes))
        elif basin == "atl":
            selected_codes = [2]
            cond1 = ds[lon] < 26.5
            cond2 = ds[lon] > -72.5
            cond3 = ds["basin"] == 1
            maskbin = ds[mask].where((basincodes == 2) | (cond1 & cond2 & cond3))
        elif basin == "pac":
            selected_codes = [3]
            cond1 = ds[lon] < -68.5
            cond2 = ds[lon] > -210.5
            cond3 = ds["basin"] == 1
            maskbin = ds[mask].where((basincodes == 3) | (cond1 & cond2 & cond3))
        elif basin == "ind":
            selected_codes = [5]
            maskbin = ds[mask].where(basincodes.isin(selected_codes))
        elif basin == "arc":
            selected_codes = [4]
            maskbin = ds[mask].where(basincodes.isin(selected_codes))
        elif basin == "antarc":
            selected_codes = [1]
            maskbin = ds[mask].where(basincodes.isin(selected_codes))
        else:
            raise ValueError("Unknown basin")
    elif isinstance(basin, list):
        for b in basin:
            assert isinstance(b, int)
        selected_codes = basin
    else:
        raise ValueError("basin must be a string or list of int")

    maskbasin = xr.where(maskbin == 1, True, False)

    return maskbasin


# ## Universal computations

# In[5]:


def zonal_mean(da, metrics):
    num = (da * metrics['dxt'] * metrics['wet']).sum(dim=['xh'])
    denom = (metrics['dxt'] * metrics['wet']).sum(dim=['xh'])
    return num/denom


# In[6]:


def horizontal_mean(da, metrics):
    num = (da * metrics['areacello'] * metrics['wet']).sum(dim=['xh', 'yh'])
    denom = (metrics['areacello'] * metrics['wet']).sum(dim=['xh', 'yh'])
    return num/denom


# In[7]:


def get_2D_yearly_avg(ds,var,long_1,long_2,lat_1,lat_2):
    ds_region = ds.sel(xh=slice(long_1,long_2),yh=slice(lat_1,lat_2))
    da_avg = horizontal_mean(ds_region[var], ds_region)
    da_avg = da_avg.groupby(da_avg['time'].dt.year).mean(dim='time')
    return da_avg


# ## Some older functions that may or may not be useful

# In[8]:


# averages data by month (i.e. averages all January data)
def get_monthly_avg(ds):
    ds_avg = ds.groupby(ds['time'].dt.month).mean(dim='time')
    ds_avg = ds_avg.assign_coords({'geolon': ds['geolon'].isel(time=0), 'geolat': ds['geolat'].isel(time=0)})
    return ds_avg


# In[9]:


# temporally averages all data
def get_time_avg(dataset):
    ds_avg = dataset.mean(dim='time')
    ds_avg = ds_avg.assign_coords({'geolon': dataset['geolon'].isel(time=0), 'geolat': dataset['geolat'].isel(time=0)})
    return ds_avg


# In[10]:


def diff_dat_raw(ds1_raw,ds2_raw,var):
    da1_raw = ds1_raw[var]
    da2_raw = ds2_raw[var]
    diff_da = da2_raw - da1_raw
    # diff_da_avg = diff_da.mean(dim='time')
    diff_da = diff_da.assign_coords({'geolon': da1_raw['geolon'], 'geolat': da1_raw['geolat']})
    return diff_da


# In[11]:


def diff_dat_time_avg(ds1_raw,ds2_raw,var):
    da1_raw = ds1_raw[var]
    da2_raw = ds2_raw[var]
    diff_da = da2_raw - da1_raw
    diff_da_avg = diff_da.mean(dim='time')
    diff_da_avg = diff_da_avg.assign_coords({'geolon': da1_raw['geolon'].isel(time=0), 'geolat': da1_raw['geolat'].isel(time=0)})
    return diff_da_avg


# In[12]:


def diff_dat_monthly_avg(ds1_raw,ds2_raw,var):
    da1_raw = ds1_raw[var]
    da2_raw = ds2_raw[var]
    diff_da = da2_raw - da1_raw
    diff_da_monthly_avg = diff_da.groupby(diff_da['time'].dt.month).mean(dim='time')
    diff_da_monthly_avg = diff_da_monthly_avg.assign_coords({'geolon': da1_raw['geolon'].mean(dim='time'), 'geolat': da1_raw['geolat'].mean(dim='time')})
    return diff_da_monthly_avg


# ## Basin and cross-section functions

# In[13]:


def get_pp_basin_dat(run_ds,basin_name,var,check_nn=True,nn_threshold=0.05,full_field_var=None,mask_ds=None,\
                     single_var_da=False,verbose=False):
    
    if mask_ds is None:
        maskbasin = selecting_basins(run_ds, basin=basin_name, verbose=False)
        if verbose:
            print("mask_ds is none ")
    else:
        maskbasin = selecting_basins(mask_ds, basin=basin_name, verbose=False)
        
    ds_basin = run_ds.where(maskbasin)

    if single_var_da == False:
        dat_slice = ds_basin[var]
    else:
        dat_slice = ds_basin
    
    if verbose:
        print(f"Min: {np.nanmin(dat_slice.values)} \t Max: {np.nanmax(dat_slice.values)}")

    if mask_ds is None:
        dat_basin_avg = zonal_mean(dat_slice, ds_basin)
        correct_lat = zonal_mean(run_ds['geolat'], run_ds)
    else:
        mask_dat_slice = mask_ds[['dxt','wet']].where(maskbasin)
        dat_basin_avg = zonal_mean(dat_slice, mask_dat_slice)
        correct_lat = zonal_mean(mask_ds['geolat'], mask_ds)
    
    dat_basin_avg = dat_basin_avg.rename({'yh': 'true_lat'})
    dat_basin_avg = dat_basin_avg.assign_coords({'true_lat': correct_lat.values})
    dat_basin_avg = dat_basin_avg.sortby('true_lat')

    if check_nn:
        if full_field_var == None:
            not_null = dat_slice.notnull()
            if verbose:
                print(f"dat_slice.sizes['xh'] = {dat_slice.sizes['xh']}")
            nn_min = int(dat_slice.sizes['xh']*nn_threshold)
        else:
            ff_dat_slice = ds_basin[full_field_var]
            not_null = ff_dat_slice.notnull()
            nn_min = int(ff_dat_slice.sizes['xh']*nn_threshold)
            
        not_null_int = not_null.astype('int')
        not_null_count = not_null_int.sum(dim=['xh'])
        not_null_count = not_null_count.rename({'yh': 'true_lat'})
        not_null_count['true_lat'] = correct_lat.values
        not_null_count = not_null_count.sortby('true_lat')
    
        # nn_min = int(dat_slice.sizes['xh']*nn_threshold)
    
        dat_basin_avg = dat_basin_avg.where(not_null_count > nn_min).isel(true_lat=slice(0,-1))
        
    else:
        dat_basin_avg = dat_basin_avg.isel(true_lat=slice(0,-1))
    
    return dat_basin_avg


# In[14]:


def get_basin_horiz_avg(ds,var,basin_name):
    ds_region = get_pp_basin_dat(ds,basin_name,var,check_nn=False)
    da_avg = horizontal_mean(ds_region[var], ds_region)
    da_avg = da_avg.groupby(da_avg['time']).mean(dim='time')
    return da_avg

