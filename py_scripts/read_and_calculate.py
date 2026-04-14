#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for reading data, universal averaging functions, and computing basin means.

import numpy as np
import xarray as xr

# modules for using datetime variables
import datetime
from datetime import time

import warnings
warnings.filterwarnings('ignore')

import cftime
from pandas.errors import OutOfBoundsDatetime  # Import the specific error

from xclim import ensembles
import cmip_basins
import momlevel as ml # Use Wright EOS

import time as time_module

# import gfdl_utils
from gfdl_utils.core import issue_dmget
from gfdl_utils.core import query_all_ondisk


# # Define functions

from functools import lru_cache

@lru_cache(maxsize=None)
def _open_static(static_path):
    return xr.open_dataset(static_path)


# ## Read netcdf files and get averages

def read_data_custom_decode(data_path, pp_type='av-annual', debug=False, dask_chunk=None, alt_time=None):
    """
    Reads a dataset and ensures time coordinates are compatible with pre-modern dates using cftime.

    Parameters:
    - data_path (str): Path to the NetCDF file.
    - debug (bool): If True, prints debug information.

    Returns:
    - xr.Dataset: Dataset with time converted to cftime-compatible format.
    """

    if alt_time:
        time_dim = alt_time
    else:
        time_dim = 'time'

    if debug:
        print("START read_data_custom_decode")
    
    # Open the dataset without decoding times
    if dask_chunk == None:
        dask_chunk = 1

    if pp_type=='av-annual':
        dataset = xr.open_dataset(data_path,decode_times=False,chunks={time_dim:dask_chunk})
    else:
        dataset = xr.open_mfdataset(
                        data_path,
                        combine="by_coords",
                        coords="minimal",
                        data_vars="minimal",
                        compat="override",
                        combine_attrs="override",
                        chunks={time_dim:dask_chunk},          # avoid overchunking
                        drop_variables=["time_bnds"],           # fine as list
                        engine="netcdf4",
                        decode_times=False,
                        parallel=False,                         # important for this error
                    )
    
    if 'time_bnds' in dataset:
        dataset = dataset.drop_vars('time_bnds')

    if debug:
        print("From read_data_custom_decode: Opened dataset")
        print(f"Initial dataset[time_dim].dtype: {dataset[time_dim].dtype}")
        print(f"Initial dataset[time_dim].attrs: {dataset[time_dim].attrs}")
    
    # Extract time attributes
    time_units = dataset[time_dim].attrs.get("units", "days since 1850-01-01")
    calendar = dataset[time_dim].attrs.get("calendar", "gregorian")
    
    # Validate time units
    if "since" not in time_units:
        time_units = "days since 1850-01-01"  # Fallback
    
    # Convert time using cftime for full compatibility
    try:
        times = cftime.num2date(dataset[time_dim].values, units=time_units, calendar=calendar)
        dataset[time_dim] = xr.DataArray(times, dims=time_dim, name=time_dim)
    except Exception as e:
        if debug:
            print(f"Error converting time with cftime: {e}")
        raise

    if debug:
        print("EXIT read_data_custom_decode")
    
    return dataset


def get_pp_av_data(exp_name,start_yr,end_yr,chunk_length,pp_type='av-annual',diag_file='ocean_monthly_z',time_decoding=True,var=None,month=None,debug=False):
    """
    Getting post-processed data from the production runs.
        Args:
            exp_name (str)
            start_yr (int)
            end_yr (int)
            chunk_length (int): number of years for av/ts period
            pp_type (str): 'av-annual', 'ts-annual', 'av-monthly', or 'ts-monthly'
            diag_file (str): pp directory name, such as 'ocean_monthly_z', 'ocean_monthly_rho2', etc.
            time_decoding (bool): if True, use xr.open_dataset() with use_cftime=True, otherwise use read_data_custom_decode()
            var (str list): required for reading 'ts' files
            month (int): value between 1 and 12
            debug (bool): if true, give verbose output
        Returns:
            dataset (xarray dataset)
    """

    # use the control for the static file every time, because sometimes static files don't get saved properly
    if diag_file == 'atmos_monthly':
        static_path = "/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/00010101.grid_spec.nc"
    else:
        static_path = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/tune_ctrl_const_200yr/gfdl.ncrc5-intel23-prod/pp/{diag_file}/{diag_file}.static.nc"
    # static_ds = xr.open_dataset(static_path)
    static_ds = _open_static(static_path)
    if debug:
        print(f"Done reading static file.")
    
    # static_path = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/{diag_file}.static.nc"
    # static_ds = xr.open_dataset(static_path)
    # if debug:
    #     print(f"Done reading static file.")

    # not automatically assigning basin codes anymore
    # if "ocean" in diag_file:
    #     basin_file = xr.open_dataset("/archive/Kiera.Lowman/basin_AM2_LM3_MOM6i_1deg.nc")
    #     basin_file = basin_file.rename({'XH': 'xh'})
    #     basin_file = basin_file.rename({'YH': 'yh'})
    #     basin_file = basin_file.assign_coords({'xh': static_ds['xh'], 'yh': static_ds['yh']})

    current_yr = start_yr
    final_start_yr = end_yr - chunk_length + 1
    if debug:
        print(f"Initial and final start years: {current_yr} and {final_start_yr}")

    ### annual averages ###
    if pp_type == 'av-annual':
        path_prefix = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/av/annual_{chunk_length}yr"
        data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.ann.nc"

        # I should remove dask time chunking for the av-annual and av-monthly data
        if time_decoding==True:
            dataset = xr.open_dataset(data_path,use_cftime=True,chunks={'time':1})
        else:
            dataset = read_data_custom_decode(data_path,pp_type=pp_type,debug=False)
        if debug:
            print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")

        while current_yr < final_start_yr:
            current_yr = current_yr + chunk_length
            data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.ann.nc"
                
            if time_decoding==True:
                chunk_data = xr.open_dataset(data_path,use_cftime=True,chunks={'time':1})
            else:
                chunk_data = read_data_custom_decode(data_path,pp_type=pp_type,debug=False)
                
            dataset = xr.concat([dataset,chunk_data],"time")
            if debug:
                print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")

    ### annual time series for one or more variables ###
    elif pp_type == 'ts-annual':
        if var is None:
            raise IOError("'var' list not specified for ts-annual data.")
        path_prefix = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/ts/annual/{chunk_length}yr"

        if var[0] == "net_dn_toa" or var[0] == "net_dn_toa_clr":
            raise IOError("'net_dn_toa' and 'net_dn_toa_clr' need to be moved to later in the variable list.")

        data_path = []
        for elem in var:
            path_suffix = f"{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{elem}.nc"
            data_path.append(f"{path_prefix}/{diag_file}.{path_suffix}")

        # path_suffix = f"{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{var[0]}.nc"
        # data_path = f"{path_prefix}/{diag_file}.{path_suffix}"

        if len(data_path) > 0:
            if debug:
                print("Issuing dmget command to migrate data to disk.", end=" ")
            issue_dmget(data_path)
            while not(query_all_ondisk(data_path)):
                time_module.sleep(0.1)
            if debug:
                print("Migration complete.")
            
        if time_decoding==True:
            # dataset = xr.open_dataset(data_path,use_cftime=True,chunks={'time':chunk_length},drop_variables='time_bnds')
            # dataset = xr.open_mfdataset(data_path,use_cftime=True,coords="minimal",combine="by_coords",chunks={'time':chunk_length},parallel=True,drop_variables='time_bnds')
            dataset = xr.open_mfdataset(
                            data_path,
                            use_cftime=True,
                            combine="by_coords",
                            coords="minimal",
                            data_vars="minimal",
                            compat="override",
                            combine_attrs="override",
                            chunks={'time':chunk_length},          # avoid overchunking
                            drop_variables=["time_bnds"],           # fine as list
                            engine="netcdf4",
                            decode_times=False,
                            parallel=False,                         # important for this error
                        )
        else:
            dataset = read_data_custom_decode(data_path,pp_type=pp_type,debug=False,dask_chunk=chunk_length)
        if debug:
            # print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data for {var[0]}.")
            # print(f"Done reading {path_suffix}.")
            print(f"Done reading year {current_yr} to {current_yr+chunk_length-1}.")
            
        # if len(var) > 1:
        #     for i in range(1,len(var)):
        #         if var[i] == "net_dn_toa" or var[i] == "net_dn_toa_clr":
        #             continue
        #         path_suffix = f"{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{var[i]}.nc"
        #         next_var_path = f"{path_prefix}/{diag_file}.{path_suffix}"
                
        #         if time_decoding==True:
        #             # next_var_data = xr.open_dataset(next_var_path,use_cftime=True,chunks="auto",drop_variables='time_bnds')
        #             next_var_data = xr.open_dataset(next_var_path,use_cftime=True,chunks={'time':chunk_length},drop_variables='time_bnds')
        #             # next_var_data = xr.open_dataset(next_var_path,use_cftime=True,chunks={'time':1},drop_variables='time_bnds')
        #         else:
        #             next_var_data = read_data_custom_decode(next_var_path,debug=False,dask_chunk=chunk_length)
        #         if debug:
        #             # print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data for {var[i]}.")
        #             print(f"Done reading {path_suffix}.")
                    
        #         dataset = xr.merge([dataset, next_var_data], compat="equals")
                
        #     if debug:
        #         print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data for all variables.")
        
        while current_yr < final_start_yr:
            current_yr = current_yr + chunk_length

            data_path = []
            for elem in var:
                path_suffix = f"{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{elem}.nc"
                data_path.append(f"{path_prefix}/{diag_file}.{path_suffix}")
                
            # path_suffix = f"{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{var[0]}.nc"
            # data_path = f"{path_prefix}/{diag_file}.{path_suffix}"

            if len(data_path) > 0:
                if debug:
                    print("Issuing dmget command to migrate data to disk.", end=" ")
                issue_dmget(data_path)
                while not(query_all_ondisk(data_path)):
                    time_module.sleep(0.1)
                if debug:
                    print("Migration complete.")
            
            if time_decoding==True:
                # chunk_data = xr.open_dataset(data_path,use_cftime=True,chunks={'time':chunk_length},drop_variables='time_bnds')
                # chunk_data = xr.open_mfdataset(data_path,use_cftime=True,coords="minimal",combine="by_coords",chunks={'time':chunk_length},parallel=True,drop_variables='time_bnds')
                chunk_data = xr.open_mfdataset(
                            data_path,
                            use_cftime=True,
                            combine="by_coords",
                            coords="minimal",
                            data_vars="minimal",
                            compat="override",
                            combine_attrs="override",
                            chunks={'time':chunk_length},          # avoid overchunking
                            drop_variables=["time_bnds"],           # fine as list
                            engine="netcdf4",
                            decode_times=False,
                            parallel=False,                         # important for this error
                        )
            else:
                chunk_data = read_data_custom_decode(data_path,pp_type=pp_type,debug=False,dask_chunk=chunk_length)

            dataset = xr.concat([dataset,chunk_data],"time")
            if debug:
                # print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data for {var[0]}.")
                # print(f"Done reading {path_suffix}.")
                print(f"Done reading year {current_yr} to {current_yr+chunk_length-1}.")
                
            # if len(var) > 1:
            #     for i in range(1,len(var)):
            #         if var[i] == "net_dn_toa" or var[i] == "net_dn_toa_clr":
            #             continue
            #         path_suffix = f"{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{var[i]}.nc"
            #         next_var_path = f"{path_prefix}/{diag_file}.{path_suffix}"
                    
            #         if time_decoding==True:
            #             # next_var_chunk = xr.open_dataset(next_var_path,use_cftime=True,chunks="auto",drop_variables='time_bnds')
            #             next_var_chunk = xr.open_dataset(next_var_path,use_cftime=True,chunks={'time':chunk_length},drop_variables='time_bnds')
            #             # next_var_chunk = xr.open_dataset(next_var_path,use_cftime=True,chunks={'time':1},drop_variables='time_bnds')
            #         else:
            #             next_var_chunk = read_data_custom_decode(next_var_path,debug=False,dask_chunk=chunk_length)
            #         if debug:
            #             # print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data for {var[i]}.")
            #             print(f"Done reading {path_suffix}.")
                        
            #         chunk_data = xr.merge([chunk_data, next_var_chunk], compat="equals")

            # dataset = xr.concat([dataset,chunk_data],"time")
            # if debug:
            #     print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data for all variables.")
                
    ### monthly averages for specific month ###
    elif pp_type == 'av-monthly' and month != None:
        path_prefix = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/av/monthly_{chunk_length}yr"
        data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{str(month).zfill(2)}.nc"
        
        if time_decoding==True:
            dataset = xr.open_dataset(data_path,use_cftime=True,chunks={'time':1})
        else:
            dataset = read_data_custom_decode(data_path,pp_type=pp_type,debug=False)
        if debug:
            print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")

        while current_yr < final_start_yr:
            current_yr = current_yr + chunk_length
            data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{str(month).zfill(2)}.nc"
            if time_decoding==True:
                chunk_data = xr.open_dataset(data_path,use_cftime=True,chunks={'time':1})
            else:
                chunk_data = read_data_custom_decode(data_path,pp_type=pp_type,debug=False)
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
                        dataset = xr.open_dataset(data_path,use_cftime=True,chunks={'time':1})
                    else:
                        dataset = read_data_custom_decode(data_path,pp_type=pp_type,debug=False)
                else:
                    data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}-{str(current_yr+chunk_length-1).zfill(4)}.{str(month).zfill(2)}.nc"
                    if time_decoding==True:
                        chunk_data = xr.open_dataset(data_path,use_cftime=True,chunks={'time':1})
                    else:
                        chunk_data = read_data_custom_decode(data_path,pp_type=pp_type,debug=False)   
                    dataset = xr.concat([dataset,chunk_data],"time")
            if debug:
                print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} month {str(month)} data.")

            current_yr = current_yr + chunk_length

    ### monthly-averaged time series for a SINGLE variable ###
    elif pp_type == 'ts-monthly':
        if var is None:
            raise IOError("'var' not specified for ts-monthly data.")
        elif len(var) > 1:
            raise IOError("Reading ts-monthly data for multiple variables not supported. Provide a list of length 1.")
        path_prefix = f"/archive/Kiera.Lowman/FMS2019.01.03_devgfdl_20201120_kiera/{exp_name}/gfdl.ncrc5-intel23-prod/pp/{diag_file}/ts/monthly/{chunk_length}yr"
        data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}01-{str(current_yr+chunk_length-1).zfill(4)}12.{var[0]}.nc"

        # should change chunks
        if time_decoding==True:
            dataset = xr.open_dataset(data_path,use_cftime=True,chunks={'time':1})
        else:
            dataset = read_data_custom_decode(data_path,pp_type=pp_type,debug=False)
        if debug:
            print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")

        while current_yr < final_start_yr:
            current_yr = current_yr + chunk_length
            data_path = f"{path_prefix}/{diag_file}.{str(current_yr).zfill(4)}01-{str(current_yr+chunk_length-1).zfill(4)}12.{var[0]}.nc"

            # should change chunks
            if time_decoding==True:
                chunk_data = xr.open_dataset(data_path,use_cftime=True,chunks={'time':1})
            else:
                chunk_data = read_data_custom_decode(data_path,pp_type=pp_type,debug=False)
            dataset = xr.concat([dataset,chunk_data],"time")
            if debug:
                print(f"Done reading year {current_yr} to {current_yr+chunk_length-1} data.")

    # I think maybe I shouldn't drop "geolon_u","geolon_v","geolon_c","geolat_u","geolat_v","geolat_c", but I think I was before because
    # xarray was getting mixed up between the different geolon and geolat variables when searching
    vars_to_drop = ["time_bnds","nv"]
    for elem in vars_to_drop:
        if elem in dataset:
            dataset = dataset.drop_vars(elem)
            
    if ("ocean" in diag_file and "scalar" not in diag_file):
        # dataset['basin'] = basin_file['basin']
        dataset['dxt'] = static_ds['dxt']
        dataset['dyt'] = static_ds['dyt']
        dataset['dxCu'] = static_ds['dxCu']
        dataset['dyCu'] = static_ds['dyCu']
        dataset['dxCv'] = static_ds['dxCv']
        dataset['dyCv'] = static_ds['dyCv']
        dataset['areacello'] = static_ds['areacello']
        dataset['areacello_cu'] = static_ds['areacello_cu']
        dataset['areacello_cv'] = static_ds['areacello_cv']
        dataset['deptho'] = static_ds['deptho']
        dataset['wet'] = static_ds['wet']
        dataset['geolon'] = static_ds['geolon']
        dataset['geolat'] = static_ds['geolat']
        dataset['wet_u'] = static_ds['wet_u']
        dataset['geolon_u'] = static_ds['geolon_u']
        dataset['geolat_u'] = static_ds['geolat_u']
        dataset['wet_v'] = static_ds['wet_v']
        dataset['geolon_v'] = static_ds['geolon_v']
        dataset['geolat_v'] = static_ds['geolat_v']
        dataset['wet_c'] = static_ds['wet_c']
        dataset['geolon_c'] = static_ds['geolon_c']
        dataset['geolat_c'] = static_ds['geolat_c']
        dataset = dataset.assign_coords({'geolon': static_ds['geolon'], 'geolat': static_ds['geolat'],
                                         'geolon_u': static_ds['geolon_u'], 'geolat_u': static_ds['geolat_u'],
                                         'geolon_v': static_ds['geolon_v'], 'geolat_v': static_ds['geolat_v'],
                                         'geolon_c': static_ds['geolon_c'], 'geolat_c': static_ds['geolat_c'],})
    elif diag_file == "ocean_monthly":
        if "volcello" in dataset:
            dataset = dataset.drop_vars("volcello")
            dataset = dataset.drop_vars("zl")

    elif diag_file == "ice":
        dataset['CELL_AREA'] = static_ds['CELL_AREA']
        dataset['COSROT'] = static_ds['COSROT']
        dataset['SINROT'] = static_ds['SINROT']
        dataset['GEOLON'] = static_ds['GEOLON']
        dataset['GEOLAT'] = static_ds['GEOLAT']
        dataset = dataset.assign_coords({'GEOLON': static_ds['GEOLON'], 'GEOLAT': static_ds['GEOLAT']})

    elif diag_file == "atmos_monthly":
        dataset['area'] = static_ds['area']

    return dataset


# modified version of select_basins function from xoverturning

def selecting_basins(
    ds,
    basin="global",
    lon="geolon",
    lat="geolat",
    mask="wet",
    verbose=False,
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
            cond1 = ds[lon] < 20.5 # originally 26.5
            cond2 = ds[lon] > -70.5
            cond3 = basincodes == 1
            maskbin = ds[mask].where((basincodes.isin(selected_codes)) | (cond1 & cond2 & cond3))
        elif basin == "atl-arc-south":
            selected_codes = [1, 2, 4, 6, 7, 8, 9]
            maskbin = ds[mask].where((basincodes.isin(selected_codes)))
        elif basin == "atl-arc-no-marg":
            selected_codes = [2, 4] # getting weird AMOC results with inclusion of Med and marginal seas
            cond1 = ds[lon] < 20.5 # originally 26.5
            cond2 = ds[lon] > -70.5
            cond3 = basincodes == 1
            maskbin = ds[mask].where((basincodes.isin(selected_codes)) | (cond1 & cond2 & cond3))
        elif basin == "indopac":
            selected_codes = [3, 5, 10, 11]
            maskbin = ds[mask].where(basincodes.isin(selected_codes))
        elif basin == "atl":
            # selected_codes = [2]
            cond1 = ds[lon] < 20.5 # originally 26.5
            cond2 = ds[lon] > -70.5
            cond3 = basincodes == 1
            maskbin = ds[mask].where((basincodes == 2) | (cond1 & cond2 & cond3))
        elif basin == "pac":
            # selected_codes = [3]
            cond1 = ds[lon] <= -70.5
            cond2 = ds[lon] > -210.5
            cond3 = basincodes == 1
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

# wet and dxt is only defined at the sea surface -- this is ultimately what's causing problems
def zonal_mean_v1(da, metrics):
    num = (da * metrics['dxt'] * metrics['wet']).sum(dim=['xh'], skipna=True)
    denom = (metrics['dxt'] * metrics['wet']).sum(dim=['xh'], skipna=True)
    return num/denom


# def zonal_mean(da, metrics):
#     dxt_3d = xr.where(da.isnull(), np.nan, metrics['dxt'])
#     num = (da * dxt_3d).sum(dim=['xh'], skipna=True)
#     denom = dxt_3d.sum(dim=['xh'], skipna=True)
#     return num/denom


def zonal_mean(da, metrics, grid_type='tracer_pt'):
    if grid_type=='u_pt':
        x_dim = 'xq'
        dx_metric = 'dxCu'
    elif grid_type=='v_pt':
        x_dim = 'xh'
        dx_metric = 'dxCv'
    else:
        x_dim = 'xh'
        dx_metric = 'dxt'
        
    dxt_3d = xr.where(da.isnull(), np.nan, metrics[dx_metric])
    num = (da * dxt_3d).sum(dim=[x_dim], skipna=True)
    denom = dxt_3d.sum(dim=[x_dim], skipna=True)
    return num/denom


def zonal_integral(da, metrics, grid_type="tracer_pt"):
    """
    Zonal integral over longitude: ∫ da dx (units: da * meters).

    Notes:
    - Uses dx metrics consistent with your zonal_mean():
        tracer_pt -> xh with dxt
        u_pt      -> xq with dxCu
        v_pt      -> xh with dxCv
    - Masks dx wherever da is NaN so basin masking behaves as expected.
    """
    if grid_type == "u_pt":
        x_dim = "xq"
        dx_metric = "dxCu"
    elif grid_type == "v_pt":
        x_dim = "xh"
        dx_metric = "dxCv"
    else:
        x_dim = "xh"
        dx_metric = "dxt"

    dx_3d = xr.where(da.isnull(), np.nan, metrics[dx_metric])
    return (da * dx_3d).sum(dim=[x_dim], skipna=True)


def ice_zonal_mean(da, metrics):
    num = (da * metrics["CELL_AREA"]).sum(dim="xT", skipna=True)
    denom = metrics["CELL_AREA"].sum(dim="xT", skipna=True)
    return num/denom


# wet, areacello is only defined at the sea surface -- this is ultimately what's causing problems
def horizontal_mean_v1(da, metrics):
    num = (da * metrics['areacello'] * metrics['wet']).sum(dim=['xh', 'yh'])
    denom = (metrics['areacello'] * metrics['wet']).sum(dim=['xh', 'yh'])
    return num/denom


# def horizontal_mean(da, metrics):
#     area_3d = xr.where(da.isnull(), np.nan, metrics['areacello'])
#     num = (da * area_3d).sum(dim=['xh', 'yh'], skipna=True)
#     denom = area_3d.sum(dim=['xh', 'yh'], skipna=True)
#     return num/denom


def horizontal_mean(da, metrics, grid_type='tracer_pt'):
    if grid_type=='u_pt':
        x_dim = 'xq'
        y_dim = 'yh'
        area_metric = 'areacello_cu'
    elif grid_type=='v_pt':
        x_dim = 'xh'
        y_dim = 'yq'
        area_metric = 'areacello_cv'
    else:
        x_dim = 'xh'
        y_dim = 'yh'
        area_metric = 'areacello'
        
    area_3d = xr.where(da.isnull(), np.nan, metrics[area_metric])
    num = (da * area_3d).sum(dim=[x_dim, y_dim], skipna=True)
    denom = area_3d.sum(dim=[x_dim, y_dim], skipna=True)
    return num/denom


# both function versions give identical answers
def atmos_horiz_mean_v1(da, metrics):
    weights = np.cos(np.deg2rad(metrics["lat"]))
    w2d, _ = xr.broadcast(weights, da)

    global_mean = da.weighted(w2d).mean(dim=["lat", "lon"])

    # # alternative that gives identical values
    # my_mean = (
    #     (da * w2d).sum(dim=["lat", "lon"]) / w2d.sum(dim=["lat", "lon"])
    # )
    
    return global_mean

def atmos_horiz_mean(da, metrics):
    x_dim = 'lon'
    y_dim = 'lat'
    area_metric = 'area'
    
    area_3d = xr.where(da.isnull(), np.nan, metrics[area_metric])
    num = (da * area_3d).sum(dim=[x_dim, y_dim], skipna=True)
    denom = area_3d.sum(dim=[x_dim, y_dim], skipna=True)
    return num/denom


def get_2D_yearly_avg(ds,var,long_1,long_2,lat_1,lat_2):
    ds_region = ds.sel(xh=slice(long_1,long_2),yh=slice(lat_1,lat_2))
    da_avg = horizontal_mean(ds_region[var], ds_region)
    da_avg = da_avg.groupby(da_avg['time'].dt.year).mean(dim='time')
    return da_avg


def calc_zrho_dat(static_rho,ds_rho,cent_out='cent',x_mean=True,ds_z=None):
    """
    Function for computing depth field from density space data to plot cross-sectional data.

    To plot a density-space field as a function of depth, such as the overturning streamfunction, add depth_field
    as a coordinate:
    # psi.coords['depth'] = depth_field

    Returns:
        depth_field: depth as a function of zl and y dimension of choice (as determined by cent_out parameter)
        
    """
    # note that the vertical coordinate of ds_rho is 'zl' and of ds_z is 'z_l'
    
    # create a grid using xgcm 
    coords = {
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'},
    }   
    metrics = {
        'X': ["dxt", "dxCu", "dxCv"],
        'Y': ["dyt", "dyCu", "dyCv"]
    }
    grid = Grid(static_rho, coords=coords, metrics=metrics, periodic=['X'])

    # calculate cell thickness from volcello and areacello (I don't have thkcello saved)
    if time in ds_rho.dims:
        thk  = ds_rho['volcello'].mean(dim='time')/static_rho['areacello']
    else:
        thk  = ds_rho['volcello']/static_rho['areacello']
    thk  = thk.rename('thkcello')

    # calculate z from the cell thickness in density space
    if x_mean is True:
        zrho = thk.mean(dim='xh').cumsum(dim='zl')
    else:
        # zrho will also be a function of xh
        zrho = thk.cumsum(dim='zl')
        
    zrho = zrho.rename('zrho')

    if cent_out == 'cent':
        depth_field = zrho
    elif cent_out == 'out':
        # create depth field defined as a function of yq and potential density (zl)
        depth_field = grid.interp(zrho, 'Y', boundary='extend')

    # # calculating the average depth
    # toz    = ds_z['temp'].mean(dim='time') 
    # soz    = ds_z['salt'].mean(dim='time') 
    # sigmaz = ml.derived.calc_pdens(toz, soz, level=2000.0) - 1000.0

    # mask   = toz/toz
    # delz   = xr.DataArray(np.diff(ds_z['z_i']), dims='z_l')
    # dvol   = delz * mask * static_rho['areacello'] 
    # sigmaz_dvol = sigmaz * dvol
    # sigmaz_xave = sigmaz_dvol.sum(dim='xh')/dvol.sum(dim='xh')

    return depth_field


# ## OHC anomaly calc

def calc_OHC_anom(thetaoga_anom_da, avg_V = 1.33287e+18):
    rho0 = 1035 # [kg m-3]
    cp = 3991.86795711963 # [J degC-1 kg-1] (https://ncar.github.io/MOM6/APIs/structmom__variables_1_1thermo__var__ptrs.html#aa0a5c6588326f0bc16576eae0aed2e0c)
    ZJ_conv = 1e21
    
    x_dim = 'xh'
    y_dim = 'yh'
    z_dim = 'z_l'
    
    # Vref = volcello_ref_da.sum(dim=[x_dim,y_dim,z_dim], skipna=True)
    # OHC_anom = cp * rho0 * Vref * thetaoga_anom_da

    OHC_anom = cp * rho0 * avg_V * thetaoga_anom_da / ZJ_conv

    return OHC_anom


def _infer_dz_from_vertical_coord(da, z_dim="z_l", z_interface_dim="z_i"):
    """
    Infer layer thickness dz [m] as a 1D DataArray on z_dim.

    Priority:
    1) Use interface coordinate z_i if present
    2) Fall back to spacing inferred from z_l cell centers
    """
    if z_interface_dim in da.coords:
        z_i = da.coords[z_interface_dim]
        if z_i.ndim != 1:
            raise ValueError(f"{z_interface_dim} must be 1D to infer dz.")
        dz_vals = np.diff(z_i.values)
        dz = xr.DataArray(
            dz_vals,
            dims=[z_dim],
            coords={z_dim: da.coords[z_dim].values},
            name="dz"
        )
        return dz

    if z_dim not in da.coords:
        raise ValueError(f"Could not infer dz: neither {z_interface_dim} nor {z_dim} coordinate found.")

    z = da.coords[z_dim].values
    if len(z) < 2:
        raise ValueError(f"Need at least 2 points in {z_dim} to infer dz from cell centers.")

    # Infer cell edges from center spacing
    edges = np.empty(len(z) + 1, dtype=float)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = z[0] - 0.5 * (z[1] - z[0])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])

    dz_vals = np.diff(edges)
    dz = xr.DataArray(
        dz_vals,
        dims=[z_dim],
        coords={z_dim: da.coords[z_dim].values},
        name="dz"
    )
    return dz


def _truncate_dz_to_depth_range(dz, z_coord, z_top=None, z_bot=None):
    """
    Optionally restrict dz to a depth interval [z_top, z_bot] in meters.
    Assumes positive-down vertical coordinate.
    """
    z = z_coord.values

    # Reconstruct layer edges from centers + dz
    edges_top = z - 0.5 * dz.values
    edges_bot = z + 0.5 * dz.values

    if z_top is None:
        z_top = float(np.nanmin(edges_top))
    if z_bot is None:
        z_bot = float(np.nanmax(edges_bot))

    overlap = np.maximum(0.0, np.minimum(edges_bot, z_bot) - np.maximum(edges_top, z_top))

    dz_trunc = xr.DataArray(
        overlap,
        dims=dz.dims,
        coords=dz.coords,
        name=dz.name
    )
    return dz_trunc


# def compute_column_ohc_anom(
#     temp_anom,
#     temp_var="temp",
#     z_dim="z_l",
#     z_interface_dim="z_i",
#     rho0=1035.0,
#     cp0=3991.86795711963,
#     z_top=None,
#     z_bot=None,
#     output_units="GJ m^-2",
#     keep_attrs=True,
# ):
#     """
#     Compute column-integrated ocean heat content anomaly from a 3D temperature anomaly.

#     Parameters
#     ----------
#     temp_anom : xr.DataArray or xr.Dataset
#         Temperature anomaly field with dimensions including z_dim.
#         Expected units: degC or K anomaly.
#     temp_var : str
#         Variable name if temp_anom is a Dataset.
#     z_dim : str
#         Vertical cell-center dimension name.
#     z_interface_dim : str
#         Vertical interface coordinate name, if available.
#     rho0 : float
#         Reference seawater density [kg m^-3].
#     cp0 : float
#         Seawater heat capacity [J kg^-1 K^-1].
#     z_top, z_bot : float or None
#         Optional depth bounds in meters, positive downward.
#         Example: z_top=0, z_bot=700 for upper-ocean OHC anomaly.
#     output_units : str
#         "J m^-2", "GJ m^-2", or "10^9 J m^-2".
#     keep_attrs : bool
#         Whether to preserve metadata where sensible.

#     Returns
#     -------
#     xr.DataArray
#         2D (or time-varying 2D) column-integrated OHC anomaly.
#     """
#     if isinstance(temp_anom, xr.Dataset):
#         if temp_var not in temp_anom:
#             raise ValueError(f"{temp_var} not found in Dataset.")
#         da = temp_anom[temp_var]
#     else:
#         da = temp_anom

#     if z_dim not in da.dims:
#         raise ValueError(f"{z_dim} must be a dimension of the temperature anomaly field.")

#     dz = _infer_dz_from_vertical_coord(da, z_dim=z_dim, z_interface_dim=z_interface_dim)

#     if z_top is not None or z_bot is not None:
#         dz = _truncate_dz_to_depth_range(dz, da.coords[z_dim], z_top=z_top, z_bot=z_bot)

#     # Broadcast dz onto the full array
#     weighted = da * dz

#     # Sum vertically, preserving NaN land mask naturally
#     ohc_anom = rho0 * cp0 * weighted.sum(dim=z_dim, skipna=True)

#     if output_units == "J m^-2":
#         scale_factor = 1.0
#         units = "J m$^{-2}$"
#     elif output_units in ("GJ m^-2", "10^9 J m^-2"):
#         scale_factor = 1e-9
#         units = "GJ m$^{-2}$"
#     else:
#         raise ValueError("output_units must be 'J m^-2', 'GJ m^-2', or '10^9 J m^-2'.")

#     ohc_anom = ohc_anom * scale_factor
#     ohc_anom.name = "ohc_anom"

#     if keep_attrs:
#         ohc_anom.attrs["long_name"] = "Column-integrated ocean heat content anomaly"
#         ohc_anom.attrs["units"] = units
#         ohc_anom.attrs["rho0"] = rho0
#         ohc_anom.attrs["cp0"] = cp0
#         if z_top is not None or z_bot is not None:
#             ohc_anom.attrs["depth_range_m"] = (
#                 f"{0 if z_top is None else z_top} to "
#                 f"{'bottom' if z_bot is None else z_bot}"
#             )

#     return ohc_anom


def compute_column_ohc_anom(
    temp_anom,
    temp_var="temp",
    z_dim="z_l",
    z_interface_dim="z_i",
    rho0=1035.0,
    cp0=3991.86795711963,
    z_top=None,
    z_bot=None,
    output_units="GJ m^-2",
    keep_attrs=True,
    regrid_output=False,
    lat_res=3 * 210,
    lon_res=3 * 360,
    reuse_regrid_weights=False,
    regridder_filename=None,
):
    """
    Compute column-integrated ocean heat content anomaly from a 3D temperature anomaly.

    Optionally regrid the final column-integrated field to a regular lat/lon grid.
    """
    if isinstance(temp_anom, xr.Dataset):
        if temp_var not in temp_anom:
            raise ValueError(f"{temp_var} not found in Dataset.")
        da = temp_anom[temp_var]
    else:
        da = temp_anom

    if z_dim not in da.dims:
        raise ValueError(f"{z_dim} must be a dimension of the temperature anomaly field.")

    dz = _infer_dz_from_vertical_coord(da, z_dim=z_dim, z_interface_dim=z_interface_dim)

    if z_top is not None or z_bot is not None:
        dz = _truncate_dz_to_depth_range(dz, da.coords[z_dim], z_top=z_top, z_bot=z_bot)

    weighted = da * dz

    valid_count = da.notnull().sum(dim=z_dim)
    ohc_anom = rho0 * cp0 * weighted.sum(dim=z_dim, skipna=True)
    ohc_anom = ohc_anom.where(valid_count > 0)

    if output_units == "J m^-2":
        scale_factor = 1.0
        units = "J m$^{-2}$"
    elif output_units in ("GJ m^-2", "10^9 J m^-2"):
        scale_factor = 1e-9
        units = "GJ m$^{-2}$"
    else:
        raise ValueError("output_units must be 'J m^-2', 'GJ m^-2', or '10^9 J m^-2'.")

    ohc_anom = ohc_anom * scale_factor
    ohc_anom.name = "ohc_anom"

    if keep_attrs:
        ohc_anom.attrs["long_name"] = "Column-integrated ocean heat content anomaly"
        ohc_anom.attrs["units"] = units
        ohc_anom.attrs["rho0"] = rho0
        ohc_anom.attrs["cp0"] = cp0
        if z_top is not None or z_bot is not None:
            ohc_anom.attrs["depth_range_m"] = (
                f"{0 if z_top is None else z_top} to "
                f"{'bottom' if z_bot is None else z_bot}"
            )

    if not regrid_output:
        return ohc_anom

    if "geolon" not in ohc_anom.coords or "geolat" not in ohc_anom.coords:
        raise ValueError("Cannot regrid output: OHC field must have geolon and geolat coordinates.")

    ohc_anom = ohc_anom.assign_coords(geolon=((ohc_anom.geolon + 360) % 360))

    target_lat = np.linspace(
        float(ohc_anom.geolat.min()),
        float(ohc_anom.geolat.max()),
        lat_res
    )
    target_lon = np.linspace(0, 360, lon_res, endpoint=False)

    ds_in = xr.Dataset({
        "lat": (["yh", "xh"], ohc_anom.geolat.values),
        "lon": (["yh", "xh"], ohc_anom.geolon.values),
    })
    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    regridder = xe.Regridder(
        ds_in,
        ds_out,
        method="bilinear",
        periodic=True,
        reuse_weights=reuse_regrid_weights,
        filename=regridder_filename,
    )

    ohc_anom_regridded = regridder(ohc_anom)
    ohc_anom_regridded.name = "ohc_anom"
    ohc_anom_regridded.attrs.update(ohc_anom.attrs)

    return ohc_anom_regridded


# ## Some older functions that may or may not be useful

# averages data by month (i.e. averages all January data)
def get_monthly_avg(ds):
    ds_avg = ds.groupby(ds['time'].dt.month).mean(dim='time')
    ds_avg = ds_avg.assign_coords({'geolon': ds['geolon'].isel(time=0), 'geolat': ds['geolat'].isel(time=0)})
    return ds_avg


# temporally averages all data
def get_time_avg(dataset):
    ds_avg = dataset.mean(dim='time')
    ds_avg = ds_avg.assign_coords({'geolon': dataset['geolon'].isel(time=0), 'geolat': dataset['geolat'].isel(time=0)})
    return ds_avg


def diff_dat_raw(ds1_raw,ds2_raw,var):
    da1_raw = ds1_raw[var]
    da2_raw = ds2_raw[var]
    diff_da = da2_raw - da1_raw
    # diff_da_avg = diff_da.mean(dim='time')
    diff_da = diff_da.assign_coords({'geolon': da1_raw['geolon'], 'geolat': da1_raw['geolat']})
    return diff_da


def diff_dat_time_avg(ds1_raw,ds2_raw,var):
    da1_raw = ds1_raw[var]
    da2_raw = ds2_raw[var]
    diff_da = da2_raw - da1_raw
    diff_da_avg = diff_da.mean(dim='time')
    diff_da_avg = diff_da_avg.assign_coords({'geolon': da1_raw['geolon'].isel(time=0), 'geolat': da1_raw['geolat'].isel(time=0)})
    return diff_da_avg


def diff_dat_monthly_avg(ds1_raw,ds2_raw,var):
    da1_raw = ds1_raw[var]
    da2_raw = ds2_raw[var]
    diff_da = da2_raw - da1_raw
    diff_da_monthly_avg = diff_da.groupby(diff_da['time'].dt.month).mean(dim='time')
    diff_da_monthly_avg = diff_da_monthly_avg.assign_coords({'geolon': da1_raw['geolon'].mean(dim='time'), 'geolat': da1_raw['geolat'].mean(dim='time')})
    return diff_da_monthly_avg


# ## Basin and cross-section functions

def get_pp_basin_dat(run_dat,var,basin_name,check_nn=False,nn_threshold=0.05,\
                     full_field_var=None,mask_ds=None,verbose=False):
    
    if mask_ds is None:
        masking_dat = run_dat
        if verbose:
            print("mask_ds is none ")
    else:
        masking_dat = mask_ds

    if var == 'u':
        lon_metric = 'geolon_u'
        lat_metric = 'geolat_u'
        mask_metric = 'wet_u'
    elif var == 'v':
        lon_metric = 'geolon_v'
        lat_metric = 'geolat_v'
        mask_metric = 'wet_v'
    else:
        lon_metric = 'geolon'
        lat_metric = 'geolat'
        mask_metric = 'wet'
        
    maskbasin = selecting_basins(masking_dat, basin=basin_name, lon=lon_metric, lat=lat_metric, mask=mask_metric, verbose=False)
    ds_basin = run_dat.where(maskbasin)

    if verbose:
        print("Before masking: ", run_dat[var].dims, run_dat[var].shape)

    dat_slice = ds_basin[var]
    
    if verbose:
        print("After masking, dat_slice: ", dat_slice.dims, dat_slice.shape)
        print(f"Raw basin DATA min and max: {np.nanmin(dat_slice.values)}, \t {np.nanmax(dat_slice.values)}")

    if mask_ds is None:
        if var == 'u':
            dat_basin_avg = zonal_mean(dat_slice, ds_basin, grid_type='u_pt')
            correct_lat = zonal_mean(run_dat['geolat_u'], run_dat, grid_type='u_pt')
        elif var == 'v':
            dat_basin_avg = zonal_mean(dat_slice, ds_basin, grid_type='v_pt')
            correct_lat = zonal_mean(run_dat['geolat_v'], run_dat, grid_type='v_pt')
        else:
            dat_basin_avg = zonal_mean(dat_slice, ds_basin)
            correct_lat = zonal_mean(run_dat['geolat'], run_dat)
    else:
        mask_dat_slice = mask_ds[['dxt','wet']].where(maskbasin)
        if var == 'u':
            dat_basin_avg = zonal_mean(dat_slice, mask_dat_slice, grid_type='u_pt')
            correct_lat = zonal_mean(mask_ds['geolat_u'], mask_ds, grid_type='u_pt')
        elif var == 'v':
            dat_basin_avg = zonal_mean(dat_slice, mask_dat_slice, grid_type='v_pt')
            correct_lat = zonal_mean(mask_ds['geolat_v'], mask_ds, grid_type='v_pt')
        else:
            dat_basin_avg = zonal_mean(dat_slice, mask_dat_slice)
            correct_lat = zonal_mean(mask_ds['geolat'], mask_ds)

    # print("dat_basin_avg.dims\n:", dat_basin_avg.dims)
    # print("dat_basin_avg\n:", dat_basin_avg)
    # print("correct_lat:\n", correct_lat)

    if verbose:
        print(f"Basin MEAN min and max: {np.nanmin(dat_basin_avg.values)}, \t {np.nanmax(dat_basin_avg.values)}")

    if var == 'v':
        dat_basin_avg = dat_basin_avg.rename({'yq': 'true_lat'})
        dat_basin_avg = dat_basin_avg.assign_coords({'true_lat': correct_lat.values})
    else:
        dat_basin_avg = dat_basin_avg.rename({'yh': 'true_lat'})
        dat_basin_avg = dat_basin_avg.assign_coords({'true_lat': correct_lat.values})
    # dat_basin_avg = dat_basin_avg.sortby('true_lat')

    # I think this check_nn method is likely checking based on the global mean not null count -- the masking step is setting values equal to null
    # everywhere outside of the basin, not changing the size of the dataset
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
        # must remove last row of true_lat values, because otherwise get the error "ValueError: The input coordinate is not sorted in
        # increasing order along axis 0. This can lead to unexpected results. Consider calling the `sortby` method on the input DataArray.
        
    # else:
        # dat_basin_avg = dat_basin_avg.isel(true_lat=slice(0,-1))

    if verbose:
        print(f"FINAL MEAN min and max: {np.nanmin(dat_basin_avg.values)}, \t {np.nanmax(dat_basin_avg.values)}")
    
    return dat_basin_avg


def get_pp_basin_zonalint_dat(
    run_dat, var, basin_name,
    full_field_var=None,
    mask_ds=None,
    verbose=False
):
    """
    Like get_pp_basin_dat(), but returns the *zonal integral* over longitude
    (∫ var dx) as a function of (true_lat, z_l).

    Returned units:
      - temp: degC * m
      - salt: PSU * m
    (unless you later rescale)
    """

    if mask_ds is None:
        masking_dat = run_dat
        if verbose:
            print("mask_ds is None; using run_dat for basin mask + metrics")
    else:
        masking_dat = mask_ds

    # Choose metrics consistent with your existing logic
    if var == "u":
        lon_metric = "geolon_u"
        lat_metric = "geolat_u"
        mask_metric = "wet_u"
        grid_type = "u_pt"
    elif var == "v":
        lon_metric = "geolon_v"
        lat_metric = "geolat_v"
        mask_metric = "wet_v"
        grid_type = "v_pt"
    else:
        lon_metric = "geolon"
        lat_metric = "geolat"
        mask_metric = "wet"
        grid_type = "tracer_pt"

    maskbasin = selecting_basins(
        masking_dat, basin=basin_name, lon=lon_metric, lat=lat_metric, mask=mask_metric, verbose=False
    )
    ds_basin = run_dat.where(maskbasin)
    dat_slice = ds_basin[var]

    if verbose:
        print("After masking:", dat_slice.dims, dat_slice.shape)
        print("Raw basin min/max:", float(np.nanmin(dat_slice.values)), float(np.nanmax(dat_slice.values)))

    # Metrics to use for dx and for defining "true_lat"
    if mask_ds is None:
        # Use run_dat's metrics
        metrics_for_int = ds_basin  # contains dxt/wet etc after masking via where()
        # "correct_lat" = zonal-mean latitude (same as you do now)
        if var == "u":
            correct_lat = zonal_mean(run_dat["geolat_u"], run_dat, grid_type="u_pt")
        elif var == "v":
            correct_lat = zonal_mean(run_dat["geolat_v"], run_dat, grid_type="v_pt")
        else:
            correct_lat = zonal_mean(run_dat["geolat"], run_dat, grid_type="tracer_pt")
    else:
        # Use mask_ds's metrics (your pattern)
        if var == "u":
            correct_lat = zonal_mean(mask_ds["geolat_u"], mask_ds, grid_type="u_pt")
        elif var == "v":
            correct_lat = zonal_mean(mask_ds["geolat_v"], mask_ds, grid_type="v_pt")
        else:
            correct_lat = zonal_mean(mask_ds["geolat"], mask_ds, grid_type="tracer_pt")

        # Build a metrics dataset that at least contains the needed dx metric, masked to basin
        # (keep it minimal; add others if you later extend to area/volume integrals)
        needed = {"u_pt": ["dxCu"], "v_pt": ["dxCv"], "tracer_pt": ["dxt"]}
        keep = needed[grid_type]
        metrics_for_int = mask_ds[keep].where(maskbasin)

    # Zonal integral over longitude
    dat_basin_int = zonal_integral(dat_slice, metrics_for_int, grid_type=grid_type)

    # Rename to true_lat like your mean version
    if var == "v":
        dat_basin_int = dat_basin_int.rename({"yq": "true_lat"})
        dat_basin_int = dat_basin_int.assign_coords({"true_lat": correct_lat.values})
    else:
        dat_basin_int = dat_basin_int.rename({"yh": "true_lat"})
        dat_basin_int = dat_basin_int.assign_coords({"true_lat": correct_lat.values})

    if verbose:
        print("FINAL ZONAL INT min/max:",
              float(np.nanmin(dat_basin_int.values)), float(np.nanmax(dat_basin_int.values)))

    return dat_basin_int


# function to try output zonal means in z-space, remapping from rho-space

def get_pp_zrho_basin_dat(run_dat,static_ds,cent_out,var_list,basin_name,check_nn=False,nn_threshold=0.05,\
                     full_field_var=None,mask_ds=None,verbose=False):
    # mask_ds contains 'dxt' and 'wet', needed if run_dat does not contain
    
    # if mask_ds is None:
    maskbasin = selecting_basins(run_dat, basin=basin_name, verbose=True)
    #     if verbose:
    #         print("mask_ds is none ")
    # else:
    #     maskbasin = selecting_basins(mask_ds, basin=basin_name, verbose=True)
        
    ds_basin = run_dat.where(maskbasin)
    dat_slice = ds_basin[var_list]

    maskbasin_static = selecting_basins(static_ds, basin=basin_name, verbose=True)
    static_basin = static_ds.where(maskbasin)

    depth_field = calc_zrho_dat(static_basin, dat_slice, cent_out=cent_out, x_mean=False)
    # print("depth_field: \n",depth_field)

    target_depth = np.linspace(float(depth_field.min()), float(depth_field.max()), 100)
    def interp_profile(temp_profile, depth_profile, target_depth):
        # depth_profile and temp_profile are 1D arrays along 'zl' for one (yh, xh) point.
        # np.interp requires the x-values (depth_profile) to be increasing. 
        # Ensure they are monotonic before using.
        return np.interp(target_depth, depth_profile, temp_profile)

    temp_on_depth = xr.apply_ufunc(
        interp_profile,
        dat_slice.temp,         # DataArray with dims ['zl', 'yh', 'xh']
        depth_field,  # DataArray with dims ['zl', 'yh', 'xh']
        target_depth, # 1D numpy array
        input_core_dims=[['zl'], ['zl'], []],
        output_core_dims=[['depth']],
        vectorize=True,
    )
    print(temp_on_depth)
    
    dat_slice = dat_slice.assign_coords(depth = depth_field)
    # dat_slice = dat_slice.swap_dims({'zl': 'depth'})
    print("dat_slice: \n", dat_slice)

    # if mask_ds is None:
    dat_basin_avg = xr.Dataset()
    for var in var_list:
        var_basin_avg = zonal_mean(dat_slice[var], dat_slice)
        dat_basin_avg[var] = var_basin_avg
    correct_lat = zonal_mean(run_dat['geolat'], run_dat)
    mean_depth = zonal_mean(dat_slice['depth'], dat_slice)
    print("mean_depth: \n", mean_depth)
        
    # else:
    #     mask_dat_slice = mask_ds[['dxt','wet']].where(maskbasin)
        
    #     dat_basin_avg = xr.Dataset()
    #     for var in var_list:
    #         var_basin_avg = zonal_mean(dat_slice[var], mask_dat_slice)
    #         dat_basin_avg[var] = var_basin_avg
    #     correct_lat = zonal_mean(mask_ds['geolat'], mask_ds)

    # if verbose:
    #     print(f"Basin MEAN min and max: {np.nanmin(dat_basin_avg.values)}, \t {np.nanmax(dat_basin_avg.values)}")
        
    dat_basin_avg = dat_basin_avg.rename({'yh': 'true_lat'})
    dat_basin_avg = dat_basin_avg.assign_coords({'true_lat': correct_lat.values})
    dat_basin_avg = dat_basin_avg.sortby('true_lat')

    mean_depth = mean_depth.rename({'yh': 'true_lat'})
    mean_depth = mean_depth.assign_coords({'true_lat': correct_lat.values})
    mean_depth = mean_depth.sortby('true_lat')

    # if check_nn:
    #     if full_field_var == None:
    #         not_null = dat_slice.notnull()
    #         if verbose:
    #             print(f"dat_slice.sizes['xh'] = {dat_slice.sizes['xh']}")
    #         nn_min = int(dat_slice.sizes['xh']*nn_threshold)
    #     else:
    #         ff_dat_slice = ds_basin[full_field_var]
    #         not_null = ff_dat_slice.notnull()
    #         nn_min = int(ff_dat_slice.sizes['xh']*nn_threshold)
            
    #     not_null_int = not_null.astype('int')
    #     not_null_count = not_null_int.sum(dim=['xh'])
    #     not_null_count = not_null_count.rename({'yh': 'true_lat'})
    #     not_null_count['true_lat'] = correct_lat.values
    #     not_null_count = not_null_count.sortby('true_lat')
    
    #     # nn_min = int(dat_slice.sizes['xh']*nn_threshold)
        
    #     dat_basin_avg = dat_basin_avg.where(not_null_count > nn_min).isel(true_lat=slice(0,-1))
    #     # must remove last row of true_lat values, because otherwise get the error "ValueError: The input coordinate is not sorted in
    #     # increasing order along axis 0. This can lead to unexpected results. Consider calling the `sortby` method on the input DataArray.
        
    # else:
    # print(f"Min and max of true_lat = -1 position: {np.nanmin(dat_basin_avg.isel(true_lat=-1))}\t{np.nanmax(dat_basin_avg.isel(true_lat=-1))}")
    # print(f"Min and max of true_lat = 0 position: {np.nanmin(dat_basin_avg.isel(true_lat=0))}\t{np.nanmax(dat_basin_avg.isel(true_lat=0))}")
    dat_basin_avg = dat_basin_avg.isel(true_lat=slice(0,-1))
    mean_depth = mean_depth.isel(true_lat=slice(0,-1))

    # if verbose:
    #     # print("After zonal_mean: ", dat_basin_avg.dims, dat_basin_avg.shape)
    #     print(f"FINAL MEAN min and max: {np.nanmin(dat_basin_avg.values)}, \t {np.nanmax(dat_basin_avg.values)}")
    
    return dat_basin_avg, depth_field


def get_basin_horiz_avg(ds,var,basin_name):
    ds_region = get_pp_basin_dat(ds,basin_name,var,check_nn=False)
    da_avg = horizontal_mean(ds_region[var], ds_region)
    da_avg = da_avg.groupby(da_avg['time']).mean(dim='time')
    return da_avg

