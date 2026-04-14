#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for reading in data and getting ensemble means for experiments. This is designed to be used with the notebook read_and_calculate.ipynb.

import numpy as np
import xarray as xr

# modules for using datetime variables
import datetime
from datetime import time
from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')

import cftime
from pandas.errors import OutOfBoundsDatetime  # Import the specific error

import copy

from xclim import ensembles


myVars = globals()


# # Functions

def create_exp_name_lists(power_str, prof, ramp_exp, lat_bound, num_ens_mem):
    
    const_exp_name_list = [None] * num_ens_mem
    doub_exp_name_list = [None] * num_ens_mem
    quad_exp_name_list = [None] * num_ens_mem

    const_exp_root = '_1860IC_200yr_'
    quad_exp_root = '_4xCO2_51-201_'
    
    if ramp_exp is True or lat_bound != None:
        doub_exp_root = "_2xCO2_200yr_"
        if (ramp_exp is True and lat_bound is None):
            const_suff = prof+const_exp_root+'ramp70yr_'+power_str
            doub_suff = prof+doub_exp_root+'ramp70yr_'+power_str
            quad_suff = prof+quad_exp_root+'ramp70yr_'+power_str
        elif lat_bound != None:
            const_suff = prof+const_exp_root+f'{lat_bound}deg_ramp70yr_'+power_str
            doub_suff = prof+doub_exp_root+f'{lat_bound}deg_ramp70yr_'+power_str
            quad_suff = prof+quad_exp_root+f'{lat_bound}deg_ramp70yr_'+power_str
        # elif (ramp_exp is False and lat_bound != None):
        #     const_suff = prof+const_exp_root+f'{lat_bound}deg_'+power_str
        #     doub_suff = prof+doub_exp_root+f'{lat_bound}deg_'+power_str
        #     quad_suff = prof+quad_exp_root+f'{lat_bound}deg_'+power_str

        if ramp_exp == False:
            const_prefix = "mem"
            doub_prefix = "ens"
        elif ramp_exp == True:
            const_prefix = "ens"
            doub_prefix = "mem"
            
        const_exp_name_list[0] = const_prefix+"1_"+const_suff
        const_exp_name_list[1] = const_prefix+"2_"+const_suff
        const_exp_name_list[2] = const_prefix+"3_"+const_suff

        doub_exp_name_list[0] = "mem1_"+doub_suff
        doub_exp_name_list[1] = doub_prefix+"2_"+doub_suff
        doub_exp_name_list[2] = doub_prefix+"3_"+doub_suff

        quad_exp_name_list[0] = "mem1_"+quad_suff
        quad_exp_name_list[1] = "mem2_"+quad_suff
        quad_exp_name_list[2] = "mem3_"+quad_suff

        if num_ens_mem == 4:
            const_exp_name_list[3] = const_prefix+"4_"+const_suff
            
            doub_exp_name_list[3] = doub_prefix+"4_"+doub_suff
            
        elif num_ens_mem == 5:
            const_exp_name_list[3] = const_prefix+"4_"+const_suff
            const_exp_name_list[4] = const_prefix+"5_"+const_suff

            doub_exp_name_list[3] = doub_prefix+"4_"+doub_suff
            doub_exp_name_list[4] = doub_prefix+"5_"+doub_suff

        elif num_ens_mem == 6:
            const_exp_name_list[3] = const_prefix+"4_"+const_suff
            const_exp_name_list[4] = const_prefix+"5_"+const_suff
            const_exp_name_list[5] = const_prefix+"6_"+const_suff

            doub_exp_name_list[3] = doub_prefix+"4_"+doub_suff
            doub_exp_name_list[4] = doub_prefix+"5_"+doub_suff
            doub_exp_name_list[5] = doub_prefix+"6_"+doub_suff
            
        
    else:
        mem1_doub_exp_root = '_2xCO2_1860IC_200yr_'
        mem2_3_doub_exp_root = '_2xCO2_200yr_'
        
        const_exp_name_list[0] = "mem1_"+prof+const_exp_root+power_str
        const_exp_name_list[1] = "ens2_"+prof+const_exp_root+power_str
        const_exp_name_list[2] = "mem3_"+prof+const_exp_root+power_str
        
        doub_exp_name_list[0] = prof+mem1_doub_exp_root+power_str
        doub_exp_name_list[1] = "ens2_"+prof+mem2_3_doub_exp_root+power_str
        doub_exp_name_list[2] = "ens3_"+prof+mem2_3_doub_exp_root+power_str

        quad_exp_name_list[0] = prof+quad_exp_root+power_str
        quad_exp_name_list[1] = "ens2_"+prof+quad_exp_root+power_str
        quad_exp_name_list[2] = "ens3_"+prof+quad_exp_root+power_str

    return const_exp_name_list, doub_exp_name_list, quad_exp_name_list


# ## Functions to return ensemble and ensemble mean data

# # function as of Jan. 11, 2026
# # added flag to omit the first ensemble member from the calculation (to use if the file/variable doesn't exist)

# def create_const_doub_ens_mean(exp_name_list,start_years,end_years,chunk_length,
#                                variable_list,pp_type='av-annual',diag_file='ocean_monthly_z',omit_mem1=False,debug=False,extra_debug=False):

#     num_ens_mem = len(exp_name_list)
#     ens_mem_list = [None] * num_ens_mem
    
#     for idx, exp_name in enumerate(exp_name_list):
#         if (omit_mem1 and idx == 0): # if not including first ens member to calculate mean
#             if debug:
#                 print(f'Omitting member #1: {exp_name}')
#             continue
#         else:
#             if idx == 0:
#                 time_decoding=True
#             else:
#                 time_decoding=False

#         ens_mem_list[idx] = get_pp_av_data(exp_name,start_years[idx],end_years[idx],chunk_length,pp_type=pp_type,\
#                                                   diag_file=diag_file,time_decoding=time_decoding,var=variable_list,debug=extra_debug)
#         if "net_dn_toa" in variable_list:
#             ens_mem_list[idx]["net_dn_toa"] = ens_mem_list[idx]["swdn_toa"] - ens_mem_list[idx]["swup_toa"] - ens_mem_list[idx]["olr"]
#         if "net_dn_toa_clr" in variable_list:
#             ens_mem_list[idx]["net_dn_toa_clr"] = ens_mem_list[idx]["swdn_toa_clr"] - ens_mem_list[idx]["swup_toa_clr"] - ens_mem_list[idx]["olr_clr"]
        
#         if debug:
#             print(f"Read ens_mem_list[{idx}] from {exp_name}")
      
#     if omit_mem1:
#         # select non-None list elements
#         ens_mem_list = ens_mem_list[1:]
#         # adjust timestamps to range of year 1 to 200
#         for idx, time_val in enumerate(ens_mem_list[0].time.values):
#             new_year = time_val.year - 200
#             new_date = time_val.replace(year=new_year)
#             for elem in ens_mem_list:
#                 elem.time.values[idx] = new_date
#     else:
#         for idx, time_val in enumerate(ens_mem_list[0].time.values):
#             for elem in ens_mem_list:
#                 elem.time.values[idx] = time_val
        
#     # ensemble mean
#     ens_mean = ensembles.create_ensemble(ens_mem_list).mean("realization")

#     ext_var_list = copy.deepcopy(variable_list)
#     if diag_file == 'ocean_monthly_z' or diag_file == 'ocean_month_rho2' or diag_file == 'ocean_monthly':
#         ext_var_list.extend(["areacello","dxt","dyt","wet","deptho",
#                             'dxCu','dxCv','dyCu','dyCv','wet_u','wet_v','areacello_cu','areacello_cv']) #,"volcello"
#     elif diag_file == 'ice':
#         ext_var_list.extend(["CELL_AREA", "COSROT", "SINROT"])
#     elif diag_file == 'atmos_monthly':
#         ext_var_list.extend(['area'])
        
#     ens_mean = ens_mean[ext_var_list]
        
#     return ens_mem_list, ens_mean


def create_const_doub_ens_mean(
    exp_name_list, start_years, end_years, chunk_length, variable_list,
    pp_type='av-annual', diag_file='ocean_monthly_z',
    omit_mem1=False, debug=False, extra_debug=False
):
    num_ens_mem = len(exp_name_list)

    # Read member datasets (still lazy if get_pp_av_data uses chunks)
    ens_mem_list = []
    member_ids = []

    for idx, exp_name in enumerate(exp_name_list):
        if omit_mem1 and idx == 0:
            if debug:
                print(f'Omitting member #1: {exp_name}')
            continue

        # Decode times once, then assign that time coord to others (no .values mutation)
        time_decoding = (idx == 0) #or (omit_mem1 and idx == 1)  # ensure first *kept* member decodes time

        ds = get_pp_av_data(
            exp_name,
            start_years[idx],
            end_years[idx],
            chunk_length,
            pp_type=pp_type,
            diag_file=diag_file,
            time_decoding=time_decoding,
            var=variable_list,
            debug=extra_debug,
        )

        # Derived variables
        if "net_dn_toa" in variable_list:
            ds["net_dn_toa"] = ds["swdn_toa"] - ds["swup_toa"] - ds["olr"]
        if "net_dn_toa_clr" in variable_list:
            ds["net_dn_toa_clr"] = ds["swdn_toa_clr"] - ds["swup_toa_clr"] - ds["olr_clr"]

        ens_mem_list.append(ds)
        member_ids.append(idx + 1)  # keep original member numbering (1..N)

        if debug:
            print(f"Read member {member_ids[-1]} from {exp_name}")

    # --- Time coordinate alignment WITHOUT touching .values ---
    ref_time = ens_mem_list[0].time

    if omit_mem1:
        # shift years by -200 (your original logic)
        # this computes only the time coord (tiny), not the data
        new_time = xr.DataArray(
            [t.replace(year=t.year - 200) for t in ref_time.values],
            dims=("time",),
            name="time"
        )
        ref_time = new_time

    # assign identical time coord to all members (cheap; doesn’t read data)
    ens_mem_list = [ds.assign_coords(time=ref_time) for ds in ens_mem_list]

    # --- Ensemble mean: concat + mean (lazy) ---
    ds_ens = xr.concat(
        ens_mem_list,
        dim=xr.DataArray(member_ids, dims="realization", name="realization"),
        coords="minimal",
        join="exact",      # fail fast if coords don’t match
        compat="override", # you already ensure same time/coords
    )

    ens_mean = ds_ens.mean("realization")

    # Add your extended variable list
    ext_var_list = copy.deepcopy(variable_list)
    if diag_file in ['ocean_monthly_z', 'ocean_month_rho2', 'ocean_monthly']:
        ext_var_list.extend([
            "areacello","dxt","dyt","wet","deptho",
            'dxCu','dxCv','dyCu','dyCv','wet_u','wet_v','areacello_cu','areacello_cv'
        ])
    elif diag_file == 'ice':
        ext_var_list.extend(["CELL_AREA", "COSROT", "SINROT"])
    elif diag_file == 'atmos_monthly':
        ext_var_list.extend(['area'])

    ens_mean = ens_mean[ext_var_list]

    return ens_mem_list, ens_mean


def create_quad_ens_mean(exp_name_list,doub_ens_mem_list,doub_ens_mean,start_years,end_years,chunk_length,
                              variable_list,pp_type='av-annual',diag_file='ocean_monthly_z',omit_mems=[],debug=False):
    
    num_ens_mem = len(exp_name_list)
    quad_ens_mem_list = [None] * num_ens_mem
    
    doub_cutoff_yr = 51
    doub_cutoff_dt = cftime.DatetimeNoLeap(doub_cutoff_yr, 1, 1, 0, 0, 0, 0, has_year_zero=True)
    post_51_start_years = [51,251,451]
    
    if start_years[0] < doub_cutoff_yr and end_years[0] < doub_cutoff_yr:
        quad_ens_mem_list = doub_ens_mem_list
        quad_ens_mean = doub_ens_mean
        if debug:
            print(f"Used quad_ens_mean = doub_ens_mean")
        
    elif start_years[0] < doub_cutoff_yr and end_years[0] > doub_cutoff_yr:
        for idx, ens_memb in enumerate(doub_ens_mem_list):
            if (1 in omit_mems and idx == 0): # if not including first ens member to calculate mean
                if debug:
                    print(f'Omitting member #1: {exp_name_list[idx]}')
                continue
            elif (2 in omit_mems and idx == 1):
                if debug:
                    print(f'Omitting member #2: {exp_name_list[idx]}')
                continue
            elif (3 in omit_mems and idx == 2):
                if debug:
                    print(f'Omitting member #3: {exp_name_list[idx]}')
                continue
            else:
                if idx == 0:
                    time_decoding=True
                else:
                    time_decoding=False

            # all the 2xCO2 ensemble members have already had their times adjusted to within 1-200 range
            pre_51_quad = ens_memb.sel(time=slice(None,doub_cutoff_dt))
            if idx == 1:
                for pre_51_idx, pre_51_time in enumerate(pre_51_quad.time.values):
                    new_year = pre_51_time.year + 200
                    new_date = pre_51_time.replace(year=new_year)
                    pre_51_quad.time.values[pre_51_idx] = new_date
            elif idx == 2:
                for pre_51_idx, pre_51_time in enumerate(pre_51_quad.time.values):
                    new_year = pre_51_time.year + 400
                    new_date = pre_51_time.replace(year=new_year)
                    pre_51_quad.time.values[pre_51_idx] = new_date
            
            if len(pre_51_quad) == 0:
                raise ValueError("len(pre_51_quad) = 0. It seems that 2xCO2 members have not had their times \
                adjusted to be within 1-200 years.")
        
            post_51_quad = get_pp_av_data(exp_name_list[idx],post_51_start_years[idx],end_years[idx],chunk_length,pp_type=pp_type,\
                                          diag_file=diag_file,time_decoding=time_decoding,var=variable_list,debug=False)
            post_51_quad = post_51_quad[variable_list]
            quad_ens_mem_list[idx] = xr.concat([pre_51_quad,post_51_quad],"time")

            if "net_dn_toa" in variable_list:
                quad_ens_mem_list[idx]["net_dn_toa"] = quad_ens_mem_list[idx]["swdn_toa"] - quad_ens_mem_list[idx]["swup_toa"] - quad_ens_mem_list[idx]["olr"]
            if "net_dn_toa_clr" in variable_list:
                quad_ens_mem_list[idx]["net_dn_toa_clr"] = quad_ens_mem_list[idx]["swdn_toa_clr"] - quad_ens_mem_list[idx]["swup_toa_clr"] - quad_ens_mem_list[idx]["olr_clr"]
        
            if debug:
                print(f"Read quad_ens_mem_list[{idx}] from {exp_name_list[idx]}")
    
    else: # a.k.a. if (start_years[0] > doub_cutoff_yr)
        for idx, ens_memb in enumerate(doub_ens_mem_list):
            if (1 in omit_mems and idx == 0): # if not including first ens member to calculate mean
                if debug:
                    print(f'Omitting member #1: {exp_name_list[idx]}')
                continue
            elif (2 in omit_mems and idx == 1):
                if debug:
                    print(f'Omitting member #2: {exp_name_list[idx]}')
                continue
            elif (3 in omit_mems and idx == 2):
                if debug:
                    print(f'Omitting member #3: {exp_name_list[idx]}')
                continue
            if idx == 0:
                time_decoding=True
            else:
                time_decoding=False
                
            quad_ens_mem_list[idx] = get_pp_av_data(exp_name_list[idx],start_years[idx],end_years[idx],chunk_length,pp_type=pp_type,\
                                                    diag_file=diag_file,time_decoding=time_decoding,var=variable_list,debug=False)

            if "net_dn_toa" in variable_list:
                quad_ens_mem_list[idx]["net_dn_toa"] = quad_ens_mem_list[idx]["swdn_toa"] - quad_ens_mem_list[idx]["swup_toa"] - quad_ens_mem_list[idx]["olr"]
            if "net_dn_toa_clr" in variable_list:
                quad_ens_mem_list[idx]["net_dn_toa_clr"] = quad_ens_mem_list[idx]["swdn_toa_clr"] - quad_ens_mem_list[idx]["swup_toa_clr"] - quad_ens_mem_list[idx]["olr_clr"]
                
            if debug:
                print(f"Read quad_ens_mem_list[{idx}] from {exp_name_list[idx]}")

    if omit_mems == [1]:
        # select non-None list elements
        quad_ens_mem_list = quad_ens_mem_list[1:]
        # adjust timestamps to range of year 1 to 200
        for idx, time_val in enumerate(quad_ens_mem_list[0].time.values):
            new_year = time_val.year - 200
            new_date = time_val.replace(year=new_year)
            quad_ens_mem_list[0].time.values[idx] = new_date
            quad_ens_mem_list[1].time.values[idx] = new_date
            if debug:
                if idx == 0:
                    print(f'time_val: {time_val}')
                    print(f'New year: {new_date}')
                    print(f'New date: {new_date}')
                    print(f'Actual xarray value: {quad_ens_mem_list[0].time.values[idx]}')
    elif omit_mems == [1,2]:
        # select non-None list elements
        quad_ens_mem_list = [quad_ens_mem_list[-1]]
        # adjust timestamps to range of year 1 to 200
        for idx, time_val in enumerate(quad_ens_mem_list[0].time.values):
            new_year = time_val.year - 400
            new_date = time_val.replace(year=new_year)
            quad_ens_mem_list[0].time.values[idx] = new_date
            if debug:
                if idx == 0:
                    print(f'time_val: {time_val}')
                    print(f'New year: {new_date}')
                    print(f'New date: {new_date}')
                    print(f'Actual xarray value: {quad_ens_mem_list[0].time.values[idx]}')
    elif omit_mems == [2]:
        # select non-None list elements
        quad_ens_mem_list = [quad_ens_mem_list[0],quad_ens_mem_list[2]]
        # adjust timestamps to range of year 1 to 200
        for idx, time_val in enumerate(quad_ens_mem_list[0].time.values):
            quad_ens_mem_list[1].time.values[idx] = time_val
    elif omit_mems == [3]:
        # select non-None list elements
        quad_ens_mem_list = quad_ens_mem_list[0,1]
        # adjust timestamps to range of year 1 to 200
        for idx, time_val in enumerate(quad_ens_mem_list[0].time.values):
            quad_ens_mem_list[1].time.values[idx] = time_val
    else:
        # adjust timestamps to range of year 1 to 200
        for idx, time_val in enumerate(quad_ens_mem_list[0].time.values):
            # quad_ens_mem_list[1].time.values[idx] = time_val
            # quad_ens_mem_list[2].time.values[idx] = time_val
            for elem in quad_ens_mem_list:
                elem.time.values[idx] = time_val
                
    # ensemble mean
    if omit_mems == [1,2]:
        quad_ens_mean = copy.deepcopy(quad_ens_mem_list[0])
    else:
        quad_ens_mean = ensembles.create_ensemble(quad_ens_mem_list).mean("realization")

    ext_var_list = copy.deepcopy(variable_list)
    if diag_file == 'ocean_monthly_z' or diag_file == 'ocean_month_rho2' or diag_file == 'ocean_monthly':
        ext_var_list.extend(["areacello","dxt","dyt","wet","deptho",
                            'dxCu','dxCv','dyCu','dyCv','wet_u','wet_v','areacello_cu','areacello_cv']) #,"volcello"
    elif diag_file == 'ice':
        ext_var_list.extend(["CELL_AREA", "COSROT", "SINROT"])
    elif diag_file == 'atmos_monthly':
        ext_var_list.extend(['area'])
        
    quad_ens_mean = quad_ens_mean[ext_var_list]
    
    return quad_ens_mem_list, quad_ens_mean


# ## Function to calculate ensemble-mean differences and horizontal mean differences

def calc_ens_diffs(diff_ens_name,ref_ens_list,perturb_ens_list,variable_list,diag_file='ocean_monthly_z',stdev_ds=None,verbose=False):

    num_ens_mem = len(ref_ens_list)
    
    if num_ens_mem != 1:
        
        diffs_mem_list = [None] * num_ens_mem
        
        # raise an error if a variable isn't found in one of the arrays
        for i in range(num_ens_mem):
            # check that each variable is in the i-th member of both the reference and perturbed simulations
            for elem in variable_list:
                if elem not in ref_ens_list[i].variables:
                    raise IOError(f"{elem} not found in ref_ens_list[{i}].")
                if elem not in perturb_ens_list[i].variables:
                    raise IOError(f"{elem} not found in perturb_ens_list[{i}].")
        
            # take difference of variables
            diffs_mem_list[i] = perturb_ens_list[i][variable_list] - ref_ens_list[i][variable_list]
        
        diff_ens = ensembles.create_ensemble(diffs_mem_list)
        
        # create hatching variable
        if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
            
            if stdev_ds != None:

                for var in variable_list:
                    arr = diff_ens[var]

                    stdev_cond = (arr > 2*stdev_ds[f"{var}_stdev"]).all(dim="realization")
                
                    # All NaNs across ensemble (e.g. land)
                    all_nan = arr.isnull().all(dim="realization")
                
                    # Agreement: all >0 OR all <0 OR all == 0
                    all_pos  = (arr > 0).all(dim="realization")
                    all_neg  = (arr < 0).all(dim="realization")
                    all_zero = (arr == 0).all(dim="realization")
                
                    agree_cond = all_pos | all_neg | all_zero
                    significant_cond = (~all_nan) & agree_cond & stdev_cond

                    # set to true where significant
                    hatch_field = xr.where(significant_cond, 1.0, np.nan)
                
                    mask_name = f"{var}_hatch"
                    diff_ens[mask_name] = hatch_field
                    
                # for var in variable_list:
                #     stdev_cond = (diff_ens[var] > stdev_ds[f"{var}_stdev"]).all(dim="realization")
                #     nan_cond = np.isnan(diff_ens[var]).all(dim="realization")

                #     # Identify regions where members all have significant change or it's NaN (i.e., land or bathymetry)
                #     # hatching_maskbin = stdev_cond | nan_cond

                #     agree_cond = ( (diff_ens[var] > 0).all(dim="realization") | (diff_ens[var] < 0).all(dim="realization") | (diff_ens[var] == 0).all(dim="realization") )
                #     disagree_cond = (agree_cond == False) & (nan_cond == False)

                #     # regions where change is significant among all ensemble members but not all members agree on sign
                #     # it seems that this isn't the case anywhere (check by taking max of <var>_hatch)
                #     hatching_maskbin = disagree_cond & stdev_cond
            
                #     # Set hatching mask: True where change not significant
                #     # hatching_mask = xr.where(hatching_maskbin, False, True)
                #     # switched for the disagree_cond
                #     hatching_mask = xr.where(hatching_maskbin, True, False)
                    
                #     # Store in dataset
                #     mask_name = f"{var}_hatch"
                #     diff_ens[mask_name] = hatching_mask
            
            elif num_ens_mem < 4:
                # agreement if all members have same sign
                for var in variable_list:
                    arr = diff_ens[var]
                
                    # All NaNs across ensemble (e.g. land)
                    all_nan = arr.isnull().all(dim="realization")
                
                    # Agreement: all >0 OR all <0 OR all == 0
                    all_pos  = (arr > 0).all(dim="realization")
                    all_neg  = (arr < 0).all(dim="realization")
                    all_zero = (arr == 0).all(dim="realization")
                
                    agree_cond = all_pos | all_neg | all_zero
                
                    # Disagreement where we have at least one non-NaN
                    disagree_cond = (~agree_cond) & (~all_nan)
                
                    # IMPORTANT: use NaN where we *don’t* want hatching,
                    # so contourf simply ignores those points.
                    hatch_field = xr.where(disagree_cond, 1.0, np.nan)
                
                    mask_name = f"{var}_hatch"
                    diff_ens[mask_name] = hatch_field

            else:
                # agreement if all or all but one members have same sign
                for var in variable_list:
                    arr = diff_ens[var]  # dims: realization, yh, xh (or similar)
                
                    # valid (non-NaN) members
                    valid = arr.notnull()
                    n_valid = valid.sum(dim="realization")  # number of non-NaN members
                
                    # Counts of positive and negative members (ignoring NaNs)
                    pos_count = (arr > 0).where(valid).sum(dim="realization")
                    neg_count = (arr < 0).where(valid).sum(dim="realization")
                
                    # Max of pos/neg counts at each grid cell (no xr.ufuncs)
                    max_sign_count = xr.where(pos_count >= neg_count, pos_count, neg_count)
                
                    # --- Agreement criteria ---
                
                    # 1) Agreement on sign if all or all-but-one share the same sign
                    #    i.e. max_sign_count >= n_valid - 1  (and at least one with that sign)
                    agree_sign = (max_sign_count >= (n_valid - 1)) & (max_sign_count >= 1)
                
                    # 2) All-zero agreement (optional)
                    all_zero = ((arr == 0).where(valid)).all(dim="realization") & (n_valid > 0)
                
                    # Final "agreement" condition
                    agree_cond = agree_sign | all_zero
                
                    # Some data present at this grid point
                    some_valid = n_valid > 0
                
                    # Disagreement where we have at least one non-NaN
                    disagree_cond = (~agree_cond) & some_valid
                
                    # NaN where we don't want hatching
                    hatch_field = xr.where(disagree_cond, 1.0, np.nan)
                
                    mask_name = f"{var}_hatch"
                    diff_ens[mask_name] = hatch_field
        
            diff_ens = diff_ens.mean("realization")

            if diag_file == 'ocean_monthly_z' or diag_file == 'ocean_month_rho2' or diag_file == 'ocean_monthly':
                metrics_to_keep = ["areacello","dxt","dyt","wet","deptho",
                                   'dxCu','dxCv','dyCu','dyCv','wet_u','wet_v','areacello_cu','areacello_cv']
                for metric in metrics_to_keep:
                    diff_ens[metric] = ref_ens_list[0][metric]
            elif diag_file == 'ice':
                metrics_to_keep = ["CELL_AREA", "COSROT", "SINROT"]
                for metric in metrics_to_keep:
                    diff_ens[metric] = ref_ens_list[0][metric]
            elif diag_file == 'atmos_monthly':
                metrics_to_keep = ["area"]
                for metric in metrics_to_keep:
                    diff_ens[metric] = ref_ens_list[0][metric]
            
        else:
            diff_ens = diff_ens.mean("realization")
        
        myVars.__setitem__(diff_ens_name, diff_ens)
        if verbose:
            print(f'{diff_ens_name}')
        
        # alternative to get more stats is:
        # diff_ens = ensembles.ensemble_mean_std_max_min(diff_ens)
        # for var in variable_list:
        #     mean_var_name = var + '_mean'
        #     diff_ens = diff_ens.rename({mean_var_name: var})
        
        if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
            horiz_avg_diff_name = f"{diff_ens_name}_mean"
            horiz_avg_diff = xr.Dataset()
            for var in variable_list:
                if diag_file == 'atmos_monthly':
                    horiz_avg_diff[var] = atmos_horiz_mean(diff_ens[var],diff_ens)
                elif diag_file == 'ice':
                    ## ZONAL MEAN ##
                    horiz_avg_diff[var] = ice_zonal_mean(diff_ens[var],diff_ens)
                else:
                    horiz_avg_diff[var] = horizontal_mean(diff_ens[var],diff_ens)
            myVars.__setitem__(horiz_avg_diff_name, horiz_avg_diff)
            if verbose:
                print(f'{horiz_avg_diff_name} done')
                
    # if there is only one ens member
    else:
        # take difference of variables
        diffs_ens = perturb_ens_list[0][variable_list] - ref_ens_list[0][variable_list]
        myVars.__setitem__(diff_ens_name, diff_ens)
        if verbose:
            print(f'{diff_ens_name}')


# ## Main functions

def get_ens_diffs(co2_scen, avg_period, mem1_start, mem1_end, var_list,
                pp_type='av-annual',
                diag_file='ocean_monthly_z',
                profiles = ['surf','therm','mid','bot'],
                power_inputs = ['0.1TW', '0.2TW', '0.3TW'],
                power_var_suff = ['0p1TW', '0p2TW', '0p3TW'],
                num_ens_mem = 3,
                omit_mem1=False,
                ramp_exp=False,
                lat_bound=None,
                stdev_ds=None,
                verbose=False,
                debug=False):

    """
    Returns variables containing the ensemble-mean raw data and variables containing the ensemble-mean anomaly data. Anomalies are
    calculated as the difference relative to the control run during the period corresponding to an ensemble member (i.e. the anomalies
    for ensemble member 2 for year 201 to 400 are taking as the difference relative to year 201 to 400 of the control run).

        Args:
            co2_scen (str): one of ['const','doub','quad','const+doub','all']; difference datasets will only be created for the co2 scenario specified, 
                            but ensembles + means may be created for control case of other CO2 scenarios
            avg_period (int): number of years for av/ts period
            mem1_start (int): start year of ens. mem. #1
            mem1_end (int): end year of ens. mem. #1
            var_list (str list): list of variables to read (e.g. var_list = ["temp", "N2", "age", "rhopot2", "salt"])
            profiles (str list): list of profiles to get data for
            power_inputs (str list): list of power inputs to get data for
            power_var_suff (str list): list of variable suffixes for each power input
            num_ens_mem (int): number of ensemble members
            ramp_exp (boolean): set to True to get data for experiments with mixing ramped up over 70 years
            lat_bound (int): for latitude-bounded experiments, specify the absolute value latitude as an int
            stdev_ds (dataset): required to compute hatching variable in calc_ens_diffs();
                                typically the standard deviation of the control run, with variables named <var>_stdev
            verbose: if True, print variable names after declaration
            
        Returns:
            has no return variables, but creates xarray datasets by using myVars = globals()
            
    """
    allowed_scen = ['const','doub','quad','const+doub','all']
    
    if co2_scen not in allowed_scen:
        raise ValueError(f"'co2_scen' must be one of {allowed_scen}.")

    all_vars = copy.deepcopy(var_list)
    if "net_dn_toa" in var_list:
        if "swdn_toa" not in var_list:
            all_vars.extend(["swdn_toa"])
        if "swup_toa" not in var_list:
            all_vars.extend(["swup_toa"])
        if "olr" not in var_list:
            all_vars.extend(["olr"])
    if "net_dn_toa_clr" in var_list:
        if "swdn_toa_clr" not in var_list:
            all_vars.extend(["swdn_toa_clr"])
        if "swup_toa_clr" not in var_list:
            all_vars.extend(["swup_toa_clr"])
        if "olr_clr" not in var_list:
            all_vars.extend(["olr_clr"])

    if verbose:
        print(all_vars)

    start_yrs = [None] * num_ens_mem
    end_yrs = [None] * num_ens_mem
    for idx in range(num_ens_mem):
        start_yrs[idx] = mem1_start + 200*idx
        end_yrs[idx] = mem1_end + 200*idx
        
    # start_yrs = [mem1_start,
    #              mem1_start+200,
    #              mem1_start+400]
    # end_yrs = [mem1_end,
    #            mem1_end+200,
    #            mem1_end+400]
    
    ##### CONTROL RUNS #####
    
    ## const CO2 control ##
    if num_ens_mem == 3:
        const_ctrl_exps = ["tune_ctrl_const_200yr",#"tune_ctrl_1860IC_200yr",
                           "ctrl_1860IC_201-2001", #tune_ctrl_1860IC_201-2001
                           "ctrl_1860IC_201-2001"]
    elif num_ens_mem == 4:
        const_ctrl_exps = ["tune_ctrl_const_200yr","ctrl_1860IC_201-2001",
                           "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001"]
    elif num_ens_mem == 5:
        const_ctrl_exps = ["tune_ctrl_const_200yr","ctrl_1860IC_201-2001",
                           "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001",
                           "ctrl_1860IC_201-2001"]
    elif num_ens_mem == 6:
        const_ctrl_exps = ["tune_ctrl_const_200yr","ctrl_1860IC_201-2001",
                           "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001",
                           "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001"]

    const_ctrl_mem_list, const_ctrl = create_const_doub_ens_mean(const_ctrl_exps,start_yrs,end_yrs,avg_period,all_vars,
                                                                 pp_type=pp_type,diag_file=diag_file,omit_mem1=omit_mem1,debug=debug)
    const_ctrl_name = f"const_ctrl_{mem1_start}_{mem1_end}"
    const_ctrl_mem_list_name = f"{const_ctrl_name}_mem_list"
    myVars.__setitem__(const_ctrl_name, const_ctrl)
    myVars.__setitem__(const_ctrl_mem_list_name, const_ctrl_mem_list)
    if verbose:
        print(f'{const_ctrl_name}, {const_ctrl_mem_list_name} done')

    if co2_scen != 'const':
        ## 2xCO2 control ##
        if num_ens_mem == 3:
            doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr"]
        elif num_ens_mem == 4:
            doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr"]
        elif num_ens_mem == 5:
            doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr",
                              "ens5_tune_ctrl_2xCO2_200yr"]
        elif num_ens_mem == 6:
            doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr",
                              "ens5_tune_ctrl_2xCO2_200yr","ens6_tune_ctrl_2xCO2_200yr"]
            
        
        doub_ctrl_mem_list, doub_ctrl = create_const_doub_ens_mean(doub_ctrl_exps,start_yrs,end_yrs,avg_period,all_vars,
                                                                   pp_type=pp_type,diag_file=diag_file,omit_mem1=omit_mem1,debug=debug)
        doub_ctrl_name = f"doub_ctrl_{mem1_start}_{mem1_end}"
        doub_ctrl_mem_list_name = f"{doub_ctrl_name}_mem_list"
        myVars.__setitem__(doub_ctrl_name, doub_ctrl)
        myVars.__setitem__(doub_ctrl_mem_list_name, doub_ctrl_mem_list)
        if verbose:
            print(f'{doub_ctrl_name}, {doub_ctrl_mem_list_name} done')

        if (co2_scen == 'doub' or co2_scen == 'const+doub' or co2_scen == 'all'):
            # differences compared to constant CO2 control #
            calc_ens_diffs(f"doub_ctrl_{mem1_start}_{mem1_end}_diff",
                           const_ctrl_mem_list,doub_ctrl_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)

        if (co2_scen == 'quad' or co2_scen == 'all'):
            ## 4xCO2 control ##
            quad_ctrl_exps = ["tune_ctrl_4xCO2_51-201",
                              "ens2_ctrl_4xCO2_51-201",
                              "ens3_ctrl_4xCO2_51-201"]

            # if 'thetaoga' in var_list:
            #     quad_ctrl_omit_mems = [2]
            # else:
            #     quad_ctrl_omit_mems = []
            quad_ctrl_omit_mems = []
        
            quad_ctrl_mem_list, quad_ctrl = create_quad_ens_mean(quad_ctrl_exps,doub_ctrl_mem_list,doub_ctrl,start_yrs,end_yrs,
                                                                 avg_period,all_vars,pp_type=pp_type,diag_file=diag_file,
                                                                 omit_mems=quad_ctrl_omit_mems,debug=debug)
            quad_ctrl_name = f"quad_ctrl_{mem1_start}_{mem1_end}"
            quad_ctrl_mem_list_name = f"{quad_ctrl_name}_mem_list"
            myVars.__setitem__(quad_ctrl_name, quad_ctrl)
            myVars.__setitem__(quad_ctrl_mem_list_name, quad_ctrl_mem_list)
            if verbose:
                print(f'{quad_ctrl_name}, {quad_ctrl_mem_list_name} done')
        
            # differences compared to constant CO2 and 2xCO2 controls #
            if quad_ctrl_omit_mems == [2]:
                calc_ens_diffs(f"quad_ctrl_{mem1_start}_{mem1_end}_diff_const_ctrl",
                               [const_ctrl_mem_list[0],const_ctrl_mem_list[2]],quad_ctrl_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                calc_ens_diffs(f"quad_ctrl_{mem1_start}_{mem1_end}_diff_2xctrl",
                               [doub_ctrl_mem_list[0],doub_ctrl_mem_list[2]],quad_ctrl_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
            else:
                calc_ens_diffs(f"quad_ctrl_{mem1_start}_{mem1_end}_diff_const_ctrl",
                               const_ctrl_mem_list,quad_ctrl_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                calc_ens_diffs(f"quad_ctrl_{mem1_start}_{mem1_end}_diff_2xctrl",
                               doub_ctrl_mem_list,quad_ctrl_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)

    
    ##### PERTURBATION RUNS #####
    
    for prof in profiles:
        for index, power_str in enumerate(power_inputs):
            if verbose:
                print(f"{prof} {power_str} experiments")

            const_exp_name_list, doub_exp_name_list, quad_exp_name_list = create_exp_name_lists(power_str, prof, ramp_exp, lat_bound, num_ens_mem)
            
            const_ens_mem_list, const_ens_mean = create_const_doub_ens_mean(const_exp_name_list,start_yrs,end_yrs,
                                                                            avg_period,all_vars,
                                                                            pp_type=pp_type,diag_file=diag_file,
                                                                            omit_mem1=omit_mem1,debug=debug)
            
            if co2_scen != 'const':
                doub_ens_mem_list, doub_ens_mean = create_const_doub_ens_mean(doub_exp_name_list,start_yrs,end_yrs,
                                                                              avg_period,all_vars,
                                                                              pp_type=pp_type,diag_file=diag_file,
                                                                              omit_mem1=omit_mem1,debug=debug)
                
                if (co2_scen == 'quad' or co2_scen == 'all'):
                    quad_omit_mems = []
                    quad_ens_mem_list, quad_ens_mean = create_quad_ens_mean(quad_exp_name_list,doub_ens_mem_list,
                                                                            doub_ens_mean,start_yrs,end_yrs,
                                                                            avg_period,all_vars,
                                                                            pp_type=pp_type,diag_file=diag_file,
                                                                            omit_mems=quad_omit_mems,debug=debug)

            ## COMPUTE DIFFERENCES ##
            diff_root = f"{prof}_{power_var_suff[index]}_{mem1_start}_{mem1_end}_diff"

            # differences wrt 1860 control
            const_diff_name = f"const_{diff_root}"
            doub_const_ctrl_diff_name = f"doub_{diff_root}_const_ctrl"
            quad_const_ctrl_diff_name = f"quad_{diff_root}_const_ctrl"

            # differences wrt 1860 experiment with same diffusivity history
            doub_1860_diff_name = f"doub_{diff_root}_1860"
            quad_1860_diff_name = f"quad_{diff_root}_1860"

            # differences wrt control for particular CO2 scenario
            doub_2xctrl_diff_name = f"doub_{diff_root}_2xctrl"
            quad_4xctrl_diff_name = f"quad_{diff_root}_4xctrl"

            ## CONST EXPERIMENTS
            if (co2_scen == 'const' or co2_scen == 'const+doub' or co2_scen == 'all'):
                calc_ens_diffs(const_diff_name,const_ctrl_mem_list,
                               const_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)

            ## 2xCO2 EXPERIMENTS
            if (co2_scen == 'doub' or co2_scen == 'const+doub' or co2_scen == 'all'):
                # differences wrt 1860 control
                calc_ens_diffs(doub_const_ctrl_diff_name,const_ctrl_mem_list,
                               doub_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                    
                # differences wrt 1860 experiment with same Kd history
                calc_ens_diffs(doub_1860_diff_name,const_ens_mem_list,
                               doub_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                
                # differences wrt control for particular CO2 scenario
                calc_ens_diffs(doub_2xctrl_diff_name,doub_ctrl_mem_list,
                               doub_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                
            ## 4xCO2 EXPERIMENTS
            if (co2_scen == 'quad' or co2_scen == 'all'):
                # differences wrt 1860 control
                if quad_omit_mems == [2]:
                    calc_ens_diffs(quad_const_ctrl_diff_name,[const_ctrl_mem_list[0],const_ctrl_mem_list[2]],
                               quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                elif quad_omit_mems == [1,2]:
                    calc_ens_diffs(quad_const_ctrl_diff_name,[const_ctrl_mem_list[2]],
                               quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                else:
                    calc_ens_diffs(quad_const_ctrl_diff_name,const_ctrl_mem_list,
                                   quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                    
                # differences wrt 1860 experiment with same Kd history
                if quad_omit_mems == [2]:
                    calc_ens_diffs(quad_1860_diff_name,[const_ens_mem_list[0],const_ens_mem_list[2]],
                                   quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                elif quad_omit_mems == [1,2]:
                    calc_ens_diffs(quad_1860_diff_name,[const_ens_mem_list[2]],
                                   quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                else:
                    calc_ens_diffs(quad_1860_diff_name,const_ens_mem_list,
                                   quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                # differences wrt control for particular CO2 scenario
                calc_ens_diffs(quad_4xctrl_diff_name,quad_ctrl_mem_list,
                               quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)

                # additional difference calcs for 4xCO2 cases #
                # difference wrt 2xCO2 ctrl
                quad_2xctrl_diff_name = f"quad_{diff_root}_2xctrl"
                calc_ens_diffs(quad_2xctrl_diff_name,doub_ctrl_mem_list,
                               quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                # difference wrt 2xCO2 experiment with same diffusivity history
                quad_2xCO2_diff_name = f"quad_{diff_root}_2xCO2"
                calc_ens_diffs(quad_2xCO2_diff_name,doub_ens_mem_list,
                               quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)


# function to get ensemble mean for every case, but not calculate any differences

def get_ens_means(co2_scen, avg_period, mem1_start, mem1_end, var_list,
                  pp_type='av-annual',
                  diag_file='ocean_monthly_z',
                  profiles = ['surf','therm','mid','bot'],
                  power_inputs = ['0.1TW', '0.2TW', '0.3TW'],
                  power_var_suff = ['0p1TW', '0p2TW', '0p3TW'],
                  num_ens_mem = 3,
                  omit_mem1=False,
                  ramp_exp=False,
                  lat_bound=None,
                  skip_ctrl=False,
                  ctrl_only=False,
                  verbose=False,
                  debug=False,
                  extra_debug=False):

    """
    Returns variables containing the ensemble-mean data only (no anomalies).

        Args:
            co2_scen (str): one of ['const','doub','quad','const+doub','all']
            avg_period (int): number of years for av/ts period
            mem1_start (int): start year of ens. mem. #1
            mem1_end (int): end year of ens. mem. #1
            var_list (str list): list of variables to read (e.g. var_list = ["Kd_int_tuned", "Kd_int_base", "Kd_interface"])
            profiles (str list): list of profiles to get data for
            power_inputs (str list): list of power inputs to get data for
            power_var_suff (str list): list of variable suffixes for each power input
            num_ens_mem (int): number of ensemble members
            ramp_exp (boolean): set to True to get data for experiments with mixing ramped up over 70 years
            lat_bound (int): for latitude-bounded experiments, specify the absolute value latitude as an int
            skip_ctrl (bool): if True, don't read control data
            ctrl_only (bool): if True, only read control data
            verbose (bool): if True, print variable names after declaration
            
        Returns:
            has no return variables, but creates xarray datasets by using myVars = globals()
            
    """
    allowed_scen = ['const','doub','quad','const+doub','all']

    if co2_scen not in allowed_scen:
        raise ValueError(f"'co2_scen' must be one of {allowed_scen}.")

    all_vars = copy.deepcopy(var_list)
    # if "net_dn_toa" in var_list:
    #     all_vars.extend(["swdn_toa","swup_toa","olr"])
    # if "net_dn_toa_clr" in var_list:
    #     all_vars.extend(["swdn_toa_clr","swup_toa_clr","olr_clr"])
    if "net_dn_toa" in var_list:
        if "swdn_toa" not in var_list:
            all_vars.extend(["swdn_toa"])
        if "swup_toa" not in var_list:
            all_vars.extend(["swup_toa"])
        if "olr" not in var_list:
            all_vars.extend(["olr"])
    if "net_dn_toa_clr" in var_list:
        if "swdn_toa_clr" not in var_list:
            all_vars.extend(["swdn_toa_clr"])
        if "swup_toa_clr" not in var_list:
            all_vars.extend(["swup_toa_clr"])
        if "olr_clr" not in var_list:
            all_vars.extend(["olr_clr"])

    ctrl_var_list = copy.deepcopy(all_vars)
    ctrl_vars_to_drop = ["Kd_int_tuned", "Kd_int_base","Kd_lay_tuned", "Kd_lay_base"]
    
    for elem in ctrl_vars_to_drop:
        if elem in ctrl_var_list:
            ctrl_var_list.remove(elem)

    if verbose:
        print(all_vars)

    start_yrs = [None] * num_ens_mem
    end_yrs = [None] * num_ens_mem
    for idx in range(num_ens_mem):
        start_yrs[idx] = mem1_start + 200*idx
        end_yrs[idx] = mem1_end + 200*idx
        
    # start_yrs = [mem1_start,
    #              mem1_start+200,
    #              mem1_start+400]
    # end_yrs = [mem1_end,
    #            mem1_end+200,
    #            mem1_end+400]

    # ##### CONTROL RUNS #####

    if skip_ctrl==False:
        ## const CO2 control ##
        if (co2_scen == 'const' or co2_scen == 'const+doub' or co2_scen == 'all'):

            ## const CO2 control ##
            if num_ens_mem == 3:
                const_ctrl_exps = ["tune_ctrl_const_200yr",#"tune_ctrl_1860IC_200yr",
                                   "ctrl_1860IC_201-2001", #tune_ctrl_1860IC_201-2001
                                   "ctrl_1860IC_201-2001"]
            elif num_ens_mem == 4:
                const_ctrl_exps = ["tune_ctrl_const_200yr","ctrl_1860IC_201-2001",
                                   "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001"]
            elif num_ens_mem == 5:
                const_ctrl_exps = ["tune_ctrl_const_200yr","ctrl_1860IC_201-2001",
                                   "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001",
                                   "ctrl_1860IC_201-2001"]
            elif num_ens_mem == 6:
                const_ctrl_exps = ["tune_ctrl_const_200yr","ctrl_1860IC_201-2001",
                                   "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001",
                                   "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001"]
        
            const_ctrl_mem_list, const_ctrl = create_const_doub_ens_mean(const_ctrl_exps,start_yrs,end_yrs,avg_period,ctrl_var_list,
                                                                         pp_type=pp_type,diag_file=diag_file,omit_mem1=omit_mem1,
                                                                         debug=debug,extra_debug=extra_debug)
            const_ctrl_name = f"const_ctrl_{mem1_start}_{mem1_end}"
            const_ctrl_mem_list_name = f"{const_ctrl_name}_mem_list"
            myVars.__setitem__(const_ctrl_name, const_ctrl)
            myVars.__setitem__(const_ctrl_mem_list_name, const_ctrl_mem_list)
            if verbose:
                print(f'{const_ctrl_name}, {const_ctrl_mem_list_name} done')
            
            if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
                horiz_avg = xr.Dataset()
                for var in ctrl_var_list:
                    if diag_file == 'atmos_monthly':
                        horiz_avg[var] = atmos_horiz_mean(const_ctrl[var],const_ctrl)
                    elif diag_file == 'ice':
                        ## ZONAL MEAN ##
                        horiz_avg[var] = ice_zonal_mean(const_ctrl[var],const_ctrl)
                    else:
                        horiz_avg[var] = horizontal_mean(const_ctrl[var],const_ctrl)
            
                horiz_avg_name = f"{const_ctrl_name}_mean"
                myVars.__setitem__(horiz_avg_name, horiz_avg)
                if verbose:
                    print(f'{horiz_avg_name} done')

        
        if co2_scen != 'const':
            ## 2xCO2 control ##
            if num_ens_mem == 3:
                doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                                  "ens3_ctrl_2xCO2_200yr"]
            elif num_ens_mem == 4:
                doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                                  "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr"]
            elif num_ens_mem == 5:
                doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                                  "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr",
                                  "ens5_tune_ctrl_2xCO2_200yr"]
            elif num_ens_mem == 6:
                doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                                  "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr",
                                  "ens5_tune_ctrl_2xCO2_200yr","ens6_tune_ctrl_2xCO2_200yr"]
            
            doub_ctrl_mem_list, doub_ctrl = create_const_doub_ens_mean(doub_ctrl_exps,start_yrs,end_yrs,avg_period,ctrl_var_list,
                                                                       pp_type=pp_type,diag_file=diag_file,omit_mem1=omit_mem1,
                                                                       debug=debug,extra_debug=extra_debug)

            if (co2_scen == 'const+doub' or co2_scen == 'doub' or co2_scen == 'all'):
                doub_ctrl_name = f"doub_ctrl_{mem1_start}_{mem1_end}"
                doub_ctrl_mem_list_name = f"{doub_ctrl_name}_mem_list"
                myVars.__setitem__(doub_ctrl_name, doub_ctrl)
                myVars.__setitem__(doub_ctrl_mem_list_name, doub_ctrl_mem_list)
                if verbose:
                    print(f'{doub_ctrl_name}, {doub_ctrl_mem_list_name} done')

                if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
                    horiz_avg = xr.Dataset()
                    for var in ctrl_var_list:
                        if diag_file == 'atmos_monthly':
                            horiz_avg[var] = atmos_horiz_mean(doub_ctrl[var],doub_ctrl)
                        elif diag_file == 'ice':
                            ## ZONAL MEAN ##
                            horiz_avg[var] = ice_zonal_mean(doub_ctrl[var],doub_ctrl)
                        else:
                            horiz_avg[var] = horizontal_mean(doub_ctrl[var],doub_ctrl)
                        
                    horiz_avg_name = f"{doub_ctrl_name}_mean"
                    myVars.__setitem__(horiz_avg_name, horiz_avg)
                    if verbose:
                        print(f'{horiz_avg_name} done')

                        
        if (co2_scen == 'quad' or co2_scen == 'all'):
            ## 4xCO2 control ##
            quad_ctrl_exps = ["tune_ctrl_4xCO2_51-201",
                              "ens2_ctrl_4xCO2_51-201",
                              "ens3_ctrl_4xCO2_51-201"]

            # if 'thetaoga' in ctrl_var_list:
            #     quad_ctrl_omit_mems = [2]
            # else:
            #     quad_ctrl_omit_mems = []
            quad_ctrl_omit_mems = []
        
            quad_ctrl_mem_list, quad_ctrl = create_quad_ens_mean(quad_ctrl_exps,doub_ctrl_mem_list,doub_ctrl,start_yrs,end_yrs,
                                                                 avg_period,ctrl_var_list,pp_type=pp_type,diag_file=diag_file,
                                                                 omit_mems=quad_ctrl_omit_mems,debug=debug)
            quad_ctrl_name = f"quad_ctrl_{mem1_start}_{mem1_end}"
            quad_ctrl_mem_list_name = f"{quad_ctrl_name}_mem_list"
            myVars.__setitem__(quad_ctrl_name, quad_ctrl)
            myVars.__setitem__(quad_ctrl_mem_list_name, quad_ctrl_mem_list)
            if verbose:
                print(f'{quad_ctrl_name}, {quad_ctrl_mem_list_name} done')

            if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
                horiz_avg = xr.Dataset()
                for var in ctrl_var_list:
                    if diag_file == 'atmos_monthly':
                        horiz_avg[var] = atmos_horiz_mean(quad_ctrl[var],quad_ctrl)
                    elif diag_file == 'ice':
                        ## ZONAL MEAN ##
                        horiz_avg[var] = ice_zonal_mean(quad_ctrl[var],quad_ctrl)
                    else:
                        horiz_avg[var] = horizontal_mean(quad_ctrl[var],quad_ctrl)
                    
                horiz_avg_name = f"{quad_ctrl_name}_mean"
                myVars.__setitem__(horiz_avg_name, horiz_avg)
                if verbose:
                    print(f'{horiz_avg_name} done')
                
    
    ##### PERTURBATION RUNS #####

    if ctrl_only==False:
        for prof in profiles:
            for index, power_str in enumerate(power_inputs):

                const_exp_name_list, doub_exp_name_list, quad_exp_name_list = create_exp_name_lists(power_str, prof, ramp_exp, lat_bound, num_ens_mem)
    
                ens_mean_root = f"{prof}_{power_var_suff[index]}_{mem1_start}_{mem1_end}"
    
                if (co2_scen == 'const' or co2_scen == 'const+doub' or co2_scen == 'all'):
                    const_ens_mem_list, const_ens_mean = create_const_doub_ens_mean(const_exp_name_list,start_yrs,end_yrs,
                                                                                    avg_period,all_vars,
                                                                                    pp_type=pp_type,diag_file=diag_file,
                                                                                    omit_mem1=omit_mem1,
                                                                                    debug=debug,extra_debug=extra_debug)
                    
                    const_ens_mean_name = f"const_{ens_mean_root}"
                    myVars.__setitem__(const_ens_mean_name, const_ens_mean)
                    print(f'{const_ens_mean_name} done')

                    if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
                        horiz_avg = xr.Dataset()
                        for var in var_list:
                            if diag_file == 'atmos_monthly':
                                horiz_avg[var] = atmos_horiz_mean(const_ens_mean[var],const_ens_mean)
                            elif diag_file == 'ice':
                                ## ZONAL MEAN ##
                                horiz_avg[var] = ice_zonal_mean(const_ens_mean[var],const_ens_mean)
                            else:
                                horiz_avg[var] = horizontal_mean(const_ens_mean[var],const_ens_mean)
                                
                        horiz_avg_name = f"{const_ens_mean_name}_mean"
                        myVars.__setitem__(horiz_avg_name, horiz_avg)
                        print(f'{horiz_avg_name} done')

                
                if co2_scen != 'const':
                    doub_ens_mem_list, doub_ens_mean = create_const_doub_ens_mean(doub_exp_name_list,start_yrs,end_yrs,
                                                                                  avg_period,all_vars,
                                                                                  pp_type=pp_type,diag_file=diag_file,
                                                                                  omit_mem1=omit_mem1,
                                                                                  debug=debug,extra_debug=extra_debug)
                    if (co2_scen != 'quad'):
                        doub_ens_mean_name = f"doub_{ens_mean_root}"
                        myVars.__setitem__(doub_ens_mean_name, doub_ens_mean)
                        print(f'{doub_ens_mean_name} done')

                        if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
                            horiz_avg = xr.Dataset()
                            for var in var_list:
                                if diag_file == 'atmos_monthly':
                                    horiz_avg[var] = atmos_horiz_mean(doub_ens_mean[var],doub_ens_mean)
                                elif diag_file == 'ice':
                                    ## ZONAL MEAN ##
                                    horiz_avg[var] = ice_zonal_mean(doub_ens_mean[var],doub_ens_mean)
                                else:
                                    horiz_avg[var] = horizontal_mean(doub_ens_mean[var],doub_ens_mean)
                                    
                            horiz_avg_name = f"{doub_ens_mean_name}_mean"
                            myVars.__setitem__(horiz_avg_name, horiz_avg)
                            print(f'{horiz_avg_name} done')
                    
                    if (co2_scen == 'quad' or co2_scen == 'all'):
                        quad_omit_mems = []
                        quad_ens_mem_list, quad_ens_mean = create_quad_ens_mean(quad_exp_name_list,doub_ens_mem_list,
                                                                                doub_ens_mean,start_yrs,end_yrs,
                                                                                avg_period,all_vars,
                                                                                pp_type=pp_type,diag_file=diag_file,
                                                                                omit_mems=quad_omit_mems,debug=debug)
                        quad_ens_mean_name = f"quad_{ens_mean_root}"
                        myVars.__setitem__(quad_ens_mean_name, quad_ens_mean)
                        print(f'{quad_ens_mean_name} done')

                        if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
                            horiz_avg = xr.Dataset()
                            for var in var_list:
                                horiz_avg[var] = horizontal_mean(quad_ens_mean[var],quad_ens_mean)
                                    
                            horiz_avg_name = f"{quad_ens_mean_name}_mean"
                            myVars.__setitem__(horiz_avg_name, horiz_avg)
                            
                            print(f'{horiz_avg_name} done')
                    


def get_ens_diff_and_means(co2_scen, avg_period, mem1_start, mem1_end, var_list,
                pp_type='av-annual',
                diag_file='ocean_monthly_z',
                profiles = ['surf','therm','mid','bot'],
                power_inputs = ['0.1TW', '0.2TW', '0.3TW'],
                power_var_suff = ['0p1TW', '0p2TW', '0p3TW'],
                num_ens_mem = 3,
                omit_mem1=False,
                ramp_exp=False,
                lat_bound=None,
                stdev_ds=None,
                verbose=False,
                debug=False):

    """
    Creates dataset variables containing the ensemble-mean raw data and anomaly data. Anomalies are calculated as the difference relative to the control run
    during the period corresponding to an ensemble member (i.e. the anomalies for ensemble member 2 for year 201 to 400 are taking as the difference relative 
    to year 201 to 400 of the control run).

        Args:
            co2_scen (str): one of ['const','doub','quad','const+doub','all']
            avg_period (int): number of years for av/ts period
            mem1_start (int): start year of ens. mem. #1
            mem1_end (int): end year of ens. mem. #1
            var_list (str list): list of variables to read (e.g. var_list = ["temp", "N2", "age", "rhopot2", "salt"])
            profiles (str list): list of profiles to get data for
            power_inputs (str list): list of power inputs to get data for
            power_var_suff (str list): list of variable suffixes for each power input
            num_ens_mem (int): number of ensemble members
            ramp_exp (boolean): set to True to get data for experiments with mixing ramped up over 70 years
            lat_bound (int): for latitude-bounded experiments, specify the absolute value latitude as an int
            stdev_ds (dataset): required to compute hatching variable in calc_ens_diffs();
                                typically the standard deviation of the control run, with variables named <var>_stdev
            verbose: if True, print variable names after declaration
            
        Returns:
            has no return variables, but creates xarray datasets by using myVars = globals()
            
    """
    allowed_scen = ['const','doub','quad','const+doub','all']
    
    if co2_scen not in allowed_scen:
        raise ValueError(f"'co2_scen' must be one of {allowed_scen}.")

    all_vars = copy.deepcopy(var_list)
    # if "net_dn_toa" in var_list:
    #     all_vars.extend(["swdn_toa","swup_toa","olr"])
    # if "net_dn_toa_clr" in var_list:
    #     all_vars.extend(["swdn_toa_clr","swup_toa_clr","olr_clr"])
    if "net_dn_toa" in var_list:
        if "swdn_toa" not in var_list:
            all_vars.extend(["swdn_toa"])
        if "swup_toa" not in var_list:
            all_vars.extend(["swup_toa"])
        if "olr" not in var_list:
            all_vars.extend(["olr"])
    if "net_dn_toa_clr" in var_list:
        if "swdn_toa_clr" not in var_list:
            all_vars.extend(["swdn_toa_clr"])
        if "swup_toa_clr" not in var_list:
            all_vars.extend(["swup_toa_clr"])
        if "olr_clr" not in var_list:
            all_vars.extend(["olr_clr"])

    ctrl_var_list = copy.deepcopy(all_vars)
    ctrl_vars_to_drop = ["Kd_int_tuned", "Kd_int_base","Kd_lay_tuned", "Kd_lay_base"]
    
    for elem in ctrl_vars_to_drop:
        if elem in ctrl_var_list:
            ctrl_var_list.remove(elem)

    if verbose:
        print(all_vars)

    start_yrs = [None] * num_ens_mem
    end_yrs = [None] * num_ens_mem
    for idx in range(num_ens_mem):
        start_yrs[idx] = mem1_start + 200*idx
        end_yrs[idx] = mem1_end + 200*idx
        
    # start_yrs = [mem1_start,
    #              mem1_start+200,
    #              mem1_start+400]
    # end_yrs = [mem1_end,
    #            mem1_end+200,
    #            mem1_end+400]
    
    ##### CONTROL RUNS #####
    
    ## const CO2 control ##
    if num_ens_mem == 3:
        const_ctrl_exps = ["tune_ctrl_const_200yr",#"tune_ctrl_1860IC_200yr",
                           "ctrl_1860IC_201-2001", #tune_ctrl_1860IC_201-2001
                           "ctrl_1860IC_201-2001"]
    elif num_ens_mem == 4:
        const_ctrl_exps = ["tune_ctrl_const_200yr","ctrl_1860IC_201-2001",
                           "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001"]
    elif num_ens_mem == 5:
        const_ctrl_exps = ["tune_ctrl_const_200yr","ctrl_1860IC_201-2001",
                           "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001",
                           "ctrl_1860IC_201-2001"]
    elif num_ens_mem == 6:
        const_ctrl_exps = ["tune_ctrl_const_200yr","ctrl_1860IC_201-2001",
                           "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001",
                           "ctrl_1860IC_201-2001","ctrl_1860IC_201-2001"]

    const_ctrl_mem_list, const_ctrl = create_const_doub_ens_mean(const_ctrl_exps,start_yrs,end_yrs,avg_period,all_vars,
                                                                 pp_type=pp_type,diag_file=diag_file,omit_mem1=omit_mem1,debug=debug)
    const_ctrl_name = f"const_ctrl_{mem1_start}_{mem1_end}"
    const_ctrl_mem_list_name = f"{const_ctrl_name}_mem_list"
    myVars.__setitem__(const_ctrl_name, const_ctrl)
    myVars.__setitem__(const_ctrl_mem_list_name, const_ctrl_mem_list)
    if verbose:
        print(f'{const_ctrl_name}, {const_ctrl_mem_list_name} done')

    if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
        horiz_avg = xr.Dataset()
        for var in ctrl_var_list:
            if diag_file == 'atmos_monthly':
                horiz_avg[var] = atmos_horiz_mean(const_ctrl[var],const_ctrl)
            elif diag_file == 'ice':
                ## ZONAL MEAN ##
                horiz_avg[var] = ice_zonal_mean(const_ctrl[var],const_ctrl)
            else:
                horiz_avg[var] = horizontal_mean(const_ctrl[var],const_ctrl)
    
        horiz_avg_name = f"{const_ctrl_name}_mean"
        myVars.__setitem__(horiz_avg_name, horiz_avg)
        if verbose:
            print(f'{horiz_avg_name} done')

    if co2_scen != 'const':
        ## 2xCO2 control ##
        if num_ens_mem == 3:
            doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr"]
        elif num_ens_mem == 4:
            doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr"]
        elif num_ens_mem == 5:
            doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr",
                              "ens5_tune_ctrl_2xCO2_200yr"]
        elif num_ens_mem == 6:
            doub_ctrl_exps = ["mem1_tune_ctrl_2xCO2_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr",
                              "ens5_tune_ctrl_2xCO2_200yr","ens6_tune_ctrl_2xCO2_200yr"]
        
        doub_ctrl_mem_list, doub_ctrl = create_const_doub_ens_mean(doub_ctrl_exps,start_yrs,end_yrs,avg_period,all_vars,
                                                                   pp_type=pp_type,diag_file=diag_file,omit_mem1=omit_mem1,debug=debug)
        doub_ctrl_name = f"doub_ctrl_{mem1_start}_{mem1_end}"
        doub_ctrl_mem_list_name = f"{doub_ctrl_name}_mem_list"
        myVars.__setitem__(doub_ctrl_name, doub_ctrl)
        myVars.__setitem__(doub_ctrl_mem_list_name, doub_ctrl_mem_list)
        if verbose:
            print(f'{doub_ctrl_name}, {doub_ctrl_mem_list_name} done')

        if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
            horiz_avg = xr.Dataset()
            for var in ctrl_var_list:
                if diag_file == 'atmos_monthly':
                    horiz_avg[var] = atmos_horiz_mean(doub_ctrl[var],doub_ctrl)
                elif diag_file == 'ice':
                    ## ZONAL MEAN ##
                    horiz_avg[var] = ice_zonal_mean(doub_ctrl[var],doub_ctrl)
                else:
                    horiz_avg[var] = horizontal_mean(doub_ctrl[var],doub_ctrl)
                
            horiz_avg_name = f"{doub_ctrl_name}_mean"
            myVars.__setitem__(horiz_avg_name, horiz_avg)
            if verbose:
                print(f'{horiz_avg_name} done')

    if (co2_scen == 'doub' or co2_scen == 'const+doub' or co2_scen == 'all'):
        # differences compared to constant CO2 control #
        calc_ens_diffs(f"doub_ctrl_{mem1_start}_{mem1_end}_diff",
                       const_ctrl_mem_list,doub_ctrl_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)

    if (co2_scen == 'quad' or co2_scen == 'all'):
        ## 4xCO2 control ##
        quad_ctrl_exps = ["tune_ctrl_4xCO2_51-201",
                          "ens2_ctrl_4xCO2_51-201",
                          "ens3_ctrl_4xCO2_51-201"]
        
        quad_ctrl_omit_mems = []
    
        quad_ctrl_mem_list, quad_ctrl = create_quad_ens_mean(quad_ctrl_exps,doub_ctrl_mem_list,doub_ctrl,start_yrs,end_yrs,
                                                             avg_period,all_vars,pp_type=pp_type,diag_file=diag_file,
                                                             omit_mems=quad_ctrl_omit_mems,debug=debug)
        quad_ctrl_name = f"quad_ctrl_{mem1_start}_{mem1_end}"
        quad_ctrl_mem_list_name = f"{quad_ctrl_name}_mem_list"
        myVars.__setitem__(quad_ctrl_name, quad_ctrl)
        myVars.__setitem__(quad_ctrl_mem_list_name, quad_ctrl_mem_list)
        if verbose:
            print(f'{quad_ctrl_name}, {quad_ctrl_mem_list_name} done')

        if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
            horiz_avg = xr.Dataset()
            for var in ctrl_var_list:
                if diag_file == 'atmos_monthly':
                    horiz_avg[var] = atmos_horiz_mean(quad_ctrl[var],quad_ctrl)
                elif diag_file == 'ice':
                    ## ZONAL MEAN ##
                    horiz_avg[var] = ice_zonal_mean(quad_ctrl[var],quad_ctrl)
                else:
                    horiz_avg[var] = horizontal_mean(quad_ctrl[var],quad_ctrl)
                
            horiz_avg_name = f"{quad_ctrl_name}_mean"
            myVars.__setitem__(horiz_avg_name, horiz_avg)
            if verbose:
                print(f'{horiz_avg_name} done')
    
        # differences compared to constant CO2 and 2xCO2 controls #
        if quad_ctrl_omit_mems == [2]:
            calc_ens_diffs(f"quad_ctrl_{mem1_start}_{mem1_end}_diff_const_ctrl",
                           [const_ctrl_mem_list[0],const_ctrl_mem_list[2]],quad_ctrl_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
            calc_ens_diffs(f"quad_ctrl_{mem1_start}_{mem1_end}_diff_2xctrl",
                           [doub_ctrl_mem_list[0],doub_ctrl_mem_list[2]],quad_ctrl_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
        else:
            calc_ens_diffs(f"quad_ctrl_{mem1_start}_{mem1_end}_diff_const_ctrl",
                           const_ctrl_mem_list,quad_ctrl_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
            calc_ens_diffs(f"quad_ctrl_{mem1_start}_{mem1_end}_diff_2xctrl",
                           doub_ctrl_mem_list,quad_ctrl_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)

    
    ##### PERTURBATION RUNS #####
    
    for prof in profiles:
        for index, power_str in enumerate(power_inputs):
            if verbose:
                print(f"{prof} {power_str} experiments")

            const_exp_name_list, doub_exp_name_list, quad_exp_name_list = create_exp_name_lists(power_str, prof, ramp_exp, lat_bound, num_ens_mem)
            
            ens_mean_root = f"{prof}_{power_var_suff[index]}_{mem1_start}_{mem1_end}"
                    
            const_ens_mem_list, const_ens_mean = create_const_doub_ens_mean(const_exp_name_list,start_yrs,end_yrs,
                                                                            avg_period,all_vars,
                                                                            pp_type=pp_type,diag_file=diag_file,
                                                                            omit_mem1=omit_mem1,debug=debug)
            const_ens_mean_name = f"const_{ens_mean_root}"
            myVars.__setitem__(const_ens_mean_name, const_ens_mean)
            print(f'{const_ens_mean_name} done')

            if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
                horiz_avg = xr.Dataset()
                for var in var_list:
                    if diag_file == 'atmos_monthly':
                        horiz_avg[var] = atmos_horiz_mean(const_ens_mean[var],const_ens_mean)
                    elif diag_file == 'ice':
                        ## ZONAL MEAN ##
                        horiz_avg[var] = ice_zonal_mean(const_ens_mean[var],const_ens_mean)
                    else:
                        horiz_avg[var] = horizontal_mean(const_ens_mean[var],const_ens_mean)
                horiz_avg_name = f"{const_ens_mean_name}_mean"
                myVars.__setitem__(horiz_avg_name, horiz_avg)
                print(f'{horiz_avg_name} done')
            
            
            if co2_scen != 'const':
                doub_ens_mem_list, doub_ens_mean = create_const_doub_ens_mean(doub_exp_name_list,start_yrs,end_yrs,
                                                                              avg_period,all_vars,
                                                                              pp_type=pp_type,diag_file=diag_file,
                                                                              omit_mem1=omit_mem1,debug=debug)
                doub_ens_mean_name = f"doub_{ens_mean_root}"
                myVars.__setitem__(doub_ens_mean_name, doub_ens_mean)
                print(f'{doub_ens_mean_name} done')

                if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
                    horiz_avg = xr.Dataset()
                    for var in var_list:
                        if diag_file == 'atmos_monthly':
                            horiz_avg[var] = atmos_horiz_mean(doub_ens_mean[var],doub_ens_mean)
                        elif diag_file == 'ice':
                            ## ZONAL MEAN ##
                            horiz_avg[var] = ice_zonal_mean(doub_ens_mean[var],doub_ens_mean)
                        else:
                            horiz_avg[var] = horizontal_mean(doub_ens_mean[var],doub_ens_mean)
                    horiz_avg_name = f"{doub_ens_mean_name}_mean"
                    myVars.__setitem__(horiz_avg_name, horiz_avg)
                    print(f'{horiz_avg_name} done')
                
            if (co2_scen == 'quad' or co2_scen == 'all'):
                quad_omit_mems = []
                    
                quad_ens_mem_list, quad_ens_mean = create_quad_ens_mean(quad_exp_name_list,doub_ens_mem_list,
                                                                        doub_ens_mean,start_yrs,end_yrs,
                                                                        avg_period,all_vars,
                                                                        pp_type=pp_type,diag_file=diag_file,
                                                                        omit_mems=quad_omit_mems,debug=debug)
                quad_ens_mean_name = f"quad_{ens_mean_root}"
                myVars.__setitem__(quad_ens_mean_name, quad_ens_mean)
                print(f'{quad_ens_mean_name} done')

                if diag_file != 'ocean_scalar_monthly' and diag_file != 'atmos_scalar_monthly':
                    horiz_avg = xr.Dataset()
                    for var in var_list:
                        if diag_file == 'atmos_monthly':
                            horiz_avg[var] = atmos_horiz_mean(quad_ens_mean[var],quad_ens_mean)
                        elif diag_file == 'ice':
                            ## ZONAL MEAN ##
                            horiz_avg[var] = ice_zonal_mean(quad_ens_mean[var],quad_ens_mean)
                        else:
                            horiz_avg[var] = horizontal_mean(quad_ens_mean[var],quad_ens_mean)
                    horiz_avg_name = f"{quad_ens_mean_name}_mean"
                    myVars.__setitem__(horiz_avg_name, horiz_avg)
                    print(f'{horiz_avg_name} done')

            ## COMPUTE DIFFERENCES ##
            diff_root = f"{prof}_{power_var_suff[index]}_{mem1_start}_{mem1_end}_diff"

            # differences wrt 1860 control
            const_diff_name = f"const_{diff_root}"
            doub_const_ctrl_diff_name = f"doub_{diff_root}_const_ctrl"
            quad_const_ctrl_diff_name = f"quad_{diff_root}_const_ctrl"

            # differences wrt 1860 experiment with same diffusivity history
            doub_1860_diff_name = f"doub_{diff_root}_1860"
            quad_1860_diff_name = f"quad_{diff_root}_1860"

            # differences wrt control for particular CO2 scenario
            doub_2xctrl_diff_name = f"doub_{diff_root}_2xctrl"
            quad_4xctrl_diff_name = f"quad_{diff_root}_4xctrl"

            ## CONST EXPERIMENTS
            if (co2_scen == 'const' or co2_scen == 'const+doub' or co2_scen == 'all'):
                calc_ens_diffs(const_diff_name,const_ctrl_mem_list,
                               const_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)

            ## 2xCO2 EXPERIMENTS
            if (co2_scen == 'doub' or co2_scen == 'const+doub' or co2_scen == 'all'):
                # differences wrt 1860 control
                calc_ens_diffs(doub_const_ctrl_diff_name,const_ctrl_mem_list,
                               doub_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                    
                # differences wrt 1860 experiment with same Kd history
                calc_ens_diffs(doub_1860_diff_name,const_ens_mem_list,
                               doub_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                
                # differences wrt control for particular CO2 scenario
                calc_ens_diffs(doub_2xctrl_diff_name,doub_ctrl_mem_list,
                               doub_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                
            ## 4xCO2 EXPERIMENTS
            if (co2_scen == 'quad' or co2_scen == 'all'):
                # differences wrt 1860 control
                if quad_omit_mems == [2]:
                    calc_ens_diffs(quad_const_ctrl_diff_name,[const_ctrl_mem_list[0],const_ctrl_mem_list[2]],
                               quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                elif quad_omit_mems == [1,2]:
                    calc_ens_diffs(quad_const_ctrl_diff_name,[const_ctrl_mem_list[2]],
                               quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                else:
                    calc_ens_diffs(quad_const_ctrl_diff_name,const_ctrl_mem_list,
                                   quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                    
                # differences wrt 1860 experiment with same Kd history
                if quad_omit_mems == [2]:
                    calc_ens_diffs(quad_1860_diff_name,[const_ens_mem_list[0],const_ens_mem_list[2]],
                                   quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                elif quad_omit_mems == [1,2]:
                    calc_ens_diffs(quad_1860_diff_name,[const_ens_mem_list[2]],
                                   quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                else:
                    calc_ens_diffs(quad_1860_diff_name,const_ens_mem_list,
                                   quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                # differences wrt control for particular CO2 scenario
                calc_ens_diffs(quad_4xctrl_diff_name,quad_ctrl_mem_list,
                               quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)

                # additional difference calcs for 4xCO2 cases #
                # difference wrt 2xCO2 ctrl
                quad_2xctrl_diff_name = f"quad_{diff_root}_2xctrl"
                calc_ens_diffs(quad_2xctrl_diff_name,doub_ctrl_mem_list,
                               quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)
                # difference wrt 2xCO2 experiment with same diffusivity history
                quad_2xCO2_diff_name = f"quad_{diff_root}_2xCO2"
                calc_ens_diffs(quad_2xCO2_diff_name,doub_ens_mem_list,
                               quad_ens_mem_list,all_vars,diag_file=diag_file,stdev_ds=stdev_ds,verbose=verbose)




