#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for reading first ensemble member of all experiments (0.1, 0.2, and 0.5 TW; surf, therm, mid, and bot; const, 2xCO2, and 4xCO2). This is designed to be used with the notebook read_and_calculate.ipynb.

import numpy as np
import xarray as xr

# modules for using datetime variables
import datetime
from datetime import time

import warnings
warnings.filterwarnings('ignore')

import cartopy.crs as ccrs
import cmocean

import subprocess as sp

import cftime
from pandas.errors import OutOfBoundsDatetime  # Import the specific error


# # Variables and functions to read

# ## Important variables

ctrl_exp_root = '_1860IC_200yr_'
doub_exp_root = '_2xCO2_1860IC_200yr_'
quad_exp_root = '_4xCO2_51-201_'

profiles = ['surf','therm','mid','bot'] #'uni',
power_inputs = ['0.1TW', '0.2TW', '0.5TW'] #, '1TW'
power_var_suff = ['0p1TW', '0p2TW', '0p5TW'] #,'1TW'

power_strings = ['0.1 TW', '0.2 TW', '0.5 TW'] #, '1 TW'
prof_strings = ["Surface-Enhanced","Thermocline-Enhanced","Middepth-Enhanced","Bottom-Enhanced"] #"Uniformly-Enhanced",


# ## Functions

def get_const_CO2_temp_data(avg_period,start_year,end_year,verbose=False):
    
    # control
    const_ctrl_name = f"const_ctrl_{start_year}_{end_year}"
    myVars[const_ctrl_name] = get_pp_av_data('tune_ctrl_1860IC_200yr',start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
    if verbose:
        print(f"{const_ctrl_name} done")
    
    # constant CO2
    for prof in profiles:
        for i in range(len(power_inputs)):
            const_ds_name = prof+f'_{power_var_suff[i]}_{start_year}_{end_year}'
            const_exp_name = prof+ctrl_exp_root+power_inputs[i]
            const_ds = get_pp_av_data(const_exp_name,start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
            myVars.__setitem__(const_ds_name, const_ds)
    
            diff_da_name = f"{const_ds_name}_diff"
            diff_da = myVars[const_ds_name]['temp'] - myVars[const_ctrl_name]['temp']
            myVars.__setitem__(diff_da_name, diff_da)
    
            mean_diff_name = f"{diff_da_name}_mean"
            mean_diff = horizontal_mean(myVars[diff_da_name],myVars[const_ds_name])
            myVars.__setitem__(mean_diff_name, mean_diff)

            if verbose:
                print(f'{const_ds_name}, {diff_da_name}, {mean_diff_name} done')

            kd_add_ds_name = f"{const_ds_name}_Kdadd_mean"
            kd_add_mean = horizontal_mean(myVars[const_ds_name]["Kd_int_tuned"],myVars[const_ds_name])
            myVars.__setitem__(kd_add_ds_name, kd_add_mean)

            kd_base_ds_name = f"{const_ds_name}_Kdbase_mean"
            kd_base_mean = horizontal_mean(myVars[const_ds_name]["Kd_int_base"],myVars[const_ds_name])
            myVars.__setitem__(kd_base_ds_name, kd_base_mean)

            kd_tot_ds_name = f"{const_ds_name}_Kdtot_mean"
            kd_tot_mean = horizontal_mean(myVars[const_ds_name]["Kd_interface"],myVars[const_ds_name])
            myVars.__setitem__(kd_tot_ds_name, kd_tot_mean)


def get_2xCO2_temp_data(avg_period,start_year,end_year,verbose=False):
    
    # control
    const_ctrl_name = f"const_ctrl_{start_year}_{end_year}"
    doub_ctrl_name = f"doub_ctrl_{start_year}_{end_year}"
    
    myVars[doub_ctrl_name] = get_pp_av_data('tune_ctrl_2xCO2_1860IC_200yr',start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
    if verbose:
        print(f"{doub_ctrl_name} done")
    
    doub_ctrl_diff_name = f"{doub_ctrl_name}_diff"
    myVars[doub_ctrl_diff_name] = myVars[doub_ctrl_name]['temp'] - myVars[const_ctrl_name]['temp']
    myVars[f"{doub_ctrl_diff_name}_mean"] = horizontal_mean(myVars[doub_ctrl_diff_name],myVars[doub_ctrl_name])
    if verbose:
        print(f"{doub_ctrl_diff_name}, {doub_ctrl_diff_name}_mean done")

    # 2xCO2
    for prof in profiles:
        for i in range(len(power_inputs)):
            doub_ds_name = prof+f'_2xCO2_{power_var_suff[i]}_{start_year}_{end_year}'
            const_ds_name = prof+f'_{power_var_suff[i]}_{start_year}_{end_year}'
            # print(f"{const_ds_name}")
            
            doub_exp_name = prof+doub_exp_root+power_inputs[i]
            
            doub_ds = get_pp_av_data(doub_exp_name,start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
            myVars.__setitem__(doub_ds_name, doub_ds)
            # print(f'{doub_ds_name} done')
    
            doub_diff_1860_name = f"{doub_ds_name}_diff_1860"
            doub_diff_1860_da = myVars[doub_ds_name]['temp'] - myVars[const_ds_name]['temp']
            myVars.__setitem__(doub_diff_1860_name, doub_diff_1860_da)
            # print(f'{doub_diff_1860_name} done')
    
            mean_diff_1860_name = f"{doub_diff_1860_name}_mean"
            mean_diff_1860 = horizontal_mean(myVars[doub_diff_1860_name],myVars[doub_ds_name])
            myVars.__setitem__(mean_diff_1860_name, mean_diff_1860)
            # print(mean_diff_1860_name)
    
            doub_diff_ctrl_name = f"{doub_ds_name}_diff_2xctrl"
            doub_diff_ctrl_da = myVars[doub_ds_name]['temp'] - myVars[doub_ctrl_name]['temp']
            myVars.__setitem__(doub_diff_ctrl_name, doub_diff_ctrl_da)
            # print(f'{doub_diff_ctrl_name} done')
    
            mean_diff_ctrl_name = f"{doub_diff_ctrl_name}_mean"
            mean_diff_ctrl = horizontal_mean(myVars[doub_diff_ctrl_name],myVars[doub_ds_name])
            myVars.__setitem__(mean_diff_ctrl_name, mean_diff_ctrl)
            
            if verbose:
                print(f'{doub_ds_name}, {doub_diff_1860_name}, {mean_diff_1860_name}, {doub_diff_ctrl_name}, {mean_diff_ctrl_name} done')

            kd_add_ds_name = f"{doub_ds_name}_Kdadd_mean"
            kd_add_mean = horizontal_mean(myVars[doub_ds_name]["Kd_int_tuned"],myVars[doub_ds_name])
            myVars.__setitem__(kd_add_ds_name, kd_add_mean)

            kd_base_ds_name = f"{doub_ds_name}_Kdbase_mean"
            kd_base_mean = horizontal_mean(myVars[doub_ds_name]["Kd_int_base"],myVars[doub_ds_name])
            myVars.__setitem__(kd_base_ds_name, kd_base_mean)

            kd_tot_ds_name = f"{doub_ds_name}_Kdtot_mean"
            kd_tot_mean = horizontal_mean(myVars[doub_ds_name]["Kd_interface"],myVars[doub_ds_name])
            myVars.__setitem__(kd_tot_ds_name, kd_tot_mean)
    
    # differences wrt const CO2 control
    for power in power_var_suff:
        ds_suffix = f'2xCO2_{power}_{start_year}_{end_year}'
        
        for i in range(len(profiles)):
            ds_name = f'{profiles[i]}_{ds_suffix}'
            diff_da = myVars[ds_name]['temp'] - myVars[const_ctrl_name]['temp']
            mean_diff_name = f"{ds_name}_diff_const_ctrl_mean"
            mean_diff = horizontal_mean(diff_da,myVars[ds_name])
            myVars.__setitem__(mean_diff_name, mean_diff)
            # print(mean_diff_name)


def get_4xCO2_temp_data(avg_period,start_year,end_year,verbose=False):
    
    # control
    const_ctrl_name = f"const_ctrl_{start_year}_{end_year}"
    quad_ctrl_name = f"quad_ctrl_{start_year}_{end_year}"
    
    myVars[quad_ctrl_name] = get_pp_av_data('tune_ctrl_4xCO2_1860IC_200yr',start_year,end_year,avg_period,pp_type='av-annual')
    if verbose:
        print(f"{quad_ctrl_name} done")
    
    quad_ctrl_diff_name = f"{quad_ctrl_name}_diff"
    myVars[quad_ctrl_diff_name] = myVars[quad_ctrl_name]['temp'] - myVars[const_ctrl_name]['temp']
    myVars[f"{quad_ctrl_diff_name}_mean"] = horizontal_mean(myVars[quad_ctrl_diff_name],myVars[quad_ctrl_name])
    if verbose:
        print(f"{quad_ctrl_diff_name}, {quad_ctrl_diff_name}_mean done")
    
    # 4xCO2
    for prof in profiles:
        for i in range(len(power_inputs)):
            quad_ds_name = prof+f'_4xCO2_{power_var_suff[i]}_{start_year}_{end_year}'
            const_ds_name = prof+f'_{power_var_suff[i]}_{start_year}_{end_year}'
            # print(f"{const_ds_name}")
            
            doub_exp_name = prof+doub_exp_root+power_inputs[i]
    
            if power_inputs[i] == '0.5TW':
                quad_exp_name = prof+'_4xCO2_1860IC_200yr_'+power_inputs[i]
                quad_ds = get_pp_av_data(quad_exp_name,start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
            elif start_year < 51:
                quad_exp_name = prof+quad_exp_root+power_inputs[i]
                pre_51_dat = get_pp_av_data(doub_exp_name,start_year,50,avg_period,pp_type='av-annual')#,debug=True)
                post_51_dat = get_pp_av_data(quad_exp_name,51,end_year,avg_period,pp_type='av-annual')#,debug=True)
                quad_ds = xr.concat([pre_51_dat,post_51_dat],"time")
            else:
                quad_exp_name = prof+quad_exp_root+power_inputs[i]
                quad_ds = get_pp_av_data(quad_exp_name,start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
    
            myVars.__setitem__(quad_ds_name, quad_ds)
            # print(f'{quad_ds_name} done')
    
            quad_diff_1860_name = f"{quad_ds_name}_diff_1860"
            quad_diff_1860_da = myVars[quad_ds_name]['temp'] - myVars[const_ds_name]['temp']
            myVars.__setitem__(quad_diff_1860_name, quad_diff_1860_da)
            # print(f'{quad_diff_1860_name} done')
    
            mean_diff_1860_name = f"{quad_diff_1860_name}_mean"
            mean_diff_1860 = horizontal_mean(myVars[quad_diff_1860_name],myVars[quad_ds_name])
            myVars.__setitem__(mean_diff_1860_name, mean_diff_1860)
    
            quad_diff_ctrl_name = f"{quad_ds_name}_diff_4xctrl"
            quad_diff_ctrl_da = myVars[quad_ds_name]['temp'] - myVars[quad_ctrl_name]['temp']
            myVars.__setitem__(quad_diff_ctrl_name, quad_diff_ctrl_da)
            # print(f'{quad_diff_ctrl_name} done')
    
            mean_diff_ctrl_name = f"{quad_diff_ctrl_name}_mean"
            mean_diff_ctrl = horizontal_mean(myVars[quad_diff_ctrl_name],myVars[quad_ds_name])
            myVars.__setitem__(mean_diff_ctrl_name, mean_diff_ctrl)

            if verbose:
                print(f'{quad_ds_name}, {quad_diff_1860_name}, {mean_diff_1860_name}, {quad_diff_ctrl_name}, {mean_diff_ctrl_name} done')

            kd_add_ds_name = f"{quad_ds_name}_Kdadd_mean"
            kd_add_mean = horizontal_mean(myVars[quad_ds_name]["Kd_int_tuned"],myVars[quad_ds_name])
            myVars.__setitem__(kd_add_ds_name, kd_add_mean)

            kd_base_ds_name = f"{quad_ds_name}_Kdbase_mean"
            kd_base_mean = horizontal_mean(myVars[quad_ds_name]["Kd_int_base"],myVars[quad_ds_name])
            myVars.__setitem__(kd_base_ds_name, kd_base_mean)

            kd_tot_ds_name = f"{quad_ds_name}_Kdtot_mean"
            kd_tot_mean = horizontal_mean(myVars[quad_ds_name]["Kd_interface"],myVars[quad_ds_name])
            myVars.__setitem__(kd_tot_ds_name, kd_tot_mean)
    
    # differences wrt constant CO2 control
    for power in power_var_suff:
        ds_suffix = f'4xCO2_{power}_{start_year}_{end_year}'
            
        for i in range(len(profiles)):
            ds_name = f'{profiles[i]}_{ds_suffix}'
            diff_da = myVars[ds_name]['temp'] - myVars[const_ctrl_name]['temp']
            mean_diff_name = f"{ds_name}_diff_const_ctrl_mean"
            mean_diff = horizontal_mean(diff_da,myVars[ds_name])
            myVars.__setitem__(mean_diff_name, mean_diff)
            # print(mean_diff_name)


# this function is doing the constant CO2, 2xCO2, and 4xCO2 data at once
# def get_all_temp_data(avg_period,start_year,end_year):
    
#     # control
#     const_ctrl_name = f"const_ctrl_{start_year}_{end_year}"
#     doub_ctrl_name = f"doub_ctrl_{start_year}_{end_year}"
#     quad_ctrl_name = f"quad_ctrl_{start_year}_{end_year}"
    
#     myVars[const_ctrl_name] = get_pp_av_data('tune_ctrl_1860IC_200yr',start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
#     myVars[doub_ctrl_name] = get_pp_av_data('tune_ctrl_2xCO2_1860IC_200yr',start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
#     myVars[quad_ctrl_name] = get_pp_av_data('tune_ctrl_4xCO2_1860IC_200yr',start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)

#     print(f"{const_ctrl_name}, {doub_ctrl_name}, {quad_ctrl_name} done")
    
#     doub_ctrl_diff_name = f"{doub_ctrl_name}_diff"
#     quad_ctrl_diff_name = f"{quad_ctrl_name}_diff"
#     myVars[doub_ctrl_diff_name] = myVars[doub_ctrl_name]['temp'] - myVars[const_ctrl_name]['temp']
#     myVars[quad_ctrl_diff_name] = myVars[quad_ctrl_name]['temp'] - myVars[const_ctrl_name]['temp']

#     myVars[f"{doub_ctrl_diff_name}_mean"] = horizontal_mean(myVars[doub_ctrl_diff_name],myVars[doub_ctrl_name])
#     myVars[f"{quad_ctrl_diff_name}_mean"] = horizontal_mean(myVars[quad_ctrl_diff_name],myVars[quad_ctrl_name])

#     print(f"{doub_ctrl_diff_name}, {quad_ctrl_diff_name}, {doub_ctrl_diff_name}_mean, {quad_ctrl_diff_name}_mean done")
    
#     # constant CO2
#     for prof in profiles:
#         for i in range(len(power_inputs)):
#             const_ds_name = prof+f'_{power_var_suff[i]}_{start_year}_{end_year}'
#             const_exp_name = prof+ctrl_exp_root+power_inputs[i]
#             const_ds = get_pp_av_data(const_exp_name,start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
#             myVars.__setitem__(const_ds_name, const_ds)
#             # print(f'{const_ds_name} done')
    
#             diff_da_name = f"{const_ds_name}_diff"
#             diff_da = myVars[const_ds_name]['temp'] - myVars[const_ctrl_name]['temp']
#             myVars.__setitem__(diff_da_name, diff_da)
#             # print(f'{diff_da_name} done')
    
#             mean_diff_name = f"{diff_da_name}_mean"
#             mean_diff = horizontal_mean(myVars[diff_da_name],myVars[const_ds_name])
#             myVars.__setitem__(mean_diff_name, mean_diff)
    
#             print(f'{const_ds_name}, {diff_da_name}, {mean_diff_name} done')

#     # 2xCO2
#     for prof in profiles:
#         for i in range(len(power_inputs)):
#             doub_ds_name = prof+f'_2xCO2_{power_var_suff[i]}_{start_year}_{end_year}'
#             const_ds_name = prof+f'_{power_var_suff[i]}_{start_year}_{end_year}'
#             # print(f"{const_ds_name}")
            
#             doub_exp_name = prof+doub_exp_root+power_inputs[i]
            
#             doub_ds = get_pp_av_data(doub_exp_name,start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
#             myVars.__setitem__(doub_ds_name, doub_ds)
#             # print(f'{doub_ds_name} done')
    
#             doub_diff_1860_name = f"{doub_ds_name}_diff_1860"
#             doub_diff_1860_da = myVars[doub_ds_name]['temp'] - myVars[const_ds_name]['temp']
#             myVars.__setitem__(doub_diff_1860_name, doub_diff_1860_da)
#             # print(f'{doub_diff_1860_name} done')
    
#             mean_diff_1860_name = f"{doub_diff_1860_name}_mean"
#             mean_diff_1860 = horizontal_mean(myVars[doub_diff_1860_name],myVars[doub_ds_name])
#             myVars.__setitem__(mean_diff_1860_name, mean_diff_1860)
#             # print(mean_diff_1860_name)
    
#             doub_diff_ctrl_name = f"{doub_ds_name}_diff_2xctrl"
#             doub_diff_ctrl_da = myVars[doub_ds_name]['temp'] - myVars[doub_ctrl_name]['temp']
#             myVars.__setitem__(doub_diff_ctrl_name, doub_diff_ctrl_da)
#             # print(f'{doub_diff_ctrl_name} done')
    
#             mean_diff_ctrl_name = f"{doub_diff_ctrl_name}_mean"
#             mean_diff_ctrl = horizontal_mean(myVars[doub_diff_ctrl_name],myVars[doub_ds_name])
#             myVars.__setitem__(mean_diff_ctrl_name, mean_diff_ctrl)
    
#             print(f'{doub_ds_name}, {doub_diff_1860_name}, {mean_diff_1860_name}, {doub_diff_ctrl_name}, {mean_diff_ctrl_name} done')
    
#     # differences wrt const CO2 control
#     for power in power_var_suff:
#         ds_suffix = f'2xCO2_{power}_{start_year}_{end_year}'
        
#         for i in range(len(profiles)):
#             ds_name = f'{profiles[i]}_{ds_suffix}'
#             diff_da = myVars[ds_name]['temp'] - myVars[const_ctrl_name]['temp']
#             mean_diff_name = f"{ds_name}_diff_const_ctrl_mean"
#             mean_diff = horizontal_mean(diff_da,myVars[ds_name])
#             myVars.__setitem__(mean_diff_name, mean_diff)
#             # print(mean_diff_name)

#     # 4xCO2
#     for prof in profiles:
#         for i in range(len(power_inputs)):
#             quad_ds_name = prof+f'_4xCO2_{power_var_suff[i]}_{start_year}_{end_year}'
#             const_ds_name = prof+f'_{power_var_suff[i]}_{start_year}_{end_year}'
#             # print(f"{const_ds_name}")
            
#             doub_exp_name = prof+doub_exp_root+power_inputs[i]
    
#             if power_inputs[i] == '0.5TW':
#                 quad_exp_name = prof+'_4xCO2_1860IC_200yr_'+power_inputs[i]
#                 quad_ds = get_pp_av_data(quad_exp_name,start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
#             elif start_year < 51:
#                 quad_exp_name = prof+quad_exp_root+power_inputs[i]
#                 pre_51_dat = get_pp_av_data(doub_exp_name,start_year,50,avg_period,pp_type='av-annual')#,debug=True)
#                 post_51_dat = get_pp_av_data(quad_exp_name,51,end_year,avg_period,pp_type='av-annual')#,debug=True)
#                 quad_ds = xr.concat([pre_51_dat,post_51_dat],"time")
#             else:
#                 quad_exp_name = prof+quad_exp_root+power_inputs[i]
#                 quad_ds = get_pp_av_data(quad_exp_name,start_year,end_year,avg_period,pp_type='av-annual')#,debug=True)
    
#             myVars.__setitem__(quad_ds_name, quad_ds)
#             # print(f'{quad_ds_name} done')
    
#             quad_diff_1860_name = f"{quad_ds_name}_diff_1860"
#             quad_diff_1860_da = myVars[quad_ds_name]['temp'] - myVars[const_ds_name]['temp']
#             myVars.__setitem__(quad_diff_1860_name, quad_diff_1860_da)
#             # print(f'{quad_diff_1860_name} done')
    
#             mean_diff_1860_name = f"{quad_diff_1860_name}_mean"
#             mean_diff_1860 = horizontal_mean(myVars[quad_diff_1860_name],myVars[quad_ds_name])
#             myVars.__setitem__(mean_diff_1860_name, mean_diff_1860)
    
#             quad_diff_ctrl_name = f"{quad_ds_name}_diff_4xctrl"
#             quad_diff_ctrl_da = myVars[quad_ds_name]['temp'] - myVars[quad_ctrl_name]['temp']
#             myVars.__setitem__(quad_diff_ctrl_name, quad_diff_ctrl_da)
#             # print(f'{quad_diff_ctrl_name} done')
    
#             mean_diff_ctrl_name = f"{quad_diff_ctrl_name}_mean"
#             mean_diff_ctrl = horizontal_mean(myVars[quad_diff_ctrl_name],myVars[quad_ds_name])
#             myVars.__setitem__(mean_diff_ctrl_name, mean_diff_ctrl)
    
#             print(f'{quad_ds_name}, {quad_diff_1860_name}, {mean_diff_1860_name}, {quad_diff_ctrl_name}, {mean_diff_ctrl_name} done')
    
#     # differences wrt constant CO2 control
#     for power in power_var_suff:
#         ds_suffix = f'4xCO2_{power}_{start_year}_{end_year}'
            
#         for i in range(len(profiles)):
#             ds_name = f'{profiles[i]}_{ds_suffix}'
#             diff_da = myVars[ds_name]['temp'] - myVars[const_ctrl_name]['temp']
#             mean_diff_name = f"{ds_name}_diff_const_ctrl_mean"
#             mean_diff = horizontal_mean(diff_da,myVars[ds_name])
#             myVars.__setitem__(mean_diff_name, mean_diff)
#             # print(mean_diff_name)

