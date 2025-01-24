#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for plotting various quantities. This is designed to be used with the notebook read_and_calculate.ipynb.

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

import cftime
from pandas.errors import OutOfBoundsDatetime  # Import the specific error


# ## Temperature map plotting functions

def plot_pp_temp_diff(prefix,title,pp_diff_da,z_idx,start_yr,end_yr,cb_max=None,non_linear_cb=False,verbose=False):

    depth = pp_diff_da.coords['z_l'].values[z_idx]
    diff_da = pp_diff_da.isel(z_l=z_idx)
    
    min_val = np.nanmin(diff_da.values)
    max_val = np.nanmax(diff_da.values)
    
    if verbose:
        print(f"Data min: {min_val:.3f}\t Data max: {max_val:.3f}")
        if np.abs(min_val) > np.abs(max_val):
            print(f"Data max mag: {np.abs(min_val):.3f}")
        else:
            print(f"Data max mag: {np.abs(max_val):.3f}")

    if cb_max != None:
        max_mag = cb_max
    elif np.abs(min_val) > np.abs(max_val):
        max_mag = np.abs(min_val)
    else:
        max_mag = np.abs(max_val)
        
    # setting plot min and max
    if max_mag <= 1:
        plot_min = -round(max_mag/0.2)*0.2
        plot_max = round(max_mag/0.2)*0.2
        num_ticks = int((plot_max-plot_min)/0.2) + 1
        num_colors = int((plot_max-plot_min)/0.1)
        tick_arr = np.linspace(plot_min,plot_max,num=num_ticks)
        for i in range(0,len(tick_arr)):
            tick_arr[i] = round(tick_arr[i]/0.1)*0.1
    elif max_mag <= 2:
        plot_min = -round(max_mag/0.4)*0.4
        plot_max = round(max_mag/0.4)*0.4
        num_ticks = int((plot_max-plot_min)/0.4) + 1
        num_colors = int((plot_max-plot_min)/0.2)
        tick_arr = np.linspace(plot_min,plot_max,num=num_ticks)
        for i in range(0,len(tick_arr)):
            tick_arr[i] = round(tick_arr[i]/0.1)*0.1
    elif max_mag <= 2.5:
        plot_min = -round(max_mag/0.5)*0.5
        plot_max = round(max_mag/0.5)*0.5
        num_ticks = int((plot_max-plot_min)/0.5) + 1
        num_colors = int((plot_max-plot_min)/0.25)
        tick_arr = np.linspace(plot_min,plot_max,num=num_ticks)
        for i in range(0,len(tick_arr)):
            tick_arr[i] = round(tick_arr[i]/0.1)*0.1
    elif max_mag <= 4.1:
        plot_min = -round(max_mag)
        plot_max = round(max_mag)
        num_ticks = int(plot_max-plot_min) + 1
        num_colors = int((plot_max-plot_min)/0.25)
        tick_arr = np.linspace(plot_min,plot_max,num=num_ticks)
    elif max_mag <= 6.1:
        plot_min = -round(max_mag)
        plot_max = round(max_mag)
        num_ticks = int(plot_max-plot_min) + 1
        num_colors = int((plot_max-plot_min)/0.5)
        tick_arr = np.linspace(plot_min,plot_max,num=num_ticks)
    elif max_mag <= 7.49:
        plot_min = -round(max_mag)
        plot_max = round(max_mag)
        num_ticks = int((plot_max-plot_min)/2) + 1
        num_colors = int((plot_max-plot_min)/0.5)
        tick_arr = np.linspace(plot_min,plot_max,num=num_ticks)
    elif max_mag <= 10:
        plot_min = -np.ceil(max_mag/2)*2
        plot_max = np.ceil(max_mag/2)*2
        num_ticks = int((plot_max-plot_min)/2) + 1
        num_colors = int((plot_max-plot_min)/1)
        tick_arr = np.linspace(plot_min,plot_max,num=num_ticks)
    elif max_mag <= 13:
        plot_min = -np.ceil(max_mag/3)*3
        plot_max = np.ceil(max_mag/3)*3
        num_ticks = int((plot_max-plot_min)/3) + 1
        num_colors = int((plot_max-plot_min)/1.5)
        tick_arr = np.linspace(plot_min,plot_max,num=num_ticks)
    elif max_mag <= 15:
        plot_min = -np.ceil(max_mag/4)*4
        plot_max = np.ceil(max_mag/4)*4
        num_ticks = int((plot_max-plot_min)/4) + 1
        num_colors = int((plot_max-plot_min)/2)
        tick_arr = np.linspace(plot_min,plot_max,num=num_ticks)
    else:
        print("Warning: plot bounds more than +/- 15")

    if verbose:
        print(f"num_colors = {num_colors}") 
        print(f"Plot min: {plot_min:.3f}\t Plot max: {plot_max:.3f}")

    plt.figure(figsize=[12, 8])

    cmap = cmocean.cm.balance  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the balance map
    # force the first color entry to be grey
    # cmaplist[0] = (.5, .5, .5, 1.0)
    
    # create the new map
    disc_bal_cmap = mcolors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    norm_bounds = np.linspace(plot_min, plot_max, num_colors + 1)
    disc_norm = mcolors.BoundaryNorm(norm_bounds, cmap.N)
        
    subplot_kws=dict(projection=ccrs.Robinson(central_longitude=209.5), facecolor='0.75') #projection=ccrs.PlateCarree(),facecolor='gray'
    # projection=ccrs.Robinson(central_longitude=180)
    
    if non_linear_cb == False:
        diff_plot = diff_da.plot(#vmin=plot_min, vmax=plot_max,
                      x='geolon', y='geolat',
                      cmap=disc_bal_cmap, norm=disc_norm,
                      subplot_kws=subplot_kws,
                          #You can pick any projection from the cartopy list but, whichever projection you use, you still have to set
                      transform=ccrs.PlateCarree(),
                      add_labels=False,
                      add_colorbar=False)

    elif non_linear_cb == True:
        norm = mcolors.SymLogNorm(linthresh=cb_max/2, linscale = 0.6, vmin=plot_min, vmax=plot_max, base=10)
        
        diff_plot = diff_da.plot(vmin=plot_min, vmax=plot_max,
                      x='geolon', y='geolat',
                      cmap=cmocean.cm.balance, norm=norm,
                      subplot_kws=subplot_kws,
                          #You can pick any projection from the cartopy list but, whichever projection you use, you still have to set
                      transform=ccrs.PlateCarree(),
                      add_labels=False,
                      add_colorbar=False)
    
    diff_plot.axes.set_title(f"{title}\nYear {start_yr}–{end_yr}, z = {depth:,.2f} m",fontdict={'fontsize':16})
    
    # diff_cb = plt.colorbar(diff_plot, fraction=0.046, pad=0.04)
    diff_cb = plt.colorbar(diff_plot, shrink=0.6, extend='both')

    tick_labels = [f"{x:.1f}" for x in tick_arr]
    
    diff_cb.set_ticks(tick_arr)
    diff_cb.ax.set_yticklabels(tick_labels)
    diff_cb.ax.tick_params(labelsize=14)
    diff_cb.set_label("Temperature Anomaly ($\degree$C)",fontdict={'fontsize':14})

    for t in diff_cb.ax.get_yticklabels():
        t.set_horizontalalignment('center')
        if plot_max < 10:
            t.set_x(2.0)
        else:
            t.set_x(2.2)
    
    # draw parallels/meridians and write labels
    # diff_gl = diff_plot.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, # draw_labels=True,
    #                       linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    # adjust labels to taste
    # diff_gl.top_labels = False
    # diff_gl.right_labels = False
    # diff_gl.bottom_labels = False
    # diff_gl.left_labels = False
    
    # diff_gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
    # diff_gl.xformatter = LONGITUDE_FORMATTER
    # diff_gl.yformatter = LATITUDE_FORMATTER
    # diff_gl.xlabel_style = {'size': 14, 'color': 'black'}
    # diff_gl.ylabel_style = {'size': 14, 'color': 'black'}

    plt.savefig(f'{prefix}_dT_{start_yr}_{end_yr}_z_{depth:.0f}.png', dpi=600, bbox_inches='tight')


def plot_pp_temp_mean(prefix,title,pp_temp_da,z_idx,start_yr,end_yr,verbose=False):

    depth = pp_temp_da.coords['z_l'].values[z_idx]
    run_da = pp_temp_da.isel(z_l=z_idx)
    
    min_val = np.nanmin(run_da.values)
    max_val = np.nanmax(run_da.values)
    
    if verbose:
        print(f"Data min: {min_val:.3f}\t Data max: {max_val:.3f}")

    plot_min = -2
    plot_max = 30
    num = int((plot_max-plot_min)/4) + 1
    tick_arr = np.linspace(plot_min,plot_max,num=num)
    
    num_colors = 4 * (num - 1)

    plt.figure(figsize=[12, 8])

    cmap = cmocean.cm.thermal  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the balance map
    # force the first color entry to be grey
    # cmaplist[0] = (.5, .5, .5, 1.0)
    
    # create the new map
    disc_bal_cmap = mcolors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    norm_bounds = np.linspace(plot_min, plot_max, num_colors + 1)
    disc_norm = mcolors.BoundaryNorm(norm_bounds, cmap.N)
        
    subplot_kws=dict(projection=ccrs.Robinson(central_longitude=209.5), facecolor='0.75')
    
    run_plot = run_da.plot(x='geolon', y='geolat',
                  cmap=disc_bal_cmap, norm=disc_norm,
                  subplot_kws=subplot_kws,
                      #You can pick any projection from the cartopy list but, whichever projection you use, you still have to set
                  transform=ccrs.PlateCarree(),
                  add_labels=False,
                  add_colorbar=False)

    run_plot.axes.set_title(f"{title}\nYear {start_yr}–{end_yr}, z = {depth:,.2f} m",fontdict={'fontsize':16})

    run_cb = plt.colorbar(run_plot, ticks=tick_arr, shrink=0.6, extend='both')
    run_cb.ax.tick_params(labelsize=14)
    run_cb.set_label("Temperature ($\degree$C)",fontdict={'fontsize':12})

    plt.show()

    plt.savefig(f'{prefix}_temp_{start_yr}_{end_yr}_z_{depth:.0f}.png', dpi=600, bbox_inches='tight')


# ### Linearity difference plotting functions

def plot_pp_linearity_temp_diff(prefix,title,pp_diff_small,pp_diff_big,linear_factor,z_idx,start_yr,end_yr,cb_max=None,non_linear_cb=False,verbose=False):

    depth = pp_diff_small.coords['z_l'].values[z_idx]
    
    small_diff = pp_diff_small.isel(z_l=z_idx)
    big_diff = pp_diff_big.isel(z_l=z_idx)

    lin_diff = big_diff - linear_factor * small_diff
    
    min_val = np.nanmin(lin_diff.values)
    max_val = np.nanmax(lin_diff.values)
    
    if verbose:
        print(f"Data min: {min_val:.3f}\t Data max: {max_val:.3f}")

    if cb_max != None:
        max_mag = cb_max
    elif np.abs(min_val) > np.abs(max_val):
        max_mag = np.abs(min_val)
    else:
        max_mag = np.abs(max_val)
        
    # setting plot min and max
    if max_mag <= 0.9:
        plot_min = -round(max_mag/0.1)*0.1
        plot_max = round(max_mag/0.1)*0.1
        num = int((plot_max-plot_min)/0.1) + 1
        tick_arr = np.linspace(plot_min,plot_max,num=num)
        for i in range(0,len(tick_arr)):
            tick_arr[i] = round(tick_arr[i]/0.1)*0.1
    elif max_mag <= 1.4:
        plot_min = -round(max_mag/0.2)*0.2
        plot_max = round(max_mag/0.2)*0.2
        num = int((plot_max-plot_min)/0.2) + 1
        tick_arr = np.linspace(plot_min,plot_max,num=num)
        for i in range(0,len(tick_arr)):
            tick_arr[i] = round(tick_arr[i]/0.1)*0.1
    elif max_mag <= 3.3:
        plot_min = -round(max_mag/0.4)*0.4
        plot_max = round(max_mag/0.4)*0.4
        num = int((plot_max-plot_min)/0.4) + 1
        tick_arr = np.linspace(plot_min,plot_max,num=num)
        # tick_arr = np.arange(plot_min,plot_max+0.4,0.4)
        for i in range(0,len(tick_arr)):
            tick_arr[i] = round(tick_arr[i]/0.1)*0.1
    elif max_mag < 5:
        plot_min = -round(max_mag)
        plot_max = round(max_mag)
        num = int((plot_max-plot_min)) + 1
        tick_arr = np.linspace(plot_min,plot_max,num=num)
    elif max_mag < 8:
        plot_min = -np.ceil(max_mag/2)*2
        plot_max = np.ceil(max_mag/2)*2
        num = int((plot_max-plot_min)/2) + 1
        tick_arr = np.linspace(plot_min,plot_max,num=num)
    elif max_mag < 12:
        plot_min = -np.ceil(max_mag/3)*3
        plot_max = np.ceil(max_mag/3)*3
        num = int((plot_max-plot_min)/3) + 1
        tick_arr = np.linspace(plot_min,plot_max,num=num)
    elif max_mag < 15:
        plot_min = -np.ceil(max_mag/4)*4
        plot_max = np.ceil(max_mag/4)*4
        num = int((plot_max-plot_min)/4) + 1
        tick_arr = np.linspace(plot_min,plot_max,num=num)
    else:
        plot_min = -16
        plot_max = 16
        tick_arr = [-16, -12, -8, -4, 0, 4 , 8, 12, 16]
        num = int((plot_max-plot_min)/4) + 1
        
    num_colors = 2 * (num - 1)

    if verbose:
        print(f"num = {num}\t num_colors = {num_colors}")  
        print(f"Plot min: {plot_min:.3f}\t Plot max: {plot_max:.3f}")

    plt.figure(figsize=[12, 8])

    cmap = cmocean.cm.balance  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the balance map
    # force the first color entry to be grey
    # cmaplist[0] = (.5, .5, .5, 1.0)
    
    # create the new map
    disc_bal_cmap = mcolors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    norm_bounds = np.linspace(plot_min, plot_max, num_colors + 1)
    disc_norm = mcolors.BoundaryNorm(norm_bounds, cmap.N)
        
    subplot_kws=dict(projection=ccrs.Robinson(central_longitude=209.5), facecolor='0.75') #projection=ccrs.PlateCarree(),facecolor='gray'
    # projection=ccrs.Robinson(central_longitude=180)
    
    if non_linear_cb == False:
        diff_plot = lin_diff.plot(#vmin=plot_min, vmax=plot_max,
                      x='geolon', y='geolat',
                      cmap=disc_bal_cmap, norm=disc_norm,
                      subplot_kws=subplot_kws,
                          #You can pick any projection from the cartopy list but, whichever projection you use, you still have to set
                      transform=ccrs.PlateCarree(),
                      add_labels=False,
                      add_colorbar=False)

    elif non_linear_cb == True:
        norm = mcolors.SymLogNorm(linthresh=cb_max/2, linscale = 0.6, vmin=plot_min, vmax=plot_max, base=10)
        
        diff_plot = lin_diff.plot(vmin=plot_min, vmax=plot_max,
                      x='geolon', y='geolat',
                      cmap=cmocean.cm.balance, norm=norm,
                      subplot_kws=subplot_kws,
                          #You can pick any projection from the cartopy list but, whichever projection you use, you still have to set
                      transform=ccrs.PlateCarree(),
                      add_labels=False,
                      add_colorbar=False)
    
    # diff_plot.axes.coastlines()
    diff_plot.axes.set_title(f"{title}: Year {start_yr}–{end_yr}, z = {depth:,.2f} m",fontdict={'fontsize':18})
    
    # diff_cb = plt.colorbar(diff_plot, fraction=0.046, pad=0.04)
    diff_cb = plt.colorbar(diff_plot, shrink=0.6, extend='both')

    tick_labels = [f"{x:.1f}" for x in tick_arr]
    
    diff_cb.set_ticks(tick_arr)
    diff_cb.ax.set_yticklabels(tick_labels)
    diff_cb.ax.tick_params(labelsize=14)
    diff_cb.set_label("Temperature Anomaly ($\degree$C)",fontdict={'fontsize':14})

    for t in diff_cb.ax.get_yticklabels():
        t.set_horizontalalignment('center')   
        t.set_x(2.0)

    plt.savefig(f'{prefix}_dT_{start_yr}_{end_yr}_z_{depth:.0f}.png', dpi=600, bbox_inches='tight')


# ## Diffusivity plotting functions

def plot_pp_Kd_map(title,pp_ds,Kd_var,z_idx,start_yr,end_yr,layer_var=False,savefig=False,cb_min=-12,\
                   cb_max=None,prefix=None,verbose=False):

    if layer_var == False:
        Kd_dat = pp_ds[Kd_var].isel(z_i=z_idx)
        depth = pp_ds[Kd_var].coords['z_i'].values[z_idx]
    else:
        Kd_dat = pp_ds[Kd_var].isel(z_l=z_idx)
        depth = pp_ds[Kd_var].coords['z_l'].values[z_idx]

    if verbose:
        print(f"Kd min: {np.nanmin(Kd_dat.values):.3e}\t Kd max: {np.nanmax(Kd_dat.values):.3e}")

    log_Kd_dat = np.log10(Kd_dat)
    log_Kd_dat = log_Kd_dat.where(log_Kd_dat != -np.inf, -50)
    
    dat_min = np.nanmin(log_Kd_dat.values)
    dat_max = np.nanmax(log_Kd_dat.values)
    
    if verbose:
        print(f"Log(Kd) min: {dat_min:.3e}\t Log(Kd) max: {dat_max:.3e}")

    if cb_max != None:
        max_val = cb_max
    else:
        max_val = dat_max

    plot_min = cb_min
    plot_max = np.ceil(max_val)
    num = int(plot_max - plot_min) + 1
    tick_arr = np.linspace(plot_min,plot_max,num=num)
    
    num_colors = 2 * (num - 1)
    
    if verbose:
        print(f"num = {num}\t num_colors = {num_colors}")  
        print(f"Plot min: {plot_min:.3f}\t Plot max: {plot_max:.3f}")
    
    plt.figure(figsize=[12, 8])
    
    cmap = cmocean.cm.dense  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the balance map
    # force the first color entry to be grey
    # cmaplist[0] = (.5, .5, .5, 1.0)
    
    # create the new map
    disc_bal_cmap = mcolors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    norm_bounds = np.linspace(plot_min, plot_max, num_colors + 1)
    disc_norm = mcolors.BoundaryNorm(norm_bounds, cmap.N)
        
    subplot_kws=dict(projection=ccrs.Robinson(central_longitude=209.5), facecolor='0.75') #projection=ccrs.PlateCarree(),facecolor='gray'
    # projection=ccrs.Robinson(central_longitude=180)
    
    Kd_plot = log_Kd_dat.plot(vmin=plot_min, vmax=plot_max,
                  x='geolon', y='geolat',
                  # cmap=cmocean.cm.dense,
                  cmap=disc_bal_cmap, norm=disc_norm,
                  subplot_kws=subplot_kws,
                      #You can pick any projection from the cartopy list but, whichever projection you use, you still have to set
                  transform=ccrs.PlateCarree(),
                  add_labels=False,
                  add_colorbar=False)
    
    # Kd_plot.axes.coastlines()
    Kd_plot.axes.set_title(f"{title}: Year {start_yr}–{end_yr}, z = {depth:,.2f} m",fontdict={'fontsize':18})
    
    # Kd_cb = plt.colorbar(Kd_plot, fraction=0.046, pad=0.04)
    Kd_cb = plt.colorbar(Kd_plot, ticks=tick_arr, shrink=0.6, extend='both') #fraction=0.046, pad=0.04,

    # tick_labels = [f"{x:.0f}" for x in tick_arr] # str(x)
    # tick_labels[np.ceil(num)] = "0"
    Kd_cb.set_ticks(tick_arr)
    Kd_cb.ax.set_yticklabels(tick_labels)
    Kd_cb.ax.tick_params(labelsize=14)
    Kd_cb.set_label("log$_{10}$ ($m^2/s$)",fontdict={'fontsize':14})

    for t in Kd_cb.ax.get_yticklabels():
        t.set_horizontalalignment('center')   
        t.set_x(2.0)

    if savefig == True:
        plt.savefig(f'{prefix}_{Kd_var}_{start_yr}_{end_yr}_z_{depth:.0f}.png', dpi=600, bbox_inches='tight')


def plot_Kd_basin(title_p1,title_p2,pp_ds,Kd_var,basin_name,max_depth,start_yr,end_yr,layer_var=False,\
                     cb_min=-12,cb_max=None,non_lin_cb_val=-5,cb_spacing=0.25,savefig=False,prefix=None,check_nn=True,nn_threshold=0.05,Kd_var_base=None,verbose=False):
    
    Kd_dat = get_pp_basin_dat(pp_ds,basin_name,Kd_var,check_nn=check_nn,nn_threshold=nn_threshold,full_field_var=Kd_var_base,verbose=verbose)
    
    if layer_var==False:
        Kd_dat = Kd_dat.sel(z_i=slice(0,max_depth))
    else:
        Kd_dat = Kd_dat.sel(z_l=slice(0,max_depth))

    if verbose:
        print(f"Kd min: {np.nanmin(Kd_dat.values):.3e}\t Kd max: {np.nanmax(Kd_dat.values):.3e}")
    
    log_Kd_dat = np.log10(Kd_dat)
    log_Kd_dat = log_Kd_dat.where(log_Kd_dat != -np.inf, -50)
    
    dat_min = np.nanmin(log_Kd_dat.values)
    dat_max = np.nanmax(log_Kd_dat.values)
    
    if verbose:
        print(f"Log(Kd) min: {dat_min:.3f}\t Log(Kd) max: {dat_max:.3f}")

    if cb_max != None:
        max_val = cb_max
    else:
        max_val = dat_max

    plot_min = cb_min
    plot_max = np.ceil(max_val)

    cmap = cmocean.cm.dense  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the balance map
    
    # create the new map
    disc_bal_cmap = mcolors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    num_col_lower = 2*int(non_lin_cb_val - plot_min)
    num_ticks_lower = int(non_lin_cb_val - plot_min)
    num_col_upper = int((plot_max - (non_lin_cb_val))/cb_spacing)
    num_ticks_upper = int((plot_max - (non_lin_cb_val))/(2*cb_spacing))

    lower_bounds = np.linspace(plot_min,non_lin_cb_val,num_col_lower,endpoint=False)
    lower_ticks = np.linspace(plot_min,non_lin_cb_val,num_ticks_lower,endpoint=False)
    upper_bounds = np.linspace(non_lin_cb_val, plot_max, num_col_upper + 1)
    upper_ticks = np.linspace(non_lin_cb_val, plot_max, num_ticks_upper + 1)
    
    norm_bounds = np.concatenate((lower_bounds,upper_bounds))
    tick_arr = np.concatenate((lower_ticks,upper_ticks))

    # print(lower_bounds)
    # print(upper_bounds)
    # print(num_col_lower)
    # print(num_col_upper)
    if verbose:
        print(norm_bounds)
    
    # for i in range(0,len(tick_arr)):
    #     tick_arr[i] = round(tick_arr[i]/0.1)*0.1

    # norm_bounds = np.linspace(plot_min, plot_max, num_colors + 1)
    disc_norm = mcolors.BoundaryNorm(norm_bounds, cmap.N)

    plt.figure(figsize=[11, 3.8])
    
    subplot_kws=dict(facecolor='black')

    if layer_var == False:
        Kd_p = log_Kd_dat.plot(x='true_lat', y='z_i',
                  cmap=disc_bal_cmap,
                  norm=disc_norm,
                  subplot_kws=subplot_kws,
                      #You can pick any projection from the cartopy list but, whichever projection you use, you still have to set
                  # transform=ccrs.PlateCarree(),
                  add_labels=False,
                  add_colorbar=False)
    else:
        Kd_p = log_Kd_dat.plot(x='true_lat', y='z_l',
                  cmap=disc_bal_cmap,
                  norm=disc_norm,
                  subplot_kws=subplot_kws,
                      #You can pick any projection from the cartopy list but, whichever projection you use, you still have to set
                  # transform=ccrs.PlateCarree(),
                  add_labels=False,
                  add_colorbar=False)
    
    Kd_p.axes.invert_yaxis()
    Kd_p.axes.minorticks_on()

    Kd_cb = plt.colorbar(Kd_p, ticks=tick_arr, fraction=0.046, pad=0.04, extend='both') #shrink=0.6

    tick_labels = [f"{x:.2f}" for x in tick_arr] # str(x)
    Kd_cb.set_ticks(tick_arr)
    Kd_cb.ax.set_yticklabels(tick_labels)
    Kd_cb.ax.tick_params(labelsize=14)
    Kd_cb.set_label("log$_{10}$ ($m^2/s$)",fontdict={'fontsize':14})

    for t in Kd_cb.ax.get_yticklabels():
        t.set_horizontalalignment('center')   
        t.set_x(2.8)
    
    Kd_p.axes.set_xlim(-60,60)
    Kd_p.axes.set_xticks(ticks=[-60,-40,-20,0,20,40,60],labels=['60$\degree$S','40$\degree$S','20$\degree$S','0$\degree$',\
                                                                '20$\degree$N','40$\degree$N','60$\degree$N'], fontsize=14)
    
    Kd_p.axes.tick_params(axis='y', labelsize=14)
    
    Kd_p.axes.set_ylabel('Depth (m)', fontsize=14)
    Kd_p.axes.set_title(f"{title_p1}\n{title_p2}: Year {start_yr}–{end_yr}",fontdict={'fontsize':16})

    if savefig == True:
        plt.savefig(f'{prefix}_{Kd_var}_{start_yr}_{end_yr}.png', dpi=600, bbox_inches='tight')

