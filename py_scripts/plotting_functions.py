#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for plotting various quantities. This is designed to be used with the notebook read_and_calculate.ipynb.

import numpy as np
import xarray as xr
import xesmf as xe

# modules for plotting datetime data
import matplotlib.dates as mdates
from matplotlib.axis import Axis

# modules for using datetime variables
import datetime
from datetime import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

# for custom legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from xgcm import Grid
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

import cartopy.crs as ccrs
import cmocean
import colorcet

import subprocess as sp

import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from matplotlib.ticker import ScalarFormatter

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import LogFormatterMathtext

from xclim import ensembles

import cftime
from pandas.errors import OutOfBoundsDatetime  # Import the specific error

import os

from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter


from matplotlib import font_manager
# Specify the path to your custom .otf font file
font_path = '/home/Kiera.Lowman/.fonts/HelveticaNeueRoman.otf'

# Add the font to the matplotlib font manager
font_manager.fontManager.addfont(font_path)

# Retrieve the font's name from the file
prop = font_manager.FontProperties(fname=font_path)
font_name = prop.get_name()

# Set the default font globally
plt.rcParams['font.family'] = font_name

# 2) make mathtext use your font
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm']     = font_name  # roman
plt.rcParams['mathtext.it']     = font_name  # italic
plt.rcParams['mathtext.bf']     = font_name  # bold

# set font sizes
plt.rcParams['axes.labelsize'] = 12    # Axis label size
plt.rcParams['xtick.labelsize'] = 10     # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 10     # Y-axis tick label size
plt.rcParams['legend.fontsize'] = 10     # Legend font size

plt.rcParams['axes.titlesize'] = 14      # Title size
plt.rcParams['figure.titlesize'] = 14


# # Useful functions for plot elements

def get_cb_spacing(per0p5,per99p5,min_bnd=1.0,min_spacing=0.1,min_n=10,max_n=20,verbose=False):
    
    data_max = max(abs(per0p5), abs(per99p5))

    # Enforce a minimum of min_bnd
    data_max = max(data_max, min_bnd)
    if verbose:
        print(f"data_max: {data_max}")

    # ------------------------------------------------------------------
    # Choose n (10..20) and stepCandidate (multiple of min_spacing, >= min_spacing)
    # so that stepCandidate*n >= 2*data_max
    # We pick the largest n for which we can find a suitable stepCandidate.
    # ------------------------------------------------------------------
    chosen_n = None
    chosen_step = None
    
    for n in range(max_n, min_n-1, -2):
        # The step we *need* to at least cover data_max
        stepNeeded = (2 * data_max) / n
        # Round that up to the nearest multiple of min_spacing
        ceil_step = np.ceil(stepNeeded / min_spacing) * min_spacing
        floor_step = np.floor(stepNeeded / min_spacing) * min_spacing
        if abs(stepNeeded - ceil_step) > abs(stepNeeded - floor_step):
            continue
        else:
            stepCandidate = ceil_step
        # Enforce stepCandidate >= min_spacing
        if stepCandidate < min_spacing:
            continue

        # This candidate max_mag
        max_magCandidate = 0.5 * n * stepCandidate

        # By construction, max_magCandidate >= data_max
        # so if we get here, it's acceptable. We'll pick the first one from n=max_n downward.
        chosen_n = n
        chosen_step = stepCandidate
        break

    # If none found (should not happen), just force n=min_n
    if chosen_n is None:
        chosen_n = min_n
        chosen_step = min_spacing
        if verbose:
            print(f"No feasible n found in [{min_n}..{max_n}]! Using fallback: n={min_n}, step={min_spacing}")

    return chosen_n, chosen_step


# def create_cb_params(max_mag, min_val, max_val, chosen_n, chosen_step, verbose=False, nonlinear=False, lin_thresh=1e-8):
#     # for nonlinear case, max_mag must be 1*10**(x), not an integer multiple of 10**(x)
    
#     vmin, vmax = -max_mag, max_mag

#     if verbose:
#         print(f"Final chosen_n: {chosen_n}, step: {chosen_step}, vmin={vmin}, vmax={vmax}")

#     # --- Enforce an odd number of colors if needed ---
#     if chosen_n % 2 == 0:
#         chosen_n += 1  # force odd so that there is a central bin for white

#     if nonlinear:
#         zero_step = lin_thresh
#     else:
#         zero_step = chosen_step/3
    
#     # Number of colored bins on one side of the central white bin.
#     # Total bins is chosen_n = 2*n_side + 1.
#     n_side = (chosen_n - 1) // 2

#     # --- boundaries: linear vs. discrete log ---
#     if nonlinear:
#         # log bins from zero_step → vmax
#         # geomspace gives n_side+1 points including endpoints
#         pos_boundaries = np.geomspace(zero_step, vmax, n_side + 1)
#         neg_boundaries = -pos_boundaries[::-1]

#         # combine: [neg..., -zero_step, +zero_step, pos...]
#         boundaries = np.concatenate([
#             neg_boundaries,
#             # [-zero_step, zero_step],
#             pos_boundaries
#         ])
        
#     else:
#         # --- Compute the boundaries with non-uniform spacing ---
#         # For the negative side: we want n_side bins ending at -zero_step.
#         neg_boundaries = np.linspace(vmin, -chosen_step, n_side)
#         neg_boundaries = np.append([neg_boundaries], [-zero_step])
        
#         # For the positive side: the white region ends at +zero_step.
#         # The first bin above white is from +zero_step to (zero_step + (chosen_step - zero_step))
#         pos_boundaries = np.linspace(chosen_step, vmax, n_side)
#         pos_boundaries = np.append([zero_step],[pos_boundaries])
    
#         # Combine the boundaries for the full colorbar.
#         # Total boundaries count will be (n_side+1) from the negative side plus (n_side+1) from the positive side
#         boundaries = np.append(neg_boundaries,pos_boundaries)
    
#     # --- Adjust the colormap ---
#     # Get the base cmocean 'balance' colormap and discretize it into chosen_n segments.
#     base_cmap = cmocean.cm.balance
#     # Sample chosen_n colors (they will be applied to the bins defined by boundaries).
#     newcolors = base_cmap(np.linspace(0, 1, chosen_n))
    
#     # Overwrite the middle color (the one for values close to zero) to white.
#     mid_index = chosen_n // 2  # integer division gives the center index
#     newcolors[mid_index] = [1, 1, 1, 1]  # RGBA for white
    
#     # Build a new discrete colormap.
#     disc_cmap = mcolors.LinearSegmentedColormap.from_list('discrete_balance', newcolors, N=chosen_n)
    
#     # --- Create a norm with the non-uniform boundaries ---
#     norm = mcolors.BoundaryNorm(boundaries, ncolors=chosen_n)

#     # extend parameter
#     if min_val < vmin and max_val > vmax:
#         extend = 'both'
#     elif min_val < vmin:
#         extend = 'min'
#     elif max_val > vmax:
#         extend = 'max'
#     else:
#         extend = 'neither'

#     if nonlinear:
#         n_ticks_side = int(np.log10(max_mag) - np.log10(zero_step)) + 1
        
#         if chosen_n < 12:
#             pos_ticks = np.geomspace(zero_step, vmax, n_ticks_side)
#             neg_ticks = -pos_ticks[::-1]
#             tick_positions = np.sort(np.concatenate((neg_ticks, pos_ticks)))
#         else:
#             pos_ticks = np.geomspace(zero_step, vmax, n_ticks_side)[1:]
#             neg_ticks = -pos_ticks[::-1]
#             zero_ticks = [0]
#             tick_positions = np.sort(np.concatenate((neg_ticks, zero_ticks, pos_ticks)))
            
            
#     else:
#         # tick positions
#         if chosen_n < 12:
#             neg_ticks = neg_boundaries
#             pos_ticks = pos_boundaries
#             tick_positions = np.sort(np.concatenate((neg_ticks, pos_ticks)))
            
#         elif (len(neg_boundaries)) % 2 == 0:
#             neg_ticks = neg_boundaries[::2]
#             # neg_ticks = neg_ticks[::2]
#             pos_ticks = pos_boundaries[1::2]
#             # pos_ticks = pos_ticks[::2]
#             tick_positions = np.sort(np.concatenate((neg_ticks, pos_ticks)))
            
#         elif (len(neg_boundaries)) % 2 == 1:
#             neg_ticks = neg_boundaries[:-1]
#             neg_ticks = neg_ticks[::2]
            
#             pos_ticks = pos_boundaries[2:]
#             pos_ticks = pos_ticks[::2]
    
#             zero_ticks = [0]
#             tick_positions = np.sort(np.concatenate((neg_ticks, zero_ticks, pos_ticks)))

#     # For debugging you can print the computed tick positions:
#     # print("Tick positions:", tick_positions)

#     return zero_step, disc_cmap, norm, boundaries, extend, tick_positions


def create_cb_params(max_mag, min_val, max_val, chosen_n, chosen_step,
                     verbose=False, nonlinear=False, lin_thresh=1e-8):
    vmin, vmax = -max_mag, max_mag

    if verbose:
        print(f"Final chosen_n: {chosen_n}, step: {chosen_step}, vmin={vmin}, vmax={vmax}")

    # Force odd number so 0 sits at the center (kept for the linear case)
    if chosen_n % 2 == 0:
        chosen_n += 1

    if nonlinear:
        # --------- NEW: continuous symmetric-log normalization ----------
        # continuous diverging cmap centered on 0
        disc_cmap = cmocean.cm.balance

        # SymLogNorm handles negative values; 'linthresh' controls the width
        # of the linear region around 0. 'base=10' gives decade-like behavior.
        norm = mcolors.SymLogNorm(linthresh=lin_thresh, linscale=1.0,
                                  vmin=vmin, vmax=vmax, base=10)

        # choose nice symmetric tick positions at decades with 0 in the middle
        lo_exp = int(np.floor(np.log10(lin_thresh)))
        hi_exp = int(np.floor(np.log10(max_mag)))
        pos_ticks = 10.0 ** np.arange(lo_exp, hi_exp + 1)
        neg_ticks = -pos_ticks[::-1]
        tick_positions = np.concatenate([neg_ticks, [0.0], pos_ticks])

        # matplotlib decides bar extension from data vs. vmin/vmax
        if min_val < vmin and max_val > vmax:
            extend = 'both'
        elif min_val < vmin:
            extend = 'min'
        elif max_val > vmax:
            extend = 'max'
        else:
            extend = 'neither'

        # boundaries are not needed for a continuous norm; keep None
        boundaries = None

        # zero_step is meaningful only for the discrete case; keep for API
        zero_step = lin_thresh

        return zero_step, disc_cmap, norm, boundaries, extend, tick_positions

    # ----------------------- ORIGINAL LINEAR (discrete) PATH -----------------------
    # (unchanged from your code)
    zero_step = chosen_step / 3
    n_side = (chosen_n - 1) // 2

    neg_boundaries = np.linspace(vmin, -chosen_step, n_side)
    neg_boundaries = np.append([neg_boundaries], [-zero_step])

    pos_boundaries = np.linspace(chosen_step, vmax, n_side)
    pos_boundaries = np.append([zero_step], [pos_boundaries])

    boundaries = np.append(neg_boundaries, pos_boundaries)

    base_cmap = cmocean.cm.balance
    newcolors = base_cmap(np.linspace(0, 1, chosen_n))
    mid_index = chosen_n // 2
    newcolors[mid_index] = [1, 1, 1, 1]
    disc_cmap = mcolors.LinearSegmentedColormap.from_list('discrete_balance', newcolors, N=chosen_n)

    norm = mcolors.BoundaryNorm(boundaries, ncolors=chosen_n)

    if min_val < vmin and max_val > vmax:
        extend = 'both'
    elif min_val < vmin:
        extend = 'min'
    elif max_val > vmax:
        extend = 'max'
    else:
        extend = 'neither'

    if chosen_n < 12:
        neg_ticks = neg_boundaries
        pos_ticks = pos_boundaries
        tick_positions = np.sort(np.concatenate((neg_ticks, pos_ticks)))
    elif (len(neg_boundaries)) % 2 == 0:
        neg_ticks = neg_boundaries[::2]
        pos_ticks = pos_boundaries[1::2]
        tick_positions = np.sort(np.concatenate((neg_ticks, pos_ticks)))
    else:
        neg_ticks = neg_boundaries[:-1][::2]
        pos_ticks = pos_boundaries[2::2]
        tick_positions = np.sort(np.concatenate((neg_ticks, [0], pos_ticks)))

    return zero_step, disc_cmap, norm, boundaries, extend, tick_positions


def bathymetry_overlay(input_ds,plot_dat,fine_lat,basin_name,depth_var='deptho',bathy_pct=85,smoothing=0.5):

    """ Returns:
    zonal_pct_bathy: 1D array of depth
    lat_vals: true latitude values to plot against

    ax.fill_between(
        lat_vals,
        max_depth, # where max_depth is the z-limit of your plot
        zonal_pct_bathy,
        color='grey',
        zorder=20 # (optional)
    )
    
    """
    basin_mask = selecting_basins(input_ds, basin=basin_name)
    bathy_dat = input_ds[depth_var].where(basin_mask)

    zonal_pct_bathy = xr.apply_ufunc(
        lambda x: np.nanpercentile(x, bathy_pct),
        bathy_dat,
        input_core_dims=[["xh"]],
        vectorize=True,
        output_dtypes=[bathy_dat.dtype],
        dask="parallelized"  # Enable handling of chunked arrays
    )

    correct_lat = zonal_mean(input_ds['geolat'], input_ds)
    zonal_pct_bathy = zonal_pct_bathy.rename({'yh': 'true_lat'})
    zonal_pct_bathy = zonal_pct_bathy.assign_coords({'true_lat': correct_lat.values})
    zonal_pct_bathy = zonal_pct_bathy.sortby('true_lat')
    zonal_pct_bathy = zonal_pct_bathy.isel(true_lat=slice(0,-1))

    # print(f"Initial zonal_pct_bathy[0]: {zonal_pct_bathy[0]}")
    zonal_pct_bathy.values[0] = 0
    # print(f"Adjusted zonal_pct_bathy[0]: {zonal_pct_bathy[0]}")

    if smoothing != None:
        # Smooth the percentile bathymetry for a nicer appearance
        zonal_pct_bathy.values = gaussian_filter1d(zonal_pct_bathy.values, sigma=smoothing)

    # Overlay the smoothed topography as a filled region
    lat_vals = plot_dat['true_lat'].values
    # min_lat = plot_dat['true_lat'].min().item()
    # max_lat = plot_dat['true_lat'].max().item()
    # print(f"min_lat: {min_lat}\t max_lat: {max_lat}")

    zonal_pct_bathy = zonal_pct_bathy.interp(true_lat=fine_lat)

    return zonal_pct_bathy, lat_vals


# # Temperature and temp anomaly map plotting functions

def plot_pp_temp_diff(
    panel_title, pp_diff_da, z_idx, start_yr, end_yr,
    cb_max=None, hatching=False, icon=None,
    savefig=True, prefix=None, verbose=False,
    fast_preview=False,
    # NEW:
    ax=None,                # if provided, draw into this axes (no new fig)
    add_colorbar=True,       # allow wrapper to suppress per-panel colorbars if desired
    # NEW:
    return_cb_params=False,   # if True, return a spec for a figure-level colorbar
    cb_label="Temperature Anomaly ($\\degree$C)"  # label reused by wrapper
):
    depth = pp_diff_da.coords['z_l'].values[z_idx]
    diff_da = pp_diff_da.temp.isel(z_l=z_idx)

    # Normalize longitudes to [0, 360)
    diff_da = diff_da.assign_coords(geolon=((diff_da.geolon + 360) % 360))

    # Target grid
    lat_res = 3 * 210
    lon_res = 3 * 360
    target_lat = np.linspace(diff_da.geolat.min(), diff_da.geolat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res, endpoint=False) #added November

    ds_in = xr.Dataset({
        "lat": (["yh", "xh"], diff_da.geolat.values),
        "lon": (["yh", "xh"], diff_da.geolon.values),
    })
    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    regridder = xe.Regridder(ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False)
    diff_da_interp = regridder(diff_da)

    # Diagnostics & cb spacing decisions (unchanged)
    min_val = float(np.nanmin(diff_da.values))
    max_val = float(np.nanmax(diff_da.values))
    per0p5 = float(np.nanpercentile(diff_da.values, 0.5))
    per99p5 = float(np.nanpercentile(diff_da.values, 99.5))
    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    extra_tick_digits = False
    if cb_max is not None:
        if cb_max == 1:
            chosen_n = 20 # 10
        elif cb_max == 1.5:
            chosen_n = 20 # 12
        elif cb_max == 2:
            chosen_n = 20 # 10
        elif cb_max == 2.5:
            chosen_n = 20
        elif cb_max == 3:
            chosen_n = 20 # 12
        elif cb_max == 4:
            chosen_n = 20 # 16
        elif cb_max == 5:
            chosen_n = 20 # 10
        elif cb_max == 6:
            chosen_n = 12
        elif cb_max == 7.5:
            chosen_n = 12
        elif cb_max == 2.222:
            extra_tick_digits = True
            cb_max = 2
            chosen_n = 12
        elif cb_max == 3.333:
            cb_max = 3
            chosen_n = 12
        elif cb_max == 4.444:
            extra_tick_digits = True
            cb_max = 4
            chosen_n = 12
        elif cb_max == 4.5:
            chosen_n = 12
        else:
            raise ValueError("cb_max is not an acceptable value.")
        data_max = cb_max
        chosen_step = 2 * data_max / chosen_n
    else:
        chosen_n, chosen_step = get_cb_spacing(
            per0p5, per99p5, min_bnd=1.0, min_spacing=0.2, min_n=10, max_n=20, verbose=verbose
        )

    max_mag = 0.5 * chosen_n * chosen_step  # final ± range

    zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
        max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
    )

    # --- NEW: figure/axes management ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=209.5), 'facecolor': 'grey'}
        )

    diff_plot = diff_da_interp.plot(
        x='lon', y='lat', cmap=disc_cmap, norm=disc_norm,
        transform=ccrs.PlateCarree(),
        add_labels=False, add_colorbar=False, ax=ax
    )

    if hatching:
        hatch_mask = pp_diff_da.temp_hatch.isel(z_l=z_idx).isel(time=0)
    
        # Use 1/NaN mask: stipple where mask == 1
        # sel = (hatch_mask.values == 1)
    
        # # geolon / geolat are already 2D (same y,x as mask)
        # lon2d = diff_da['geolon'].values  # shape (y, x)
        # lat2d = diff_da['geolat'].values  # shape (y, x)
    
        # # (Optional sanity check)
        # # print(hatch_mask.shape, lon2d.shape, lat2d.shape)
    
        # ax.scatter(
        #     lon2d[sel],
        #     lat2d[sel],
        #     s=3,
        #     marker='.',
        #     alpha=0.5,
        #     transform=ccrs.PlateCarree(),
        # )
    
        # keep only every 3rd grid point in y and x
        step_y, step_x = 3, 3
        hatch_sub = hatch_mask.isel(yh=slice(0, None, step_y),
                                    xh=slice(0, None, step_x))
        lon_sub = diff_da['geolon'].isel(yh=slice(0, None, step_y),
                                         xh=slice(0, None, step_x))
        lat_sub = diff_da['geolat'].isel(yh=slice(0, None, step_y),
                                         xh=slice(0, None, step_x))

        #### STIPPLING ####
        sel = (hatch_sub.values == 1)   # or >0.5 etc.
    
        ax.scatter(
            lon_sub.values[sel],
            lat_sub.values[sel],
            s=2,              # smaller points
            marker='x',#'.',       # or ',' for tiny pixels
            color='k',
            alpha=0.4,       # more transparent
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        # #### HATCHING ####
        # # 3) Define hatch levels: 1 → hatch, NaN → ignored
        # hatch_levels = [0.5, 1.5]   # will draw hatch anywhere =1
        
        # # 4) Plot hatching
        # ax.contourf(
        #     lon_sub.values,
        #     lat_sub.values,
        #     hatch_sub.values,
        #     levels=hatch_levels,
        #     colors='none',
        #     hatches=['////'],     # change to e.g. '.' or '\\\\' if desired
        #     linewidth=0,
        #     transform=ccrs.PlateCarree(),
        #     zorder=3,
        # )

    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Colorbar (optional so wrapper can control layout)
    diff_cb = None
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, shrink=0.8, pad=0.04, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing='proportional'
        )
        tick_labels = []
        for val in tick_positions:
            if (np.abs(val) == 0.05 or np.abs(val) == 0.25):
                tick_labels.append(f"{val:.2f}")
            elif np.abs(val) == 0.125:
                tick_labels.append(f"{val:.3f}")
            elif extra_tick_digits:
                tick_labels.append(f"{val:.2f}")
            else:
                tick_labels.append(f"{val:.1f}")

        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.set_yticklabels(tick_labels)
        diff_cb.ax.tick_params(labelsize=10)
        diff_cb.set_label("Temperature Anomaly ($\\degree$C)", fontdict={'fontsize': 12})
        if zero_step < 0.1 or max_mag > 10 or extra_tick_digits:
            plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.2)
        else:
            plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.0)

    # --- NEW: package colorbar spec for the wrapper ---
    cb_params = None
    if return_cb_params:
        # Build tick labels the same way as panel bars
        tick_labels = []
        for val in tick_positions:
            if (np.abs(val) == 0.05 or np.abs(val) == 0.25):
                tick_labels.append(f"{val:.2f}")
            elif np.abs(val) == 0.125:
                tick_labels.append(f"{val:.3f}")
            elif extra_tick_digits:
                tick_labels.append(f"{val:.2f}")
            else:
                tick_labels.append(f"{val:.1f}")

        cb_params = dict(
            mappable=diff_plot,        # carries cmap+norm
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing='proportional',
            ticks=tick_positions,
            ticklabels=tick_labels,
            label=cb_label
        )

    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(imagebox, (0.95, 1.00), xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # Only save/close if this function created the figure
    if savefig and created_fig is not None:
        if fast_preview:
            image_dpi = 200
        else: 
            image_dpi = 600
        created_fig.savefig(
            f'{prefix}_dT_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}_z_{depth:.0f}.png',
            dpi=image_dpi, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params  # (cb_params is None unless requested)


def create_temp_diff_plots(diff_type,fig_dir,start_year,end_year,z_idx,
                           profiles = ['surf','therm','mid','bot'],
                           prof_strings = ["Surf","Therm","Mid","Bot"],
                           power_var_suff = ['0p1TW', '0p2TW', '0p3TW'],
                           power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                           dT_max=None,
                           hatching=False,
                           extra_prefix=None,
                           savefig=True,
                           extra_verbose=False):
    """
    Inputs:
    diff_type (str): one of
                    ['const-1860ctrl',
                    'doub-1860exp','doub-2xctrl','doub-1860ctrl',
                    'quad-1860exp','quad-4xctrl','quad-1860ctrl']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    z_idx (int): z-index for depth of temp anomalies to plot
    dT_max (int/float): input for plot_pp_temp_diff
    hatching (boolean): input for plot_pp_temp_diff
    extra_verbose (boolean): input for plot_pp_temp_diff
    """

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # control cases
    if diff_type == 'doub-1860exp':
        ds_name = f'doub_ctrl_{start_year}_{end_year}_diff'
        title_str = f"1pct2xCO\u2082 Control"
        fig_name = f"2xCO2-const-ctrl"
        fig_prefix = fig_dir+fig_name
        plot_pp_temp_diff(fig_prefix, title_str, myVars[ds_name], z_idx, \
                start_year, end_year, cb_max=dT_max, hatching=hatching, verbose=extra_verbose)
        print(f"Done {fig_name}.")
        
    elif diff_type == 'quad-1860exp':
        ds_name = f'quad_ctrl_{start_year}_{end_year}_diff_const_ctrl'
        title_str = f"1pct4xCO\u2082 Control"
        fig_name = f"4xCO2-const-ctrl"
        fig_prefix = fig_dir+fig_name
        plot_pp_temp_diff(fig_prefix, title_str, myVars[ds_name], z_idx, \
                start_year, end_year, cb_max=dT_max, hatching=hatching, verbose=extra_verbose)
        print(f"Done {fig_name}.")
        
    # perturbation cases
    for i, power_str in enumerate(power_strings):
        for j, prof in enumerate(profiles):

            if diff_type == 'const-1860ctrl':
                ds_root = f'const_{prof}_{power_var_suff[i]}_{start_year}_{end_year}_diff'
            elif (diff_type == 'doub-1860exp' or diff_type == 'doub-2xctrl' or diff_type == 'doub-1860ctrl'):
                ds_root = f'doub_{prof}_{power_var_suff[i]}_{start_year}_{end_year}_diff'
            elif (diff_type == 'quad-1860exp' or diff_type == 'quad-4xctrl' or diff_type == 'quad-1860ctrl'):
                ds_root = f'quad_{prof}_{power_var_suff[i]}_{start_year}_{end_year}_diff'
            
            if diff_type == 'const-1860ctrl':
                title_str = f"Const {prof_strings[j]} {power_str}"
                ds_name = ds_root
                fig_name = f"{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'doub-1860exp':
                title_str = f"1pct2xCO\u2082 — Const CO\u2082: {prof_strings[j]} {power_str}"
                ds_name = f'{ds_root}_1860'
                fig_name = f"2xCO2-const_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'doub-2xctrl':
                title_str = f"1pct2xCO\u2082 {prof_strings[j]} {power_str} — 1pct2xCO\u2082 Control"
                ds_name = f'{ds_root}_2xctrl'
                fig_name = f"2xCO2-2xctrl_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'doub-1860ctrl':
                title_str = f"1pct2xCO\u2082 {prof_strings[j]} {power_str} — Const CO\u2082 Control"
                ds_name = f'{ds_root}_const_ctrl'
                fig_name = f"2xCO2-const-ctrl_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'quad-1860exp':
                title_str = f"1pct4xCO\u2082 — Const CO\u2082: {prof_strings[j]} {power_str}"
                ds_name = f'{ds_root}_1860'
                fig_name = f"4xCO2-const_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'quad-4xctrl':
                title_str = f"1pct4xCO\u2082 {prof_strings[j]} {power_str} — 1pct4xCO\u2082 Control"
                ds_name = f'{ds_root}_4xctrl'
                fig_name = f"4xCO2-4xctrl_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'quad-1860ctrl':
                title_str = f"1pct4xCO\u2082 {prof_strings[j]} {power_str} — Const CO\u2082 Control"
                ds_name = f'{ds_root}_const_ctrl'
                fig_name = f"4xCO2-const-ctrl_{prof}_{power_var_suff[i]}"

            if extra_prefix != None:
                fig_prefix = fig_dir+extra_prefix+'_'+fig_name
            else:
                fig_prefix = fig_dir+fig_name

            plot_pp_temp_diff(title_str, myVars[ds_name], z_idx, start_year, end_year, \
                              cb_max=dT_max, hatching=hatching, 
                              icon=prof, prefix=fig_prefix, savefig=savefig, verbose=extra_verbose)

            print(f"Done {fig_name}.")


def plot_pp_temp_mean(prefix,title,pp_temp_da,z_idx,start_yr,end_yr,savefig=True,verbose=False):

    depth = pp_temp_da.coords['z_l'].values[z_idx]
    run_da = pp_temp_da.isel(z_l=z_idx)

    # Step 1: Normalize geolon to [0, 360) to avoid wraparound issues
    run_da = run_da.assign_coords(
        geolon=((run_da.geolon + 360) % 360)
    )
    
    # Step 2: Define target lat/lon grid resolution
    lat_res = 3 * 210  # e.g., 630 points from -76.75 to 89.75
    lon_res = 3 * 360  # e.g., 1080 points from 0 to 360
    
    target_lat = np.linspace(run_da.geolat.min(), run_da.geolat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res)
    
    # Step 3: Build source and target grid datasets
    ds_in = xr.Dataset({
        "lat": (["yh", "xh"], run_da.geolat.values),
        "lon": (["yh", "xh"], run_da.geolon.values),
    })
    
    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })
    
    # Step 4: Create the regridder (periodic=True for wrapping at 0/360)
    regridder = xe.Regridder(ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False)
    
    # Step 5: Apply the regridder to your data
    run_da_interp = regridder(run_da)
    
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
    
    run_plot = run_da_interp.plot(x='lon', y='lat',
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
    
    if savefig:
        plt.savefig(f'{prefix}_temp_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}_z_{depth:.0f}.png', dpi=600, bbox_inches='tight')
        plt.close()


# # Column-integrated OHC anomaly plotting

def plot_pp_ohc_anom(
    panel_title,
    ohc_da,
    start_yr,
    end_yr,
    cb_max=None,
    icon=None,
    savefig=True,
    prefix=None,
    verbose=False,
    fast_preview=False,
    ax=None,
    add_colorbar=True,
    return_cb_params=False,
    cb_label="OHC Anomaly (GJ m$^{-2}$)"
):
    """
    Plot a 2D map of column-integrated OHC anomaly.

    Parameters
    ----------
    ohc_da : xr.DataArray or xr.Dataset
        2D field with horizontal coords and optionally geolon/geolat.
        If Dataset, it must contain variable 'ohc_anom'.
    """
    if isinstance(ohc_da, xr.Dataset):
        if "ohc_anom" not in ohc_da:
            raise ValueError("Dataset input must contain variable 'ohc_anom'.")
        plot_da = ohc_da["ohc_anom"]
    else:
        plot_da = ohc_da

    # Handle time dimension automatically
    if "time" in plot_da.dims:
        if plot_da.sizes["time"] > 1:
            raise ValueError(
                f"plot_pp_ohc_anom expected a single time index, but got time dimension "
                f"of length {plot_da.sizes['time']}."
            )
        plot_da = plot_da.isel(time=0)

    if "lon" in plot_da.dims and "lat" in plot_da.dims:
        plot_da_interp = plot_da
    else:
        if "geolon" not in plot_da.coords or "geolat" not in plot_da.coords:
            raise ValueError(
                "ohc_da must either already be on a lat/lon grid or have 2D geolon/geolat coordinates."
            )

        plot_da = plot_da.assign_coords(geolon=((plot_da.geolon + 360) % 360))

        lat_res = 3 * 210
        lon_res = 3 * 360
        target_lat = np.linspace(float(plot_da.geolat.min()), float(plot_da.geolat.max()), lat_res)
        target_lon = np.linspace(0, 360, lon_res, endpoint=False)

        ds_in = xr.Dataset({
            "lat": (["yh", "xh"], plot_da.geolat.values),
            "lon": (["yh", "xh"], plot_da.geolon.values),
        })
        ds_out = xr.Dataset({
            "lat": (["lat"], target_lat),
            "lon": (["lon"], target_lon),
        })

        regridder = xe.Regridder(ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False)
        plot_da_interp = regridder(plot_da)
        
    # Diagnostics
    min_val = float(np.nanmin(plot_da.values))
    max_val = float(np.nanmax(plot_da.values))
    per0p5 = float(np.nanpercentile(plot_da.values, 0.5))
    per99p5 = float(np.nanpercentile(plot_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    extra_tick_digits = False
    chosen_n = 20

    if cb_max is not None:
        if not np.isscalar(cb_max) or not np.isfinite(cb_max) or cb_max <= 0:
            raise ValueError("cb_max must be a positive finite number.")
        data_max = float(cb_max)
        chosen_step = 2 * data_max / chosen_n
    else:
        data_max = max(abs(per0p5), abs(per99p5), 1.0)
        raw_step = 2 * data_max / chosen_n
        chosen_step = np.ceil(raw_step / 0.1) * 0.1

    max_mag = 0.5 * chosen_n * chosen_step

    zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
        max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
    )

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={'projection': ccrs.Robinson(central_longitude=209.5), 'facecolor': 'grey'}
        )

    diff_plot = plot_da_interp.plot(
        x='lon', y='lat',
        cmap=disc_cmap, norm=disc_norm,
        transform=ccrs.PlateCarree(),
        add_labels=False, add_colorbar=False, ax=ax
    )

    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    diff_cb = None
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, shrink=0.8, pad=0.04, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing='proportional'
        )

        tick_labels = []
        for val in tick_positions:
            if np.abs(val) in (0.05, 0.25):
                tick_labels.append(f"{val:.2f}")
            elif np.abs(val) == 0.125:
                tick_labels.append(f"{val:.3f}")
            elif extra_tick_digits:
                tick_labels.append(f"{val:.2f}")
            else:
                tick_labels.append(f"{val:.1f}")

        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.set_yticklabels(tick_labels)
        diff_cb.ax.tick_params(labelsize=10)
        diff_cb.set_label(cb_label, fontdict={'fontsize': 12})

        if zero_step < 0.1 or max_mag > 10 or extra_tick_digits:
            plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.2)
        else:
            plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.0)

    cb_params = None
    if return_cb_params:
        tick_labels = []
        for val in tick_positions:
            if np.abs(val) in (0.05, 0.25):
                tick_labels.append(f"{val:.2f}")
            elif np.abs(val) == 0.125:
                tick_labels.append(f"{val:.3f}")
            elif extra_tick_digits:
                tick_labels.append(f"{val:.2f}")
            else:
                tick_labels.append(f"{val:.1f}")

        cb_params = dict(
            mappable=diff_plot,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing='proportional',
            ticks=tick_positions,
            ticklabels=tick_labels,
            label=cb_label
        )

    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(imagebox, (0.95, 1.00), xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    if savefig and created_fig is not None:
        image_dpi = 200 if fast_preview else 600
        created_fig.savefig(
            f'{prefix}_OHC_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
            dpi=image_dpi,
            bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


# # Function for multi-plot figure

def plot_pp_grid(
    panels,
    suptitle=None,
    outfile=None,
    figsize=None,
    cbar_label=None,
    fast_preview=False,
    panel_func=None,          # e.g., plot_temp_diff_basin, plot_N2_diff_basin
    shared_cbar=True,
    default_projection=ccrs.Robinson(central_longitude=209.5),
    cb_max=None,
    hatching=None,
    **panel_func_kwargs,   # <- extra args for panel_func
):
    if panel_func is None:
        raise ValueError("panel_func must be provided.")

    n = len(panels)
    if n not in (2, 3, 4):
        raise ValueError("panels must have length 2, 3, or 4.")

    # Figure + gridspec
    if n == 2:
        figsize = figsize or (11, 2.75)
    elif n == 3:
        figsize = figsize or (11, 5.5)
    else:  # n == 4
        figsize = figsize or (11, 5.5)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=4/72, h_pad=4/72)

    # I THINK I SHOULD ADD A LITTLE HORIZONTAL PADDING
    if n == 2:
        gs = GridSpec(1, 2, figure=fig, wspace=0, hspace=0)
        slots = [(0, slice(0,1)), (0, slice(1,2))]
    elif n == 3:
        gs = GridSpec(2, 4, figure=fig, wspace=0, hspace=0)
        slots = [(0, slice(0,2)), (0, slice(2,4)), (1, slice(1,3))]
    else:
        gs = GridSpec(2, 2, figure=fig, wspace=0, hspace=0)
        slots = [(0, slice(0,1)), (0, slice(1,2)), (1, slice(0,1)), (1, slice(1,2))]

    # Axes
    ax_list = []
    for (r, c), panel_kwargs in zip(slots, panels):
        proj = panel_kwargs.get("projection", default_projection)
        if proj is None:
            ax = fig.add_subplot(gs[r, c])
        else:
            ax = fig.add_subplot(gs[r, c], projection=proj)
            ax.set_facecolor('grey')
        ax_list.append(ax)

    # Draw panels
    cb_spec = None
    extension_param = 'neither'
    per_panel_specs = []

    for ax, panel_kwargs in zip(ax_list, panels):
        # kw = dict(panel_kwargs)
        
        # Start with wrapper-level kwargs, then let per-panel kwargs override them
        kw = dict(panel_func_kwargs)
        kw.update(panel_kwargs)

        # Apply global cb_max only if panel didn't specify one
        if cb_max is not None and 'cb_max' not in kw:
            kw['cb_max'] = cb_max

        # Similarly for hatching
        if hatching is not None and 'hatching' not in kw:
            kw['hatching'] = hatching

        # kw.update({
        #     'ax': ax,
        #     'savefig': False,
        #     # IMPORTANT: per-panel colorbars only when not sharing
        #     'add_colorbar': (not shared_cbar),
        #     # Only ask panels for cb params when we will draw a shared bar
        #     'return_cb_params': shared_cbar,
        # })
        kw.update({
            'ax': ax,
            'savefig': False,
            'add_colorbar': False,
            'return_cb_params': True,
        })

        _, _mappable, _, cb_params = panel_func(**kw)

        if shared_cbar:
            # keep last panel's params as the template; merge 'extend'
            if cb_params is not None:
                cb_spec = cb_params
                if extension_param == 'neither':
                    extension_param = cb_spec.get('extend', 'neither')
                elif extension_param == 'min':
                    if cb_spec.get('extend') in ('both', 'max'):
                        extension_param = 'both'
                elif extension_param == 'max':
                    if cb_spec.get('extend') in ('both', 'min'):
                        extension_param = 'both'
        else:
            # for per-panel bars later
            if cb_params is not None:
                per_panel_specs.append((ax, cb_params))

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    # ---------------- Shared colorbar ----------------
    if shared_cbar and cb_spec is not None:
        cbar = fig.colorbar(
            cb_spec['mappable'],
            ax=ax_list,
            orientation='vertical',
            fraction=0.04, pad=0.03, shrink=0.8,
            boundaries=cb_spec.get('boundaries', None),
            norm=cb_spec['norm'],
            spacing=cb_spec.get('spacing', 'uniform'),
            extend=extension_param
        )

        ticks = np.asarray(cb_spec.get('ticks', []))
        if ticks.size:
            cbar.set_ticks(ticks)

        labels = cb_spec.get('ticklabels', None)
        if labels is not None:
            cbar.set_ticklabels(labels)
            for t in cbar.ax.get_yticklabels():
                t.set_ha("center")
                # t.set_x(1.5)
                # t.set_x(2.2)

        # cbar.ax.tick_params(labelsize=10)
        cbar.ax.tick_params(labelsize=10, pad=14)
        cbar.set_label(cbar_label or cb_spec.get('label', ''), fontsize=12,
                      rotation=270,labelpad=12) # added Jan. 21, 2026
        cbar.ax.yaxis.set_label_position("right") # added Jan. 21, 2026
        cbar.ax.yaxis.label.set_verticalalignment("center") # added Jan. 21, 2026

        # Optional legacy path: if a panel provided a scale exponent, draw ×10^k
        if cb_spec.get('scale_exponent') is not None:
            k = int(cb_spec['scale_exponent'])
            cbar.ax.text(
                1.12, 1.06, r"$\times$ $10^{%d}$" % k,
                transform=cbar.ax.transAxes, ha="left", va="bottom", fontsize=10
            )

    # ------------- Per-panel colorbars (only if not shared) -------------
    if not shared_cbar and per_panel_specs:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        fig.canvas.draw()
        for ax, spec in per_panel_specs:
            cax = ax.inset_axes([1.03, 0.0, 0.035, 1.0])
            # cax = inset_axes(
            #     ax, width="3%", height="80%", loc='center left',
            #     bbox_to_anchor=(1.03, 0.5, 1, 1), bbox_transform=ax.transAxes, borderpad=0
            # )
            cb = fig.colorbar(
                spec['mappable'], cax=cax, orientation='vertical',
                boundaries=spec.get('boundaries', None),
                norm=spec['norm'],
                spacing=spec.get('spacing', 'uniform'),
                extend=spec.get('extend', 'neither')
            )
            ticks = np.asarray(spec.get('ticks', []))
            if ticks.size:
                cb.set_ticks(ticks)
            labels = spec.get('ticklabels', None)
            if labels is not None:
                cb.set_ticklabels(labels)
                for t in cb.ax.get_yticklabels():
                    t.set_ha("center")
                    # t.set_x(1.5)
                    
            cb.ax.tick_params(labelsize=8, pad=12)
            cb.set_label(cbar_label or spec.get('label', ''), fontsize=10)

    # Save / return
    dpi_out = 200 if fast_preview else 600
    if outfile:
        fig.savefig(outfile, dpi=dpi_out)#, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, ax_list


# def plot_pp_grid(
#     panels,
#     suptitle=None,
#     outfile=None,
#     figsize=None,
#     cbar_label=None,
#     fast_preview=False,
#     panel_func=None,          # e.g., plot_temp_diff_basin, plot_N2_diff_basin
#     shared_cbar=True,
#     default_projection=ccrs.Robinson(central_longitude=209.5),
#     cb_max=None,
# ):
#     if panel_func is None:
#         raise ValueError("panel_func must be provided.")

#     n = len(panels)
#     if n not in (2, 3, 4, 5):
#         raise ValueError("panels must have length 2, 3, 4, or 5.")

#     # Figure + gridspec
#     if n == 2:
#         figsize = figsize or (11, 2.25)
#     elif n == 3:
#         figsize = figsize or (11, 5.5)
#     elif n == 4:
#         figsize = figsize or (11, 5.5)
#     elif n == 5:
#         figsize = figsize or (11, 8.0)

#     fig = plt.figure(figsize=figsize, constrained_layout=True)
#     fig.set_constrained_layout_pads(w_pad=4/72, h_pad=4/72, wspace=0.0, hspace=0.0)

#     if n == 2:
#         # gs = GridSpec(1, 2, figure=fig, wspace=0, hspace=0)
#         # slots = [(0, slice(0,1)), (0, slice(1,2))]
#         gs = GridSpec(1, 4, figure=fig, wspace=0, hspace=0)
#         slots = [(0, slice(0,2)), (0, slice(2,4))]
#     elif n == 3:
#         gs = GridSpec(2, 4, figure=fig, wspace=0, hspace=0)
#         slots = [(0, slice(0,2)), (0, slice(2,4)), (1, slice(1,3))]
#     elif n == 4:
#         # gs = GridSpec(2, 2, figure=fig, wspace=0, hspace=0)
#         # slots = [(0, slice(0,1)), (0, slice(1,2)), (1, slice(0,1)), (1, slice(1,2))]
#         gs = GridSpec(2, 4, figure=fig, wspace=0, hspace=0)
#         slots = [(0, slice(0,2)), (0, slice(2,4)), (1, slice(0,2)), (1, slice(2,4))]
#     elif n == 5:
#         gs = GridSpec(3, 4, figure=fig, wspace=0, hspace=0)
#         slots = [(0, slice(0,2)), (0, slice(2,4)), (1, slice(0,2)), (1, slice(2,4)), (2, slice(1,3))]

#     # Axes
#     ax_list = []
#     for (r, c), panel_kwargs in zip(slots, panels):
#         proj = panel_kwargs.get("projection", default_projection)
#         if proj is None:
#             ax = fig.add_subplot(gs[r, c])
#         else:
#             ax = fig.add_subplot(gs[r, c], projection=proj)
#             ax.set_facecolor('grey')
#         ax_list.append(ax)

#     # Draw panels
#     cb_spec = None
#     extension_param = 'neither'
#     per_panel_specs = []

#     for ax, panel_kwargs in zip(ax_list, panels):
#         kw = dict(panel_kwargs)

#         # Apply global cb_max only if panel didn't specify one
#         if cb_max is not None and 'cb_max' not in kw:
#             kw['cb_max'] = cb_max

#         kw.update({
#             'ax': ax,
#             'savefig': False,
#             # IMPORTANT: per-panel colorbars only when not sharing
#             'add_colorbar': (not shared_cbar),
#             # Only ask panels for cb params when we will draw a shared bar
#             'return_cb_params': shared_cbar,
#         })

#         _, _mappable, _, cb_params = panel_func(**kw)

#         if shared_cbar:
#             # keep last panel's params as the template; merge 'extend'
#             if cb_params is not None:
#                 cb_spec = cb_params
#                 if extension_param == 'neither':
#                     extension_param = cb_spec.get('extend', 'neither')
#                 elif extension_param == 'min':
#                     if cb_spec.get('extend') in ('both', 'max'):
#                         extension_param = 'both'
#                 elif extension_param == 'max':
#                     if cb_spec.get('extend') in ('both', 'min'):
#                         extension_param = 'both'
#         else:
#             # for per-panel bars later
#             if cb_params is not None:
#                 per_panel_specs.append((ax, cb_params))

#     if suptitle:
#         fig.suptitle(suptitle, fontsize=14)

#     # ---------------- Shared colorbar ----------------
#     if shared_cbar and cb_spec is not None:
#         cbar = fig.colorbar(
#             cb_spec['mappable'],
#             ax=ax_list,
#             orientation='vertical',
#             fraction=0.04, pad=0.03, shrink=0.8,
#             boundaries=cb_spec.get('boundaries', None),
#             norm=cb_spec['norm'],
#             spacing=cb_spec.get('spacing', 'uniform'),
#             extend=extension_param
#         )

#         ticks = np.asarray(cb_spec.get('ticks', []))
#         if ticks.size:
#             cbar.set_ticks(ticks)

#         labels = cb_spec.get('ticklabels', None)
#         if labels is not None:
#             cbar.set_ticklabels(labels)

#         cbar.ax.tick_params(labelsize=10)
#         cbar.set_label(cbar_label or cb_spec.get('label', ''), fontsize=12)

#         # Optional legacy path: if a panel provided a scale exponent, draw ×10^k
#         if 'scale_exponent' in cb_spec:
#             k = int(cb_spec['scale_exponent'])
#             cbar.ax.text(
#                 1.12, 1.02, r"$\times 10^{%d}$" % k,
#                 transform=cbar.ax.transAxes, ha="left", va="bottom", fontsize=10
#             )

#     # ------------- Per-panel colorbars (only if not shared) -------------
#     if not shared_cbar and per_panel_specs:
#         from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#         fig.canvas.draw()
#         for ax, spec in per_panel_specs:
#             cax = inset_axes(
#                 ax, width="3%", height="80%", loc='center left',
#                 bbox_to_anchor=(1.03, 0.5, 1, 1), bbox_transform=ax.transAxes, borderpad=0
#             )
#             cb = fig.colorbar(
#                 spec['mappable'], cax=cax, orientation='vertical',
#                 boundaries=spec.get('boundaries', None),
#                 norm=spec['norm'],
#                 spacing=spec.get('spacing', 'uniform'),
#                 extend=spec.get('extend', 'neither')
#             )
#             ticks = np.asarray(spec.get('ticks', []))
#             if ticks.size:
#                 cb.set_ticks(ticks)
#             labels = spec.get('ticklabels', None)
#             if labels is not None:
#                 cb.set_ticklabels(labels)
#             cb.ax.tick_params(labelsize=8)
#             cb.set_label(cbar_label or spec.get('label', ''), fontsize=10)

#     # Save / return
#     dpi_out = 200 if fast_preview else 600
#     if outfile:
#         fig.savefig(outfile, dpi=dpi_out, bbox_inches='tight')
#         plt.close(fig)
#     else:
#         return fig, ax_list


# def plot_pp_grid_subfig(
#     panels,
#     suptitle=None,
#     outfile=None,
#     figsize=None,
#     cbar_label=None,
#     fast_preview=False,
#     panel_func=None,                  # used for the first n-1 panels (shared group)
#     last_panel_func=None,             # used for the last panel
#     default_projection=ccrs.Robinson(central_longitude=209.5),
#     cb_max=None,
#     shared_cbar_width_ratio=0.08,     # width of the RHS colorbar column (as a gridspec ratio)
# ):
#     """
#     Two-subfigure layout with shared colorbar on the RIGHT:

#       Subfigure A (top): first n-1 panels laid out in a 1x5 or 2x5 grid
#                          (first 4 columns for panels, 5th column for shared colorbar).
#       Subfigure B (bottom): last panel in a 1x5 grid
#                             (first 4 columns for the panel, 5th column is an invisible spacer).
#     => Because both subfigures have *identical* panel columns (4 equal columns) and
#        subfigure height ratios are [2,1], each panel has the SAME width and SAME height.

#     Each `panels[i]` is a dict of kwargs forwarded to the plotting function(s).
#     The plotting functions must accept:
#         ax=..., savefig=False, add_colorbar=..., return_cb_params=...
#     and return: (_, mappable, _, cb_params)
#       where cb_params contains at least 'mappable' and 'norm', plus optional:
#       'ticks', 'ticklabels', 'boundaries', 'spacing', 'extend', 'label', 'scale_exponent'.
#     """
#     if panel_func is None:
#         raise ValueError("panel_func must be provided.")

#     n = len(panels)
#     if n not in (2, 3, 4, 5):
#         raise ValueError("panels must have length 2, 3, 4, or 5.")

#     # Figure size with some headroom (you can tweak)
#     if n == 2:
#         figsize = figsize or (11, 2.75)
#     elif n == 3:
#         figsize = figsize or (11, 5.5)
#     elif n == 4:
#         figsize = figsize or (11, 5.5)
#     elif n == 5:
#         figsize = figsize or (11, 7.75)

#     # Parent figure with constrained layout
#     try:
#         fig = plt.figure(figsize=figsize, layout='constrained')  # mpl>=3.6
#     except TypeError:
#         fig = plt.figure(figsize=figsize, constrained_layout=True)
#     fig.set_constrained_layout_pads(w_pad=4/72, h_pad=6/72, wspace=0.0, hspace=0.0)

#     # Two subfigures; choose height ratios so each "row" height matches
#     sfA, sfB = fig.subfigures(2, 1, height_ratios=[2, 1])

#     # ---------------- SUBFIGURE A (shared group) ----------------
#     shared_count = n - 1
#     shared_panels = panels[:-1]

#     # rows: 1 if <=2 panels, else 2
#     rowsA = 1 if shared_count <= 2 else 2
#     # 5 columns: 4 for panels (equal), 1 narrow for the shared cbar on the right
#     wrA = [1, 1, 1, 1, shared_cbar_width_ratio]
#     gsA = sfA.add_gridspec(rowsA, 5, width_ratios=wrA)

#     # slots for the first 4 columns only (panels always span 2 of the first 4 columns)
#     if shared_count == 1:
#         slotsA = [(0, slice(1, 3))]
#     elif shared_count == 2:
#         slotsA = [(0, slice(0, 2)), (0, slice(2, 4))]
#     elif shared_count == 3:
#         slotsA = [(0, slice(0, 2)), (0, slice(2, 4)), (1, slice(1, 3))]
#     elif shared_count == 4:
#         slotsA = [(0, slice(0, 2)), (0, slice(2, 4)), (1, slice(0, 2)), (1, slice(2, 4))]
#     else:
#         raise ValueError("Internal: unsupported shared_count")

#     axA_list = []
#     for (r, c), pk in zip(slotsA, shared_panels):
#         proj = pk.get("projection", default_projection)
#         if proj is None:
#             ax = sfA.add_subplot(gsA[r, c])
#         else:
#             ax = sfA.add_subplot(gsA[r, c], projection=proj)
#             ax.set_facecolor('grey')
#         ax.set_anchor('W')  # keep consistent anchoring
#         axA_list.append(ax)

#     # Draw shared-group panels; collect shared colorbar params
#     shared_cb_spec = None
#     extension_param = 'neither'
#     for ax, panel_kwargs in zip(axA_list, shared_panels):
#         kw = dict(panel_kwargs)
#         if cb_max is not None and 'cb_max' not in kw:
#             kw['cb_max'] = cb_max
#         kw.update(dict(ax=ax, savefig=False, add_colorbar=False, return_cb_params=True))
#         _, _mappable, _, cb_params = panel_func(**kw)
#         if cb_params is not None:
#             shared_cb_spec = cb_params
#             ext = cb_params.get('extend', 'neither')
#             if extension_param == 'neither':
#                 extension_param = ext
#             elif extension_param == 'min' and ext in ('both', 'max'):
#                 extension_param = 'both'
#             elif extension_param == 'max' and ext in ('both', 'min'):
#                 extension_param = 'both'

#     # Shared VERTICAL colorbar in the 5th column, spanning all rows of sfA
#     if shared_cb_spec is not None and len(axA_list) > 0:
#         ax_cbarA = sfA.add_subplot(gsA[:, 4])  # dedicated column
#         ax_cbarA.set_in_layout(True)           # subfigure A owns this space

#         # Use the subfigure's colorbar if available (older MPL falls back to fig)
#         colorbar_callable = sfA.colorbar if hasattr(sfA, "colorbar") else fig.colorbar
#         cbarA = colorbar_callable(
#             shared_cb_spec['mappable'],
#             cax=ax_cbarA,
#             orientation='vertical',
#             boundaries=shared_cb_spec.get('boundaries', None),
#             norm=shared_cb_spec['norm'],
#             spacing=shared_cb_spec.get('spacing', 'uniform'),
#             extend=extension_param,
#         )
#         ticks = np.asarray(shared_cb_spec.get('ticks', []))
#         if ticks.size:
#             cbarA.set_ticks(ticks)
#         labels = shared_cb_spec.get('ticklabels', None)
#         if labels is not None:
#             cbarA.set_ticklabels(labels)
#         cbarA.ax.tick_params(labelsize=10)
#         cbarA.set_label(cbar_label or shared_cb_spec.get('label', ''), fontsize=12)

#     # ---------------- SUBFIGURE B (last panel + spacer column) ----------------
#     last_kwargs = panels[-1]
#     last_proj = last_kwargs.get("projection", default_projection)

#     # Same 5-column geometry so panel widths match Subfig A exactly; last column is a spacer
#     wrB = [1, 1, 1, 1, shared_cbar_width_ratio]
#     gsB = sfB.add_gridspec(1, 5, width_ratios=wrB)

#     # Place the last panel in the middle two panel columns -> same width as others
#     slotB = (0, slice(1, 3))
#     if last_proj is None:
#         axB = sfB.add_subplot(gsB[slotB])
#     else:
#         axB = sfB.add_subplot(gsB[slotB], projection=last_proj)
#         axB.set_facecolor('grey')
#     axB.set_anchor('W')

#     # Create an invisible spacer axis in column 4 for symmetry (keeps exact widths)
#     ax_spacer = sfB.add_subplot(gsB[:, 4])
#     ax_spacer.set_visible(False)
#     ax_spacer.set_in_layout(True)  # reserve the column so widths match subfig A

#     fn_last = last_panel_func or panel_func
#     kwB = dict(last_kwargs)
#     if cb_max is not None and 'cb_max' not in kwB:
#         kwB['cb_max'] = cb_max
#     kwB.update(dict(ax=axB, savefig=False, add_colorbar=False, return_cb_params=True))
#     _, _mappableB, _, cb_paramsB = fn_last(**kwB)

#     # Last panel's own colorbar (inset; ignored by layout; centered on the panel)
#     if cb_paramsB is not None:
#         caxB = inset_axes(
#             axB,
#             width="3%", height="80%",        # percent of the *anchor box* size
#             loc='center left',
#             bbox_to_anchor=(1.02, 0, 1, 1),# <-- non-zero w,h so % has meaning
#             bbox_transform=axB.transAxes,
#             borderpad=0
#         )
#         cbB = fig.colorbar(
#             cb_paramsB['mappable'],
#             cax=caxB,
#             orientation='vertical',
#             boundaries=cb_paramsB.get('boundaries', None),
#             norm=cb_paramsB['norm'],
#             spacing=cb_paramsB.get('spacing', 'uniform'),
#             extend=cb_paramsB.get('extend', 'neither'),
#         )
        
#         # keep the inset out of constrained_layout so panel sizes stay identical
#         caxB.set_in_layout(False)
#         cbB.ax.set_in_layout(False)
        
#         ticks = np.asarray(cb_paramsB.get('ticks', []))
#         if ticks.size:
#             cbB.set_ticks(ticks)
#         labels = cb_paramsB.get('ticklabels', None)
#         if labels is not None:
#             cbB.set_ticklabels(labels)
#         cbB.ax.tick_params(labelsize=9)
#         cbB.ax.set_ylabel(cbar_label or cb_paramsB.get('label', ''), fontsize=11)
#         cbB.ax.yaxis.set_label_position('right')
#         cbB.ax.yaxis.tick_right()

#     if suptitle:
#         fig.suptitle(suptitle, fontsize=14)

#     dpi_out = 200 if fast_preview else 600
#     if outfile:
#         fig.savefig(outfile, dpi=dpi_out, bbox_inches='tight')
#         plt.close(fig)
#     else:
#         return fig, (axA_list, axB)


from PIL import Image

def stitch_images_vertical_keep_size(img_top_path, img_bottom_path, out_path, pad=0, pad_color=(255, 255, 255, 255)):
    """
    Vertically stitches two PNG images together while preserving their original sizes.
    The narrower image is padded on the sides with pad_color to match widths.

    Parameters
    ----------
    img_top_path : str
        Path to the top image.
    img_bottom_path : str
        Path to the bottom image.
    out_path : str
        Output file path for the stitched image.
    pad : int, optional
        Vertical padding (in pixels) between the two images. Default 0.
    pad_color : tuple, optional
        Background color for padding and side fill. Default white (255,255,255,255).
    """

    # Load both images
    top = Image.open(img_top_path).convert("RGBA")
    bottom = Image.open(img_bottom_path).convert("RGBA")

    print(f"top.height: {top.height}\t top.width: {top.width}", flush=True)
    print(f"bottom.height: {bottom.height}\t bottom.width: {bottom.width}", flush=True)

    # Determine max width
    final_width = max(top.width, bottom.width)

    # Pad the narrower image (no resizing)
    if top.width < final_width:
        padded_top = Image.new("RGBA", (final_width, top.height), pad_color)
        x_offset = (final_width - top.width) // 2
        padded_top.paste(top, (x_offset, 0))
        top = padded_top

    if bottom.width < final_width:
        padded_bottom = Image.new("RGBA", (final_width, bottom.height), pad_color)
        x_offset = (final_width - bottom.width) // 2
        padded_bottom.paste(bottom, (x_offset, 0))
        bottom = padded_bottom

    # Compute final height
    total_height = top.height + pad + bottom.height

    # Create final image canvas
    stitched = Image.new("RGBA", (final_width, total_height), pad_color)

    # Paste both images with exact pixel heights preserved
    stitched.paste(top, (0, 0))
    stitched.paste(bottom, (0, top.height + pad))

    # Save output
    stitched.save(out_path)
    print(f"✅ Stitched image saved to: {out_path}")


from PIL import Image

def stitch_images_vertical_with_height_ratio(
    img_top_path,
    img_bottom_path,
    out_path,
    height_ratio=(1, 1),           # e.g., (2, 1) -> top twice the height of bottom
    pad=0,                          # vertical gap in pixels between the two images
    pad_color=(255, 255, 255),      # RGB pad/fill color (white default)
    avoid_upscale=True,             # if True, do not enlarge either image (downscale only)
    keep_alpha=False,               # if True, keep transparency; else flatten to RGB background
    png_compress_level=6,           # 0..9 (lower=faster, larger files)
    png_optimize=False              # True = smaller file but slower
):
    """
    Stitch two PNGs vertically with a *specified height ratio*, preserving aspect ratios.
    If scaled widths differ, the narrower one is padded left/right with `pad_color`.
    
    Guarantees:
      - Each image is resized ONLY by a single uniform scale factor (aspect preserved).
      - The final displayed heights satisfy: top:bottom = height_ratio[0] : height_ratio[1].
      - No width warping; horizontal sizes are equalized by side-padding, not stretching.
      - If avoid_upscale=True, we downscale at most one image so that no image is enlarged.

    Notes:
      - Height ratio is exact; scaling factors are chosen so that one image stays at 1×
        and the other is <= 1× when avoid_upscale=True.
      - If keep_alpha=False, images are flattened onto `pad_color` (useful for white padding).
    """
    r1, r2 = height_ratio
    if r1 <= 0 or r2 <= 0:
        raise ValueError("height_ratio values must be positive.")

    # --- Load images ---
    top_raw = Image.open(img_top_path)
    bottom_raw = Image.open(img_bottom_path)

    # Normalize color mode
    if keep_alpha:
        if top_raw.mode != "RGBA":    top_raw = top_raw.convert("RGBA")
        if bottom_raw.mode != "RGBA": bottom_raw = bottom_raw.convert("RGBA")
        mode = "RGBA"
        bg = pad_color + ((0,) if len(pad_color) == 3 else ())
    else:
        # Flatten any alpha onto solid background so padding is visible
        def flatten(im, bg_color):
            if im.mode in ("RGBA", "LA"):
                base = Image.new("RGB", im.size, bg_color)
                im = im.convert("RGBA")
                base.paste(im, (0, 0), im)
                return base
            return im.convert("RGB") if im.mode != "RGB" else im
        top_raw = flatten(top_raw, pad_color)
        bottom_raw = flatten(bottom_raw, pad_color)
        mode = "RGB"
        bg = pad_color

    w1, h1 = top_raw.width, top_raw.height
    w2, h2 = bottom_raw.width, bottom_raw.height

    # --- Compute scale factors s1, s2 so that (s1*h1) : (s2*h2) = r1 : r2 ---
    # Ratio constraint -> s1/s2 = (r1*h2)/(r2*h1) = q
    q = (r1 * h2) / (r2 * h1)

    if avoid_upscale:
        # Keep the larger scale at 1.0, the other <= 1.0 (downscale-only).
        if q >= 1.0:
            s1 = 1.0
            s2 = 1.0 / q
        else:
            s1 = q
            s2 = 1.0
    else:
        # Let bottom be the anchor; scale top to satisfy the ratio exactly.
        s2 = 1.0
        s1 = q

    # --- Pick resampling filters (fast & sensible) ---
    def pick_resample(scale):
        # Downscale heavy -> BOX; modest -> BILINEAR; Upscale -> LANCZOS
        if scale < 0.7:
            return Image.Resampling.BOX
        elif scale <= 1.0:
            return Image.Resampling.BILINEAR
        else:
            return Image.Resampling.LANCZOS

    # --- Resize uniformly per image (aspect preserved) ---
    new_h1 = max(1, int(round(h1 * s1)))
    new_h2 = max(1, int(round(h2 * s2)))
    new_w1 = max(1, int(round(w1 * s1)))
    new_w2 = max(1, int(round(w2 * s2)))

    if (new_w1, new_h1) != (w1, h1):
        top_resized = top_raw.resize((new_w1, new_h1), pick_resample(s1))
    else:
        top_resized = top_raw

    if (new_w2, new_h2) != (w2, h2):
        bottom_resized = bottom_raw.resize((new_w2, new_h2), pick_resample(s2))
    else:
        bottom_resized = bottom_raw

    # Height ratio is now exact by construction:
    #   new_h1 / new_h2 == r1 / r2  (up to rounding)

    # --- Equalize widths by padding the narrower image (no stretching) ---
    final_width = max(top_resized.width, bottom_resized.width)

    def pad_to_width(img, target_w, bg_color):
        if img.width == target_w:
            return img
        x_left = (target_w - img.width) // 2
        canvas = Image.new(mode, (target_w, img.height), bg_color)
        canvas.paste(img, (x_left, 0))
        return canvas

    top_padded    = pad_to_width(top_resized, final_width, bg)
    bottom_padded = pad_to_width(bottom_resized, final_width, bg)

    # --- Compose final canvas ---
    total_height = top_padded.height + pad + bottom_padded.height
    stitched = Image.new(mode, (final_width, total_height), bg)
    stitched.paste(top_padded, (0, 0))
    stitched.paste(bottom_padded, (0, top_padded.height + pad))

    # --- Save ---
    save_kwargs = dict(optimize=png_optimize, compress_level=png_compress_level)
    stitched.save(out_path, **save_kwargs)
    # Optionally return sizes for verification
    return {
        "top_scaled": (top_resized.width, top_resized.height),
        "bottom_scaled": (bottom_resized.width, bottom_resized.height),
        "final_size": (final_width, total_height),
        "width_padding_applied": (
            final_width - top_resized.width,
            final_width - bottom_resized.width
        )
    }


# # Function for plotting horizontal mean anomaly profiles

def plot_horiz_mean_diff(diff_type,fig_dir,start_year,end_year,anom_var="temp",xlabel="Temperature Anomaly ($\degree$C)",
                         omit_title=True,
                         x_bounds=[None,None],
                         xlims_dict = {'0p05TW': (-0.5,1.0),'0p1TW': (-0.8,1.2), 
                                       '0p2TW': (-1.2,2.0), '0p3TW': (-1.6,2.6), 
                                       '0p5TW':(-2,3.4)},
                       max_z = 6000,
                       profiles = ['surf','therm','mid','bot'], 
                       power_inputs = ['0.1TW', '0.2TW', '0.3TW'], 
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                         prefix=None
                       ):
    """
    Inputs:
    diff_type (str): one of
                    ['const-1860ctrl',
                    'doub-1860exp','doub-2xctrl','doub-1860ctrl',
                    'quad-1860exp','quad-4xctrl','quad-1860ctrl']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    """

    prof_dict = {'surf': 'b',
                 'therm': 'm',
                 'mid': 'g',
                 'bot': 'r'}

    # separate plot for each power input
    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(4, 4.8)) #original (5,6)
        
        ds_root = f'{power}_{start_year}_{end_year}_diff'
        for prof in profiles:
            if diff_type == 'const-1860ctrl':
                ds_name = f'const_{prof}_{ds_root}_mean'
            elif diff_type == 'doub-1860exp':
                ds_name = f'doub_{prof}_{ds_root}_1860_mean'
            elif diff_type == 'doub-2xctrl':
                ds_name = f'doub_{prof}_{ds_root}_2xctrl_mean'
            elif diff_type == 'doub-1860ctrl':
                ds_name = f'doub_{prof}_{ds_root}_const_ctrl_mean'
            elif diff_type == 'quad-1860exp':
                ds_name = f'quad_{prof}_{ds_root}_1860_mean'
            elif diff_type == 'quad-4xctrl':
                ds_name = f'quad_{prof}_{ds_root}_4xctrl_mean'
            elif diff_type == 'quad-1860ctrl':
                ds_name = f'quad_{prof}_{ds_root}_const_ctrl_mean'

            depth = myVars[ds_name]['z_l']
            ax.plot(myVars[ds_name][anom_var][0,:],depth,label=prof,color=prof_dict[prof])
            
        ax.set_xlabel(xlabel, fontdict={'fontsize': 12})
        ax.set_ylabel("Depth (m)", fontdict={'fontsize': 12})
        ax.tick_params(labelsize=10)
        if x_bounds == [None,None]:
            if start_year == 176:
                ax.set_xlim(xlims_dict[power])
        else:
            ax.set_xlim(x_bounds)
        ax.set_ylim(0,max_z)
        ax.invert_yaxis()
        ax.legend(loc='lower right', fontsize=10)
        ax.grid("both")
    
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

        if diff_type == 'const-1860ctrl':
            title_str = f"Const CO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"const_{power}"
            
        elif diff_type == 'doub-1860exp':
            title_str = f"Radiative $\Delta T$: 1pct2xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-const_{power}"
            
        elif diff_type == 'doub-2xctrl':
            title_str = f"Mixing $\Delta T$: 1pct2xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-2xctrl_{power}"
            
        elif diff_type == 'doub-1860ctrl':
            title_str = f"Total $\Delta T$: 1pct2xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-const-ctrl_{power}"
            
        elif diff_type == 'quad-1860exp':
            title_str = f"Radiative $\Delta T$: 1pct4xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-const_{power}"
            
        elif diff_type == 'quad-4xctrl':
            title_str = f"Mixing $\Delta T$: 1pct4xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-4xctrl_{power}"
            
        elif diff_type == 'quad-1860ctrl':
            title_str = f"Total $\Delta T$: 1pct4xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-const-ctrl_{power}"

        if omit_title is False:
            ax.set_title(title_str+f"Year {start_year} to {end_year}",fontdict={'fontsize':14})
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if prefix != None:
            save_path = fig_dir + prefix + f'_mean_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf'
        else:
            save_path = fig_dir + f'_mean_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf'
        
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
    
    # plot for all power inputs
    fig, ax = plt.subplots(figsize=(4, 4.8))
    line_list = ['solid','dashed','dotted']

    for pow_idx, power_str in enumerate(power_strings):
        ds_root = f'{power_var_suff[pow_idx]}_{start_year}_{end_year}_diff'
        for prof in profiles:
            if diff_type == 'const-1860ctrl':
                ds_name = f'const_{prof}_{ds_root}_mean'
            elif diff_type == 'doub-1860exp':
                ds_name = f'doub_{prof}_{ds_root}_1860_mean'
            elif diff_type == 'doub-2xctrl':
                ds_name = f'doub_{prof}_{ds_root}_2xctrl_mean'
            elif diff_type == 'doub-1860ctrl':
                ds_name = f'doub_{prof}_{ds_root}_const_ctrl_mean'
            elif diff_type == 'quad-1860exp':
                ds_name = f'quad_{prof}_{ds_root}_1860_mean'
            elif diff_type == 'quad-4xctrl':
                ds_name = f'quad_{prof}_{ds_root}_4xctrl_mean'
            elif diff_type == 'quad-1860ctrl':
                ds_name = f'quad_{prof}_{ds_root}_const_ctrl_mean'

            depth = myVars[ds_name]['z_l']
            ax.plot(myVars[ds_name][anom_var][0,:],depth,label=f'{power_str} {prof}',linestyle=line_list[pow_idx],color=prof_dict[prof])
        
    ax.set_xlabel(xlabel, fontdict={'fontsize': 12})
    ax.set_ylabel("Depth (m)", fontdict={'fontsize': 12})
    ax.tick_params(labelsize=10)
    if x_bounds == [None,None]:
        if start_year == 176:
            ax.set_xlim(xlims_dict['0p3TW'])
    else:
        ax.set_xlim(x_bounds)
    
    ax.set_ylim(0,max_z)
    ax.invert_yaxis()

    # Define custom legend entries
    leg_1 = [  # First column (4 labels)
        Line2D([0], [0], color='b', lw=2),  # surf
        Line2D([0], [0], color='m', lw=2),  # therm
        Line2D([0], [0], color='g', lw=2),  # mid
        Line2D([0], [0], color='r', lw=2),  # bot
    ]
    labels_1 = ['surf', 'therm', 'mid', 'bot']
    
    leg_2 = [Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
            Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
            Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
    labels_2 = power_strings

    # First legend (4 labels)
    legend1 = ax.legend(
        leg_1, labels_1,
        loc='lower right',
        fontsize=10, labelspacing=0.1,
        bbox_to_anchor=(1.0, 0.17),  # Adjust position as needed
        frameon=True
    )
    # Second legend (3 labels, positioned below the first)
    legend2 = ax.legend(
        leg_2, labels_2,
        loc='lower right',
        fontsize=10, labelspacing=0.1,
        bbox_to_anchor=(1.0, 0.0),  # Adjust position as needed
        frameon=True
    )
    
    # Add the first legend back to the axis
    ax.add_artist(legend1)
    
    ax.grid("both")
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

    if diff_type == 'const-1860ctrl':
        title_str = f"Const CO\u2082 Cases\n"
        fig_name = f"const"
        
    elif diff_type == 'doub-1860exp':
        title_str = f"Radiative $\Delta T$: 1pct2xCO\u2082 Cases\n"
        fig_name = f"2xCO2-const"
        
    elif diff_type == 'doub-2xctrl':
        title_str = f"Mixing $\Delta T$: 1pct2xCO\u2082 Cases\n"
        fig_name = f"2xCO2-2xctrl"
        
    elif diff_type == 'doub-1860ctrl':
        title_str = f"Total $\Delta T$: 1pct2xCO\u2082 Cases\n"
        fig_name = f"2xCO2-const-ctrl"
        
    elif diff_type == 'quad-1860exp':
        title_str = f"Radiative $\Delta T$: 1pct4xCO\u2082 Cases\n"
        fig_name = f"4xCO2-const"
        
    elif diff_type == 'quad-4xctrl':
        title_str = f"Mixing $\Delta T$: 1pct4xCO\u2082 Cases\n"
        fig_name = f"4xCO2-4xctrl"
        
    elif diff_type == 'quad-1860ctrl':
        title_str = f"Total $\Delta T$: 1pct4xCO\u2082 Cases\n"
        fig_name = f"4xCO2-const-ctrl"

    if omit_title is False:
        ax.set_title(title_str+f"Year {start_year} to {end_year}",fontdict={'fontsize':14})

    if prefix != None:
        save_path = fig_dir + prefix + f'_mean_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf'
    else:
        save_path = fig_dir + f'_mean_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf'
        
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


# # Functions for basin means

def transform_depth(z, max_depth, axis_split):
    """
    Custom transformation for depth axis:
    - Expands top portion of plot
    - Compresses lower depths
    """
    compress_factor = axis_split/(max_depth - axis_split)
    
    # return np.where(z <= 1000, z, 1000 + (z - 1000) * 0.2)
    return np.where(z <= axis_split, z, axis_split + (z - axis_split) * compress_factor)


def refine_grid(coord, factor=3):
    """Return a refined version of a 1D coordinate array by inserting points between original values."""
    refined = []
    for i in range(len(coord) - 1):
        start = coord[i]
        end = coord[i + 1]
        refined += list(np.linspace(start, end, factor + 1)[:-1])  # skip endpoint to avoid duplication
    refined.append(coord[-1])  # include the last original point
    return np.array(refined)


def get_depth_labels(axis_split, max_depth):
    
    # Define original depth values and their transformed positions
    if axis_split == None:
        if (max_depth == 4000 or max_depth == 5000 or max_depth == 6000):
            depth_labels = np.arange(0,max_depth+1000,1000)
        else:
            raise ValueError(f"{max_depth} not an acceptable value.")
    elif axis_split == 1000:
        if max_depth >= 6000:
            depth_labels = [0, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000]
        elif max_depth >= 5000:
            depth_labels = [0, 250, 500, 750, 1000, 2000, 3000, 4000, 5000]
        else:
            raise ValueError(f"{max_depth} not an acceptable value.")
    elif axis_split == 1500:
        if max_depth >= 6000:
            depth_labels = [0, 500, 1000, 1500, 2500, 3500, 4500, 5500]
        elif max_depth >= 5000:
            depth_labels = [0, 500, 1000, 1500, 2000, 3000, 4000, 5000]
        else:
            raise ValueError(f"{max_depth} not an acceptable value.")
    elif axis_split == 1600:
        if max_depth >= 6000:
            depth_labels = [0, 400, 800, 1200, 1600, 2000, 2800, 3600, 4400, 5200, 6000]
        elif max_depth >= 5000:
            depth_labels = [0, 400, 800, 1200, 1600, 2000, 2800, 3600, 4400, 5200]
        else:
            raise ValueError(f"{max_depth} not an acceptable value.")
    elif axis_split == 2000:
        if max_depth >= 6000:
            depth_labels = [0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]
        elif max_depth >= 5000:
            depth_labels = [0, 500, 1000, 1500, 2000, 3000, 4000, 5000]
        else:
            raise ValueError(f"{max_depth} not an acceptable value.")
    elif axis_split == 3000:
        if max_depth >= 6000:
            depth_labels = [0, 1000, 2000, 3000, 4000, 5000, 6000]
        elif max_depth >= 5000:
            depth_labels = [0, 1000, 2000, 3000, 4000, 5000]
        else:
            raise ValueError(f"{max_depth} not an acceptable value.")
    else:
        raise ValueError(f"{axis_split} not an acceptable value. Must be in list [1000, 1500, 1600, 2000, 3000]")

    if axis_split == None:
        depth_positions = np.arange(0, max_depth+1000, 1000)
    else:
        depth_positions = [transform_depth(d, max_depth, axis_split) for d in depth_labels]

    return depth_positions, depth_labels


def get_basin_xlims(basin_name, MOC_override=False):
    """ X limits & labels in degrees with N/S suffixes. """
    
    if basin_name == 'global':
        xmin, xmax, xstep = -77, 77, 20
    elif basin_name == 'atl':
        xmin, xmax, xstep = -77, 66, 20
    elif basin_name == 'atl-arc' or basin_name == 'atl-arc-no-marg':
        if MOC_override:
            xmin, xmax, xstep = -30, 77, 20
        else:
            xmin, xmax, xstep = -77, 77, 20
    elif basin_name == 'pac':
        xmin, xmax, xstep = -77, 63, 20
    elif basin_name == 'indopac':
        xmin, xmax, xstep = -30, 63, 20
    elif basin_name == 'antarc':
        xmin, xmax, xstep = -77, -33, 20
    else:
        xmin, xmax, xstep = -80, 80, 20

    # Major ticks (labeled every 20°)
    if basin_name == 'indopac':
        xticks_major = np.arange(-20, 61, xstep)
    elif (basin_name == 'atl-arc' and MOC_override) or (basin_name == 'atl-arc-no-marg' and MOC_override):
        xticks_major = np.arange(-20, 61, xstep)
    else:
        xticks_major = np.arange(-60, 61, xstep)
    xlabels_major = [
        (f"{abs(v)}$\\degree$S" if v < 0 else ("0$\\degree$" if v == 0 else f"{v}$\\degree$N"))
        for v in xticks_major
    ]

    # Minor ticks (every 5°, no labels)
    if basin_name == 'indopac':
        xticks_minor = np.arange(-30, xmax + 1, 5)
    elif (basin_name == 'atl-arc' and MOC_override) or (basin_name == 'atl-arc-no-marg' and MOC_override):
        xticks_minor = np.arange(-30, xmax + 1, 5)
    else:
        xticks_minor = np.arange(-75, xmax + 1, 5)
    
    return xmin, xmax, xticks_major, xlabels_major, xticks_minor


# ## Basin mean anomaly functions (T, S, N2, rhopot2)

# def plot_temp_diff_basin(
#     panel_title,               # renamed from `title` to match panel API
#     diff_ds, basin_name, max_depth, axis_split, start_yr, end_yr,
#     icon=None,
#     check_nn=False, nn_threshold=0.05, cb_max=None, mask_dataset=None,
#     run_ds=None,  # must be passed to plot overlays
#     savefig=True, fig_dir=None, prefix=None, verbose=False,
#     # NEW (wrapper hooks):
#     ax=None,                  # if provided, draw into this axes (no new fig)
#     add_colorbar=True,        # allow wrapper to suppress per-panel colorbars
#     return_cb_params=False,   # if True, return a spec for a figure-level colorbar
#     cb_label="Temperature Anomaly ($\\degree$C)",  # used by wrapper if shared
#     # BACKWARD-COMPAT alias (optional):
#     title=None                # if someone still passes old name
# ):
#     # ---- title alias (back-compat) ----
#     if panel_title is None and title is not None:
#         panel_title = title

#     # --- data prep (unchanged) ---
#     # lat_res = 210 * 3
#     # z_res   = 33 * 3

#     if len(diff_ds.time.values) > 1:
#         raise ValueError("diff_ds cannot be a time series.")

#     diff_ds = diff_ds.isel(time=0)
#     diff_dat = get_pp_basin_dat(
#         diff_ds, "temp", basin_name, check_nn=check_nn,
#         nn_threshold=nn_threshold, mask_ds=mask_dataset
#     )
#     diff_dat = diff_dat.sel(z_l=slice(0, max_depth))

#     # Refine target grids
#     fine_lat   = refine_grid(diff_dat.true_lat.values, factor=10)
#     fine_depth = refine_grid(diff_dat.z_l.values,      factor=10)
#     if verbose:
#         print("fine_depth:\n", fine_depth)

#     # Interpolate onto refined grid
#     diff_dat = diff_dat.interp(true_lat=fine_lat, z_l=fine_depth)

#     # Optional depth-axis transform (split axis)
#     if axis_split is not None:
#         transformed_z = xr.apply_ufunc(
#             transform_depth, diff_dat.z_l,
#             kwargs={"max_depth": max_depth, "axis_split": axis_split}
#         )
#         diff_dat = diff_dat.assign_coords(z_l=transformed_z)

#     # Optional overlay (mean temperature contours)
#     overlay_dat = None
#     if run_ds is not None:
#         if len(run_ds.time.values) > 1:
#             raise ValueError("run_ds cannot be a time series.")
#         run_da = get_pp_basin_dat(
#             run_ds.isel(time=0), "temp", basin_name, check_nn=check_nn,
#             nn_threshold=nn_threshold, mask_ds=mask_dataset
#         ).sel(z_l=slice(0, max_depth))
#         ##### DEBUGGING SO COMMENTED OUT THIS STEP
#         overlay_dat = run_da.interp(true_lat=fine_lat, z_l=fine_depth)
#         # overlay_dat = run_da
#         if axis_split is not None:
#             overlay_transformed_z = xr.apply_ufunc(
#                 transform_depth, overlay_dat.z_l,
#                 kwargs={"max_depth": max_depth, "axis_split": axis_split}
#             )
#             overlay_dat = overlay_dat.assign_coords(z_l=overlay_transformed_z)

#     # --- diagnostics & colorbar spacing (mostly unchanged) ---
#     min_val = float(np.nanmin(diff_dat.values))
#     max_val = float(np.nanmax(diff_dat.values))
#     if verbose:
#         print(f"Min and max temp anomaly: {min_val:.2f}, {max_val:.2f}")

#     p0p5  = float(np.nanpercentile(diff_dat.values,  0.5))
#     p99p5 = float(np.nanpercentile(diff_dat.values, 99.5))
#     if verbose:
#         if abs(p0p5) > abs(p99p5):
#             print(f"0.5 to 99.5th percentile data max mag: {abs(p0p5):.3f}")
#         else:
#             print(f"0.5 to 99.5th percentile data max mag: {abs(p99p5):.3f}")

#     extra_tick_digits = False
#     if cb_max is not None:
#         if cb_max in (1, 1.5, 2, 2.5, 3, 4, 5, 6):
#             chosen_n = 20
#         elif cb_max == 7.5:
#             chosen_n = 12
#         elif cb_max == 2.222:
#             extra_tick_digits = True; cb_max = 2; chosen_n = 12
#         elif cb_max == 3.333:
#             cb_max = 3; chosen_n = 12
#         elif cb_max == 4.444:
#             extra_tick_digits = True; cb_max = 4; chosen_n = 12
#         elif cb_max == 4.5:
#             chosen_n = 12
#         else:
#             raise ValueError("cb_max is not an acceptable value.")
#         data_max   = cb_max
#         chosen_step = 2 * data_max / chosen_n
#     else:
#         chosen_n, chosen_step = get_cb_spacing(
#             p0p5, p99p5, min_bnd=1.0, min_spacing=0.1, min_n=10, max_n=20, verbose=verbose
#         )

#     max_mag = 0.5 * chosen_n * chosen_step  # final ± range

#     zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
#         max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
#     )

#     # --- figure/axes management (NEW) ---
#     created_fig = None
#     if ax is None:
#         created_fig, ax = plt.subplots(figsize=(7.5, 3))

#     # Draw the main cross-section (no subplot_kws needed since we have an Axes)
#     # diff_plot = diff_dat.plot(
#     #     x='true_lat', y='z_l',
#     #     cmap=disc_cmap, norm=disc_norm,
#     #     add_labels=False, add_colorbar=False, ax=ax,
#     #     edgecolors='face'
#     # )

#     diff_plot = diff_dat.plot(
#         x='true_lat', y='z_l',
#         cmap=disc_cmap, norm=disc_norm,
#         add_labels=False, add_colorbar=False, ax=ax,
#         infer_intervals=True,     # <- compute cell edges from centers
#         rasterized=True,          # <- avoids vector “stairs” on save
#         antialiased=False,        # <- no edge antialiasing lines
#         # edgecolors='none',      # (or drop edgecolors entirely)
#     )

#     # Axis direction and limits
#     ax.invert_yaxis()
#     if axis_split is None:
#         ax.set_ylim(max_depth, 0)
#     else:
#         y_top    = transform_depth(0,          max_depth, axis_split)
#         y_bottom = transform_depth(max_depth,  max_depth, axis_split)
#         ax.set_ylim(y_bottom, y_top)

#     # Keep spines/ticks above the quadmesh
#     for spine in ax.spines.values():
#         spine.set_zorder(30)
#     for tick in ax.get_xticklines():
#         tick.set_zorder(30)
#     for label in ax.get_xticklabels():
#         label.set_zorder(30)

#     # Y ticks (depth labels)
#     depth_positions, depth_labels = get_depth_labels(axis_split, max_depth)
#     ax.set_yticks(ticks=depth_positions, labels=[str(d) for d in depth_labels])
#     ax.set_ylabel('Depth (m)')

#     # Optional overlay contours (mean temperature)
#     if overlay_dat is not None:
#         contour_levels = [-2, 0, 2, 4, 6, 8, 10, 15, 20, 25, 30]
#         if verbose:
#             min_overlay_temp = float(np.nanmin(overlay_dat.values))
#             max_overlay_temp = float(np.nanmax(overlay_dat.values))
#             print(f"Min and max mean temp: {min_overlay_temp:.2f}, {max_overlay_temp:.2f}\n")
#         overlay_plot = ax.contour(
#             overlay_dat["true_lat"], overlay_dat["z_l"], overlay_dat,
#             levels=contour_levels, colors="k", linewidths=0.8
#         )
#         ax.clabel(overlay_plot, inline=True, fontsize=10)

#     # Bathymetry overlay
#     zonal_pct_bathy, lat_vals = bathymetry_overlay(diff_ds, diff_dat, fine_lat, basin_name)
#     if axis_split is not None:
#         zonal_pct_bathy = xr.apply_ufunc(
#             transform_depth, zonal_pct_bathy,
#             kwargs={"max_depth": max_depth, "axis_split": axis_split}
#         )
#     ax.fill_between(lat_vals, max_depth, zonal_pct_bathy, color='grey', zorder=20)
    
#     xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin_name)
#     ax.set_xlim(xmin, xmax)
#     ax.set_xticks(xticks_major)
#     ax.set_xticklabels(xlabels_major)
#     ax.set_xticks(xticks_minor, minor=True)
#     # Optional: make minor ticks smaller and unlabeled
#     ax.tick_params(axis='x', which='minor', length=4)
#     ax.tick_params(axis='x', which='major', length=6)

#     # Title
#     if created_fig is None:
#         ax.set_title(f"{panel_title}")
#     else:
#         ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

#     # Icon (optional)
#     if icon is not None:
#         image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
#         img = mpimg.imread(image_path)
#         imagebox = OffsetImage(img, zoom=0.08)
#         ab = AnnotationBbox(imagebox, (0.95, 1.10), xycoords="axes fraction", frameon=False)
#         ax.add_artist(ab)

#     # --- Colorbar (optional local bar so wrapper can turn it off) ---
#     diff_cb = None
#     tick_labels = []
#     for val in tick_positions:
#         if (abs(val) == 0.05 or abs(val) == 0.25):
#             tick_labels.append(f"{val:.2f}")
#         elif abs(val) == 0.125:
#             tick_labels.append(f"{val:.3f}")
#         elif extra_tick_digits:
#             tick_labels.append(f"{val:.2f}")
#         else:
#             tick_labels.append(f"{val:.1f}")

#     if add_colorbar:
#         diff_cb = plt.colorbar(
#             diff_plot, ax=ax, fraction=0.046, pad=0.04, extend=extend,
#             boundaries=boundaries, norm=disc_norm, spacing='proportional'
#         )
#         diff_cb.set_ticks(tick_positions)
#         diff_cb.ax.set_yticklabels(tick_labels)
#         diff_cb.ax.tick_params(labelsize=10)
#         diff_cb.set_label(cb_label, fontdict={'fontsize': 12})
#         if zero_step < 0.1 or max_mag > 10 or extra_tick_digits:
#             plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.2)
#         else:
#             plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.0)

#     # --- Package colorbar spec for shared-bar wrapper (NEW) ---
#     cb_params = None
#     if return_cb_params:
#         cb_params = dict(
#             mappable=diff_plot,        # carries cmap+norm for a shared colorbar
#             cmap=disc_cmap,
#             norm=disc_norm,
#             boundaries=boundaries,
#             extend=extend,
#             spacing='proportional',
#             ticks=tick_positions,
#             ticklabels=tick_labels,
#             label=cb_label
#         )

#     # Save/close only if we created the figure here
#     if savefig and created_fig is not None:
#         if fig_dir is None:
#             raise ValueError("Must specify 'fig_dir' = <directory>.")
#         if prefix is None:
#             raise ValueError("Must specify prefix for figure file name.")
#         if not os.path.exists(fig_dir):
#             os.makedirs(fig_dir)
#         created_fig.savefig(
#             os.path.join(fig_dir, f'{prefix}_dT_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png'),
#             dpi=600, bbox_inches='tight'
#         )
#         plt.close(created_fig)

#     return ax, diff_plot, diff_cb, cb_params


def plot_temp_diff_basin(
    panel_title,               # renamed from `title` to match panel API
    diff_ds, basin_name, max_depth, axis_split, start_yr, end_yr,
    zonal_int=False,
    icon=None,
    check_nn=False, nn_threshold=0.05, cb_max=None, mask_dataset=None,
    run_ds=None,  # must be passed to plot overlays
    savefig=True, fig_dir=None, prefix=None, verbose=False,
    # NEW (wrapper hooks):
    ax=None,                  # if provided, draw into this axes (no new fig)
    add_colorbar=True,        # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,   # if True, return a spec for a figure-level colorbar
    cb_label="Temperature Anomaly ($\\degree$C)",  # used by wrapper if shared
    # BACKWARD-COMPAT alias (optional):
    title=None                # if someone still passes old name
):
    # ---- title alias (back-compat) ----
    if panel_title is None and title is not None:
        panel_title = title

    # --- data prep (unchanged) ---
    # lat_res = 210 * 3
    # z_res   = 33 * 3

    if len(diff_ds.time.values) > 1:
        raise ValueError("diff_ds cannot be a time series.")

    diff_ds = diff_ds.isel(time=0)
    if zonal_int:
        diff_dat = get_pp_basin_zonalint_dat(
            diff_ds, "temp", basin_name, mask_ds=mask_dataset
        )
    else:
        diff_dat = get_pp_basin_dat(
            diff_ds, "temp", basin_name, check_nn=check_nn,
            nn_threshold=nn_threshold, mask_ds=mask_dataset
        )
        
    diff_dat = diff_dat.sel(z_l=slice(0, max_depth))

    # Refine target grids
    fine_lat   = refine_grid(diff_dat.true_lat.values, factor=10)
    fine_depth = refine_grid(diff_dat.z_l.values,      factor=10)
    if verbose:
        print("fine_depth:\n", fine_depth)

    # Interpolate onto refined grid
    diff_dat = diff_dat.interp(true_lat=fine_lat, z_l=fine_depth)

    # Optional depth-axis transform (split axis)
    if axis_split is not None:
        transformed_z = xr.apply_ufunc(
            transform_depth, diff_dat.z_l,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
        diff_dat = diff_dat.assign_coords(z_l=transformed_z)

    # Optional overlay (mean temperature contours)
    overlay_dat = None
    if run_ds is not None:
        if len(run_ds.time.values) > 1:
            raise ValueError("run_ds cannot be a time series.")
        if zonal_int:
            run_da = get_pp_basin_zonalint_dat(
                        run_ds.isel(time=0), "temp", basin_name, mask_ds=mask_dataset
                    ).sel(z_l=slice(0, max_depth))
        else:
            run_da = get_pp_basin_dat(
                        run_ds.isel(time=0), "temp", basin_name, check_nn=check_nn,
                        nn_threshold=nn_threshold, mask_ds=mask_dataset
                    ).sel(z_l=slice(0, max_depth))
            
        ##### DEBUGGING SO COMMENTED OUT THIS STEP
        overlay_dat = run_da.interp(true_lat=fine_lat, z_l=fine_depth)
        # overlay_dat = run_da
        if axis_split is not None:
            overlay_transformed_z = xr.apply_ufunc(
                transform_depth, overlay_dat.z_l,
                kwargs={"max_depth": max_depth, "axis_split": axis_split}
            )
            overlay_dat = overlay_dat.assign_coords(z_l=overlay_transformed_z)

    # --- diagnostics & colorbar spacing (mostly unchanged) ---
    min_val = float(np.nanmin(diff_dat.values))
    max_val = float(np.nanmax(diff_dat.values))
    if verbose:
        if zonal_int:
            print(f"Min and max zonal int temp anomaly: {min_val:.3g}, {max_val:.3g}")
        else:
            print(f"Min and max temp anomaly: {min_val:.2f}, {max_val:.2f}")

    p0p5  = float(np.nanpercentile(diff_dat.values,  0.5))
    p99p5 = float(np.nanpercentile(diff_dat.values, 99.5))
    if verbose:
        if abs(p0p5) > abs(p99p5):
            print(f"0.5 to 99.5th percentile data max mag: {abs(p0p5):.3g}")
        else:
            print(f"0.5 to 99.5th percentile data max mag: {abs(p99p5):.3g}")

    extra_tick_digits = False
    if cb_max is not None:
        if zonal_int:
            chosen_n = 20
        else:
            if cb_max in (1, 1.5, 2, 2.5, 3, 4, 5, 6):
                chosen_n = 20
            elif cb_max == 7.5:
                chosen_n = 12
            elif cb_max == 2.222:
                extra_tick_digits = True; cb_max = 2; chosen_n = 12
            elif cb_max == 3.333:
                cb_max = 3; chosen_n = 12
            elif cb_max == 4.444:
                extra_tick_digits = True; cb_max = 4; chosen_n = 12
            elif cb_max == 4.5:
                chosen_n = 12
            else:
                raise ValueError("cb_max is not an acceptable value.")
        data_max   = cb_max
        chosen_step = 2 * data_max / chosen_n
    else:
        if zonal_int:
            chosen_n, chosen_step = get_cb_spacing(
                p0p5, p99p5, min_bnd=1*10**4, min_spacing=1*10**2, min_n=10, max_n=20, verbose=verbose
            )
        else:
            chosen_n, chosen_step = get_cb_spacing(
                p0p5, p99p5, min_bnd=1.0, min_spacing=0.1, min_n=10, max_n=20, verbose=verbose
            )

    max_mag = 0.5 * chosen_n * chosen_step  # final ± range

    zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
        max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
    )

    # --- figure/axes management (NEW) ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7.5, 3))

    # Draw the main cross-section (no subplot_kws needed since we have an Axes)
    # diff_plot = diff_dat.plot(
    #     x='true_lat', y='z_l',
    #     cmap=disc_cmap, norm=disc_norm,
    #     add_labels=False, add_colorbar=False, ax=ax,
    #     edgecolors='face'
    # )

    diff_plot = diff_dat.plot(
        x='true_lat', y='z_l',
        cmap=disc_cmap, norm=disc_norm,
        add_labels=False, add_colorbar=False, ax=ax,
        infer_intervals=True,     # <- compute cell edges from centers
        rasterized=True,          # <- avoids vector “stairs” on save
        antialiased=False,        # <- no edge antialiasing lines
        # edgecolors='none',      # (or drop edgecolors entirely)
    )

    # Axis direction and limits
    ax.invert_yaxis()
    if axis_split is None:
        ax.set_ylim(max_depth, 0)
    else:
        y_top    = transform_depth(0,          max_depth, axis_split)
        y_bottom = transform_depth(max_depth,  max_depth, axis_split)
        ax.set_ylim(y_bottom, y_top)

    # Keep spines/ticks above the quadmesh
    for spine in ax.spines.values():
        spine.set_zorder(30)
    for tick in ax.get_xticklines():
        tick.set_zorder(30)
    for label in ax.get_xticklabels():
        label.set_zorder(30)

    # Y ticks (depth labels)
    depth_positions, depth_labels = get_depth_labels(axis_split, max_depth)
    ax.set_yticks(ticks=depth_positions, labels=[str(d) for d in depth_labels])
    ax.set_ylabel('Depth (m)')

    # Optional overlay contours (mean temperature)
    if (overlay_dat is not None) and (zonal_int == False):
        contour_levels = [-2, 0, 2, 4, 6, 8, 10, 15, 20, 25, 30]
        if verbose:
            min_overlay_temp = float(np.nanmin(overlay_dat.values))
            max_overlay_temp = float(np.nanmax(overlay_dat.values))
            print(f"Min and max mean temp: {min_overlay_temp:.2f}, {max_overlay_temp:.2f}\n")
        overlay_plot = ax.contour(
            overlay_dat["true_lat"], overlay_dat["z_l"], overlay_dat,
            levels=contour_levels, colors="k", linewidths=0.8
        )
        ax.clabel(overlay_plot, inline=True, fontsize=10)

    # Bathymetry overlay
    zonal_pct_bathy, lat_vals = bathymetry_overlay(diff_ds, diff_dat, fine_lat, basin_name)
    if axis_split is not None:
        zonal_pct_bathy = xr.apply_ufunc(
            transform_depth, zonal_pct_bathy,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
    ax.fill_between(lat_vals, max_depth, zonal_pct_bathy, color='grey', zorder=20)
    
    xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin_name)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(xlabels_major)
    ax.set_xticks(xticks_minor, minor=True)
    # Optional: make minor ticks smaller and unlabeled
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=6)

    # Title
    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Icon (optional)
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.08)
        ab = AnnotationBbox(imagebox, (0.95, 1.10), xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # --- Colorbar (optional local bar so wrapper can turn it off) ---
    diff_cb = None
    scale_exponent = None
    
    if zonal_int:
        max_tick = np.max(np.abs(tick_positions))
        if max_tick > 0:
            scale_exponent = int(np.floor(np.log10(max_tick)))
            scale_factor = 10.0 ** (scale_exponent - 1)
            tick_labels = [f"{v/scale_factor:.0f}" for v in tick_positions]
        else:
            tick_labels = ["0" for _ in tick_positions]

    else:
        tick_labels = []
        for val in tick_positions:
            if (abs(val) == 0.05 or abs(val) == 0.25):
                tick_labels.append(f"{val:.2f}")
            elif abs(val) == 0.125:
                tick_labels.append(f"{val:.3f}")
            elif extra_tick_digits:
                tick_labels.append(f"{val:.2f}")
            else:
                tick_labels.append(f"{val:.1f}")

    if zonal_int:
        cb_label="Integrated Temperature Anomaly ($\\degree$C m)"
    
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, fraction=0.046, pad=0.04, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing='proportional'
        )
        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.set_yticklabels(tick_labels)
        for t in diff_cb.ax.get_yticklabels():
            t.set_ha("center")
        diff_cb.ax.tick_params(labelsize=10)
        diff_cb.set_label(cb_label, fontdict={'fontsize': 12})
        # if zonal_int:
        #     plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.0)
        # else:
        #     if zero_step < 0.1 or max_mag > 10 or extra_tick_digits:
        #         plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.2)
        #     else:
        #         plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.0)

    # --- Package colorbar spec for shared-bar wrapper (NEW) ---
    cb_params = None
    if return_cb_params:
        cb_params = dict(
            mappable=diff_plot,        # carries cmap+norm for a shared colorbar
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing='proportional',
            ticks=tick_positions,
            ticklabels=tick_labels,
            label=cb_label,
            scale_exponent=scale_exponent
        )

    # Save/close only if we created the figure here
    if savefig and created_fig is not None:
        if fig_dir is None:
            raise ValueError("Must specify 'fig_dir' = <directory>.")
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        created_fig.savefig(
            os.path.join(fig_dir, f'{prefix}_dT_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


def plot_salt_diff_basin(
    panel_title,               # renamed from `title` to match panel API
    diff_ds, basin_name, max_depth, axis_split, start_yr, end_yr,
    icon=None,
    check_nn=False, nn_threshold=0.05, cb_max=None, mask_dataset=None,
    run_ds=None,  # must be passed to plot overlays
    savefig=True, fig_dir=None, prefix=None, verbose=False,
    # NEW (wrapper hooks):
    ax=None,                  # if provided, draw into this axes (no new fig)
    add_colorbar=True,        # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,   # if True, return a spec for a figure-level colorbar
    cb_label="Salinity Anomaly (psu)",  # used by wrapper if shared
    # BACKWARD-COMPAT alias (optional):
    title=None                # if someone still passes old name
):
    # ---- title alias (back-compat) ----
    if panel_title is None and title is not None:
        panel_title = title

    # --- data prep (mirrors temp) ---
    if len(diff_ds.time.values) > 1:
        raise ValueError("diff_ds cannot be a time series.")

    diff_ds = diff_ds.isel(time=0)
    diff_dat = get_pp_basin_dat(
        diff_ds, "salt", basin_name, check_nn=check_nn,
        nn_threshold=nn_threshold, mask_ds=mask_dataset
    )
    diff_dat = diff_dat.sel(z_l=slice(0, max_depth))

    # Refine target grids
    fine_lat   = refine_grid(diff_dat.true_lat.values, factor=10)
    fine_depth = refine_grid(diff_dat.z_l.values,      factor=10)
    if verbose:
        print("fine_depth:\n", fine_depth)

    # Interpolate onto refined grid
    diff_dat = diff_dat.interp(true_lat=fine_lat, z_l=fine_depth)

    # Optional depth-axis transform (split axis)
    if axis_split is not None:
        transformed_z = xr.apply_ufunc(
            transform_depth, diff_dat.z_l,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
        diff_dat = diff_dat.assign_coords(z_l=transformed_z)

    # Optional overlay (mean salinity contours)
    overlay_dat = None
    if run_ds is not None:
        if len(run_ds.time.values) > 1:
            raise ValueError("run_ds cannot be a time series.")
        run_da = get_pp_basin_dat(
            run_ds.isel(time=0), "salt", basin_name, check_nn=check_nn,
            nn_threshold=nn_threshold, mask_ds=mask_dataset
        ).sel(z_l=slice(0, max_depth))
        overlay_dat = run_da.interp(true_lat=fine_lat, z_l=fine_depth)
        if axis_split is not None:
            overlay_transformed_z = xr.apply_ufunc(
                transform_depth, overlay_dat.z_l,
                kwargs={"max_depth": max_depth, "axis_split": axis_split}
            )
            overlay_dat = overlay_dat.assign_coords(z_l=overlay_transformed_z)

    # --- diagnostics & colorbar spacing (tuned for salt) ---
    min_val = float(np.nanmin(diff_dat.values))
    max_val = float(np.nanmax(diff_dat.values))
    if verbose:
        print(f"Min and max salt anomaly: {min_val:.3f}, {max_val:.3f}")

    p0p5  = float(np.nanpercentile(diff_dat.values,  0.5))
    p99p5 = float(np.nanpercentile(diff_dat.values, 99.5))
    if verbose:
        if abs(p0p5) > abs(p99p5):
            print(f"0.5 to 99.5th percentile data max mag: {abs(p0p5):.3f}")
        else:
            print(f"0.5 to 99.5th percentile data max mag: {abs(p99p5):.3f}")

    # Support a curated set of cb_max (small ranges typical for salinity)
    extra_tick_digits = False
    if cb_max is not None:
        if cb_max in (0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75):
            chosen_n = 20
        else:
            raise ValueError("cb_max is not an acceptable value.")
        data_max    = cb_max
        chosen_step = 2 * data_max / chosen_n
    else:
        # tighter defaults than temperature
        chosen_n, chosen_step = get_cb_spacing(
            p0p5, p99p5, min_bnd=0.025, min_spacing=0.005, min_n=10, max_n=20, verbose=verbose
        )

    max_mag = 0.5 * chosen_n * chosen_step  # final ± range

    zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
        max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
    )

    # --- figure/axes management (same as temp) ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7.5, 3))

    # Draw the main cross-section with the same rasterization/AA settings as temp
    diff_plot = diff_dat.plot(
        x='true_lat', y='z_l',
        cmap=disc_cmap, norm=disc_norm,
        add_labels=False, add_colorbar=False, ax=ax,
        infer_intervals=True,
        rasterized=True,
        antialiased=False,
    )

    # Axis direction and limits
    ax.invert_yaxis()
    if axis_split is None:
        ax.set_ylim(max_depth, 0)
    else:
        y_top    = transform_depth(0,          max_depth, axis_split)
        y_bottom = transform_depth(max_depth,  max_depth, axis_split)
        ax.set_ylim(y_bottom, y_top)

    # Keep spines/ticks above the quadmesh
    for spine in ax.spines.values():
        spine.set_zorder(30)
    for tick in ax.get_xticklines():
        tick.set_zorder(30)
    for label in ax.get_xticklabels():
        label.set_zorder(30)

    # Y ticks (depth labels) — use same helper as temp
    depth_positions, depth_labels = get_depth_labels(axis_split, max_depth)
    ax.set_yticks(ticks=depth_positions, labels=[str(d) for d in depth_labels])
    ax.set_ylabel('Depth (m)')

    # Optional overlay contours (mean salinity)
    if overlay_dat is not None:
        contour_levels = np.arange(30.0, 40.1, 0.5)
        if verbose:
            min_overlay_salt = float(np.nanmin(overlay_dat.values))
            max_overlay_salt = float(np.nanmax(overlay_dat.values))
            print(f"Min and max mean salt: {min_overlay_salt:.2f}, {max_overlay_salt:.2f}\n")
        overlay_plot = ax.contour(
            overlay_dat["true_lat"], overlay_dat["z_l"], overlay_dat,
            levels=contour_levels, colors="k", linewidths=0.8
        )
        ax.clabel(overlay_plot, inline=True, fontsize=10, levels=contour_levels[::2])

    # Bathymetry overlay (same)
    zonal_pct_bathy, lat_vals = bathymetry_overlay(diff_ds, diff_dat, fine_lat, basin_name)
    if axis_split is not None:
        zonal_pct_bathy = xr.apply_ufunc(
            transform_depth, zonal_pct_bathy,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
    ax.fill_between(lat_vals, max_depth, zonal_pct_bathy, color='grey', zorder=20)

    xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin_name)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(xlabels_major)
    ax.set_xticks(xticks_minor, minor=True)
    # Optional: make minor ticks smaller and unlabeled
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=6)

    # Title
    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Icon (optional; same placement)
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.08)
        ab = AnnotationBbox(imagebox, (0.95, 1.10), xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # --- Colorbar (optional local bar so wrapper can turn it off) ---
    diff_cb = None
    tick_labels = []
    for val in tick_positions:
        # mimic temp panel’s readable tick logic, but salt ranges are small
        if (abs(val) == 0.05 or abs(val) == 0.25):
            tick_labels.append(f"{val:.2f}")
        elif abs(val) == 0.125:
            tick_labels.append(f"{val:.3f}")
        elif extra_tick_digits or chosen_step < 0.1:
            # show more precision for tight ranges typical of salinity
            tick_labels.append(f"{val:.2f}")
        else:
            tick_labels.append(f"{val:.1f}")

    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, fraction=0.046, pad=0.04, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing='proportional'
        )
        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.set_yticklabels(tick_labels)
        diff_cb.ax.tick_params(labelsize=10)
        diff_cb.set_label(cb_label, fontdict={'fontsize': 12})
        if zero_step < 0.1 or max_mag > 10 or extra_tick_digits:
            plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.2)
        else:
            plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.0)

    # --- Package colorbar spec for shared-bar wrapper (NEW) ---
    cb_params = None
    if return_cb_params:
        cb_params = dict(
            mappable=diff_plot,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing='proportional',
            ticks=tick_positions,
            ticklabels=tick_labels,
            label=cb_label
        )

    # Save/close only if we created the figure here
    if savefig and created_fig is not None:
        if fig_dir is None:
            raise ValueError("Must specify 'fig_dir' = <directory>.")
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        created_fig.savefig(
            os.path.join(fig_dir, f'{prefix}_dS_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


def plot_N2_diff_basin(
    panel_title,
    diff_ds, basin_name, max_depth, axis_split, start_yr, end_yr,
    icon=None,
    check_nn=False, nn_threshold=0.05, cb_max=None, mask_dataset=None,
    run_ds=None,
    savefig=True, fig_dir=None, prefix=None, verbose=False,
    ax=None,
    add_colorbar=True,
    return_cb_params=False,
    cb_label="N$^2$ Anomaly (s$^{-2}$)",
    title=None
):
    """
    Plot N² anomaly (raw values) with symmetric-log color normalization.
    Colormap and colorbar are continuous, but ticks appear at decade values
    formatted as ±10^k.
    """

    if panel_title is None and title is not None:
        panel_title = title

    # --- Data prep ---
    if len(diff_ds.time.values) > 1:
        raise ValueError("diff_ds cannot be a time series.")
    diff_ds = diff_ds.isel(time=0)

    diff_dat = get_pp_basin_dat(
        diff_ds, "N2", basin_name,
        check_nn=check_nn, nn_threshold=nn_threshold, mask_ds=mask_dataset
    )

    diff_dat = diff_dat.sel(z_i=slice(0, max_depth + (500 if max_depth > 2000 else 0)))

    fine_lat   = refine_grid(diff_dat.true_lat.values, factor=10)
    fine_depth = refine_grid(diff_dat.z_i.values, factor=10)
    diff_dat   = diff_dat.interp(true_lat=fine_lat, z_i=fine_depth)

    if axis_split is not None:
        diff_dat = diff_dat.assign_coords(
            z_i=xr.apply_ufunc(
                transform_depth, diff_dat.z_i,
                kwargs={"max_depth": max_depth, "axis_split": axis_split}
            )
        )

    # Optional overlay dataset
    overlay_dat = None
    if run_ds is not None:
        run_da = get_pp_basin_dat(
            run_ds.isel(time=0), "N2", basin_name,
            check_nn=check_nn, nn_threshold=nn_threshold, mask_ds=mask_dataset
        )
        run_da = run_da.sel(z_i=slice(0, max_depth))
        overlay_dat = run_da.interp(true_lat=fine_lat, z_i=fine_depth)
        if axis_split is not None:
            overlay_dat = overlay_dat.assign_coords(
                z_i=xr.apply_ufunc(transform_depth, overlay_dat.z_i,
                                   kwargs={"max_depth": max_depth, "axis_split": axis_split})
            )

    # ------------------------------------------------------------------
    # 1️⃣ Symmetric log normalization, continuous cmap
    # ------------------------------------------------------------------
    min_val = float(np.nanmin(diff_dat.values))
    max_val = float(np.nanmax(diff_dat.values))
    if verbose:
        print(f"N² anomaly range: {min_val:.3e} to {max_val:.3e}")

    if cb_max is None:
        cb_max = np.nanmax(np.abs([min_val, max_val]))
    if not np.isfinite(cb_max) or cb_max == 0:
        cb_max = 1e-12

    linthresh = cb_max / 100.0
    disc_cmap = cmocean.cm.balance
    disc_norm = mcolors.SymLogNorm(
        linthresh=linthresh, linscale=1.0, vmin=-cb_max, vmax=cb_max, base=10
    )

    # Tick positions and formatter (decade labels)
    exp_min = int(np.floor(np.log10(linthresh)))
    exp_max = int(np.ceil(np.log10(cb_max)))
    pos_decades = 10.0 ** np.arange(exp_min, exp_max + 1)
    tick_positions = np.concatenate([-pos_decades[::-1], [0.0], pos_decades])

    from matplotlib.ticker import FuncFormatter
    def decade_tick_fmt(v, pos=None):
        if v == 0:
            return "0"
        exp = int(np.round(np.log10(abs(v))))
        sign = "-" if v < 0 else ""
        return fr"${sign}10^{{{exp}}}$"

    extend = (
        "both" if (min_val < -cb_max and max_val > cb_max)
        else "min" if min_val < -cb_max
        else "max" if max_val > cb_max
        else "neither"
    )

    # ------------------------------------------------------------------
    # 2️⃣ Create figure / axes
    # ------------------------------------------------------------------
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7.5, 3))

    # Continuous plot
    diff_plot = diff_dat.plot(
        x='true_lat', y='z_i',
        cmap=disc_cmap, norm=disc_norm,
        add_labels=False, add_colorbar=False, ax=ax,
        infer_intervals=True, rasterized=True, antialiased=False,
    )

    # ------------------------------------------------------------------
    # 3️⃣ Axes cosmetics
    # ------------------------------------------------------------------
    ax.invert_yaxis()
    if axis_split is None:
        ax.set_ylim(max_depth, 0)
    else:
        ax.set_ylim(
            transform_depth(max_depth, max_depth, axis_split),
            transform_depth(0, max_depth, axis_split)
        )

    depth_positions, depth_labels = get_depth_labels(axis_split, max_depth)
    ax.set_yticks(depth_positions)
    ax.set_yticklabels([str(d) for d in depth_labels])
    ax.set_ylabel("Depth (m)")

    if overlay_dat is not None:
        if verbose:
            print("Min N² contour:", np.nanmin(overlay_dat))
            print("Max N² contour:", np.nanmax(overlay_dat))
        levels = np.arange(0, 4.0e-4 + 2.5e-5, 2.5e-5)
        overlay_plot = ax.contour(
            overlay_dat["true_lat"], overlay_dat["z_i"], overlay_dat,
            levels=levels, colors="k", linewidths=0.8
        )
        ax.clabel(overlay_plot, inline=True, fontsize=10, fmt="%.2e", levels=levels[::2])

    zonal_pct_bathy, lat_vals = bathymetry_overlay(diff_ds, diff_dat, fine_lat, basin_name)
    if axis_split is not None:
        zonal_pct_bathy = xr.apply_ufunc(
            transform_depth, zonal_pct_bathy,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
    ax.fill_between(lat_vals, max_depth, zonal_pct_bathy, color="grey", zorder=20)

    xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin_name)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(xlabels_major)
    ax.set_xticks(xticks_minor, minor=True)
    # Optional: make minor ticks smaller and unlabeled
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=6)

    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    if icon is not None:
        img = mpimg.imread(f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png")
        ab = AnnotationBbox(OffsetImage(img, zoom=0.08), (0.95, 1.10),
                            xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # ------------------------------------------------------------------
    # 4️⃣ Continuous colorbar with decade labels
    # ------------------------------------------------------------------
    diff_cb = None
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, fraction=0.046, pad=0.04,
            extend=extend, spacing='proportional'
        )
        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.yaxis.set_major_formatter(FuncFormatter(decade_tick_fmt))
        diff_cb.ax.tick_params(labelsize=10)
        diff_cb.set_label(cb_label, fontsize=12)

    cb_params = None
    if return_cb_params:
        cb_params = dict(
            mappable=diff_plot,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=None,
            extend=extend,
            spacing='proportional',
            ticks=tick_positions,
            ticklabels=[decade_tick_fmt(v) for v in tick_positions],
            label=cb_label,
        )

    if savefig and created_fig is not None:
        os.makedirs(fig_dir, exist_ok=True)
        created_fig.savefig(
            os.path.join(fig_dir, f"{prefix}_dN2_{start_yr:04d}_{end_yr:04d}.png"),
            dpi=600, bbox_inches="tight"
        )
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


def plot_rhopot2_diff_basin(
    panel_title,               # renamed from `title` to match panel API
    diff_ds, basin_name, max_depth, axis_split, start_yr, end_yr,
    icon=None,
    check_nn=False, nn_threshold=0.05, cb_max=None, mask_dataset=None,
    run_ds=None,  # must be passed to plot overlays
    savefig=True, fig_dir=None, prefix=None, verbose=False,
    # NEW (wrapper hooks):
    ax=None,                  # if provided, draw into this axes (no new fig)
    add_colorbar=True,        # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,   # if True, return a spec for a figure-level colorbar
    cb_label=r"$\sigma_2$ Anomaly (kg m$^{-3}$)",  # used by wrapper if shared
    # BACKWARD-COMPAT alias (optional):
    title=None                # if someone still passes old name
):
    # ---- title alias (back-compat) ----
    if panel_title is None and title is not None:
        panel_title = title

    # --- data prep (mirrors temp panel) ---
    if len(diff_ds.time.values) > 1:
        raise ValueError("diff_ds cannot be a time series.")

    diff_ds = diff_ds.isel(time=0)
    if verbose:
        min_rhopot2 = float(np.nanmin(diff_ds.rhopot2.values))
        max_rhopot2 = float(np.nanmax(diff_ds.rhopot2.values))
        print(f"Min and max rhopot2 anomaly: {min_rhopot2:.3f}, {max_rhopot2:.3f}")

    diff_dat = get_pp_basin_dat(
        diff_ds, "rhopot2", basin_name, check_nn=check_nn,
        nn_threshold=nn_threshold, mask_ds=mask_dataset
    ).sel(z_l=slice(0, max_depth))

    # Refine target grids
    fine_lat   = refine_grid(diff_dat.true_lat.values, factor=10)
    fine_depth = refine_grid(diff_dat.z_l.values,      factor=10)
    if verbose:
        print("fine_depth (z_l):\n", fine_depth)

    # Interpolate onto refined grid
    diff_dat = diff_dat.interp(true_lat=fine_lat, z_l=fine_depth)

    # Optional depth-axis transform (split axis)
    if axis_split is not None:
        transformed_z = xr.apply_ufunc(
            transform_depth, diff_dat.z_l,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
        diff_dat = diff_dat.assign_coords(z_l=transformed_z)

    # Optional overlay (mean sigma2 contours)
    overlay_dat = None
    if run_ds is not None:
        if len(run_ds.time.values) > 1:
            raise ValueError("run_ds cannot be a time series.")
        run_da = get_pp_basin_dat(
            run_ds.isel(time=0), "rhopot2", basin_name, check_nn=check_nn,
            nn_threshold=nn_threshold, mask_ds=mask_dataset
        ).sel(z_l=slice(0, max_depth))
        overlay_dat = run_da.interp(true_lat=fine_lat, z_l=fine_depth)
        if axis_split is not None:
            overlay_transformed_z = xr.apply_ufunc(
                transform_depth, overlay_dat.z_l,
                kwargs={"max_depth": max_depth, "axis_split": axis_split}
            )
            overlay_dat = overlay_dat.assign_coords(z_l=overlay_transformed_z)

    # --- diagnostics & colorbar spacing (tuned for density anomalies) ---
    min_val = float(np.nanmin(diff_dat.values))
    max_val = float(np.nanmax(diff_dat.values))

    p0p5  = float(np.nanpercentile(diff_dat.values,  0.5))
    p99p5 = float(np.nanpercentile(diff_dat.values, 99.5))
    if verbose:
        if abs(p0p5) > abs(p99p5):
            print(f"0.5 to 99.5th percentile data max mag: {abs(p0p5):.3f}")
        else:
            print(f"0.5 to 99.5th percentile data max mag: {abs(p99p5):.3f}")

    # Support a curated set of cb_max typical for sigma2 anomalies
    extra_tick_digits = False
    if cb_max is not None:
        if cb_max in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.4, 2.0, 2.5):
            chosen_n = 20
        else:
            raise ValueError("cb_max is not an acceptable value.")
        data_max    = cb_max
        chosen_step = 2 * data_max / chosen_n   # gives 0.03–0.25 steps across range
    else:
        # Defaults roughly match your previous bin logic (finer than temp)
        # min_bnd ~ smallest plausible half-range; min_spacing ~ smallest step
        chosen_n, chosen_step = get_cb_spacing(
            p0p5, p99p5, min_bnd=0.3, min_spacing=0.025, min_n=10, max_n=20, verbose=verbose
        )

    max_mag = 0.5 * chosen_n * chosen_step  # final ± range

    zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
        max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
    )

    # --- figure/axes management (same as temp) ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7.5, 3))

    # Main cross-section (rasterized, infer_intervals, no AA) for consistency
    diff_plot = diff_dat.plot(
        x='true_lat', y='z_l',
        cmap=disc_cmap, norm=disc_norm,
        add_labels=False, add_colorbar=False, ax=ax,
        infer_intervals=True,
        rasterized=True,
        antialiased=False,
    )

    # Axis direction and limits
    ax.invert_yaxis()
    if axis_split is None:
        ax.set_ylim(max_depth, 0)
    else:
        y_top    = transform_depth(0,          max_depth, axis_split)
        y_bottom = transform_depth(max_depth,  max_depth, axis_split)
        ax.set_ylim(y_bottom, y_top)

    # Keep spines/ticks above the quadmesh
    for spine in ax.spines.values():
        spine.set_zorder(30)
    for tick in ax.get_xticklines():
        tick.set_zorder(30)
    for label in ax.get_xticklabels():
        label.set_zorder(30)

    # Y ticks (depth labels) — use same helper as temp
    depth_positions, depth_labels = get_depth_labels(axis_split, max_depth)
    ax.set_yticks(ticks=depth_positions, labels=[str(d) for d in depth_labels])
    ax.set_ylabel('Depth (m)')

    # Optional overlay contours (mean sigma2)
    if overlay_dat is not None:
        # contour_transition = 1035.0
        # low_contour_levels = np.arange(1028.0, contour_transition, 1.0)
        # upp_contour_levels = np.arange(contour_transition, 1040.0 + 0.25, 0.25)
        # contour_levels = np.concatenate([low_contour_levels, upp_contour_levels])

        contour_transition = 1035.5
        low_contour_levels = np.arange(1030.0, contour_transition, 1.0)
        upp_contour_levels = np.arange(contour_transition, 1040.0 + 0.25, 0.25)
        contour_levels = np.concatenate([low_contour_levels, upp_contour_levels])

        if verbose:
            min_overlay = float(np.nanmin(overlay_dat.values))
            max_overlay = float(np.nanmax(overlay_dat.values))
            print(f"Min and max mean rhopot2: {min_overlay:.3f}, {max_overlay:.3f}\n")

        overlay_plot = ax.contour(
            overlay_dat["true_lat"], overlay_dat["z_l"], overlay_dat,
            levels=contour_levels, colors="k", linewidths=0.8
        )

        # Custom label formatting across the transition
        custom_fmt = {}
        for lev in contour_levels:
            if lev < contour_transition:
                custom_fmt[lev] = f"{lev:.0f}"
            else:
                if np.isclose(lev % 1.0, 0.0, atol=1e-9):
                    custom_fmt[lev] = f"{lev:.0f}"
                elif np.isclose(lev % 0.5, 0.0, atol=1e-9):
                    custom_fmt[lev] = f"{lev:.1f}"
                else:
                    custom_fmt[lev] = f"{lev:.2f}"
        ax.clabel(overlay_plot, inline=True, fontsize=10, fmt=custom_fmt, levels=contour_levels[::2])

    # Bathymetry overlay
    zonal_pct_bathy, lat_vals = bathymetry_overlay(diff_ds, diff_dat, fine_lat, basin_name)
    if axis_split is not None:
        zonal_pct_bathy = xr.apply_ufunc(
            transform_depth, zonal_pct_bathy,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
    ax.fill_between(lat_vals, max_depth, zonal_pct_bathy, color='grey', zorder=20)

    xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin_name)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(xlabels_major)
    ax.set_xticks(xticks_minor, minor=True)
    # Optional: make minor ticks smaller and unlabeled
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=6)

    # Title
    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Icon (optional; same placement)
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.08)
        ab = AnnotationBbox(imagebox, (0.95, 1.10), xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # --- Colorbar (optional local bar so wrapper can turn it off) ---
    diff_cb = None
    # Build readable tick labels (slightly more precision than temp if tight spacing)
    tick_labels = []
    for val in tick_positions:
        if (abs(val) == 0.05 or abs(val) == 0.25):
            tick_labels.append(f"{val:.2f}")
        elif abs(val) == 0.125:
            tick_labels.append(f"{val:.3f}")
        elif chosen_step < 0.1:
            tick_labels.append(f"{val:.2f}")
        elif chosen_step < 0.01:
            tick_labels.append(f"{val:.3f}")
        else:
            tick_labels.append(f"{val:.1f}")

    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, fraction=0.046, pad=0.04, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing='proportional'
        )
        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.set_yticklabels(tick_labels)
        diff_cb.ax.tick_params(labelsize=10)
        diff_cb.set_label(cb_label, fontdict={'fontsize': 12})
        if extra_tick_digits or chosen_step < 0.1 or max_mag > 10:
            plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.2)
        else:
            plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.0)

    # --- Package colorbar spec for shared-bar wrapper (NEW) ---
    cb_params = None
    if return_cb_params:
        cb_params = dict(
            mappable=diff_plot,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing='proportional',
            ticks=tick_positions,
            ticklabels=tick_labels,
            label=cb_label
        )

    # Save/close only if we created the figure here
    if savefig and created_fig is not None:
        if fig_dir is None:
            raise ValueError("Must specify 'fig_dir' = <directory>.")
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        created_fig.savefig(
            os.path.join(fig_dir, f'{prefix}_dSigma2_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


# ## Basin mean plot functions (T, S, N2, rhopot2)

def plot_temp_mean_basin(
    panel_title,               # renamed from `title` to match panel API
    mean_ds, basin_name, max_depth, axis_split, start_yr, end_yr,
    icon=None,
    check_nn=False, nn_threshold=0.05, cb_max=None, mask_dataset=None,
    # NEW (wrapper hooks):
    ax=None,                  # if provided, draw into this axes (no new fig)
    add_colorbar=True,        # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,   # if True, return a spec for a figure-level colorbar
    cb_label="Temperature ($\\degree$C)",  # used by wrapper if shared
    # BACKWARD-COMPAT alias (optional):
    title=None,
    # Saving (only if this function creates the figure)
    savefig=False, fig_dir=None, prefix=None,
    verbose=False
):
    """
    Plot basin-mean absolute temperature cross-section (not anomaly)
    with a discrete cmocean 'thermal' colorbar split into 30 bins
    from -2°C to plot_max (default 28°C or cb_max if provided).
    Labels appear every 6°C on the colorbar. Compatible with plot_pp_grid().
    """

    # ---- title alias (back-compat) ----
    if panel_title is None and title is not None:
        panel_title = title

    # --- data prep ---
    if len(mean_ds.time.values) > 1:
        raise ValueError("mean_ds cannot be a time series.")

    mean_ds  = mean_ds.isel(time=0)
    mean_dat = get_pp_basin_dat(
        mean_ds, "temp", basin_name, check_nn=check_nn,
        nn_threshold=nn_threshold, mask_ds=mask_dataset
    ).sel(z_l=slice(0, max_depth))

    # Refine target grids
    fine_lat   = refine_grid(mean_dat.true_lat.values, factor=10)
    fine_depth = refine_grid(mean_dat.z_l.values,      factor=10)

    # Interpolate onto refined grid
    mean_dat = mean_dat.interp(true_lat=fine_lat, z_l=fine_depth)

    # Optional depth-axis transform (split axis)
    if axis_split is not None:
        transformed_z = xr.apply_ufunc(
            transform_depth, mean_dat.z_l,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
        mean_dat = mean_dat.assign_coords(z_l=transformed_z)

    # Diagnostics for colorbar arrows
    min_val = float(np.nanmin(mean_dat.values))
    max_val = float(np.nanmax(mean_dat.values))
    if verbose:
        print(f"Min and max temp: {min_val:.2f}, {max_val:.2f}")

    # -------- Discrete colormap: 30 bins from -2 to plot_max --------
    plot_min = -2.0
    plot_max = float(cb_max) if (cb_max is not None) else 28.0

    n_bins = 30  # << requested number of discrete bins
    boundaries = np.linspace(plot_min, plot_max, n_bins + 1)  # length 31
    # Major ticks every 6°C, starting at -2
    major_ticks = np.arange(-2.0, plot_max + 1e-9, 6.0)

    cmap      = cmocean.cm.thermal
    # keep as-is; BoundaryNorm will discretize into the 30 bins via "boundaries"
    disc_cmap = cmap
    disc_norm = mcolors.BoundaryNorm(boundaries, disc_cmap.N, clip=False)

    # Determine the extend setting for colorbar arrows
    if min_val < plot_min and max_val > plot_max:
        extend = 'both'
    elif min_val < plot_min:
        extend = 'min'
    elif max_val > plot_max:
        extend = 'max'
    else:
        extend = 'neither'

    # --- figure/axes management ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7.5, 3))

    # Main cross-section (rasterized, infer_intervals, no AA for consistency)
    mean_p = mean_dat.plot(
        x='true_lat', y='z_l',
        cmap=disc_cmap, norm=disc_norm,
        add_labels=False, add_colorbar=False, ax=ax,
        infer_intervals=True, rasterized=True, antialiased=False,
    )

    # Axes cosmetics
    ax.invert_yaxis()
    if axis_split is None:
        ax.set_ylim(max_depth, 0)
    else:
        y_top    = transform_depth(0,          max_depth, axis_split)
        y_bottom = transform_depth(max_depth,  max_depth, axis_split)
        ax.set_ylim(y_bottom, y_top)

    for spine in ax.spines.values():
        spine.set_zorder(30)
    for tick in ax.get_xticklines():
        tick.set_zorder(30)
    for label in ax.get_xticklabels():
        label.set_zorder(30)

    # Y ticks (depth labels)
    depth_positions, depth_labels = get_depth_labels(axis_split, max_depth)
    ax.set_yticks(ticks=depth_positions, labels=[str(d) for d in depth_labels])
    ax.set_ylabel('Depth (m)')

    # Mean-temperature contours for reference
    contour_levels = [-2, 0, 2, 4, 6, 8, 10, 15, 20, 25, 30]
    CS = ax.contour(
        mean_dat["true_lat"], mean_dat["z_l"], mean_dat,
        levels=contour_levels, colors="white", linewidths=0.8
    )
    ax.clabel(CS, fmt="%2.0f", inline=True, fontsize=8, colors="white")

    # Bathymetry overlay
    zonal_pct_bathy, lat_vals = bathymetry_overlay(mean_ds, mean_dat, fine_lat, basin_name)
    if axis_split is not None:
        zonal_pct_bathy = xr.apply_ufunc(
            transform_depth, zonal_pct_bathy,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
    ax.fill_between(lat_vals, max_depth, zonal_pct_bathy, color='grey', zorder=20)

    xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin_name)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(xlabels_major)
    ax.set_xticks(xticks_minor, minor=True)
    # Optional: make minor ticks smaller and unlabeled
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=6)

    # Title
    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Icon (optional)
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        ab = AnnotationBbox(OffsetImage(img, zoom=0.08), (0.95, 1.10),
                            xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # --- Colorbar (optional; discrete, 30 bins; labels every 6°C; no minor ticks) ---
    mean_cb = None
    if add_colorbar:
        mean_cb = plt.colorbar(
            mean_p, ax=ax,
            boundaries=boundaries, norm=disc_norm, spacing='proportional',
            ticks=major_ticks, fraction=0.046, pad=0.04, extend=extend
        )
        mean_cb.ax.minorticks_off()
        mean_cb.ax.tick_params(labelsize=10)
        mean_cb.set_label(cb_label, fontdict={'fontsize': 12})

    # --- Package colorbar spec for shared-bar wrapper ---
    cb_params = None
    if return_cb_params:
        cb_params = dict(
            mappable=mean_p,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing='proportional',
            ticks=major_ticks,
            ticklabels=[f"{t:.0f}" for t in major_ticks],
            label=cb_label
        )

    # Save only if we created the figure here
    if savefig and created_fig is not None:
        if fig_dir is None:
            raise ValueError("Must specify 'fig_dir' = <directory>.")
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        created_fig.savefig(
            os.path.join(fig_dir, f'{prefix}_mean_temp_{start_yr:04d}_{end_yr:04d}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, mean_p, mean_cb, cb_params


def plot_salt_mean_basin(
    panel_title,               # renamed from `title` to match panel API
    mean_ds, basin_name, max_depth, axis_split, start_yr, end_yr,
    icon=None,
    check_nn=False, nn_threshold=0.05, cb_max=None, mask_dataset=None,
    # NEW (wrapper hooks):
    ax=None,                  # if provided, draw into this axes (no new fig)
    add_colorbar=True,        # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,   # if True, return a spec for a figure-level colorbar
    cb_label="Salinity (psu)",# used by wrapper if shared
    # BACKWARD-COMPAT alias (optional):
    title=None,
    # Saving (only if this function creates the figure)
    savefig=False, fig_dir=None, prefix=None,
    verbose=False
):
    """
    Plot basin-mean absolute salinity cross-section (not anomaly),
    compatible with plot_pp_grid().
    Discrete cmocean 'haline' colorbar with ~0.1 psu bins across the chosen range.
    """

    # ---- title alias (back-compat) ----
    if panel_title is None and title is not None:
        panel_title = title

    # --- data prep ---
    if len(mean_ds.time.values) > 1:
        raise ValueError("mean_ds cannot be a time series.")

    mean_ds  = mean_ds.isel(time=0)
    mean_dat = get_pp_basin_dat(
        mean_ds, "salt", basin_name, check_nn=check_nn,
        nn_threshold=nn_threshold, mask_ds=mask_dataset
    ).sel(z_l=slice(0, max_depth))

    # Refine target grids (consistent look)
    fine_lat   = refine_grid(mean_dat.true_lat.values, factor=10)
    fine_depth = refine_grid(mean_dat.z_l.values,      factor=10)

    # Interpolate onto refined grid
    mean_dat = mean_dat.interp(true_lat=fine_lat, z_l=fine_depth)

    # Optional depth-axis transform (split axis)
    if axis_split is not None:
        transformed_z = xr.apply_ufunc(
            transform_depth, mean_dat.z_l,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
        mean_dat = mean_dat.assign_coords(z_l=transformed_z)

    # Diagnostics for colorbar arrows
    min_val = float(np.nanmin(mean_dat.values))
    max_val = float(np.nanmax(mean_dat.values))
    if verbose:
        print(f"Min and max salt: {min_val:.2f}, {max_val:.2f}")

    # -------- Discrete colormap & ticks --------
    # Default plotting range; allow cb_max to override the upper bound.
    if basin_name == 'global':
        plot_min = 30.5
        plot_max = float(cb_max) if (cb_max is not None) else 36.5
    elif basin_name == 'atl':
        plot_min = 32.5
        plot_max = float(cb_max) if (cb_max is not None) else 36.5
    elif basin_name == 'pac':
        plot_min = 31.5
        plot_max = float(cb_max) if (cb_max is not None) else 35.5
    else:
        plot_min = 30.5
        plot_max = float(cb_max) if (cb_max is not None) else 36.5

    # Build boundaries at ~0.1 psu resolution (like your original logic)
    step_psu    = 0.1
    n_bins      = int(np.round((plot_max - plot_min) / step_psu))
    boundaries  = np.linspace(plot_min, plot_max, n_bins + 1)

    # Major ticks every 0.5 psu
    major_ticks = np.arange(np.ceil(plot_min * 2) / 2, plot_max + 0.001, 0.5)

    cmap      = cmocean.cm.haline
    disc_cmap = cmap
    disc_norm = mcolors.BoundaryNorm(boundaries, disc_cmap.N, clip=False)

    # Determine the extend setting for colorbar arrows
    if min_val < plot_min and max_val > plot_max:
        extend = 'both'
    elif min_val < plot_min:
        extend = 'min'
    elif max_val > plot_max:
        extend = 'max'
    else:
        extend = 'neither'

    # --- figure/axes management ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7.5, 3))

    # Main cross-section (rasterized, infer_intervals, no AA for consistency)
    mean_p = mean_dat.plot(
        x='true_lat', y='z_l',
        cmap=disc_cmap, norm=disc_norm,
        add_labels=False, add_colorbar=False, ax=ax,
        infer_intervals=True, rasterized=True, antialiased=False,
    )

    # Axes cosmetics
    ax.invert_yaxis()
    if axis_split is None:
        ax.set_ylim(max_depth, 0)
    else:
        y_top    = transform_depth(0,          max_depth, axis_split)
        y_bottom = transform_depth(max_depth,  max_depth, axis_split)
        ax.set_ylim(y_bottom, y_top)

    for spine in ax.spines.values():
        spine.set_zorder(30)
    for tick in ax.get_xticklines():
        tick.set_zorder(30)
    for label in ax.get_xticklabels():
        label.set_zorder(30)

    # Y ticks (depth labels)
    depth_positions, depth_labels = get_depth_labels(axis_split, max_depth)
    ax.set_yticks(ticks=depth_positions, labels=[str(d) for d in depth_labels])
    ax.set_ylabel('Depth (m)')

    # White salinity contours (every 0.5 psu)
    contour_levels = np.arange(plot_min, plot_max + 0.5, 0.5)
    CS = ax.contour(
        mean_dat["true_lat"], mean_dat["z_l"], mean_dat,
        levels=contour_levels, colors="white", linewidths=0.8
    )
    # Label every other level to reduce clutter
    ax.clabel(CS, fmt="%g", inline=True, fontsize=8, colors="white",
              levels=contour_levels[::2])

    # Bathymetry overlay
    zonal_pct_bathy, lat_vals = bathymetry_overlay(mean_ds, mean_dat, fine_lat, basin_name)
    if axis_split is not None:
        zonal_pct_bathy = xr.apply_ufunc(
            transform_depth, zonal_pct_bathy,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
    ax.fill_between(lat_vals, max_depth, zonal_pct_bathy, color='grey', zorder=20)

    xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin_name)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(xlabels_major)
    ax.set_xticks(xticks_minor, minor=True)
    # Optional: make minor ticks smaller and unlabeled
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=6)

    # Title
    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Icon (optional)
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        ab = AnnotationBbox(OffsetImage(img, zoom=0.08), (0.95, 1.10),
                            xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # --- Colorbar (discrete; ~0.1 psu bins; ticks every 0.5 psu) ---
    mean_cb = None
    if add_colorbar:
        mean_cb = plt.colorbar(
            mean_p, ax=ax,
            boundaries=boundaries, norm=disc_norm, spacing='proportional',
            ticks=major_ticks, fraction=0.046, pad=0.04, extend=extend
        )
        mean_cb.ax.minorticks_off()
        mean_cb.ax.tick_params(labelsize=10)
        mean_cb.set_label(cb_label, fontdict={'fontsize': 12})

    # --- Package colorbar spec for shared-bar wrapper ---
    cb_params = None
    if return_cb_params:
        cb_params = dict(
            mappable=mean_p,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing='proportional',
            ticks=major_ticks,
            ticklabels=[f"{t:g}" for t in major_ticks],
            label=cb_label
        )

    # Save only if we created the figure here
    if savefig and created_fig is not None:
        if fig_dir is None:
            raise ValueError("Must specify 'fig_dir' = <directory>.")
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        created_fig.savefig(
            os.path.join(fig_dir, f'{prefix}_mean_salt_{start_yr:04d}_{end_yr:04d}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, mean_p, mean_cb, cb_params


def plot_N2_mean_basin(
    panel_title,               # renamed from `title` to match panel API
    mean_ds, basin_name, max_depth, axis_split, start_yr, end_yr,
    icon=None,
    check_nn=False, nn_threshold=0.05, cb_max=None, mask_dataset=None,
    # NEW (wrapper hooks):
    ax=None,                  # if provided, draw into this axes (no new fig)
    add_colorbar=True,        # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,   # if True, return a spec for a figure-level colorbar
    cb_label="N$^2$ (s$^{-2}$)",  # used by wrapper if shared
    # BACKWARD-COMPAT alias (optional):
    title=None,
    # Saving (only if this function creates the figure)
    savefig=False, fig_dir=None, prefix=None,
    # Log/symlog options
    cb_min=None,              # lower bound for positive-only log case
    linthresh=1e-9,          # <= your “preserve sign if |N²|>1e-9” threshold
    sym_cmap=None,            # diverging cmap for symlog (defaults to cmocean.balance)
    pos_cmap=None,            # sequential cmap for positive-only (defaults to cmocean.deep)
    verbose=False
):
    """
    Plot basin-mean N² cross-section on a logarithmic scale.

    - If any negative values with |N²| > linthresh exist -> SymLogNorm (sign preserved).
    - Else -> LogNorm (positive-only).
    - Colorbars are continuous (no boundaries), so no mid-bar white gap.
    """

    # ---- title alias (back-compat) ----
    if panel_title is None and title is not None:
        panel_title = title

    # --- data prep ---
    if len(mean_ds.time.values) > 1:
        raise ValueError("mean_ds cannot be a time series.")

    mean_ds  = mean_ds.isel(time=0)
    mean_dat = get_pp_basin_dat(
        mean_ds, "N2", basin_name, check_nn=check_nn,
        nn_threshold=nn_threshold, mask_ds=mask_dataset
    )

    # Include a bit deeper for smoother edges if very deep
    mean_dat = mean_dat.sel(z_i=slice(0, max_depth + 500 if max_depth > 2000 else max_depth))

    # Refine target grids
    fine_lat   = refine_grid(mean_dat.true_lat.values, factor=10)
    fine_depth = refine_grid(mean_dat.z_i.values,      factor=10)

    # Interpolate onto refined grid
    mean_dat = mean_dat.interp(true_lat=fine_lat, z_i=fine_depth)

    # Optional depth-axis transform (split axis)
    if axis_split is not None:
        mean_dat = mean_dat.assign_coords(
            z_i=xr.apply_ufunc(
                transform_depth, mean_dat.z_i,
                kwargs={"max_depth": max_depth, "axis_split": axis_split}
            )
        )

    # ---- Inspect data range ----
    data_min = float(np.nanmin(mean_dat.values))
    data_max = float(np.nanmax(mean_dat.values))
    if verbose:
        print(f"[N2] min={data_min:.2e}, max={data_max:.2e}")

    use_symlog = np.isfinite(data_min) and (data_min < -abs(linthresh))

    # --- figure/axes management ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7.5, 3))

    mean_p = None
    mean_cb = None
    cb_params = None

    # ====================== BRANCH 1: SYMLOG (signed) ======================
    if use_symlog:
        # Diverging colormap
        cmap = sym_cmap if sym_cmap is not None else cmocean.cm.balance

        # Choose symmetric limits around 0
        default_max = 2e-4
        vmax = float(cb_max) if (cb_max is not None) else max(default_max, abs(data_max))
        vbound = max(vmax, abs(data_min))  # symmetric
        norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=1.0,
                                  vmin=-vbound, vmax=vbound, base=10)

        # Plot (no masking near zero -> no white band)
        mean_p = mean_dat.plot(
            x='true_lat', y='z_i',
            cmap=cmap, norm=norm,
            add_labels=False, add_colorbar=False, ax=ax,
            infer_intervals=True, rasterized=True, antialiased=False,
        )

        # Contours: symmetric log-spaced plus 0
        def _log_levels(vmin_abs, vmax_abs, n):
            return np.logspace(np.log10(max(vmin_abs, linthresh)), np.log10(vmax_abs), n)
        n_contours_side = 10
        pos_levels = _log_levels(linthresh, vbound, n_contours_side)
        neg_levels = -pos_levels[::-1]
        levels = np.concatenate([neg_levels, [0.0], pos_levels])
        CS = ax.contour(
            mean_dat["true_lat"], mean_dat["z_i"], mean_dat,
            levels=levels, colors="black", linewidths=0.8
        )
        ax.clabel(CS, fmt="%.1e", inline=True, fontsize=8, colors="black", levels=levels[::2])

        if add_colorbar:
            mean_cb = plt.colorbar(mean_p, ax=ax, fraction=0.046, pad=0.04, extend='both')
            # Proper symlog ticks via locator/formatter (no manual yscale change!)
            locator   = mticker.SymmetricalLogLocator(base=10, linthresh=linthresh, subs=(1.0, ))
            formatter = mticker.LogFormatterMathtext(base=10)
            mean_cb.locator   = locator
            mean_cb.formatter = formatter
            mean_cb.update_ticks()
            mean_cb.ax.tick_params(labelsize=10)
            mean_cb.set_label(cb_label, fontdict={'fontsize': 12})

        # if return_cb_params:
        #     # Provide enough info for a shared bar (wrapper can rebuild norm)
        #     cb_params = dict(
        #         cmap=cmocean.cm.balance,
        #         vmin=-vbound, vmax=vbound,
        #         scale="symlog", linthresh=linthresh,
        #         label=cb_label
        #     )
        if return_cb_params:
            cb_params = dict(
                mappable=mean_p,      # <-- the artist returned by .plot(...)
                norm=norm,            # <-- the Normalize used
                # optional extras your wrapper might use:
                boundaries=None,
                spacing="uniform",
                extend="both",        # symlog case (you used extend='both' above)
                label=cb_label,
            )

    # ==================== BRANCH 2: POSITIVE-ONLY LOG =====================
    else:
        # Mask non-positive for log
        dat_pos = mean_dat.where(mean_dat > 0)
        pos_min = float(np.nanmin(dat_pos.values))
        pos_max = float(np.nanmax(dat_pos.values))
        if not np.isfinite(pos_min) or not np.isfinite(pos_max):
            raise ValueError("N² has no positive values to plot on a log scale.")

        vmin = float(cb_min) if (cb_min is not None) else max(1e-7, pos_min)
        vmax = float(cb_max) if (cb_max is not None) else 2e-4
        if vmin <= 0 or vmax <= vmin:
            raise ValueError("Invalid cb_min/cb_max for LogNorm.")

        cmap = pos_cmap if pos_cmap is not None else cmocean.cm.deep
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        mean_p = dat_pos.plot(
            x='true_lat', y='z_i',
            cmap=cmap, norm=norm,
            add_labels=False, add_colorbar=False, ax=ax,
            infer_intervals=True, rasterized=True, antialiased=False,
        )

        # Log-spaced contours
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 12)
        CS = ax.contour(
            dat_pos["true_lat"], dat_pos["z_i"], dat_pos,
            levels=levels, colors="black", linewidths=0.8
        )
        ax.clabel(CS, fmt="%.2e", inline=True, fontsize=8, colors="black", levels=levels[::2])

        if add_colorbar:
            mean_cb = plt.colorbar(mean_p, ax=ax, fraction=0.046, pad=0.04)
            # Decade ticks + mathtext formatting; no gaps
            mean_cb.ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
            mean_cb.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2,3,5)))
            mean_cb.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            mean_cb.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
            mean_cb.ax.tick_params(labelsize=10)
            mean_cb.set_label(cb_label, fontdict={'fontsize': 12})

        # if return_cb_params:
        #     cb_params = dict(
        #         cmap=cmocean.cm.deep,
        #         vmin=vmin, vmax=vmax,
        #         scale="log",
        #         label=cb_label
        #     )
        if return_cb_params:
            cb_params = dict(
                mappable=mean_p,
                norm=norm,
                boundaries=None,
                spacing="uniform",
                extend=None,          # or "max" if you prefer
                label=cb_label,
            )

    # ---------------- common axes cosmetics ----------------
    ax.invert_yaxis()
    if axis_split is None:
        ax.set_ylim(max_depth, 0)
    else:
        ax.set_ylim(
            transform_depth(max_depth, max_depth, axis_split),
            transform_depth(0, max_depth, axis_split)
        )

    for spine in ax.spines.values():
        spine.set_zorder(30)
    for tick in ax.get_xticklines():
        tick.set_zorder(30)
    for label in ax.get_xticklabels():
        label.set_zorder(30)

    depth_positions, depth_labels = get_depth_labels(axis_split, max_depth)
    ax.set_yticks(ticks=depth_positions, labels=[str(d) for d in depth_labels])
    ax.set_ylabel('Depth (m)')

    # Bathymetry overlay
    zonal_pct_bathy, lat_vals = bathymetry_overlay(mean_ds, mean_dat, fine_lat, basin_name)
    if axis_split is not None:
        zonal_pct_bathy = xr.apply_ufunc(
            transform_depth, zonal_pct_bathy,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
    ax.fill_between(lat_vals, max_depth, zonal_pct_bathy, color='grey', zorder=20)

    xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin_name)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(xlabels_major)
    ax.set_xticks(xticks_minor, minor=True)
    # Optional: make minor ticks smaller and unlabeled
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=6)

    # Title
    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Icon (optional)
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        ab = AnnotationBbox(OffsetImage(img, zoom=0.08), (0.95, 1.10),
                            xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # Save only if we created the figure here
    if savefig and created_fig is not None:
        if fig_dir is None:
            raise ValueError("Must specify 'fig_dir' = <directory>.")
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")
        fname = f'{prefix}_mean_N2_log{"_signed" if use_symlog else ""}_{start_yr:04d}_{end_yr:04d}.png'
        os.makedirs(fig_dir, exist_ok=True)
        created_fig.savefig(os.path.join(fig_dir, fname), dpi=600, bbox_inches='tight')
        plt.close(created_fig)

    return ax, mean_p, mean_cb, cb_params


def plot_rhopot2_mean_basin(
    panel_title,               # renamed from `title` to match panel API
    mean_ds, basin_name, max_depth, axis_split, start_yr, end_yr,
    icon=None,
    check_nn=False, nn_threshold=0.05, cb_max=None, mask_dataset=None,
    # NEW (wrapper hooks):
    ax=None,                  # if provided, draw into this axes (no new fig)
    add_colorbar=True,        # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,   # if True, return a spec for a figure-level colorbar
    cb_label=r"$\sigma_2$ (kg m$^{-3}$)",  # used by wrapper if shared
    # BACKWARD-COMPAT alias (optional):
    title=None,
    # Saving (only if this function creates the figure)
    savefig=False, fig_dir=None, prefix=None,
    verbose=False
):
    """
    Plot basin-mean absolute potential density (sigma_2) cross-section
    using a discrete cmocean 'dense' colorbar with 0.5 kg m^-3 bins
    from 1028.0 to plot_max (default 1038 or cb_max if provided).
    Labels appear every 2.0 on the colorbar. Compatible with plot_pp_grid().
    """

    # ---- title alias (back-compat) ----
    if panel_title is None and title is not None:
        panel_title = title

    # --- data prep ---
    if len(mean_ds.time.values) > 1:
        raise ValueError("mean_ds cannot be a time series.")

    mean_ds  = mean_ds.isel(time=0)
    mean_dat = get_pp_basin_dat(
        mean_ds, "rhopot2", basin_name, check_nn=check_nn,
        nn_threshold=nn_threshold, mask_ds=mask_dataset
    ).sel(z_l=slice(0, max_depth))

    # Refine target grids
    fine_lat   = refine_grid(mean_dat.true_lat.values, factor=10)
    fine_depth = refine_grid(mean_dat.z_l.values,      factor=10)

    # Interpolate onto refined grid
    mean_dat = mean_dat.interp(true_lat=fine_lat, z_l=fine_depth)

    # Optional depth-axis transform (split axis)
    if axis_split is not None:
        transformed_z = xr.apply_ufunc(
            transform_depth, mean_dat.z_l,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
        mean_dat = mean_dat.assign_coords(z_l=transformed_z)

    # Diagnostics for colorbar arrows
    min_val = float(np.nanmin(mean_dat.values))
    max_val = float(np.nanmax(mean_dat.values))
    if verbose:
        print(f"Min and max rhopot2: {min_val:.3f}, {max_val:.3f}")

    # -------- Discrete colormap: 0.5-bin sigma2 from 1028.0 to plot_max --------
    plot_min = 1028.0
    plot_max = float(cb_max) if (cb_max is not None) else 1038.0

    # Ensure the range makes sense
    if plot_max <= plot_min:
        raise ValueError("cb_max must be > 1028.0 for rhopot2 mean panel.")

    # Build 0.5-spaced boundaries including the top end
    boundaries = np.arange(plot_min, plot_max + 0.5, 0.5)  # 0.5 kg m^-3 bins
    # Major ticks every 2.0
    # Start on the first multiple of 2.0 >= plot_min
    first_major = np.ceil((plot_min - 0.0) / 2.0) * 2.0
    major_ticks = np.arange(first_major, plot_max + 1e-9, 2.0)

    cmap      = cmocean.cm.dense
    disc_cmap = cmap
    disc_norm = mcolors.BoundaryNorm(boundaries, disc_cmap.N, clip=False)

    # Determine the extend setting for colorbar arrows
    if min_val < plot_min and max_val > plot_max:
        extend = 'both'
    elif min_val < plot_min:
        extend = 'min'
    elif max_val > plot_max:
        extend = 'max'
    else:
        extend = 'neither'

    # --- figure/axes management ---
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7.5, 3))

    # Main cross-section (rasterized, infer_intervals, no AA for consistency)
    mean_p = mean_dat.plot(
        x='true_lat', y='z_l',
        cmap=disc_cmap, norm=disc_norm,
        add_labels=False, add_colorbar=False, ax=ax,
        infer_intervals=True, rasterized=True, antialiased=False,
    )

    # Axes cosmetics
    ax.invert_yaxis()
    if axis_split is None:
        ax.set_ylim(max_depth, 0)
    else:
        y_top    = transform_depth(0,          max_depth, axis_split)
        y_bottom = transform_depth(max_depth,  max_depth, axis_split)
        ax.set_ylim(y_bottom, y_top)

    for spine in ax.spines.values():
        spine.set_zorder(30)
    for tick in ax.get_xticklines():
        tick.set_zorder(30)
    for label in ax.get_xticklabels():
        label.set_zorder(30)

    # Y ticks (depth labels)
    depth_positions, depth_labels = get_depth_labels(axis_split, max_depth)
    ax.set_yticks(ticks=depth_positions, labels=[str(d) for d in depth_labels])
    ax.set_ylabel('Depth (m)')

    # Reference sigma2 contours (white), coarser at 1.0 kg m^-3
    # contour_levels = np.arange(max(1028.0, plot_min), plot_max + 1e-9, 1.0)
    contour_transition = 1035.5
    low_contour_levels = np.arange(1030.0, contour_transition, 1.0)
    upp_contour_levels = np.arange(contour_transition, 1040.0 + 0.25, 0.25)
    contour_levels = np.concatenate([low_contour_levels, upp_contour_levels])
    
    CS = ax.contour(
        mean_dat["true_lat"], mean_dat["z_l"], mean_dat,
        levels=contour_levels, colors="white", linewidths=0.8
    )
    ax.clabel(CS, fmt="%.1f", inline=True, fontsize=8, colors="white", levels=contour_levels[::2])

    # Bathymetry overlay
    zonal_pct_bathy, lat_vals = bathymetry_overlay(mean_ds, mean_dat, fine_lat, basin_name)
    if axis_split is not None:
        zonal_pct_bathy = xr.apply_ufunc(
            transform_depth, zonal_pct_bathy,
            kwargs={"max_depth": max_depth, "axis_split": axis_split}
        )
    ax.fill_between(lat_vals, max_depth, zonal_pct_bathy, color='grey', zorder=20)

    xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin_name)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(xlabels_major)
    ax.set_xticks(xticks_minor, minor=True)
    # Optional: make minor ticks smaller and unlabeled
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=6)

    # Title
    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Icon (optional)
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        ab = AnnotationBbox(OffsetImage(img, zoom=0.08), (0.95, 1.10),
                            xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # --- Colorbar (optional; discrete, 0.5 bins; labels every 2.0; no minor ticks) ---
    mean_cb = None
    if add_colorbar:
        mean_cb = plt.colorbar(
            mean_p, ax=ax,
            boundaries=boundaries, norm=disc_norm, spacing='proportional',
            ticks=major_ticks, fraction=0.046, pad=0.04, extend=extend
        )
        mean_cb.ax.minorticks_off()
        mean_cb.ax.tick_params(labelsize=10)
        mean_cb.set_label(cb_label, fontdict={'fontsize': 12})

    # --- Package colorbar spec for shared-bar wrapper ---
    cb_params = None
    if return_cb_params:
        cb_params = dict(
            mappable=mean_p,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing='proportional',
            ticks=major_ticks,
            ticklabels=[f"{t:.0f}" for t in major_ticks],
            label=cb_label
        )

    # Save only if we created the figure here
    if savefig and created_fig is not None:
        if fig_dir is None:
            raise ValueError("Must specify 'fig_dir' = <directory>.")
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        created_fig.savefig(
            os.path.join(fig_dir, f'{prefix}_mean_rhopot2_{start_yr:04d}_{end_yr:04d}.png'),
            dpi=600, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, mean_p, mean_cb, cb_params


# # Diffusivity plotting functions

# ## Global mean profiles

# plotting Kd variable with continuous y-axis

def plot_Kd_cont_yaxis(co2_scen,fig_dir,start_yr,end_yr,Kd_var,max_Kd,
                       max_z = 6250, 
                       profiles = ['surf','therm','mid','bot'],
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                       savefig=True,
                       fig_suff=None):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_scientific(True)
    sci_formatter.set_powerlimits((0, 0))
    
    if (Kd_var == "Kd_int_tuned" or Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
        depth = myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]['z_i']
    elif (Kd_var == "Kd_lay_tuned" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        depth = myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]['z_l']

    prof_colors = ['b','m','g','r']
    
    if Kd_var == "Kd_int_tuned":
        title_pref = r"Mean $\kappa_{\mathregular{add}}$"
    elif Kd_var == "Kd_interface":
        title_pref = r"Mean $\kappa_{\mathregular{tot}}$"
    elif Kd_var == "Kd_int_base":
        title_pref = r"Mean $\kappa_{\mathregular{base}}$"
    elif Kd_var == "Kd_lay_tuned":
        title_pref = r"Mean $\kappa_{\mathregular{add}}$"
    elif Kd_var == "Kd_layer":
        title_pref = r"Mean $\kappa_{\mathregular{tot}}$"
    elif Kd_var == "Kd_lay_base":
        title_pref = r"Mean $\kappa_{\mathregular{base}}$"

    if co2_scen == "const":
        co2_str = "Const CO\u2082"
    elif co2_scen == "doub":
        co2_str = "1pct2xCO\u2082"
    elif co2_scen == "quad":
        co2_str = "1pct4xCO\u2082"
        
    # plot for each power input
    for pow_idx in range(len(power_var_suff)):
        fig, ax = plt.subplots(figsize=(4,4.8))

        if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
            ax.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_interface"][0,:],depth,label='control',color='k')
        elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
            ax.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_layer"][0,:],depth,label='control',color='k')
        
        for i in range(len(profiles)):
            ax.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start_yr}_{end_yr}_mean"][Kd_var][0,:],
                    depth,label=f'{profiles[i]}',color=prof_colors[i])

        ax.set_xlabel(r"$\kappa_d$ (m/s$^2$)")
        ax.set_ylabel("Depth (m)")
        
        ax.xaxis.set_major_formatter(sci_formatter)
        ax.set_xlim(0,max_Kd)
        if Kd_var != 'Kd_int_tuned' and Kd_var != 'Kd_lay_tuned':
            ax.set_ylim(-50,max_z)
        else:
            ax.set_ylim(0,max_z)
            
        ax.invert_yaxis()
        
        if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
            ax.legend(loc='lower right', fontsize=10, labelspacing=0.1)
        else:
            ax.legend(loc='best', fontsize=10, labelspacing=0.1)
        ax.grid("both")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.set_title(title_pref+f" {power_strings[pow_idx]} {co2_str}\nYear {start_yr} to {end_yr}")

        if savefig:
            if fig_dir is None:
                raise ValueError("Must specify 'fig_dir' = <directory>.")
                
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            if fig_suff is None:
                plt.savefig(fig_dir+f'{co2_scen}_{power_var_suff[pow_idx]}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.pdf', dpi=600, bbox_inches='tight')
            else:
                plt.savefig(fig_dir+f'{co2_scen}_{power_var_suff[pow_idx]}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}_{fig_suff}.pdf', dpi=600, bbox_inches='tight')
                
            plt.close()
            
    # plot with all data
    fig, ax = plt.subplots(figsize=(4,4.8))
    
    line_list = ['solid','dashed','dotted']

    if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        custom_leg_1 = [Line2D([0], [0], color='k', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2)]
        custom_leg_2 = [Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
        
        leg_labels_1 = ['control']
        for elem in profiles:
            leg_labels_1.append(elem)
            
        leg_labels_2 = copy.deepcopy(power_strings)
        
        if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
            ax.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_interface"][0,:],depth,label='control',color='k')
        elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
            ax.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_layer"][0,:],depth,label='control',color='k')
    else:
        custom_leg = [Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
        leg_labels = copy.deepcopy(profiles)
        for elem in power_strings:
            leg_labels.append(elem)
    
    for pow_idx, power_str in enumerate(power_strings):
        for i in range(len(profiles)):
            ax.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start_yr}_{end_yr}_mean"][Kd_var][0,:],depth,label=f'{power_str} {profiles[i]}',
                    linestyle=line_list[pow_idx],color=prof_colors[i])
        
    ax.set_xlabel(r"$\kappa_d$ (m/s$^2$)")
    ax.set_ylabel("Depth (m)")
    
    ax.xaxis.set_major_formatter(sci_formatter)
    ax.set_xlim(0,max_Kd)
    if Kd_var != 'Kd_int_tuned' and Kd_var != 'Kd_lay_tuned':
        ax.set_ylim(-50,max_z)
    else:
        ax.set_ylim(0,max_z)
            
    ax.invert_yaxis()
    
    # ax.legend(loc='best',ncol=2)
    if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        # First legend (5 labels)
        legend1 = ax.legend(
            custom_leg_1, leg_labels_1,
            loc='lower right',
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(0.67, 0.0),  # Adjust position as needed
            frameon=True
        )
        # Second legend (3 labels, positioned below the first)
        legend2 = ax.legend(
            custom_leg_2, leg_labels_2,
            loc='lower right',
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(1.0, 0.0),  # Adjust position as needed
            frameon=True
        )
        
        # Add the first legend back to the axis
        ax.add_artist(legend1)
    else:
        ax.legend(custom_leg, leg_labels, loc='best', fontsize=10, ncol = 2, labelspacing=0.1)
    
    ax.grid("both")
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    ax.set_title(title_pref+f" {co2_str}\nYear {start_yr} to {end_yr}")

    if savefig:
        if fig_dir is None:
            raise ValueError("Must specify 'fig_dir' = <directory>.")
            
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if fig_suff is None:
            plt.savefig(fig_dir+f'{co2_scen}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'{co2_scen}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}_{fig_suff}.pdf', dpi=600, bbox_inches='tight')
        plt.close()


# plotting Kd variable with split axis plot (abrupt change in y-axis)

def plot_Kd_split_yaxis(co2_scen,fig_dir,start_yr,end_yr,Kd_var,max_Kd,
                        axis_break = 850,
                        max_z = 6250,
                       profiles = ['surf','therm','mid','bot'],
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                       savefig=True,
                       fig_suff=None):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_scientific(True)
    sci_formatter.set_powerlimits((0, 0))

    # depth = myVars[f"{co2_scen}_{profiles[0]}_{power_var_suff[0]}_{start_yr}_{end_yr}_mean"]['z_i']
    if (Kd_var == "Kd_int_tuned" or Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
        depth = myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]['z_i']
    elif (Kd_var == "Kd_lay_tuned" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        depth = myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]['z_l']

    prof_colors = ['b','m','g','r']
    
    if Kd_var == "Kd_int_tuned":
        title_pref = r"Mean $\kappa_{\mathregular{add}}$"
    elif Kd_var == "Kd_interface":
        title_pref = r"Mean $\kappa_{\mathregular{tot}}$"
    elif Kd_var == "Kd_int_base":
        title_pref = r"Mean $\kappa_{\mathregular{base}}$"
    elif Kd_var == "Kd_lay_tuned":
        title_pref = r"Mean $\kappa_{\mathregular{add}}$"
    elif Kd_var == "Kd_layer":
        title_pref = r"Mean $\kappa_{\mathregular{tot}}$"
    elif Kd_var == "Kd_lay_base":
        title_pref = r"Mean $\kappa_{\mathregular{base}}$"

    if co2_scen == "const":
        co2_str = "Const CO\u2082"
    elif co2_scen == "doub":
        co2_str = "1pct2xCO\u2082"
    elif co2_scen == "quad":
        co2_str = "1pct4xCO\u2082"

    # plot for each power input
    for pow_idx in range(len(power_var_suff)):
        # Create a figure with GridSpec
        fig = plt.figure(figsize=(4,4.8))
        gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0)  # Adjust height_ratios
        
        # Top subplot
        ax1 = fig.add_subplot(gs[0])
    
        if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
            ax1.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_interface"][0,:].sel(z_i=slice(0,axis_break)),
                     depth.sel(z_i=slice(0,axis_break)),label='control',color='k')
        elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
            ax1.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_layer"][0,:].sel(z_l=slice(0,axis_break)),
                     depth.sel(z_l=slice(0,axis_break)),label='control',color='k')
    
        for i in range(len(profiles)):
            if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
                ax1.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start_yr}_{end_yr}_mean"][Kd_var][0,:].sel(z_i=slice(0,axis_break)),
                        depth.sel(z_i=slice(0,axis_break)),label=f'{profiles[i]}',color=prof_colors[i])
            elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
                ax1.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start_yr}_{end_yr}_mean"][Kd_var][0,:].sel(z_l=slice(0,axis_break)),
                        depth.sel(z_l=slice(0,axis_break)),label=f'{profiles[i]}',color=prof_colors[i])
    
        ax1.spines['bottom'].set_visible(False)  # Hide bottom spine
        ax1.tick_params(bottom=True, labelbottom=False)  # Enable ticks but hide labels
        # ax1.set_ylim(0, axis_break)
        if Kd_var != 'Kd_int_tuned' and Kd_var != 'Kd_lay_tuned':
            ax1.set_ylim(-50,axis_break)
        else:
            ax1.set_ylim(0,axis_break)
            
        ax1.invert_yaxis()
        
        # Bottom subplot
        ax2 = fig.add_subplot(gs[1])
    
        if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
            ax2.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_interface"][0,:].sel(z_i=slice(axis_break,None)),
                     depth.sel(z_i=slice(axis_break,None)),label='control',color='k')
        elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
            ax2.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_layer"][0,:].sel(z_l=slice(axis_break,None)),
                     depth.sel(z_l=slice(axis_break,None)),label='control',color='k')

        for i in range(len(profiles)):
            if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_int_tuned"):
                ax2.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start_yr}_{end_yr}_mean"][Kd_var][0,:].sel(z_i=slice(axis_break,None)),
                        depth.sel(z_i=slice(axis_break,None)),label=f'{profiles[i]}',color=prof_colors[i])
            elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base" or "Kd_lay_tuned"):
                ax2.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start_yr}_{end_yr}_mean"][Kd_var][0,:].sel(z_l=slice(axis_break,None)),
                        depth.sel(z_l=slice(axis_break,None)),label=f'{profiles[i]}',color=prof_colors[i])
    
        ax2.set_ylim(axis_break,max_z)
        ax2.invert_yaxis()
        
        # Synchronize the x-axis limits
        ax1.xaxis.set_major_formatter(sci_formatter)
        ax2.xaxis.set_major_formatter(sci_formatter)
        ax1.set_xlim(0,max_Kd)
        ax2.set_xlim(0,max_Kd)
    
        ax1.grid("both")
        ax2.grid("both")
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax2.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        
        if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
            ax2.legend(loc='lower right', fontsize=10, labelspacing=0.1)
        else:
            ax2.legend(loc='best', fontsize=10, labelspacing=0.1)
        
        ax2.set_xlabel(r"$\kappa_d$ (m/s$^2$)")
        fig.text(0, 0.5, "Depth (m)", va='center', rotation='vertical')
        ax1.set_title(title_pref+f" {power_strings[pow_idx]} {co2_str}\nYear {start_yr} to {end_yr}")

        if savefig:
            if fig_dir is None:
                raise ValueError("Must specify 'fig_dir' = <directory>.")
                
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            if fig_suff is None:
                plt.savefig(fig_dir+f'{co2_scen}_{power_var_suff[pow_idx]}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.pdf', dpi=600, bbox_inches='tight')
            else:
                plt.savefig(fig_dir+f'{co2_scen}_{power_var_suff[pow_idx]}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}_{fig_suff}.pdf', dpi=600, bbox_inches='tight')
                
            plt.close()

    # plot with all data
    # Create a figure with GridSpec
    fig = plt.figure(figsize=(4,4.8))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0)  # Adjust height_ratios

    line_list = ['solid','dashed','dotted']
    
    # Top subplot
    ax1 = fig.add_subplot(gs[0])

    if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        custom_leg_1 = [Line2D([0], [0], color='k', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2)]
        custom_leg_2 = [Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
        
        leg_labels_1 = ['control']
        for elem in profiles:
            leg_labels_1.append(elem)
            
        leg_labels_2 = copy.deepcopy(power_strings)

        if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
            ax1.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_interface"][0,:].sel(z_i=slice(0,axis_break)),
                     depth.sel(z_i=slice(0,axis_break)),label=f'control',color='k')
        elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
            ax1.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_layer"][0,:].sel(z_l=slice(0,axis_break)),
                     depth.sel(z_l=slice(0,axis_break)),label=f'control',color='k')
        
    else:
        custom_leg = [Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
        leg_labels = copy.deepcopy(profiles)
        for elem in power_strings:
            leg_labels.append(elem)

    for pow_idx in range(len(power_var_suff)):
        for i in range(len(profiles)):
            if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_int_tuned"):
                ax1.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start_yr}_{end_yr}_mean"][Kd_var][0,:].sel(z_i=slice(0,axis_break)),
                        depth.sel(z_i=slice(0,axis_break)),label=f'{power_strings[pow_idx]} {profiles[i]}',
                        linestyle=line_list[pow_idx],color=prof_colors[i])
            elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base" or "Kd_lay_tuned"):
                ax1.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start_yr}_{end_yr}_mean"][Kd_var][0,:].sel(z_l=slice(0,axis_break)),
                        depth.sel(z_l=slice(0,axis_break)),label=f'{power_strings[pow_idx]} {profiles[i]}',
                        linestyle=line_list[pow_idx],color=prof_colors[i])

    ax1.spines['bottom'].set_visible(False)  # Hide bottom spine
    ax1.tick_params(bottom=True, labelbottom=False)  # Enable ticks but hide labels
    # ax1.set_ylim(0, axis_break)
    if Kd_var != 'Kd_int_tuned' and Kd_var != 'Kd_lay_tuned':
        ax1.set_ylim(-50,axis_break)
    else:
        ax1.set_ylim(0,axis_break)
    
    ax1.invert_yaxis()
    
    # Bottom subplot
    ax2 = fig.add_subplot(gs[1])

    if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
        ax2.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_interface"][0,:].sel(z_i=slice(axis_break,None)),
                 depth.sel(z_i=slice(axis_break,None)),label=f'control',color='k')
    elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        ax2.plot(myVars[f"{co2_scen}_ctrl_{start_yr}_{end_yr}_mean"]["Kd_layer"][0,:].sel(z_l=slice(axis_break,None)),
                 depth.sel(z_l=slice(axis_break,None)),label=f'control',color='k')

    for pow_idx in range(len(power_var_suff)):
        for i in range(len(profiles)):
            if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_int_tuned"):
                ax2.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start_yr}_{end_yr}_mean"][Kd_var][0,:].sel(z_i=slice(axis_break,None)),
                        depth.sel(z_i=slice(axis_break,None)),label=f'{power_strings[pow_idx]} {profiles[i]}',
                        linestyle=line_list[pow_idx],color=prof_colors[i])
            elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base" or "Kd_lay_tuned"):
                ax2.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start_yr}_{end_yr}_mean"][Kd_var][0,:].sel(z_l=slice(axis_break,None)),
                        depth.sel(z_l=slice(axis_break,None)),label=f'{power_strings[pow_idx]} {profiles[i]}',
                        linestyle=line_list[pow_idx],color=prof_colors[i])

    ax2.set_ylim(axis_break,max_z)
    ax2.invert_yaxis()
    
    # Synchronize the x-axis limits
    ax1.xaxis.set_major_formatter(sci_formatter)
    ax2.xaxis.set_major_formatter(sci_formatter)
    ax1.set_xlim(0,max_Kd)
    ax2.set_xlim(0,max_Kd)

    ax1.grid("both")
    ax2.grid("both")
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    
    # ax2.legend(loc='best',ncol=2)
    if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        # First legend (5 labels)
        legend1 = ax2.legend(
            custom_leg_1, leg_labels_1,
            loc='lower right',
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(0.67, 0.0),  # Adjust position as needed
            frameon=True
        )
        # Second legend (3 labels, positioned below the first)
        legend2 = ax2.legend(
            custom_leg_2, leg_labels_2,
            loc='lower right',
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(1.0, 0.0),  # Adjust position as needed
            frameon=True
        )
        
        # Add the first legend back to the axis
        ax2.add_artist(legend1)
    else:
        ax2.legend(custom_leg, leg_labels, loc='best', fontsize=10, ncol = 2, labelspacing=0.1)
    
    ax2.set_xlabel(r"$\kappa_d$ (m/s$^2$)")
    fig.text(0, 0.5, "Depth (m)", va='center', rotation='vertical')
    ax1.set_title(title_pref+f" {co2_str}\nYear {start_yr} to {end_yr}")

    if savefig:
        if fig_dir is None:
            raise ValueError("Must specify 'fig_dir' = <directory>.")
            
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if fig_suff is None:
            plt.savefig(fig_dir+f'{co2_scen}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'{co2_scen}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}_{fig_suff}.pdf', dpi=600, bbox_inches='tight')
        plt.close()


# ### Functions for creating figure with multiple subplots (by power and by time)

# plotting Kd for three time periods (one subplot per power)

def plot_Kd_multi_time(co2_scen,fig_dir,starts,ends,Kd_var,max_Kd,
                       max_z = 6250, 
                       profiles = ['surf','therm','mid','bot'],
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                       savefig=True,
                       fig_suff=None):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_scientific(True)
    sci_formatter.set_powerlimits((0, 0))
    
    if (Kd_var == "Kd_int_tuned" or Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
        depth = myVars[f"{co2_scen}_ctrl_{starts[0]}_{ends[0]}_mean"]['z_i']
    elif (Kd_var == "Kd_lay_tuned" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        depth = myVars[f"{co2_scen}_ctrl_{starts[0]}_{ends[0]}_mean"]['z_l']

    prof_colors = ['b','m','g','r']
    time_strings = []
    for time_idx in range(len(starts)):
        time_strings.append(f'yr {starts[time_idx]}-{ends[time_idx]}')
    
    if Kd_var == "Kd_int_tuned":
        title_pref = r"Mean $\kappa_{\mathregular{add}}$"
    elif Kd_var == "Kd_interface":
        title_pref = r"Mean $\kappa_{\mathregular{tot}}$"
    elif Kd_var == "Kd_int_base":
        title_pref = r"Mean $\kappa_{\mathregular{base}}$"
    elif Kd_var == "Kd_lay_tuned":
        title_pref = r"Mean $\kappa_{\mathregular{add}}$"
    elif Kd_var == "Kd_layer":
        title_pref = r"Mean $\kappa_{\mathregular{tot}}$"
    elif Kd_var == "Kd_lay_base":
        title_pref = r"Mean $\kappa_{\mathregular{base}}$"

    if co2_scen == "const":
        co2_str = "Const CO\u2082"
    elif co2_scen == "doub":
        co2_str = "1pct2xCO\u2082"
    elif co2_scen == "quad":
        co2_str = "1pct4xCO\u2082"
        
    # plot for each power input
    
    line_list = ['solid','dashed','dotted']

    if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        custom_leg_1 = [Line2D([0], [0], color='k', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2)]
        custom_leg_2 = [Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
        
        leg_labels_1 = ['control']
        for elem in profiles:
            leg_labels_1.append(elem)

        leg_labels_2 = copy.deepcopy(time_strings)

    else:
        custom_leg = [Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
        
        leg_labels = copy.deepcopy(profiles)
        for elem in time_strings:
            leg_labels.append(elem)

    for pow_idx in range(len(power_var_suff)):
        fig, ax = plt.subplots(figsize=(4,4.8))

        for time_idx in range(len(starts)):
            if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
                ax.plot(myVars[f"{co2_scen}_ctrl_{starts[time_idx]}_{ends[time_idx]}_mean"]["Kd_interface"][0,:],depth,label='control',color='k',linestyle=line_list[time_idx])
            elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
                ax.plot(myVars[f"{co2_scen}_ctrl_{starts[time_idx]}_{ends[time_idx]}_mean"]["Kd_layer"][0,:],depth,label='control',color='k',linestyle=line_list[time_idx])

            for i in range(len(profiles)):
                ax.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{starts[time_idx]}_{ends[time_idx]}_mean"][Kd_var][0,:],depth,
                        linestyle=line_list[time_idx],color=prof_colors[i])
            
        ax.set_xlabel(r"$\kappa_d$ (m/s$^2$)")
        ax.set_ylabel("Depth (m)")
        
        ax.xaxis.set_major_formatter(sci_formatter)
        ax.set_xlim(0,max_Kd)
        if Kd_var != 'Kd_int_tuned' and Kd_var != 'Kd_lay_tuned':
            ax.set_ylim(-50,max_z)
        else:
            ax.set_ylim(0,max_z)
                
        ax.invert_yaxis()
        
        # ax.legend(loc='best',ncol=2)
        if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
            # First legend (5 labels)
            legend1 = ax.legend(
                custom_leg_1, leg_labels_1,
                loc='lower right',
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(0.6, 0.0),  # Adjust position as needed
                frameon=True
            )
            # Second legend (3 labels, positioned below the first)
            legend2 = ax.legend(
                custom_leg_2, leg_labels_2,
                loc='lower right',
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(1.0, 0.0),  # Adjust position as needed
                frameon=True
            )
            
            # Add the first legend back to the axis
            ax.add_artist(legend1)
        else:
            ax.legend(custom_leg, leg_labels, loc='best', fontsize=10, ncol = 2, labelspacing=0.1)
        
        ax.grid("both")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.set_title(title_pref+f" {co2_str} {power_strings[pow_idx]}")
    
        if savefig:
            if fig_dir is None:
                raise ValueError("Must specify 'fig_dir' = <directory>.")
                
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            if fig_suff is None:
                plt.savefig(fig_dir+f'{co2_scen}_{power_var_suff[pow_idx]}_{Kd_var}.pdf', dpi=600, bbox_inches='tight')
            else:
                plt.savefig(fig_dir+f'{co2_scen}_{power_var_suff[pow_idx]}_{Kd_var}_{fig_suff}.pdf', dpi=600, bbox_inches='tight')
            plt.close()

    # --- combined figure with subplots and ONE shared legend ---
    n = len(power_var_suff)
    if n != 1:
        if Kd_var in ["Kd_interface", "Kd_int_base", "Kd_layer", "Kd_lay_base"]:
            fig_height = 6.4
        else:
            fig_height = 6.0
        
        # Give the legend its own dedicated row
        fig = plt.figure(figsize=(3.5 * n, fig_height))
        gs = GridSpec(2, n, height_ratios=[20, 3], hspace=0.25, figure=fig)
        
        axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis("off")
        
        for pow_idx, ax in enumerate(axes):
            for time_idx in range(len(starts)):
                if Kd_var in ["Kd_interface", "Kd_int_base"]:
                    ax.plot(
                        myVars[f"{co2_scen}_ctrl_{starts[time_idx]}_{ends[time_idx]}_mean"]["Kd_interface"][0,:],
                        depth, label="control", color="k", linestyle=line_list[time_idx]
                    )
                elif Kd_var in ["Kd_layer", "Kd_lay_base"]:
                    ax.plot(
                        myVars[f"{co2_scen}_ctrl_{starts[time_idx]}_{ends[time_idx]}_mean"]["Kd_layer"][0,:],
                        depth, label="control", color="k", linestyle=line_list[time_idx]
                    )
                for i in range(len(profiles)):
                    ax.plot(
                        myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{starts[time_idx]}_{ends[time_idx]}_mean"][Kd_var][0,:],
                        depth, linestyle=line_list[time_idx], color=prof_colors[i]
                    )
        
            ax.set_xlabel(r"$\kappa_d$ (m/s$^2$)")
            ax.xaxis.set_major_formatter(sci_formatter)
            ax.set_xlim(0, max_Kd)
            if Kd_var not in ["Kd_int_tuned", "Kd_lay_tuned"]:
                ax.set_ylim(-50, max_z)
            else:
                ax.set_ylim(0, max_z)
            ax.invert_yaxis()
            ax.grid("both")
            ax.minorticks_on()
            ax.grid(which="major", linestyle="-", linewidth=0.5, color="gray")
            ax.set_title(f"{power_strings[pow_idx]}")
        
        fig.supylabel("Depth (m)", x=0.02)
        fig.subplots_adjust(left=0.08)
        
        # Build the shared legend(s) on the dedicated legend row
        if Kd_var in ["Kd_interface", "Kd_int_base", "Kd_layer", "Kd_lay_base"]:
            if n == 1:
                x_loc_1 = 0.3
                x_loc_2 = 0.7
            elif n == 2:
                x_loc_1 = 0.35
                x_loc_2 = 0.65
            else:
                x_loc_1 = 0.4
                x_loc_2 = 0.6
                
            # two grouped legends: profiles/colors + time/linestyles
            leg1 = legend_ax.legend(
                custom_leg_1, leg_labels_1,
                loc="center", bbox_to_anchor=(x_loc_1, 0.5),
                ncol=2,
                fontsize=10, frameon=True, labelspacing=0.1
            )
            legend_ax.add_artist(leg1)
            legend_ax.legend(
                custom_leg_2, leg_labels_2,
                loc="center", bbox_to_anchor=(x_loc_2, 0.5),
                fontsize=10, frameon=True, labelspacing=0.1
            )
        else:
            legend_ax.legend(
                custom_leg, leg_labels,
                loc="center", ncol=2,
                fontsize=10, frameon=True, labelspacing=0.1
            )
        
        if savefig:
            if fig_dir is None:
                raise ValueError("When savefig=True, provide fig_dir.")
            os.makedirs(fig_dir, exist_ok=True)
            if fig_suff is None:
                saved_path = os.path.join(fig_dir, f"{co2_scen}_{Kd_var}_over_time.pdf")
            else:
                saved_path = os.path.join(fig_dir, f"{co2_scen}_{Kd_var}_{fig_suff}_over_time.pdf")
            fig.savefig(saved_path, dpi=600, bbox_inches='tight')
            plt.close(fig)


# plotting Kd for three powers (one subplot per time period)

def plot_Kd_multi_power(co2_scen,fig_dir,starts,ends,Kd_var,max_Kd,
                       max_z = 6250, 
                       profiles = ['surf','therm','mid','bot'],
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                       savefig=True,
                       fig_suff=None):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_scientific(True)
    sci_formatter.set_powerlimits((0, 0))
    
    if (Kd_var == "Kd_int_tuned" or Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
        depth = myVars[f"{co2_scen}_ctrl_{starts[0]}_{ends[0]}_mean"]['z_i']
    elif (Kd_var == "Kd_lay_tuned" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        depth = myVars[f"{co2_scen}_ctrl_{starts[0]}_{ends[0]}_mean"]['z_l']

    prof_colors = ['b','m','g','r']
    time_strings = []
    for time_idx in range(len(starts)):
        time_strings.append(f'yr {starts[time_idx]}-{ends[time_idx]}')
    
    if Kd_var == "Kd_int_tuned":
        title_pref = r"Mean $\kappa_{\mathregular{add}}$"
    elif Kd_var == "Kd_interface":
        title_pref = r"Mean $\kappa_{\mathregular{tot}}$"
    elif Kd_var == "Kd_int_base":
        title_pref = r"Mean $\kappa_{\mathregular{base}}$"
    elif Kd_var == "Kd_lay_tuned":
        title_pref = r"Mean $\kappa_{\mathregular{add}}$"
    elif Kd_var == "Kd_layer":
        title_pref = r"Mean $\kappa_{\mathregular{tot}}$"
    elif Kd_var == "Kd_lay_base":
        title_pref = r"Mean $\kappa_{\mathregular{base}}$"

    if co2_scen == "const":
        co2_str = "Const CO\u2082"
    elif co2_scen == "doub":
        co2_str = "1pct2xCO\u2082"
    elif co2_scen == "quad":
        co2_str = "1pct4xCO\u2082"
        
    # plot for each power input
    
    line_list = ['solid','dashed','dotted']

    if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        custom_leg_1 = [Line2D([0], [0], color='k', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2)]
        custom_leg_2 = [Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
        
        leg_labels_1 = ['control']
        for elem in profiles:
            leg_labels_1.append(elem)

        leg_labels_2 = copy.deepcopy(power_strings)

    else:
        custom_leg = [Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
        
        leg_labels = copy.deepcopy(profiles)
        for elem in power_strings:
            leg_labels.append(elem)

    for time_idx in range(len(starts)):
        fig, ax = plt.subplots(figsize=(4,4.8))

        for pow_idx in range(len(power_var_suff)):
            if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
                ax.plot(myVars[f"{co2_scen}_ctrl_{starts[time_idx]}_{ends[time_idx]}_mean"]["Kd_interface"][0,:],depth,label='control',color='k',linestyle=line_list[pow_idx])
            elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
                ax.plot(myVars[f"{co2_scen}_ctrl_{starts[time_idx]}_{ends[time_idx]}_mean"]["Kd_layer"][0,:],depth,label='control',color='k',linestyle=line_list[pow_idx])

            for i in range(len(profiles)):
                ax.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{starts[time_idx]}_{ends[time_idx]}_mean"][Kd_var][0,:],depth,
                        linestyle=line_list[pow_idx],color=prof_colors[i])
            
        ax.set_xlabel(r"$\kappa_d$ (m/s$^2$)")
        ax.set_ylabel("Depth (m)")
        
        ax.xaxis.set_major_formatter(sci_formatter)
        ax.set_xlim(0,max_Kd)
        if Kd_var != 'Kd_int_tuned' and Kd_var != 'Kd_lay_tuned':
            ax.set_ylim(-50,max_z)
        else:
            ax.set_ylim(0,max_z)
                
        ax.invert_yaxis()
        
        # ax.legend(loc='best',ncol=2)
        if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
            # First legend (5 labels)
            legend1 = ax.legend(
                custom_leg_1, leg_labels_1,
                loc='lower right',
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(0.67, 0.0),  # Adjust position as needed
                frameon=True
            )
            # Second legend (3 labels, positioned below the first)
            legend2 = ax.legend(
                custom_leg_2, leg_labels_2,
                loc='lower right',
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(1.0, 0.0),  # Adjust position as needed
                frameon=True
            )
            
            # Add the first legend back to the axis
            ax.add_artist(legend1)
        else:
            ax.legend(custom_leg, leg_labels, loc='best', fontsize=10, ncol = 2, labelspacing=0.1)
        
        ax.grid("both")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.set_title(title_pref+f" {co2_str} Year {starts[time_idx]}-{ends[time_idx]}")
    
        if savefig:
            if fig_dir is None:
                raise ValueError("Must specify 'fig_dir' = <directory>.")
                
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            if fig_suff is None:
                plt.savefig(fig_dir+f'{co2_scen}_{Kd_var}_{str(starts[time_idx]).zfill(4)}_{str(ends[time_idx]).zfill(4)}.pdf', dpi=600, bbox_inches='tight')
            else:
                plt.savefig(fig_dir+f'{co2_scen}_{Kd_var}_{fig_suff}_{str(starts[time_idx]).zfill(4)}_{str(ends[time_idx]).zfill(4)}.pdf', dpi=600, bbox_inches='tight')
            plt.close()

    # --- combined figure with subplots and ONE shared legend ---
    n = len(starts)
    if n != 1:
        if (Kd_var in ["Kd_interface", "Kd_int_base", "Kd_layer", "Kd_lay_base"]):
            fig_height = 6.4
        else:
            fig_height = 6.0
        
        # Make a figure with two rows: plots (row 0) + legend (row 1)
        fig = plt.figure(figsize=(3.5 * n, fig_height))
        gs = GridSpec(2, n, height_ratios=[20, 3], hspace=0.25, figure=fig)
        
        axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis("off")  # pure legend canvas
        
        for time_idx, ax in enumerate(axes):
            for pow_idx in range(len(power_var_suff)):
                if Kd_var in ["Kd_interface", "Kd_int_base"]:
                    ax.plot(
                        myVars[f"{co2_scen}_ctrl_{starts[time_idx]}_{ends[time_idx]}_mean"]["Kd_interface"][0,:],
                        depth, label="control", color="k", linestyle=line_list[pow_idx]
                    )
                elif Kd_var in ["Kd_layer", "Kd_lay_base"]:
                    ax.plot(
                        myVars[f"{co2_scen}_ctrl_{starts[time_idx]}_{ends[time_idx]}_mean"]["Kd_layer"][0,:],
                        depth, label="control", color="k", linestyle=line_list[pow_idx]
                    )
                for i in range(len(profiles)):
                    ax.plot(
                        myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{starts[time_idx]}_{ends[time_idx]}_mean"][Kd_var][0,:],
                        depth, linestyle=line_list[pow_idx], color=prof_colors[i]
                    )
        
            ax.set_xlabel(r"$\kappa_d$ (m/s$^2$)")
            ax.xaxis.set_major_formatter(sci_formatter)
            ax.set_xlim(0, max_Kd)
            if Kd_var not in ["Kd_int_tuned", "Kd_lay_tuned"]:
                ax.set_ylim(-50, max_z)
            else:
                ax.set_ylim(0, max_z)
            ax.invert_yaxis()
            ax.grid("both")
            ax.minorticks_on()
            ax.grid(which="major", linestyle="-", linewidth=0.5, color="gray")
            ax.set_title(f"Year {starts[time_idx]}-{ends[time_idx]}")
        
        # after creating axes
        fig.supylabel("Depth (m)", x=0.02)  # move label closer to the panels
        fig.subplots_adjust(left=0.08)       # shrink the left gutter
        
        # Build legends on the dedicated legend axis so they never overlap plots
        if Kd_var in ["Kd_interface", "Kd_int_base", "Kd_layer", "Kd_lay_base"]:
            if n == 1:
                x_loc_1 = 0.3
                x_loc_2 = 0.7
            elif n == 2:
                x_loc_1 = 0.35
                x_loc_2 = 0.65
            else:
                x_loc_1 = 0.4
                x_loc_2 = 0.6
            
            # two separate legend groups, side-by-side
            leg1 = legend_ax.legend(
                custom_leg_1, leg_labels_1,
                loc="center", bbox_to_anchor=(x_loc_1, 0.5),
                ncol=2,
                fontsize=10, frameon=True, labelspacing=0.1
            )
            legend_ax.add_artist(leg1)
            legend_ax.legend(
                custom_leg_2, leg_labels_2,
                loc="center", bbox_to_anchor=(x_loc_2, 0.5),
                fontsize=10, frameon=True, labelspacing=0.1
            )
        else:
            legend_ax.legend(
                custom_leg, leg_labels,
                loc="center", bbox_to_anchor=(0.5, 0.35), ncol=2,
                fontsize=10, frameon=True, labelspacing=0.1
            )
        
        # Save
        if savefig:
            if fig_dir is None:
                raise ValueError("When savefig=True, provide fig_dir.")
            os.makedirs(fig_dir, exist_ok=True)
            if fig_suff is None:
                saved_path = os.path.join(fig_dir, f"{co2_scen}_{Kd_var}_by_power.pdf")
            else:
                saved_path = os.path.join(fig_dir, f"{co2_scen}_{Kd_var}_{fig_suff}_by_power.pdf")
            fig.savefig(saved_path, dpi=600, bbox_inches="tight")
            plt.close(fig)


# plotting Kd for three co2 scenarios (one subplot per power, only a single time chosen)

def plot_Kd_multi_co2(start,end,fig_dir,Kd_var,max_Kd,
                       max_z = 6250, 
                       profiles = ['surf','therm','mid','bot'],
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                       savefig=True,
                       fig_suff=None):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_scientific(True)
    sci_formatter.set_powerlimits((0, 0))
    
    if (Kd_var == "Kd_int_tuned" or Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
        depth = myVars[f"const_ctrl_{starts[0]}_{ends[0]}_mean"]['z_i']
    elif (Kd_var == "Kd_lay_tuned" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        depth = myVars[f"const_ctrl_{starts[0]}_{ends[0]}_mean"]['z_l']

    prof_colors = ['b','m','g','r']
    co2_types = ["const","doub","quad"]
    
    co2_leg_strings = ["Const CO\u2082", "1pct2xCO\u2082","1pct4xCO\u2082"]
    
    if Kd_var == "Kd_int_tuned":
        title_pref = r"Mean $\kappa_{\mathregular{add}}$"
    elif Kd_var == "Kd_interface":
        title_pref = r"Mean $\kappa_{\mathregular{tot}}$"
    elif Kd_var == "Kd_int_base":
        title_pref = r"Mean $\kappa_{\mathregular{base}}$"
    elif Kd_var == "Kd_lay_tuned":
        title_pref = r"Mean $\kappa_{\mathregular{add}}$"
    elif Kd_var == "Kd_layer":
        title_pref = r"Mean $\kappa_{\mathregular{tot}}$"
    elif Kd_var == "Kd_lay_base":
        title_pref = r"Mean $\kappa_{\mathregular{base}}$"
        
    # plot for each power input
    
    line_list = ['solid','dashed','dotted']

    if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        custom_leg_1 = [Line2D([0], [0], color='k', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2)]
        custom_leg_2 = [Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
        
        leg_labels_1 = ['control']
        for elem in profiles:
            leg_labels_1.append(elem)

        leg_labels_2 = copy.deepcopy(co2_leg_strings)

    else:
        custom_leg = [Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], linestyle=line_list[0], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[1], lw=2, color='k'),
                Line2D([0], [0], linestyle=line_list[2], lw=2, color='k')]
        
        leg_labels = copy.deepcopy(profiles)
        for elem in co2_leg_strings:
            leg_labels.append(elem)

    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(4,4.8))

        for co2_idx, co2_scen in enumerate(co2_types):
            if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
                ax.plot(myVars[f"{co2_scen}_ctrl_{start}_{end}_mean"]["Kd_interface"][0,:],depth,label='control',color='k',linestyle=line_list[co2_idx])
            elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
                ax.plot(myVars[f"{co2_scen}_ctrl_{start}_{end}_mean"]["Kd_layer"][0,:],depth,label='control',color='k',linestyle=line_list[co2_idx])

            for i in range(len(profiles)):
                ax.plot(myVars[f"{co2_scen}_{profiles[i]}_{power}_{start}_{end}_mean"][Kd_var][0,:],depth,
                        linestyle=line_list[co2_idx],color=prof_colors[i])
            
        ax.set_xlabel(r"$\kappa_d$ (m/s$^2$)")
        ax.set_ylabel("Depth (m)")
        
        ax.xaxis.set_major_formatter(sci_formatter)
        ax.set_xlim(0,max_Kd)
        if Kd_var != 'Kd_int_tuned' and Kd_var != 'Kd_lay_tuned':
            ax.set_ylim(-50,max_z)
        else:
            ax.set_ylim(0,max_z)
                
        ax.invert_yaxis()
        
        # ax.legend(loc='best',ncol=2)
        if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
            # First legend (5 labels)
            legend1 = ax.legend(
                custom_leg_1, leg_labels_1,
                loc='lower right',
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(0.67, 0.0),  # Adjust position as needed
                frameon=True
            )
            # Second legend (3 labels, positioned below the first)
            legend2 = ax.legend(
                custom_leg_2, leg_labels_2,
                loc='lower right',
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(1.0, 0.0),  # Adjust position as needed
                frameon=True
            )
            
            # Add the first legend back to the axis
            ax.add_artist(legend1)
        else:
            ax.legend(custom_leg, leg_labels, loc='best', fontsize=10, ncol = 2, labelspacing=0.1)
        
        ax.grid("both")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.set_title(title_pref+f" {power_strings[pow_idx]} Year {start}–{end}")
    
        if savefig:
            if fig_dir is None:
                raise ValueError("Must specify 'fig_dir' = <directory>.")
                
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            if fig_suff is None:
                plt.savefig(fig_dir+f'{start}_{end}_{power}_{Kd_var}.pdf', dpi=600, bbox_inches='tight')
            else:
                plt.savefig(fig_dir+f'{start}_{end}_{power}_{Kd_var}_{fig_suff}.pdf', dpi=600, bbox_inches='tight')
            plt.close()
            

    # --- combined figure with subplots and ONE shared legend ---
    n = len(power_inputs)
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 4.8), sharey='row')

    # Make axes iterable even if n == 1
    axes = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
    
    for pow_idx, ax in enumerate(axes):
        for co2_idx, co2_scen in enumerate(co2_types):
            if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base"):
                ax.plot(myVars[f"{co2_scen}_ctrl_{start}_{end}_mean"]["Kd_interface"][0,:],depth,label='control',color='k',linestyle=line_list[co2_idx])
            elif (Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
                ax.plot(myVars[f"{co2_scen}_ctrl_{start}_{end}_mean"]["Kd_layer"][0,:],depth,label='control',color='k',linestyle=line_list[co2_idx])

            for i in range(len(profiles)):
                ax.plot(myVars[f"{co2_scen}_{profiles[i]}_{power_var_suff[pow_idx]}_{start}_{end}_mean"][Kd_var][0,:],depth,
                        linestyle=line_list[co2_idx],color=prof_colors[i])
            
        ax.set_xlabel(r"$\kappa_d$ (m/s$^2$)")
        
        ax.xaxis.set_major_formatter(sci_formatter)
        ax.set_xlim(0,max_Kd)
        if Kd_var != 'Kd_int_tuned' and Kd_var != 'Kd_lay_tuned':
            ax.set_ylim(-50,max_z)
        else:
            ax.set_ylim(0,max_z)
                
        ax.invert_yaxis()

        ax.grid("both")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        
        ax.set_title(f"Year {start}–{end}")

    fig.supylabel("Depth (m)", x=0.06)
    # # Overall title
    # fig.suptitle(f"{title_pref}", y=0.98, fontsize=14)

    # Put the shared legend centered below the title (outside the axes)
    # Adjust rect to leave headroom for suptitle + legend.

    if (Kd_var == "Kd_interface" or Kd_var == "Kd_int_base" or Kd_var == "Kd_layer" or Kd_var == "Kd_lay_base"):
        # First legend (5 labels)
        legend1 = fig.legend(
            custom_leg_1, leg_labels_1,
            loc='lower right',
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(0.67, 0.0),  # Adjust position as needed
            frameon=True
        )
        # Second legend (3 labels, positioned below the first)
        legend2 = fig.legend(
            custom_leg_2, leg_labels_2,
            loc='lower right',
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(1.0, 0.0),  # Adjust position as needed
            frameon=True
        )
        
        # Add the first legend back to the axis
        axes.add_artist(legend1)
    else:
        fig.legend(
            custom_leg, leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            fontsize=10, ncol = 2, labelspacing=0.1, frameon=True
        )

    # fig.tight_layout(rect=[0, 0, 1, 0.90])

    if savefig:
        if fig_dir is None:
            raise ValueError("When savefig=True, provide fig_dir.")
        os.makedirs(fig_dir, exist_ok=True)
        if fig_suff is None:
            saved_path = os.path.join(fig_dir, f"{start}_{end}_{Kd_var}_co2_comparison.pdf")
        else:
            saved_path = os.path.join(fig_dir, f"{start}_{end}_{Kd_var}_{fig_suff}_co2_comparison.pdf")
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


# ## Global maps

def plot_pp_Kd_map(title,pp_ds,Kd_var,z_idx,start_yr,end_yr,layer_var=False,savefig=False,cb_min=-10,\
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

    # Step 1: Normalize geolon to [0, 360) to avoid wraparound issues
    log_Kd_dat = log_Kd_dat.assign_coords(
        geolon=((log_Kd_dat.geolon + 360) % 360)
    )
    
    # Step 2: Define target lat/lon grid resolution
    lat_res = 3 * 210  # e.g., 630 points from -76.75 to 89.75
    lon_res = 3 * 360  # e.g., 1080 points from 0 to 360
    
    target_lat = np.linspace(log_Kd_dat.geolat.min(), log_Kd_dat.geolat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res)
    
    # Step 3: Build source and target grid datasets
    ds_in = xr.Dataset({
        "lat": (["yh", "xh"], log_Kd_dat.geolat.values),
        "lon": (["yh", "xh"], log_Kd_dat.geolon.values),
    })
    
    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })
    
    # Step 4: Create the regridder (periodic=True for wrapping at 0/360)
    regridder = xe.Regridder(ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False)
    
    # Step 5: Apply the regridder to your data
    log_Kd_dat_interp = regridder(log_Kd_dat)
    
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
    
    cmap = cmocean.cm.matter  # define the colormap
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
    
    Kd_plot = log_Kd_dat_interp.plot(vmin=plot_min, vmax=plot_max,
                  x='geolon', y='geolat',
                  cmap=disc_bal_cmap, norm=disc_norm,
                  subplot_kws=subplot_kws,
                      #You can pick any projection from the cartopy list but, whichever projection you use, you still have to set
                  transform=ccrs.PlateCarree(),
                  add_labels=False,
                  add_colorbar=False)
    
    # Kd_plot.axes.coastlines()
    Kd_plot.axes.set_title(f"{title}: Year {start_yr}–{end_yr}, z = {depth:,.2f} m",fontdict={'fontsize':18})

    # Determine the extend setting for the colorbar arrows
    if dat_min < plot_min and dat_max > plot_max:
        extend = 'both'
    elif dat_min < plot_min:
        extend = 'min'
    elif dat_max > plot_max:
        extend = 'max'
    else:
        extend = 'neither'
    
    # Kd_cb = plt.colorbar(Kd_plot, fraction=0.046, pad=0.04)
    Kd_cb = plt.colorbar(Kd_plot, ticks=tick_arr, shrink=0.6, extend=extend) #fraction=0.046, pad=0.04,

    # tick_labels = [f"{x:.0f}" for x in tick_arr] # str(x)
    # tick_labels[np.ceil(num)] = "0"
    Kd_cb.set_ticks(tick_arr)
    Kd_cb.ax.set_yticklabels(tick_labels)
    Kd_cb.ax.tick_params(labelsize=14)
    Kd_cb.set_label("log$_{10}$ ($m^2/s$)",fontdict={'fontsize':14})

    for t in Kd_cb.ax.get_yticklabels():
        t.set_horizontalalignment('center')   
        t.set_x(2.0)

    if savefig is True:
        plt.savefig(f'{prefix}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}_z_{depth:.0f}.pdf', dpi=600, bbox_inches='tight')
        plt.close()


# ## Basin mean diffusivity

def plot_Kd_basin(
    title, pp_ds, Kd_var, basin_name, max_depth, start_yr, end_yr, 
    icon=None,
    layer_var=False,
    cb_min=-7, cb_max=None, cb_spacing=0.25, non_lin_cb_val=None,
    savefig=False, fig_dir=None, prefix=None,
    check_nn=True, nn_threshold=0.00, full_field_var=None, verbose=False,
    # NEW:
    ax=None,                         # draw onto this axes if provided
    cmap=None, norm=None, tick_arr=None, extend=None,  # allow external color scaling
    add_colorbar=False,              # keep default single-panel behavior
    ):

    lat_res = 1000
    z_res = 200
    
    Kd_dat = get_pp_basin_dat(pp_ds, Kd_var, basin_name,
                              check_nn=check_nn, nn_threshold=nn_threshold,
                              full_field_var=full_field_var)

    log_Kd_dat = np.log10(Kd_dat)
    log_Kd_dat = log_Kd_dat.where(log_Kd_dat != -np.inf, -100)

    dat_min = np.nanmin(log_Kd_dat.values)
    dat_max = np.nanmax(log_Kd_dat.values)
    
    # ------- Interpolation (unchanged) -------
    if layer_var is False:
        fine_lat   = np.linspace(log_Kd_dat.true_lat.min(), log_Kd_dat.true_lat.max(), lat_res)
        fine_depth = np.linspace(log_Kd_dat.z_i.min(),       log_Kd_dat.z_i.max(),       z_res)
        log_Kd_dat = log_Kd_dat.interp(true_lat=fine_lat, z_i=fine_depth)
        y_name = 'z_i'
    else:
        fine_lat   = np.linspace(log_Kd_dat.true_lat.min(), log_Kd_dat.true_lat.max(), lat_res)
        fine_depth = np.linspace(log_Kd_dat.z_l.min(),       log_Kd_dat.z_l.max(),       z_res)
        log_Kd_dat = log_Kd_dat.interp(true_lat=fine_lat, z_l=fine_depth)
        y_name = 'z_l'

    # ------- Color scaling (compute only if not supplied) -------
    if cmap is None:
        base_cmap = cmocean.cm.matter
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'Custom cmap', [base_cmap(i) for i in range(base_cmap.N)], base_cmap.N
        )

    if (norm is None) or (tick_arr is None) or (extend is None):
        plot_min = cb_min if cb_min is not None else np.floor(dat_min)
        plot_max = cb_max if cb_max is not None else np.ceil(dat_max)

        if non_lin_cb_val is not None:
            num_col_lower  = 2*int(non_lin_cb_val - plot_min)
            num_ticks_lower = int(non_lin_cb_val - plot_min)
            num_col_upper  = int((plot_max - non_lin_cb_val)/cb_spacing)
            num_ticks_upper = int((plot_max - (non_lin_cb_val))/(2*cb_spacing))
            lower_bounds = np.linspace(plot_min, non_lin_cb_val, num_col_lower, endpoint=False)
            lower_ticks  = np.linspace(plot_min, non_lin_cb_val, num_ticks_lower, endpoint=False)
            upper_bounds = np.linspace(non_lin_cb_val, plot_max, num_col_upper + 1)
            upper_ticks  = np.linspace(non_lin_cb_val, plot_max, num_ticks_upper + 1)
            norm_bounds  = np.concatenate((lower_bounds, upper_bounds))
            tick_arr     = np.concatenate((lower_ticks, upper_ticks))
        else:
            num_col  = int((plot_max - plot_min)/cb_spacing)
            num_ticks = int((plot_max - plot_min)/(2*cb_spacing))
            norm_bounds = np.linspace(plot_min, plot_max, num_col + 1)
            tick_arr    = np.linspace(plot_min, plot_max, num_ticks + 1)

        norm = mcolors.BoundaryNorm(norm_bounds, cmap.N)

        if dat_min < plot_min and dat_max > plot_max:
            extend = 'both'
        elif dat_min < plot_min:
            extend = 'min'
        elif dat_max > plot_max:
            extend = 'max'
        else:
            extend = 'neither'

    # ------- Create figure/axes only if needed -------
    created_fig = False
    if ax is None:
        plt.figure(figsize=[7,3])
        ax = plt.gca()
        created_fig = True

    # ------- Plot onto the provided axes -------
    Kd_p = log_Kd_dat.plot(
        x='true_lat', y=y_name, cmap=cmap, norm=norm,
        facecolor='grey', add_labels=False, add_colorbar=False, ax=ax
    )

    # ------- Bathymetry overlay -------
    zonal_pct_bathy, lat_vals = bathymetry_overlay(pp_ds, log_Kd_dat, fine_lat,
                                                   basin_name, depth_var='deptho')
    ax.fill_between(lat_vals, max_depth, zonal_pct_bathy, color='grey', zorder=20)

    # ------- Ax cosmetics -------
    ax.set_ylim(0, max_depth)
    ax.invert_yaxis()
    ax.minorticks_on()
    ax.tick_params(axis='y')#, labelsize=12)
    ax.set_ylabel('Depth (m)')#, fontsize=12)
    ax.set_title(title)#, fontsize=14)

    xticks = np.arange(-60, 61, 20)
    xlabels = [f"{abs(e)}$\\degree$S" if e<0 else ("0$\\degree$" if e==0 else f"{e}$\\degree$N") for e in xticks]
    ax.set_xticks(ticks=xticks, labels=xlabels)#, fontsize=12)

    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        ab = AnnotationBbox(OffsetImage(img, zoom=0.065), (0.95, 1.09),#zoom=0.085
                            xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # ------- Optional panel-specific colorbar (default off) -------
    if add_colorbar:
        cbar = plt.colorbar(Kd_p, ax=ax, ticks=tick_arr, fraction=0.046, pad=0.04, extend=extend)
        cbar.set_ticks(tick_arr)
        cbar.ax.set_yticklabels([f"{x:.1f}" for x in tick_arr])
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(r"log$_{10}$ ($\kappa_d$) ($m^2/s$)", fontsize=12)

    # ------- Optional save/close for single-panel usage -------
    if savefig:
        if fig_dir is None or prefix is None:
            raise ValueError("When savefig=True, provide fig_dir and prefix.")
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(fig_dir + f'{prefix}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
                    dpi=600, bbox_inches='tight')
        if created_fig:
            plt.close()

    # Return bits needed for shared colorbar in a grid
    return {"mappable": Kd_p, "cmap": cmap, "norm": norm, "ticks": tick_arr, "extend": extend}


def plot_Kd_basin_2x2(
    Kd_var, basin_name, max_depth, start_yr, end_yr, power_suff, co2_suff,
    cb_min=-7, cb_max=None, cb_spacing=0.25, non_lin_cb_val=None,
    profiles=['surf','therm','mid','bot'],
    prof_strings = ["Surf","Therm","Mid","Bot"],
    figsize=(12, 8), sharex='col', sharey='row', tight=True,
    layer_var=False,
    fig_title=None,
    savefig=False, fig_dir=None, prefix=None
    ):
    """
    All 4 subplots will share the same cmap/norm/ticks and a single colorbar.
    """

    basin_dict = {'global': 'Global',
                  'atl-arc': 'Atlantic',
                  'atl-arc-no-marg': 'Atlantic',
                  'pac': 'Pacific'
                 }
    Kd_dict = {'Kd_int_tuned': r'$\kappa_{\mathregular{add}}$',
               'Kd_int_base': r'$\kappa_{\mathregular{base}}$',
               'Kd_interface': r'$\kappa_{\mathregular{tot}}$'
              }

    # First pass (lightweight) to compute global bounds for color scaling
    # We’ll just reuse panel[0]’s computed cmap/norm/ticks and force others to match,
    # OR you can precompute your own cb_min/max and pass them in as function args.
    # Here we compute using panel 0 call (no drawing to screen: we create a temp Axes).
    
    fig_tmp, ax_tmp = plt.subplots(figsize=(1,1))
    
    ds0_name = f"{co2_suff}_{profiles[3]}_{power_suff}_{start_yr}_{end_yr}"
    
    info0 = plot_Kd_basin(prof_strings[3], myVars[ds0_name], Kd_var, basin_name, max_depth, start_yr, end_yr,
                          icon=profiles[3],
                          layer_var=layer_var,
                          ax=ax_tmp,
                          cb_min=cb_min, cb_max=cb_max,cb_spacing=cb_spacing, non_lin_cb_val=non_lin_cb_val,
                          add_colorbar=False)
    plt.close(fig_tmp)

    cmap, norm, tick_arr, extend = info0["cmap"], info0["norm"], info0["ticks"], info0["extend"]

    # Build the 2x2 grid with shared axes
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=sharex, sharey=sharey,
                            constrained_layout=tight)
    axs = axs.ravel()

    mappables = []
    for i in range(4):
        ds_name = f"{co2_suff}_{profiles[i]}_{power_suff}_{start_yr}_{end_yr}"
        info = plot_Kd_basin(prof_strings[i], myVars[ds_name], Kd_var, basin_name, max_depth, start_yr, end_yr,
                             icon=profiles[i],
                             layer_var=layer_var,
                             ax=axs[i],
                             cmap=cmap, norm=norm, tick_arr=tick_arr, extend=extend,
                             add_colorbar=False)
        mappables.append(info["mappable"])

        # # Optionally add panel letters
        # axs[i].text(0.02, 1.1, f"({chr(97+i)})", transform=axs[i].transAxes,#0.02, 0.95
        #             va='top', ha='left', fontsize=12, fontweight='bold')
        
        # Optional: simplify labels on inner axes
        row, col = divmod(i, 2)
        if row == 0:
            axs[i].set_xticklabels([])  # hide top-row x tick labels
        if col == 1:
            axs[i].set_ylabel("")       # hide right-column y label

    # Shared colorbar
    cbar = fig.colorbar(mappables[0], ax=axs, ticks=tick_arr,
                        fraction=0.046, pad=0.02, extend=extend)
    cbar.ax.set_yticklabels([f"{x:.1f}" for x in tick_arr])
    cbar.set_label(r"log$_{10}$ ($\kappa_d$) ($m^2/s$)", fontsize=12)

    # # Figure-level title
    # if fig_title != None:
    #     # With constrained_layout=True, a slightly lower y avoids clipping
    #     fig.suptitle(rf"{fig_title}\n{basin_dict[basin_name]} Mean {Kd_dict[Kd_var]}: Year {start_yr}–{end_yr}", y=1.025)
    # else:
    #     fig.suptitle(rf"{basin_dict[basin_name]} Mean {Kd_dict[Kd_var]}: Year {start_yr}–{end_yr}", y=1.025)
        

    # Optional save
    saved_path = None
    if savefig:
        if fig_dir is None or prefix is None:
            raise ValueError("When savefig=True, provide fig_dir and prefix.")
        os.makedirs(fig_dir, exist_ok=True)
        saved_path = os.path.join(fig_dir, f"{prefix}_{Kd_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png")
        fig.savefig(saved_path, dpi=600, bbox_inches='tight')
        plt.close(fig)

    return fig, axs, saved_path




