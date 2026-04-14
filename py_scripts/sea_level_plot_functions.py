#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for plotting sea level anomaly maps and time series.

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
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

from xgcm import Grid
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

import cartopy.crs as ccrs
import cmocean

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import subprocess as sp

import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from matplotlib.ticker import ScalarFormatter

from xclim import ensembles
from xoverturning import calcmoc
import cmip_basins
import momlevel

import cftime
from pandas.errors import OutOfBoundsDatetime  # Import the specific error

import os


get_ipython().run_line_magic('run', '/home/Kiera.Lowman/Kd-sensitivity-analysis/notebooks/plotting_functions.ipynb')


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

plt.rcParams['axes.labelsize'] = 12    # Axis label size
plt.rcParams['xtick.labelsize'] = 10     # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 10     # Y-axis tick label size
plt.rcParams['axes.titlesize'] = 14      # Title size
plt.rcParams['legend.fontsize'] = 10     # Legend font size


# # Sea level maps

def compute_SL_diff(
    ref_ds=None,
    exp_ds=None,
    z_vals_ds=None,
    variant="steric",
    interp_method="bilinear",
    lat_res=None,
    lon_res=None,
    periodic=True,
    reuse_weights=False,
):

    if ref_ds is None or exp_ds is None:
        raise ValueError("ref_ds and exp_ds must both be provided.")

    ref_ds = ref_ds.copy(deep=False)
    exp_ds = exp_ds.copy(deep=False)
    
    if z_vals_ds:
        ref_ds["z_i"] = z_vals_ds["z_i"]
        exp_ds["z_i"] = z_vals_ds["z_i"]
    
    # ---- your original SLA computation ----

    rename_map = {}
    if "temp" in ref_ds: rename_map["temp"] = "thetao"
    if "salt" in ref_ds: rename_map["salt"] = "so"
    if rename_map:
        ref_ds = ref_ds.rename(rename_map)
        
    if not ref_ds.attrs.get("_area_weighted", False):
        ref_ds["areacello"] = ref_ds["areacello"] * ref_ds["wet"]
        ref_ds.attrs["_area_weighted"] = True
    
    # ref_ds["areacello"] = ref_ds["areacello"] * ref_ds["wet"]
    # ref_ds = ref_ds.rename({"temp": "thetao", "salt": "so"})
    
    ref_state = momlevel.reference.setup_reference_state(ref_ds)

    rename_map = {}
    if "temp" in exp_ds: rename_map["temp"] = "thetao"
    if "salt" in exp_ds: rename_map["salt"] = "so"
    if rename_map:
        exp_ds = exp_ds.rename(rename_map)

    if not exp_ds.attrs.get("_area_weighted", False):
        exp_ds["areacello"] = exp_ds["areacello"] * exp_ds["wet"]
        exp_ds.attrs["_area_weighted"] = True
        
    # exp_ds["areacello"] = exp_ds["areacello"] * exp_ds["wet"]
    # exp_ds = exp_ds.rename({"temp": "thetao", "salt": "so"})

    exp_result = momlevel.steric(exp_ds, reference=ref_state, variant=variant)
    exp_slr_native = exp_result[0][variant]

    global_result = momlevel.steric(exp_ds, reference=ref_state, variant=variant, domain="global")
    global_slr = global_result[0][variant]

    # ---- normalize & regrid to regular (lat, lon) ----
    exp_slr_native = exp_slr_native.assign_coords(geolon=((exp_slr_native.geolon + 360) % 360))
    # lat_res = 3 * 210
    # lon_res = 3 * 360
    # ---- interpolation target ----
    if lat_res is None:
        lat_res = 3 * 210
    if lon_res is None:
        lon_res = 3 * 360
    target_lat = np.linspace(exp_slr_native.geolat.min(), exp_slr_native.geolat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res, endpoint=False) # added November

    ds_in = xr.Dataset({
        "lat": (["yh", "xh"], exp_slr_native.geolat.values),
        "lon": (["yh", "xh"], exp_slr_native.geolon.values),
    })
    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })
    regridder = xe.Regridder(ds_in, ds_out, method=interp_method, periodic=periodic, reuse_weights=reuse_weights)
    exp_slr_interp = regridder(exp_slr_native)
    

    # ---- package result ----
    out_ds = xr.Dataset(
        data_vars={
            "sla_native": exp_slr_native,
            "sla_interp": exp_slr_interp,
            "global_mean_sla": global_slr,
        },
        attrs={
            "variant": variant,
            "interp_method": interp_method,
            "periodic": periodic,
            "lat_res": lat_res,
            "lon_res": lon_res,
        }
    )

    # Keep some descriptive metadata
    out_ds["sla_native"].attrs["long_name"] = f"{variant} sea level anomaly"
    out_ds["sla_native"].attrs["units"] = "m"

    out_ds["sla_interp"].attrs["long_name"] = f"{variant} sea level anomaly (interpolated)"
    out_ds["sla_interp"].attrs["units"] = "m"

    out_ds["global_mean_sla"].attrs["long_name"] = f"global mean {variant} sea level anomaly"
    out_ds["global_mean_sla"].attrs["units"] = "m"

    return out_ds


def plot_SL_diff( #_from_processed
    sl_ds=None,
    variant="steric",
    panel_title=None,
    start_yr=None,
    end_yr=None,
    cb_max=None,
    icon=None,
    savefig=False,
    fig_dir=None,
    prefix=None,
    verbose=False,

    # grid hooks
    ax=None,
    add_colorbar=True,
    return_cb_params=False,
    cb_label="Sea Level Anomaly (m)",

    # plotting choices
    use_interp=True,
    title=None,
):

    if sl_ds is None:
        raise ValueError("sl_ds must be provided.")

    panel_title = panel_title or title or ""
    check_variant = sl_ds.attrs.get("variant")
    if variant != check_variant:
        raise ValueError(f'Variant mismatch: requested "{variant}", 'f'but dataset has "{check_variant}"')

    plot_da = sl_ds["sla_interp"] if use_interp else sl_ds["sla_native"]

    # ---- color scaling ----
    # Match your old behavior by basing color scaling on the native field
    native_da = sl_ds["sla_native"]

    min_val = float(np.nanmin(native_da.values))
    max_val = float(np.nanmax(native_da.values))
    per0p5 = float(np.nanpercentile(native_da.values, 0.5))
    per99p5 = float(np.nanpercentile(native_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    extra_tick_digits = False
    if cb_max is not None:
        chosen_n = 20
        data_max = cb_max
        chosen_step = 2 * data_max / chosen_n
        if cb_max in (0.06, 0.075, 0.08):
            extra_tick_digits = True
    else:
        chosen_n, chosen_step = get_cb_spacing(
            per0p5, per99p5,
            min_bnd=0.2, min_spacing=0.02,
            min_n=10, max_n=20,
            verbose=verbose
        )

    max_mag = 0.5 * chosen_n * chosen_step

    zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
        max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
    )

    # ---- figure/axes management ----
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=209.5)}
        )

    # ---- draw the panel ----
    if use_interp:
        diff_plot = plot_da.plot(
            x="lon", y="lat",
            cmap=disc_cmap, norm=disc_norm,
            transform=ccrs.PlateCarree(),
            add_labels=False, add_colorbar=False, ax=ax
        )
    else:
        diff_plot = plot_da.plot(
            x="geolon", y="geolat",
            cmap=disc_cmap, norm=disc_norm,
            transform=ccrs.PlateCarree(),
            add_labels=False, add_colorbar=False, ax=ax
        )

    # ---- titles ----
    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\n{variant.capitalize()} SLR: Year {start_yr}–{end_yr}")

    # ---- optional per-panel colorbar ----
    diff_cb = None
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, shrink=0.58, pad=0.04, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing="proportional"
        )

        tick_labels = []
        for val in tick_positions:
            if extra_tick_digits:
                tick_labels.append(f"{val:.3f}")
            elif chosen_step < 0.1:
                tick_labels.append(f"{val:.2f}")
            else:
                tick_labels.append(f"{val:.1f}")

        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.set_yticklabels(tick_labels)
        diff_cb.ax.tick_params(labelsize=10, pad=14)
        diff_cb.set_label(cb_label, labelpad=12)

    # ---- package params for a figure-level colorbar ----
    cb_params = None
    if return_cb_params:
        tick_labels = []
        for val in tick_positions:
            if np.abs(val) in {0.125, 1/3, 2/3, 5/8}:
                tick_labels.append(f"{val:.3f}")
            elif chosen_step < 0.1:
                tick_labels.append(f"{val:.2f}")
            else:
                tick_labels.append(f"{val:.1f}")

        cb_params = dict(
            mappable=diff_plot,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing="proportional",
            ticks=tick_positions,
            ticklabels=tick_labels,
            label=cb_label
        )

    # ---- annotation: global mean ----
    mean_val = 1e3 * sl_ds["global_mean_sla"].isel(time=0).values
    ax.text(
        0.17, 0.8, f"{mean_val:.1f} mm",
        transform=ax.transAxes,
        fontsize=12, va="top", ha="center",
        color="white", fontweight="bold", alpha=1
    )

    # ---- optional corner icon ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(imagebox, (0.95, 1.00), xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # ---- save if we created the figure ----
    if savefig and created_fig is not None:
        out = fig_dir + f"{prefix}_{variant}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png"
        created_fig.savefig(out, dpi=600, bbox_inches="tight")
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


def plot_SL_diff_region( #_from_processed
    sl_ds=None,
    panel_title=None,
    start_yr=None,
    end_yr=None,
    cb_max=None,
    icon=None,
    savefig=False,
    fig_dir=None,
    prefix=None,
    verbose=False,

    # region definition
    lon_bounds=None,            # e.g. (-85, 20)
    lat_bounds=None,            # e.g. (0, 65)
    region_name=None,

    # plotting options
    use_interp=True,
    projection=None,
    central_longitude=0,
    add_coastlines=True,
    coastline_res="50m",
    coastline_lw=0.6,
    add_gridlines=False,

    # grid hooks
    ax=None,
    add_colorbar=True,
    return_cb_params=False,
    cb_label="Sea Level Anomaly (m)",

    # backwards-compat
    title=None,
):

    if sl_ds is None:
        raise ValueError("sl_ds must be provided.")
    if lon_bounds is None or lat_bounds is None:
        raise ValueError("lon_bounds and lat_bounds must both be provided.")

    panel_title = panel_title or title or ""
    variant = sl_ds.attrs.get("variant", "steric")

    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds

    if projection is None:
        projection = ccrs.PlateCarree(central_longitude=central_longitude)

    # ---- choose plotting field ----
    plot_da = sl_ds["sla_interp"] if use_interp else sl_ds["sla_native"]

    # ---- subset to region ----
    if use_interp:
        # sla_interp is assumed to use lon in [0, 360)
        lon_min_360 = lon_min % 360
        lon_max_360 = lon_max % 360

        if lon_min_360 <= lon_max_360:
            plot_region = plot_da.sel(
                lon=slice(lon_min_360, lon_max_360),
                lat=slice(lat_min, lat_max)
            )
        else:
            # Handle wraparound across 0E / dateline
            left = plot_da.sel(lon=slice(lon_min_360, 360), lat=slice(lat_min, lat_max))
            right = plot_da.sel(lon=slice(0, lon_max_360), lat=slice(lat_min, lat_max))
            plot_region = xr.concat([left, right], dim="lon")

        # Convert lon from [0, 360) to [-180, 180) and sort for plotting
        plot_region = plot_region.assign_coords(
            lon=((plot_region.lon + 180) % 360) - 180
        ).sortby("lon")
        
    else:
        # Native grid case: do a mask instead of a slice
        geolon = (plot_da["geolon"] + 360) % 360
        lon_min_360 = lon_min % 360
        lon_max_360 = lon_max % 360

        if lon_min_360 <= lon_max_360:
            lon_mask = (geolon >= lon_min_360) & (geolon <= lon_max_360)
        else:
            lon_mask = (geolon >= lon_min_360) | (geolon <= lon_max_360)

        lat_mask = (plot_da["geolat"] >= lat_min) & (plot_da["geolat"] <= lat_max)
        plot_region = plot_da.where(lon_mask & lat_mask)

    # ---- color scaling ----
    # Keep same philosophy as your original function:
    # color limits based on full field, not regional subset
    native_da = sl_ds["sla_native"]

    min_val = float(np.nanmin(native_da.values))
    max_val = float(np.nanmax(native_da.values))
    per0p5 = float(np.nanpercentile(native_da.values, 0.5))
    per99p5 = float(np.nanpercentile(native_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    extra_tick_digits = False
    if cb_max is not None:
        chosen_n = 20
        data_max = cb_max
        chosen_step = 2 * data_max / chosen_n
        if cb_max in (0.06, 0.075, 0.08):
            extra_tick_digits = True
    else:
        chosen_n, chosen_step = get_cb_spacing(
            per0p5, per99p5,
            min_bnd=0.2, min_spacing=0.02,
            min_n=10, max_n=20,
            verbose=verbose
        )

    max_mag = 0.5 * chosen_n * chosen_step

    zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
        max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
    )

    # ---- figure/axes management ----
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={"projection": projection}
        )

    # ---- draw the panel ----
    if use_interp:
        diff_plot = plot_region.plot(
            x="lon", y="lat",
            cmap=disc_cmap, norm=disc_norm,
            transform=ccrs.PlateCarree(),
            add_labels=False, add_colorbar=False, ax=ax
        )
    else:
        diff_plot = plot_region.plot(
            x="geolon", y="geolat",
            cmap=disc_cmap, norm=disc_norm,
            transform=ccrs.PlateCarree(),
            add_labels=False, add_colorbar=False, ax=ax
        )

    # Make the panel a geographic rectangle
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    if add_coastlines:
        ax.coastlines(resolution=coastline_res, linewidth=coastline_lw)

    if add_gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    # ---- titles ----
    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        region_text = f"\n{region_name}" if region_name else ""
        ax.set_title(f"{panel_title}{region_text}\n{variant.capitalize()} SLR: Year {start_yr}–{end_yr}")

    # ---- optional per-panel colorbar ----
    diff_cb = None
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, shrink=0.58, pad=0.04, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing="proportional"
        )

        tick_labels = []
        for val in tick_positions:
            if extra_tick_digits:
                tick_labels.append(f"{val:.3f}")
            elif chosen_step < 0.1:
                tick_labels.append(f"{val:.2f}")
            else:
                tick_labels.append(f"{val:.1f}")

        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.set_yticklabels(tick_labels)
        diff_cb.ax.tick_params(labelsize=10, pad=14)
        diff_cb.set_label(cb_label, labelpad=12)

    # ---- package params for a figure-level colorbar ----
    cb_params = None
    if return_cb_params:
        tick_labels = []
        for val in tick_positions:
            if np.abs(val) in {0.125, 1/3, 2/3, 5/8}:
                tick_labels.append(f"{val:.3f}")
            elif chosen_step < 0.1:
                tick_labels.append(f"{val:.2f}")
            else:
                tick_labels.append(f"{val:.1f}")

        cb_params = dict(
            mappable=diff_plot,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing="proportional",
            ticks=tick_positions,
            ticklabels=tick_labels,
            label=cb_label
        )

    # ---- annotation: global mean ----
    mean_val = 1e3 * sl_ds["global_mean_sla"].isel(time=0).values
    ax.text(
        0.17, 0.8, f"{mean_val:.1f} mm",
        transform=ax.transAxes,
        fontsize=12, va="top", ha="center",
        color="white", fontweight="bold", alpha=1
    )

    # ---- optional corner icon ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(imagebox, (0.95, 1.00), xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # ---- save if we created the figure ----
    if savefig and created_fig is not None:
        region_tag = f"_{region_name}" if region_name else ""
        out = fig_dir + f"{prefix}{region_tag}_{variant}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png"
        created_fig.savefig(out, dpi=600, bbox_inches="tight")
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


# def plot_SL_diff(
#     panel_title=None,                       # short title when used inside a grid
#     ref_ds=None, exp_ds=None, z_vals_ds=None,
#     start_yr=None, end_yr=None, variant="steric",
#     cb_max=None, icon=None,
#     savefig=False, fig_dir=None, prefix=None,
#     verbose=False,
#     # grid hooks:
#     ax=None,                 # if provided, draw into this axes (no new fig)
#     add_colorbar=True,       # grid sets this False (figure-level bar instead)
#     return_cb_params=False,  # grid sets this True (to build figure-level bar)
#     cb_label="Sea Level Anomaly (m)",
#     # backwards-compat if you still call with `title=...`
#     title=None
# ):
#     # ---- titles / labels ----
#     panel_title = panel_title or title or ""

#     ref_ds = ref_ds.copy(deep=False)
#     exp_ds = exp_ds.copy(deep=False)
    
#     if z_vals_ds:
#         ref_ds["z_i"] = z_vals_ds["z_i"]
#         exp_ds["z_i"] = z_vals_ds["z_i"]
    
#     # ---- your original SLA computation ----

#     rename_map = {}
#     if "temp" in ref_ds: rename_map["temp"] = "thetao"
#     if "salt" in ref_ds: rename_map["salt"] = "so"
#     if rename_map:
#         ref_ds = ref_ds.rename(rename_map)
        
#     if not ref_ds.attrs.get("_area_weighted", False):
#         ref_ds["areacello"] = ref_ds["areacello"] * ref_ds["wet"]
#         ref_ds.attrs["_area_weighted"] = True
    
#     # ref_ds["areacello"] = ref_ds["areacello"] * ref_ds["wet"]
#     # ref_ds = ref_ds.rename({"temp": "thetao", "salt": "so"})
    
#     ref_state = momlevel.reference.setup_reference_state(ref_ds)

#     rename_map = {}
#     if "temp" in exp_ds: rename_map["temp"] = "thetao"
#     if "salt" in exp_ds: rename_map["salt"] = "so"
#     if rename_map:
#         exp_ds = exp_ds.rename(rename_map)

#     if not exp_ds.attrs.get("_area_weighted", False):
#         exp_ds["areacello"] = exp_ds["areacello"] * exp_ds["wet"]
#         exp_ds.attrs["_area_weighted"] = True
        
#     # exp_ds["areacello"] = exp_ds["areacello"] * exp_ds["wet"]
#     # exp_ds = exp_ds.rename({"temp": "thetao", "salt": "so"})

#     exp_result = momlevel.steric(exp_ds, reference=ref_state, variant=variant)
#     exp_slr = exp_result[0][variant]

#     global_result = momlevel.steric(exp_ds, reference=ref_state, variant=variant, domain="global")
#     global_slr = global_result[0][variant]

#     # ---- normalize & regrid to regular (lat, lon) ----
#     exp_slr = exp_slr.assign_coords(geolon=((exp_slr.geolon + 360) % 360))
#     lat_res = 3 * 210
#     lon_res = 3 * 360
#     target_lat = np.linspace(exp_slr.geolat.min(), exp_slr.geolat.max(), lat_res)
#     target_lon = np.linspace(0, 360, lon_res, endpoint=False) # added November

#     ds_in = xr.Dataset({
#         "lat": (["yh", "xh"], exp_slr.geolat.values),
#         "lon": (["yh", "xh"], exp_slr.geolon.values),
#     })
#     ds_out = xr.Dataset({
#         "lat": (["lat"], target_lat),
#         "lon": (["lon"], target_lon),
#     })
#     regridder = xe.Regridder(ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False)
#     exp_slr_interp = regridder(exp_slr)

#     # ---- color scaling (unchanged logic) ----
#     min_val = float(np.nanmin(exp_slr.values))
#     max_val = float(np.nanmax(exp_slr.values))
#     per0p5  = float(np.nanpercentile(exp_slr.values, 0.5))
#     per99p5 = float(np.nanpercentile(exp_slr.values, 99.5))
#     if verbose:
#         print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
#         print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

#     extra_tick_digits = False
#     if cb_max is not None:
#         # if cb_max in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 4):
#         chosen_n = 20
#         # elif cb_max in (1.5, 3):
#         #     chosen_n = 12
#         # else:
#         #     raise ValueError("cb_max is not an acceptable value.")
#         data_max = cb_max
#         chosen_step = 2 * data_max / chosen_n
#         if cb_max == 0.06 or cb_max == 0.075 or cb_max == 0.08:
#             extra_tick_digits = True
#     else:
#         chosen_n, chosen_step = get_cb_spacing(
#             per0p5, per99p5, min_bnd=0.2, min_spacing=0.02, min_n=10, max_n=20, verbose=verbose
#         )

#     max_mag = 0.5 * chosen_n * chosen_step

#     zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
#         max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
#     )

#     # ---- figure/axes management (grid-compatible) ----
#     created_fig = None
#     if ax is None:
#         created_fig, ax = plt.subplots(
#             figsize=(7.5, 5),
#             # subplot_kw={'projection': ccrs.Robinson(central_longitude=209.5), 'facecolor': 'grey'}
#             subplot_kw={'projection': ccrs.Robinson(central_longitude=209.5)}
#         )

#     # ---- draw the panel ----
#     diff_plot = exp_slr_interp.plot(
#         x='lon', y='lat',
#         cmap=disc_cmap, norm=disc_norm,
#         transform=ccrs.PlateCarree(),
#         add_labels=False, add_colorbar=False, ax=ax
#     )
#     # # testing to see non-interpolated data
#     # diff_plot = exp_slr.plot(
#     #     x='geolon', y='geolat',
#     #     cmap=disc_cmap, norm=disc_norm,
#     #     transform=ccrs.PlateCarree(),
#     #     add_labels=False, add_colorbar=False, ax=ax
#     # )

#     # titles: short for subplots, longer when standalone
#     if created_fig is None:
#         ax.set_title(f"{panel_title}")
#     else:
#         ax.set_title(f"{panel_title}\n{variant.capitalize()} SLR: Year {start_yr}–{end_yr}")

#     # ---- optional per-panel colorbar (suppressed by grid) ----
#     diff_cb = None
#     if add_colorbar:
#         diff_cb = plt.colorbar(
#             diff_plot, ax=ax, shrink=0.58, pad=0.04, extend=extend,
#             boundaries=boundaries, norm=disc_norm, spacing='proportional'
#         )
#         tick_labels = []
#         for val in tick_positions:
#             if extra_tick_digits: #(np.abs(val) in {0.125, 1/3, 2/3, 5/8})
#                 tick_labels.append(f"{val:.3f}")
#             elif chosen_step < 0.1:
#                 tick_labels.append(f"{val:.2f}")
#             else:
#                 tick_labels.append(f"{val:.1f}")
#         diff_cb.set_ticks(tick_positions)
#         diff_cb.ax.set_yticklabels(tick_labels)
#         diff_cb.ax.tick_params(labelsize=10, pad=14)
#         diff_cb.set_label(cb_label,labelpad=12)
#         # fixed: old code referenced undefined `vmax`
#         # if zero_step < 0.1 or max_mag > 10 or extra_tick_digits:
#         #     plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.2)
#         # else:
#         #     plt.setp(diff_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.0)

#     # ---- package params for a figure-level colorbar (used by the grid) ----
#     cb_params = None
#     if return_cb_params:
#         tick_labels = []
#         for val in tick_positions:
#             if (np.abs(val) in {0.125, 1/3, 2/3, 5/8}):
#                 tick_labels.append(f"{val:.3f}")
#             elif chosen_step < 0.1:
#                 tick_labels.append(f"{val:.2f}")
#             else:
#                 tick_labels.append(f"{val:.1f}")
#         cb_params = dict(
#             mappable=diff_plot,      # carries cmap+norm
#             cmap=disc_cmap,
#             norm=disc_norm,
#             boundaries=boundaries,
#             extend=extend,
#             spacing='proportional',
#             ticks=tick_positions,
#             ticklabels=tick_labels,
#             label=cb_label
#         )

#     # ---- annotation: global mean (mm) ----
#     mean_val = 1e3 * global_slr.isel(time=0).values
#     ax.text(0.17, 0.8, f"{mean_val:.1f} mm", transform=ax.transAxes,
#             fontsize=12, va='top', ha='center', color='white', fontweight='bold', alpha=1)

#     # ---- optional corner icon ----
#     if icon is not None:
#         image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
#         img = mpimg.imread(image_path)
#         imagebox = OffsetImage(img, zoom=0.09)
#         ab = AnnotationBbox(imagebox, (0.95, 1.00), xycoords="axes fraction", frameon=False)
#         ax.add_artist(ab)

#     # ---- save if we created the figure ----
#     if savefig and created_fig is not None:
#         out = fig_dir + f'{prefix}_{variant}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png'
#         created_fig.savefig(out, dpi=600, bbox_inches='tight')
#         plt.close(created_fig)

#     # what the grid expects:
#     return ax, diff_plot, diff_cb, cb_params


# def plot_SL_diff_region(
#     panel_title=None,
#     ref_ds=None, exp_ds=None, z_vals_ds=None,
#     start_yr=None, end_yr=None, variant="steric",
#     cb_max=None, icon=None,
#     savefig=False, fig_dir=None, prefix=None,
#     verbose=False,

#     # region definition
#     lon_bounds=None,          # tuple: (lon_min, lon_max), expected in degrees east
#     lat_bounds=None,          # tuple: (lat_min, lat_max)
#     region_name=None,         # optional string for save name / title
#     central_longitude=0,      # map projection central longitude
#     plot_projection=None,     # optional override projection

#     # # optional map styling
#     # add_coastlines=True,
#     # coastline_res="50m",
#     # coastline_lw=0.6,
#     # add_gridlines=False,

#     # grid hooks
#     ax=None,
#     add_colorbar=True,
#     return_cb_params=False,
#     cb_label="Sea Level Anomaly (m)",

#     # backwards-compat
#     title=None
# ):

#     panel_title = panel_title or title or ""

#     if lon_bounds is None or lat_bounds is None:
#         raise ValueError("lon_bounds and lat_bounds must both be provided.")

#     lon_min, lon_max = lon_bounds
#     lat_min, lat_max = lat_bounds

#     if plot_projection is None:
#         plot_projection = ccrs.PlateCarree(central_longitude=central_longitude)

#     ref_ds = ref_ds.copy(deep=False)
#     exp_ds = exp_ds.copy(deep=False)

#     if z_vals_ds is not None:
#         ref_ds["z_i"] = z_vals_ds["z_i"]
#         exp_ds["z_i"] = z_vals_ds["z_i"]

#     # ---- reference-state prep ----
#     rename_map = {}
#     if "temp" in ref_ds:
#         rename_map["temp"] = "thetao"
#     if "salt" in ref_ds:
#         rename_map["salt"] = "so"
#     if rename_map:
#         ref_ds = ref_ds.rename(rename_map)

#     if not ref_ds.attrs.get("_area_weighted", False):
#         ref_ds["areacello"] = ref_ds["areacello"] * ref_ds["wet"]
#         ref_ds.attrs["_area_weighted"] = True

#     ref_state = momlevel.reference.setup_reference_state(ref_ds)

#     rename_map = {}
#     if "temp" in exp_ds:
#         rename_map["temp"] = "thetao"
#     if "salt" in exp_ds:
#         rename_map["salt"] = "so"
#     if rename_map:
#         exp_ds = exp_ds.rename(rename_map)

#     if not exp_ds.attrs.get("_area_weighted", False):
#         exp_ds["areacello"] = exp_ds["areacello"] * exp_ds["wet"]
#         exp_ds.attrs["_area_weighted"] = True

#     # ---- compute SLA ----
#     exp_result = momlevel.steric(exp_ds, reference=ref_state, variant=variant)
#     exp_slr = exp_result[0][variant]

#     global_result = momlevel.steric(exp_ds, reference=ref_state, variant=variant, domain="global")
#     global_slr = global_result[0][variant]

#     # ---- normalize longitudes to 0..360 and regrid ----
#     exp_slr = exp_slr.assign_coords(geolon=((exp_slr.geolon + 360) % 360))

#     lat_res = 3 * 210
#     lon_res = 3 * 360
#     target_lat = np.linspace(float(exp_slr.geolat.min()), float(exp_slr.geolat.max()), lat_res)
#     target_lon = np.linspace(0, 360, lon_res, endpoint=False)

#     ds_in = xr.Dataset({
#         "lat": (["yh", "xh"], exp_slr.geolat.values),
#         "lon": (["yh", "xh"], exp_slr.geolon.values),
#     })
#     ds_out = xr.Dataset({
#         "lat": (["lat"], target_lat),
#         "lon": (["lon"], target_lon),
#     })

#     regridder = xe.Regridder(
#         ds_in, ds_out,
#         method="bilinear",
#         periodic=True,
#         reuse_weights=False
#     )
#     exp_slr_interp = regridder(exp_slr)

#     # ---- subset to region ----
#     # Work in 0..360 coordinates internally
#     lon_min_360 = lon_min % 360
#     lon_max_360 = lon_max % 360

#     if lon_min_360 <= lon_max_360:
#         exp_slr_region = exp_slr_interp.sel(
#             lon=slice(lon_min_360, lon_max_360),
#             lat=slice(lat_min, lat_max)
#         )
#     else:
#         # region crosses the dateline / 0E
#         left = exp_slr_interp.sel(lon=slice(lon_min_360, 360), lat=slice(lat_min, lat_max))
#         right = exp_slr_interp.sel(lon=slice(0, lon_max_360), lat=slice(lat_min, lat_max))
#         exp_slr_region = xr.concat([left, right], dim="lon")

#     # ---- color scaling ----
#     # Keep same philosophy as your global plot:
#     # scaling based on full-field values, not just local subset
#     min_val = float(np.nanmin(exp_slr.values))
#     max_val = float(np.nanmax(exp_slr.values))
#     per0p5 = float(np.nanpercentile(exp_slr.values, 0.5))
#     per99p5 = float(np.nanpercentile(exp_slr.values, 99.5))

#     if verbose:
#         print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
#         print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

#     extra_tick_digits = False
#     if cb_max is not None:
#         chosen_n = 20
#         data_max = cb_max
#         chosen_step = 2 * data_max / chosen_n
#         if cb_max in (0.06, 0.075, 0.08):
#             extra_tick_digits = True
#     else:
#         chosen_n, chosen_step = get_cb_spacing(
#             per0p5, per99p5,
#             min_bnd=0.2, min_spacing=0.02,
#             min_n=10, max_n=20,
#             verbose=verbose
#         )

#     max_mag = 0.5 * chosen_n * chosen_step

#     zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
#         max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
#     )

#     # ---- figure / axes ----
#     created_fig = None
#     if ax is None:
#         created_fig, ax = plt.subplots(
#             figsize=(7.5, 5),
#             subplot_kw={"projection": plot_projection}
#         )

#     # ---- plot region ----
#     diff_plot = exp_slr_region.plot(
#         x="lon", y="lat",
#         cmap=disc_cmap, norm=disc_norm,
#         transform=ccrs.PlateCarree(),
#         add_labels=False, add_colorbar=False,
#         ax=ax
#     )

#     # Set rectangular map extent in true geographic coordinates
#     ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

#     # if add_coastlines:
#     #     ax.coastlines(resolution=coastline_res, linewidth=coastline_lw)

#     # if add_gridlines:
#     #     gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
#     #     gl.top_labels = False
#     #     gl.right_labels = False

#     # ---- titles ----
#     if created_fig is None:
#         ax.set_title(f"{panel_title}")
#     else:
#         region_line = f"\n{region_name}" if region_name else ""
#         ax.set_title(
#             f"{panel_title}{region_line}\n{variant.capitalize()} SLR: Year {start_yr}–{end_yr}"
#         )

#     # ---- optional per-panel colorbar ----
#     diff_cb = None
#     if add_colorbar:
#         diff_cb = plt.colorbar(
#             diff_plot, ax=ax,
#             shrink=0.58, pad=0.04, extend=extend,
#             boundaries=boundaries, norm=disc_norm, spacing="proportional"
#         )

#         tick_labels = []
#         for val in tick_positions:
#             if extra_tick_digits:
#                 tick_labels.append(f"{val:.3f}")
#             elif chosen_step < 0.1:
#                 tick_labels.append(f"{val:.2f}")
#             else:
#                 tick_labels.append(f"{val:.1f}")

#         diff_cb.set_ticks(tick_positions)
#         diff_cb.ax.set_yticklabels(tick_labels)
#         diff_cb.ax.tick_params(labelsize=10, pad=14)
#         diff_cb.set_label(cb_label, labelpad=12)

#     # ---- cb params for figure-level colorbar ----
#     cb_params = None
#     if return_cb_params:
#         tick_labels = []
#         for val in tick_positions:
#             if np.abs(val) in {0.125, 1/3, 2/3, 5/8}:
#                 tick_labels.append(f"{val:.3f}")
#             elif chosen_step < 0.1:
#                 tick_labels.append(f"{val:.2f}")
#             else:
#                 tick_labels.append(f"{val:.1f}")

#         cb_params = dict(
#             mappable=diff_plot,
#             cmap=disc_cmap,
#             norm=disc_norm,
#             boundaries=boundaries,
#             extend=extend,
#             spacing="proportional",
#             ticks=tick_positions,
#             ticklabels=tick_labels,
#             label=cb_label
#         )

#     # ---- annotation: global mean ----
#     mean_val = 1e3 * global_slr.isel(time=0).values
#     ax.text(
#         0.17, 0.8, f"{mean_val:.1f} mm",
#         transform=ax.transAxes,
#         fontsize=12, va="top", ha="center",
#         color="white", fontweight="bold", alpha=1
#     )

#     # ---- optional icon ----
#     if icon is not None:
#         image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
#         img = mpimg.imread(image_path)
#         imagebox = OffsetImage(img, zoom=0.09)
#         ab = AnnotationBbox(imagebox, (0.95, 1.00), xycoords="axes fraction", frameon=False)
#         ax.add_artist(ab)

#     # ---- save ----
#     if savefig and created_fig is not None:
#         region_tag = f"_{region_name}" if region_name else ""
#         out = (
#             fig_dir
#             + f"{prefix}{region_tag}_{variant}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png"
#         )
#         created_fig.savefig(out, dpi=600, bbox_inches="tight")
#         plt.close(created_fig)

#     return ax, diff_plot, diff_cb, cb_params


def create_SLR_plots(diff_type,start_year,end_year,
                     omit_title=True,
                     variant="steric",
                     profiles = ['surf','therm','mid','bot'],
                     prof_strings = ["Surf","Therm","Mid","Bot"],
                     power_var_suff = ['0p1TW', '0p2TW', '0p3TW'],
                     power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                     pref_addition=None,
                     plot_max=None,
                     savefig=False,fig_dir=None,
                     extra_verbose=False):
    """
    Inputs:
    diff_type (str): one of
                    ['const-1860ctrl',
                    'doub-1860exp','doub-2xctrl','doub-1860ctrl',
                    'quad-1860exp','quad-4xctrl','quad-1860ctrl']
    fig_dir (str): path of parent directory in which to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    variant (str): type of SLR (either "steric", "thermosteric", or "halosteric")
    plot_max (int/float): input for plot_SL_diff
    savefig (boolean): input for plot_SL_diff
    fig_dir (boolean): input for plot_SL_diff
    extra_verbose (boolean): input for plot_SL_diff
    """

    if savefig:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

    # control cases
    if diff_type == 'doub-1860exp':
        ref_name = f"const_ctrl_{start_year}_{end_year}"
        ds_name = f'doub_ctrl_{start_year}_{end_year}'
        title_str = f"1pct2xCO\u2082 Control"
        fig_name = f"2xCO2-const-ctrl"
        fig_prefix = fig_dir+fig_name
        # plot_SL_diff(title_str, myVars[ref_name], myVars[ds_name], start_year, end_year, variant=variant,
        #                  cb_max=plot_max, icon=prof,
        #                  savefig=savefig, fig_dir=fig_path, prefix=fig_name,
        #                  verbose=extra_verbose)
        plot_SL_diff(panel_title=title_str, ref_ds=myVars[ref_name], exp_ds=myVars[ds_name], start_yr=start_year, end_yr=end_year, variant=variant,
                         cb_max=plot_max, icon=prof,
                         savefig=savefig, fig_dir=fig_path, prefix=fig_name,
                         verbose=extra_verbose)
        print(f"Done {fig_name}.")
        
    elif diff_type == 'quad-1860exp':
        ref_name = f"const_ctrl_{start_year}_{end_year}"
        ds_name = f'quad_ctrl_{start_year}_{end_year}'
        title_str = f"1pct4xCO\u2082 Control"
        fig_name = f"4xCO2-const-ctrl"
        fig_prefix = fig_dir+fig_name
        # plot_SL_diff(title_str, myVars[ref_name], myVars[ds_name], start_year, end_year, variant=variant,
        #                  cb_max=plot_max, icon=prof,
        #                  savefig=savefig, fig_dir=fig_path, prefix=fig_name,
        #                  verbose=extra_verbose)
        plot_SL_diff(panel_title=title_str, ref_ds=myVars[ref_name], exp_ds=myVars[ds_name], start_yr=start_year, end_yr=end_year, variant=variant,
                         cb_max=plot_max, icon=prof,
                         savefig=savefig, fig_dir=fig_path, prefix=fig_name,
                         verbose=extra_verbose)
        print(f"Done {fig_name}.")
        
    # perturbation cases
    for i, power_str in enumerate(power_strings):
        for j, prof in enumerate(profiles):

            # get name of perturbation ds
            if diff_type == 'const-1860ctrl':
                ds_name = f'const_{prof}_{power_var_suff[i]}_{start_year}_{end_year}'
                fig_path = f"{fig_dir}/piControl/"
            elif (diff_type == 'doub-1860exp' or diff_type == 'doub-2xctrl' or diff_type == 'doub-1860ctrl'):
                ds_name = f'doub_{prof}_{power_var_suff[i]}_{start_year}_{end_year}'
                fig_path = f"{fig_dir}/2xCO2/"
            elif (diff_type == 'quad-1860exp' or diff_type == 'quad-4xctrl' or diff_type == 'quad-1860ctrl'):
                ds_name = f'quad_{prof}_{power_var_suff[i]}_{start_year}_{end_year}'
                fig_path = f"{fig_dir}/4xCO2/"

            # get name of reference ds
            if (diff_type == 'const-1860ctrl' or diff_type == 'doub-1860ctrl' or diff_type == 'quad-1860ctrl'):
                ref_name = f"const_ctrl_{start_year}_{end_year}"
            elif diff_type == 'doub-2xctrl':
                ref_name = f"doub_ctrl_{start_year}_{end_year}"
            elif diff_type == 'quad-4xctrl':
                ref_name = f"quad_ctrl_{start_year}_{end_year}"
            elif (diff_type == 'doub-1860exp' or diff_type == 'quad-1860exp'):
                ref_name = f"const_{prof}_{power_var_suff[i]}_{start_year}_{end_year}"

            # assign plot title and fig name
            if diff_type == 'const-1860ctrl':
                title_str = f"Const {prof_strings[j]} {power_str}"
                fig_name = f"{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'doub-1860exp':
                title_str = f"1pct2xCO\u2082 — Const CO2: {prof_strings[j]} {power_str}"
                fig_name = f"2xCO2-const_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'doub-2xctrl':
                title_str = f"1pct2xCO\u2082 {prof_strings[j]} {power_str} — 1pct2xCO\u2082 Control"
                fig_name = f"2xCO2-2xctrl_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'doub-1860ctrl':
                title_str = f"1pct2xCO\u2082 {prof_strings[j]} {power_str} — Const CO\u2082 Control"
                fig_name = f"2xCO2-const-ctrl_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'quad-1860exp':
                title_str = f"1pct4xCO\u2082 — Const CO2: {prof_strings[j]} {power_str}"
                fig_name = f"4xCO2-const_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'quad-4xctrl':
                title_str = f"1pct4xCO\u2082 {prof_strings[j]} {power_str} — 1pct4xCO\u2082 Control"
                fig_name = f"4xCO2-4xctrl_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'quad-1860ctrl':
                title_str = f"1pct4xCO\u2082 {prof_strings[j]} {power_str} — Const CO\u2082 Control"
                fig_name = f"4xCO2-const-ctrl_{prof}_{power_var_suff[i]}"
            
            if savefig:
                fig_path = fig_path + f"{str(start_year).zfill(4)}_{str(end_year).zfill(4)}/"
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)
            else:
                fig_path = None

            if pref_addition != None:
                fig_name = fig_name + "_" + pref_addition
            
            # plot_SL_diff(title_str, myVars[ref_name], myVars[ds_name], start_year, end_year, variant=variant,
            #              cb_max=plot_max, icon=prof,
            #              savefig=savefig, fig_dir=fig_path, prefix=fig_name,
            #              verbose=extra_verbose)

            plot_SL_diff(panel_title=title_str, ref_ds=myVars[ref_name], exp_ds=myVars[ds_name], start_yr=start_year, end_yr=end_year, variant=variant,
                         cb_max=plot_max, icon=prof,
                         savefig=savefig, fig_dir=fig_path, prefix=fig_name,
                         verbose=extra_verbose)

            print(f"Done {fig_name}.")


# # Calculating sea level time series

# ## Helper functions

def generate_all_needed_names(diff_types, start_year, end_year, profiles, power_var_suff):
    """
    Generate a sorted list of all dataset names (reference and experiment) needed
    for a given diff_type, time window, profiles, and power suffixes.

    Parameters:
    - diff_types (str): string of list of strings. Options are
     ['const-1860ctrl', 'doub-1860exp', 'doub-2xctrl', 'doub-1860ctrl',
      'quad-1860exp', 'quad-4xctrl', 'quad-1860ctrl']
    - start_year (int): start of averaging period
    - end_year (int): end of averaging period
    - profiles (list of str): e.g. ['surf','therm','mid','bot']
    - power_var_suff (list of str): e.g. ['0p1TW','0p2TW','0p3TW']

    Returns:
    - List[str]: sorted unique dataset names to preprocess
    """
    if isinstance(diff_types, str):
        diff_types = [diff_types]
        
    names = set()
    # Helper to format the common control names
    ctrl_const = f"const_ctrl_{start_year}_{end_year}"
    ctrl_doub  = f"doub_ctrl_{start_year}_{end_year}"
    ctrl_quad  = f"quad_ctrl_{start_year}_{end_year}"

    for diff_type in diff_types:
        for power in power_var_suff:
            ds_root = f"{power}_{start_year}_{end_year}"
            for prof in profiles:
                if diff_type == 'const-1860ctrl':
                    ref = ctrl_const
                    exp = f"const_{prof}_{ds_root}"
                elif diff_type == 'doub-1860exp':
                    ref = f"const_{prof}_{ds_root}"
                    exp = f"doub_{prof}_{ds_root}"
                elif diff_type == 'doub-2xctrl':
                    ref = f"doub_ctrl_{start_year}_{end_year}"
                    exp = f"doub_{prof}_{ds_root}"
                elif diff_type == 'doub-1860ctrl':
                    ref = ctrl_const
                    exp = f"doub_{prof}_{ds_root}"
                elif diff_type == 'quad-1860exp':
                    ref = f"const_{prof}_{ds_root}"
                    exp = f"quad_{prof}_{ds_root}"
                elif diff_type == 'quad-4xctrl':
                    ref = ctrl_quad
                    exp = f"quad_{prof}_{ds_root}"
                elif diff_type == 'quad-1860ctrl':
                    ref = ctrl_const
                    exp = f"quad_{prof}_{ds_root}"
                else:
                    raise ValueError(f"Unknown diff_type: {diff_type}")
    
                names.add(ref)
                names.add(exp)

        # Add the extra control experiment line for specific diff_types
        if diff_type in ('doub-1860exp', 'doub-2xctrl', 'doub-1860ctrl'):
            names.update({ctrl_const, ctrl_doub})

    return sorted(names)


def preprocess(ds, time_chunk, z_vals_ds=None):
    # ds = ds.chunk({'time': time_chunk,
    #                'yh': 105,
    #                'xh': 180})
    ds = ds.copy(deep=False)

    # preprocess only if needed
    rename_map = {}
    if "temp" in ds: rename_map["temp"] = "thetao"
    if "salt" in ds: rename_map["salt"] = "so"
    if rename_map:
        ds = ds.rename(rename_map)

    if not ds.attrs.get("_area_weighted", False):
        ds["areacello"] = ds["areacello"] * ds["wet"]
        ds.attrs["_area_weighted"] = True
        
    # ds['areacello'] = ds.areacello * ds.wet
    # ds = ds.rename({'temp':'thetao','salt':'so'})
    
    if z_vals_ds:
        ds["z_i"] = z_vals_ds["z_i"]
        
    return ds#.persist()


# ## Main function

# def calculate_GMSLR(diff_types,start_year,end_year,variant_list=["steric"],
#                  profiles = ['surf','therm','mid','bot'],
#                  power_var_suff = ['0p1TW', '0p2TW', '0p3TW'],
#                  # skip_prep = False,
#                  z_vals_ds=None):

#     """
#     Function to plot calculate global-mean steric, thermosteric, and halosteric sea level anomalies over time.
    
#     Inputs:
#     diff_type (str): one of
#                     ['const-1860ctrl',
#                     'doub-1860exp','doub-2xctrl','doub-1860ctrl',
#                     'quad-1860exp','quad-4xctrl','quad-1860ctrl']
#     start_year (int): start year of avg period
#     end_year (int): end year of avg period
#     variant_list (list of str): one of more of "steric", "thermosteric", "halosteric". Global-mean halosteric SLA should always be zero
#                                 (neglecting round-off errors), so only steric or thermosteric component is important on the global-mean.
#     """

#     n_yrs = (end_year - start_year + 1)
#     # time = np.linspace(start_year,end_year,num=n_yrs)

#     new_chunk = int(n_yrs/5)

#     # if skip_prep==False:
    
#     # 1) preprocess all needed datasets
#     all_needed_names = generate_all_needed_names(diff_types, start_year, end_year, profiles, power_var_suff)
#     print(all_needed_names)
#     prepped = {}
#     for name in all_needed_names:
#         print(f"Preprocessing {name}…")
#         prepped[name] = preprocess(myVars[name], new_chunk, z_vals_ds=z_vals_ds)
#         print(f"Finished {name}")
#     print("All datasets preprocessed!")

#     # prebuild control reference states once
#     ref_state_ctrl = {
#         "const": momlevel.reference.setup_reference_state(prepped[f"const_ctrl_{start_year}_{end_year}"].load()),
#         "doub" : momlevel.reference.setup_reference_state(prepped[f"doub_ctrl_{start_year}_{end_year}"].load()) if f"doub_ctrl_{start_year}_{end_year}" in prepped else None,
#         "quad" : momlevel.reference.setup_reference_state(prepped[f"quad_ctrl_{start_year}_{end_year}"].load()) if f"quad_ctrl_{start_year}_{end_year}" in prepped else None,
#     }
    
#     for diff_type in (diff_types if isinstance(diff_types, (list,tuple)) else [diff_types]):
#         print(f'Starting {diff_type}')
        
#         # looping over power inputs
#         for pow_idx, power in enumerate(power_var_suff):
#             print(f'Starting {power}')
#             ds_root = f'{power}_{start_year}_{end_year}'
    
#             # looping over profiles
#             for prof in profiles:
#                 print(f'Starting {prof}')
#                 # choose exp + output key + reference state
#                 if diff_type == "const-1860ctrl":
#                     exp_ds_name = f"const_{prof}_{ds_root}"
#                     sla_name    = f"const_{prof}_{ds_root}_SLA"
#                     ref_ds_name_this = f"const_ctrl_{start_year}_{end_year}"
#                     ref_state_this = ref_state_ctrl["const"]

#                 elif diff_type == "doub-2xctrl":
#                     exp_ds_name = f"doub_{prof}_{ds_root}"
#                     sla_name    = f"doub_{prof}_{ds_root}_2xctrl_SLA"
#                     ref_ds_name_this = f"doub_ctrl_{start_year}_{end_year}"
#                     ref_state_this = ref_state_ctrl["doub"]

#                 elif diff_type == "doub-1860ctrl":
#                     exp_ds_name = f"doub_{prof}_{ds_root}"
#                     sla_name    = f"doub_{prof}_{ds_root}_const_ctrl_SLA"
#                     ref_ds_name_this = f"const_ctrl_{start_year}_{end_year}"
#                     ref_state_this = ref_state_ctrl["const"]

#                 elif diff_type == "doub-1860exp":
#                     exp_ds_name = f"doub_{prof}_{ds_root}"
#                     sla_name    = f"doub_{prof}_{ds_root}_1860_SLA"
#                     ref_ds_name_this = f"const_{prof}_{ds_root}"
#                     ref_state_this = momlevel.reference.setup_reference_state(prepped[ref_ds_name_this].load())
                    
#                 else:
#                     raise ValueError(diff_type)

#                 exp_ds = prepped[exp_ds_name]  # optionally .load() while debugging
#                 myVars[sla_name] = xr.Dataset()
#                 print(f"[{diff_type}] power={power} prof={prof} EXP={exp_ds_name} REF={ref_ds_name_this}")
#                 for variant in variant_list:
#                     global_slr = momlevel.steric(exp_ds, reference=ref_state_this,
#                                                 variant=variant, domain="global")[0][variant].load()
#                     myVars[sla_name][variant] = global_slr
#                     print(f"Done {sla_name} {variant}")
                    
#             # end of profile loop
        
#         # end of power loop
        
#         # ctrl diagnostic (optional)
#         if diff_type in ("doub-1860exp","doub-1860ctrl"):
#             sla_name = f"doub_ctrl_{start_year}_{end_year}_SLA"
            
#             # ---- GUARD: skip if already computed ----
#             if sla_name in myVars:
#                 print(f"[{diff_type}] skipping existing {sla_name}")
#             else:
#                 exp_ds_name = f"doub_ctrl_{start_year}_{end_year}"
#                 ref_ds_name_this = f"const_ctrl_{start_year}_{end_year}"
#                 exp_ds = prepped[exp_ds_name]
#                 ref_state_this = ref_state_ctrl["const"]
    
#                 myVars[sla_name] = xr.Dataset()
#                 print(f"[{diff_type}] power={power} prof={prof} EXP={exp_ds_name} REF={ref_ds_name_this}")
#                 for variant in variant_list:
#                     myVars[sla_name][variant] = momlevel.steric(
#                         exp_ds, reference=ref_state_this, variant=variant, domain="global"
#                     )[0][variant].load()
#                     print(f"Done {sla_name} {variant}")

#     # end of diff_type loop


def calculate_GMSLR(
    diff_types,
    start_year,
    end_year,
    variant_list=("steric",),
    profiles=("surf", "therm", "mid", "bot"),
    power_var_suff=("0p1TW", "0p2TW", "0p3TW"),
    z_vals_ds=None,
    abs_ref_name=None,          # if None -> const_ctrl_{start}_{end}
    compute_ctrl_diag=True,     # keep your doub_ctrl vs const_ctrl diagnostic
    verbose=True,
):
    """
    Compute global-mean sea level anomalies (steric / thermosteric / halosteric)
    for one or more diff_types. Anomaly is computed as:

        anom(t) = SLR(exp_ds; abs_ref_state) - SLR(ref_ds; abs_ref_state)

    where abs_ref_state is built once from abs_ref_name.

    Stores results into global dict `myVars` with the same keys you used before:
      - const_{prof}_{power}_{start}_{end}_SLA
      - doub_{prof}_{power}_{start}_{end}_2xctrl_SLA
      - doub_{prof}_{power}_{start}_{end}_const_ctrl_SLA
      - doub_{prof}_{power}_{start}_{end}_1860_SLA
      (and similarly for quad if you extend)

    Parameters
    ----------
    diff_types : str or list/tuple of str
    start_year, end_year : int
    variant_list : iterable of str
    profiles : iterable of str
    power_var_suff : iterable of str
    z_vals_ds : xr.Dataset or None
    abs_ref_name : str or None
        Dataset name in myVars to use as the absolute reference.
        If None, uses f"const_ctrl_{start_year}_{end_year}".
    compute_ctrl_diag : bool
        If True, compute doub_ctrl vs const_ctrl diagnostic once (if relevant).
    verbose : bool
    """

    # Normalize diff_types to a list
    if isinstance(diff_types, str):
        diff_types = [diff_types]
    diff_set = set(diff_types)

    n_yrs = (end_year - start_year + 1)
    new_chunk = max(1, int(n_yrs / 5))

    # Choose absolute reference dataset name
    if abs_ref_name is None:
        abs_ref_name = f"const_ctrl_{start_year}_{end_year}"

    # ---- helper: build (ref_ds_name, exp_ds_name, sla_name) for a given diff_type/prof/power
    def resolve_names(diff_type, prof, ds_root):
        if diff_type == "const-1860ctrl":
            ref_ds_name = f"const_ctrl_{start_year}_{end_year}"
            exp_ds_name = f"const_{prof}_{ds_root}"
            sla_name    = f"const_{prof}_{ds_root}_SLA"

        elif diff_type == "doub-2xctrl":
            ref_ds_name = f"doub_ctrl_{start_year}_{end_year}"
            exp_ds_name = f"doub_{prof}_{ds_root}"
            sla_name    = f"doub_{prof}_{ds_root}_2xctrl_SLA"

        elif diff_type == "doub-1860ctrl":
            ref_ds_name = f"const_ctrl_{start_year}_{end_year}"
            exp_ds_name = f"doub_{prof}_{ds_root}"
            sla_name    = f"doub_{prof}_{ds_root}_const_ctrl_SLA"

        elif diff_type == "doub-1860exp":
            ref_ds_name = f"const_{prof}_{ds_root}"
            exp_ds_name = f"doub_{prof}_{ds_root}"
            sla_name    = f"doub_{prof}_{ds_root}_1860_SLA"

        elif diff_type == "quad-4xctrl":
            ref_ds_name = f"quad_ctrl_{start_year}_{end_year}"
            exp_ds_name = f"quad_{prof}_{ds_root}"
            sla_name    = f"quad_{prof}_{ds_root}_4xctrl_SLA"

        elif diff_type == "quad-1860ctrl":
            ref_ds_name = f"const_ctrl_{start_year}_{end_year}"
            exp_ds_name = f"quad_{prof}_{ds_root}"
            sla_name    = f"quad_{prof}_{ds_root}_const_ctrl_SLA"

        elif diff_type == "quad-1860exp":
            ref_ds_name = f"const_{prof}_{ds_root}"
            exp_ds_name = f"quad_{prof}_{ds_root}"
            sla_name    = f"quad_{prof}_{ds_root}_1860_SLA"

        else:
            raise ValueError(f"Unknown diff_type: {diff_type}")

        return ref_ds_name, exp_ds_name, sla_name

    # ---- preprocess union of all needed datasets
    all_needed_names = generate_all_needed_names(diff_types, start_year, end_year, list(profiles), list(power_var_suff))
    # ensure abs_ref_name included even if not requested elsewhere
    all_needed_names = sorted(set(all_needed_names) | {abs_ref_name})

    if verbose:
        print(all_needed_names)

    prepped = {}
    for name in all_needed_names:
        if verbose:
            print(f"Preprocessing {name}…")
        prepped[name] = preprocess(myVars[name], new_chunk, z_vals_ds=z_vals_ds)
        if verbose:
            print(f"Finished {name}")
    if verbose:
        print("All datasets preprocessed!")

    # ---- build absolute reference state ONCE
    abs_ref_ds = prepped[abs_ref_name]#.load()
    abs_ref_state = momlevel.reference.setup_reference_state(abs_ref_ds)

    # ---- compute requested diffs
    for diff_type in diff_types:
        if verbose:
            print(f"Starting {diff_type}")

        for power in power_var_suff:
            if verbose:
                print(f"  Starting {power}")

            ds_root = f"{power}_{start_year}_{end_year}"

            for prof in profiles:
                ref_ds_name, exp_ds_name, sla_name = resolve_names(diff_type, prof, ds_root)

                ref_ds = prepped[ref_ds_name]
                exp_ds = prepped[exp_ds_name]

                if verbose:
                    print(f"    [{diff_type}] prof={prof} EXP={exp_ds_name} REF={ref_ds_name} ABS={abs_ref_name}")

                myVars[sla_name] = xr.Dataset()

                for variant in variant_list:
                    # compute both relative to the SAME absolute reference state, then subtract
                    slr_exp = momlevel.steric(exp_ds, reference=abs_ref_state, variant=variant, domain="global")[0][variant]
                    slr_ref = momlevel.steric(ref_ds, reference=abs_ref_state, variant=variant, domain="global")[0][variant]
                    myVars[sla_name][variant] = (slr_exp - slr_ref)#.load()

                    if verbose:
                        print(f"      Done {sla_name} {variant}")

    # ---- optional: ctrl diagnostic (doub_ctrl vs const_ctrl), compute once
    if compute_ctrl_diag and ({"doub-1860exp", "doub-1860ctrl", "doub-2xctrl"} & diff_set):
        diag_name = f"doub_ctrl_{start_year}_{end_year}_SLA"
        if diag_name not in myVars:
            ref_ds_name = f"const_ctrl_{start_year}_{end_year}"
            exp_ds_name = f"doub_ctrl_{start_year}_{end_year}"

            ref_ds = prepped[ref_ds_name]
            exp_ds = prepped[exp_ds_name]

            if verbose:
                print(f"[ctrl-diagnostic] EXP={exp_ds_name} REF={ref_ds_name} ABS={abs_ref_name}")

            myVars[diag_name] = xr.Dataset()
            for variant in variant_list:
                slr_exp = momlevel.steric(exp_ds, reference=abs_ref_state, variant=variant, domain="global")[0][variant]
                slr_ref = momlevel.steric(ref_ds, reference=abs_ref_state, variant=variant, domain="global")[0][variant]
                myVars[diag_name][variant] = (slr_exp - slr_ref)#.load()
                if verbose:
                    print(f"  Done {diag_name} {variant}")
        else:
            if verbose:
                print(f"[ctrl-diagnostic] skipping existing {diag_name}")


# # Plotting SL time series

def plot_SLA_ts(diff_type,fig_dir,start_year,end_year,variant="steric",
                 fig_pref=None,
                 omit_title=True,
                 ylimits = None,
                 # ylimits = [-0.1,0.5],
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                 profiles = ['surf','therm','mid','bot'],
                 power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                 power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                 savefig=False,
                 verbose=False):
    """
    Function to plot a specific component of the global-mean SL anomaly over time (each power input on separate plot).
    
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

    n_yrs = (end_year - start_year + 1)
    time = np.linspace(start_year,end_year,num=n_yrs)
    
    # separate plot for each power input
    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(6,3))

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                
        ds_root = f'{power}_{start_year}_{end_year}'
        for prof in profiles:
            if verbose:
                print(f'Starting {prof}')
                
            if diff_type == 'const-1860ctrl':
                exp_ds_name = f'const_{prof}_{ds_root}_SLA'
                
            elif diff_type == 'doub-2xctrl':
                exp_ds_name = f'doub_{prof}_{ds_root}_2xctrl_SLA'
                
            elif diff_type == 'doub-1860ctrl':
                exp_ds_name = f'doub_{prof}_{ds_root}_const_ctrl_SLA'

            elif diff_type == 'quad-4xctrl':
                exp_ds_name = f'quad_{prof}_{ds_root}_4xctrl_SLA'
                
            elif diff_type == 'quad-1860ctrl':
                exp_ds_name = f'quad_{prof}_{ds_root}_const_ctrl_SLA'

            elif diff_type == 'doub-1860exp':
                exp_ds_name = f'doub_{prof}_{ds_root}_1860_SLA'
                
            elif diff_type == 'quad-1860exp':
                exp_ds_name = f'quad_{prof}_{ds_root}_1860_SLA'

            ax.plot(time,myVars[exp_ds_name][variant].load(),label=prof,color=prof_dict[prof])
            if verbose:
                print(f'Ending {prof}')

        if diff_type == 'doub-1860exp' or diff_type == 'doub-1860ctrl':
            if verbose:
                print(f'Starting ctrl')
            exp_ds_name = f'doub_ctrl_{start_year}_{end_year}_SLA'
            ax.plot(time,myVars[exp_ds_name][variant].load(),label='control',color='k')
            if verbose:
                print(f'Ending ctrl')

        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Sea Level Anomaly (m)")
        ax.set_xlim(start_year-1,end_year)
        if ylimits:
            ax.set_ylim(ylimits)
            
        ax.legend(loc=leg_loc,ncols=leg_ncols)
        
        ax.grid("both")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

        if diff_type == 'const-1860ctrl':
            title_str = f"Const CO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"const_{power}"
            
        elif diff_type == 'doub-1860exp':
            title_str = f"1pct2xCO\u2082 Radiative: {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-const_{power}"
            
        elif diff_type == 'doub-2xctrl':
            title_str = f"1pct2xCO\u2082 Mixing: {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-2xctrl_{power}"
            
        elif diff_type == 'doub-1860ctrl':
            title_str = f"1pct2xCO\u2082 Total: {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-const-ctrl_{power}"
            
        elif diff_type == 'quad-1860exp':
            title_str = f"1pct4xCO\u2082 Radiative: {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-const_{power}"
            
        elif diff_type == 'quad-4xctrl':
            title_str = f"1pct4xCO\u2082 Mixing: {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-4xctrl_{power}"
            
        elif diff_type == 'quad-1860ctrl':
            title_str = f"1pct4xCO\u2082 Total: {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-const-ctrl_{power}"

        if omit_title is False:
            title_suff = f"Global Mean {variant.capitalize()} SLA"
            ax.set_title(title_str+title_suff)

        if savefig:
            if fig_pref != None:
                plt.savefig(fig_dir+f'{fig_pref}_SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
            else:
                plt.savefig(fig_dir+f'SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
            # plt.close()


def plot_mixing_SLA_ts(co2_scen,fig_dir,start_year,end_year,variant="steric",
                 fig_pref=None,
                 omit_ctrl=True,
                 omit_title=True,
                 ystep_list = [0.02, 0.05, 0.05], ymin_frac_list = [0.25, 0.2, 0.2], ylims_list = [None,None,None],
                 # ylimits = [-0.1,0.5],
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                 profiles = ['surf','therm','mid','bot'],
                 power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                 power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                 savefig=False,
                 verbose=False):
    """
    Function to plot mixing-induced global-mean SL anomalies over time for multiple radiative forcing scenarios (each power input on separate plot).
    
    Inputs:
    co2_scen (str): one of ['doub','quad','all']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    """

    prof_dict = {'surf': 'b',
                 'therm': 'm',
                 'mid': 'g',
                 'bot': 'r'}
    
    if co2_scen == 'all':
        mixing_leg = [Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='m', lw=2),
                    Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], linestyle='solid', lw=2, color='k'),
                    Line2D([0], [0], linestyle='dashed', lw=2, color='k'),
                    Line2D([0], [0], linestyle='dotted', lw=2, color='k')]
        
        mixing_leg_labels = ['surf', 'therm', 'mid', 'bot', 'const CO\u2082', '1pct2xCO\u2082', '1pct4xCO\u2082']

    elif co2_scen == 'doub':
        mixing_leg = [Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='m', lw=2),
                    Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], linestyle='solid', lw=2, color='k'),
                    Line2D([0], [0], linestyle='dashed', lw=2, color='k')]
        
        mixing_leg_labels = ['surf', 'therm', 'mid', 'bot', 
                             'const CO\u2082', '1pct2xCO\u2082']

    elif co2_scen == 'quad':
        mixing_leg = [Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='m', lw=2),
                    Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], linestyle='solid', lw=2, color='k'),
                    Line2D([0], [0], linestyle='dotted', lw=2, color='k')]
        
        mixing_leg_labels = ['surf', 'therm', 'mid', 'bot', 
                             'const CO\u2082', '1pct4xCO\u2082']

    mix_leg_2 = [Line2D([0], [0], alpha=0.6, lw=4, color='tab:gray')]

    if co2_scen == 'doub':
        labels_2 = ['1pct2xCO\u2082 ctrl']
        ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_SLA'
    elif co2_scen == 'quad':
        labels_2 = ['1pct4xCO\u2082 ctrl']
        ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_const_ctrl_SLA'

    n_yrs = (end_year - start_year + 1)
    time = np.linspace(start_year,end_year,num=n_yrs)
    
    # separate plot for each power input
    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(6,3))

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        plotted_vals = []   # collect values used in this panel

        if omit_ctrl == False:
            ax.plot(time,myVars[ctrl_rad_name][variant].load(),color='tab:gray',alpha=0.6,lw=4)
                
        ds_root = f'{power}_{start_year}_{end_year}'
        for prof in profiles:
            if verbose:
                print(f'Starting {prof}')

            const_ds_name = f'const_{prof}_{ds_root}_SLA'
            doub_ds_name = f'doub_{prof}_{ds_root}_2xctrl_SLA'
            quad_ds_name = f'quad_{prof}_{ds_root}_4xctrl_SLA'

            const_anom = myVars[const_ds_name][variant].load()
            ax.plot(time,const_anom,label=prof,color=prof_dict[prof])
            plotted_vals.append(np.asarray(const_anom))
            
            if co2_scen == 'doub' or co2_scen == 'all':
                doub_anom = myVars[doub_ds_name][variant].load()
                ax.plot(time,doub_anom,linestyle='dashed',color=prof_dict[prof])
                plotted_vals.append(np.asarray(doub_anom))
                
            elif co2_scen == 'quad' or co2_scen == 'all':
                quad_anom = myVars[quad_ds_name][variant].load()
                ax.plot(time,quad_anom,linestyle='dotted',color=prof_dict[prof])
                plotted_vals.append(np.asarray(quad_anom))
                
            if verbose:
                print(f'Ending {prof}')
            
        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Sea Level Anomaly (m)")
        ax.set_xlim(start_year-1,end_year)
        
        # --- y-axis grid/ticks every ystep, bounds forced to multiples of ystep ---
        ystep = ystep_list[pow_idx]
        ymin = -ymin_frac_list[pow_idx] * ystep
        
        if ylims_list[pow_idx] is not None:
            ymin, ymax = ylims_list[pow_idx]
        else:
            all_vals = np.concatenate([arr[np.isfinite(arr)] for arr in plotted_vals])
            ymax = np.ceil(all_vals.max() / ystep) * ystep
        
        if ymax <= 0:
            ymax = ystep
        
        ax.set_ylim(ymin, ymax)
    
        # major ticks/gridlines at multiples of ystep within the displayed range
        tick_start = ystep * np.ceil(np.round(ymin / ystep, 10))
        tick_end   = ystep * np.floor(np.round(ymax / ystep, 10))
        major_ticks = np.arange(tick_start, tick_end + ystep/2, ystep)
        major_ticks = np.round(major_ticks, 10)
        
        # clean up floating-point noise
        major_ticks = np.round(major_ticks, 10)
        
        ax.yaxis.set_major_locator(FixedLocator(major_ticks))

        if omit_ctrl == False:
            legend1 = ax.legend(
                mixing_leg, mixing_leg_labels,
                loc=leg_loc,
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(0.0, 1.0),  # Adjust position as needed
                frameon=True
            )
            legend2 = ax.legend(
                mix_leg_2, labels_2,
                loc=leg_loc,
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(0.27, 1.0),  # Adjust position as needed
                frameon=True
            )
            # Add the first legend back to the axis
            ax.add_artist(legend1)
        else:
            ax.legend(mixing_leg, mixing_leg_labels, loc=leg_loc, fontsize=10, ncol = leg_ncols, labelspacing=0.1)
        
        ax.grid("both")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        
        if co2_scen == 'doub':
            fig_name = f"const_2xCO2_mixing_{power}"
        elif co2_scen == 'quad':
            fig_name = f"const_4xCO2_mixing_{power}"
        elif co2_scen == 'all':
            fig_name = f"all_mixing_{power}"

        if omit_title is False:
            title_suff = f"Global Mean {variant.capitalize()} SLA\nComparison: {power_strings[pow_idx]} Cases"
            ax.set_title(title_str+title_suff)

        if savefig:
            if fig_pref != None:
                plt.savefig(fig_dir+f'{fig_pref}_SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
            else:
                plt.savefig(fig_dir+f'SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
            # plt.close()


def plot_rad_SLA_ts(co2_scen,fig_dir,start_year,end_year,variant="steric",
                 fig_pref=None,
                 omit_title=True,
                 ylimits = None, ystep = 0.05, ymin_frac = 0.2,
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                 profiles = ['surf','therm','mid','bot'],
                 power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                 power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                 savefig=False,
                 verbose=False):
    """
    Function to plot mixing-induced global-mean SL anomalies over time for multiple radiative forcing scenarios.
    
    Inputs:
    co2_scen (str): one of ['doub','quad','all']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    """

    prof_dict = {'surf': 'b',
                 'therm': 'm',
                 'mid': 'g',
                 'bot': 'r'}
    
    rad_leg = [Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='m', lw=2),
                Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], linestyle='solid', lw=2, color='k'),
                Line2D([0], [0], linestyle='dashed', lw=2, color='k'),
                Line2D([0], [0], linestyle='dotted', lw=2, color='k'),
                Line2D([0], [0], color='tab:gray',alpha=0.6,lw=4)]
    
    rad_leg_labels = ['surf', 'therm', 'mid', 'bot']
    for elem in power_strings:
        rad_leg_labels.append(elem)

    if co2_scen == 'doub':
        rad_leg_labels.append('1pct2xCO\u2082 ctrl')
    elif co2_scen == 'quad':
        rad_leg_labels.append('1pct4xCO\u2082 ctrl')

    n_yrs = (end_year - start_year + 1)
    time = np.linspace(start_year,end_year,num=n_yrs)
    
    fig, ax = plt.subplots(figsize=(6,3))

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    power_line_types = ['solid','dashed','dotted']

    plotted_vals = []   # collect values used in this panel

    if co2_scen == 'doub':
        ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_SLA'
    elif co2_scen == 'quad':
        ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_const_ctrl_SLA'

    ctrl_rad_anom = myVars[ctrl_rad_name][variant].load()
    ax.plot(time,ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)
    plotted_vals.append(np.asarray(ctrl_rad_anom))

    for pow_idx, power in enumerate(power_var_suff):
        ds_root = f'{power}_{start_year}_{end_year}'
        
        for prof in profiles:
            if verbose:
                print(f'Starting {prof}')

            const_ds_name = f'const_{prof}_{ds_root}_SLA'
            if co2_scen == 'doub':
                ds_name = f'doub_{prof}_{ds_root}_1860_SLA'
            elif co2_scen == 'quad':
                ds_name = f'quad_{prof}_{ds_root}_1860_SLA'

            rad_anom = myVars[ds_name][variant].load()

            ax.plot(time,rad_anom,color=prof_dict[prof],linestyle=power_line_types[pow_idx])
            plotted_vals.append(np.asarray(rad_anom))
                
            if verbose:
                print(f'Ending {prof}')
            
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Sea Level Anomaly (m)")
    ax.set_xlim(start_year-1,end_year)
    
    # --- y-axis grid/ticks every ystep, bounds forced to multiples of ystep ---
    ymin = -ymin_frac * ystep
    
    if ylimits is not None:
        # ymax = np.ceil(ylimits[1] / ystep) * ystep
        ymin, ymax = ylimits
    else:
        all_vals = np.concatenate([arr[np.isfinite(arr)] for arr in plotted_vals])
        ymax = np.ceil(all_vals.max() / ystep) * ystep
    
    if ymax <= 0:
        ymax = ystep
    
    ax.set_ylim(ymin, ymax)

    # major ticks/gridlines at multiples of ystep within the displayed range
    tick_start = ystep * np.ceil(np.round(ymin / ystep, 10))
    tick_end   = ystep * np.floor(np.round(ymax / ystep, 10))
    major_ticks = np.arange(tick_start, tick_end + ystep/2, ystep)
    major_ticks = np.round(major_ticks, 10)
    
    # clean up floating-point noise
    major_ticks = np.round(major_ticks, 10)
    
    ax.yaxis.set_major_locator(FixedLocator(major_ticks))

    ax.legend(rad_leg, rad_leg_labels, loc=leg_loc, fontsize=10, ncol = leg_ncols, labelspacing=0.1)
    
    ax.grid("both")
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    
    if co2_scen == 'doub':
        fig_name = f"2xCO2_rad"
    elif diff_type == 'quad':
        fig_name = f"4xCO2_rad"

    if omit_title is False:
        title_suff = f"Global Mean Radiative {variant.capitalize()} SLA Comparison"
        ax.set_title(title_suff)

    if savefig:
        if fig_pref != None:
            plt.savefig(fig_dir+f'{fig_pref}_SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        # plt.close()


def plot_mix_pow_SLA_ts(co2_scen,fig_dir,start_year,end_year,variant="steric",
                 fig_pref=None,
                 omit_title=True,
                 omit_ctrl=False,
                 ylimits = None, ystep = 0.05, ymin_frac = 0.2,
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                 profiles = ['surf','therm','mid','bot'],
                 power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                 power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                 savefig=False,
                 verbose=False):
    """
    Function to plot mixing-induced global-mean SL anomalies over time for multiple power inputs.
    
    Inputs:
    co2_scen (str): one of ['doub','quad','all']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    """

    prof_dict = {'surf': 'b',
                 'therm': 'm',
                 'mid': 'g',
                 'bot': 'r'}
    
    # Define custom legend entries
    mix_leg_1 = [  # First column (4 labels)
        Line2D([0], [0], color='b', lw=2),  # surf
        Line2D([0], [0], color='m', lw=2),  # therm
        Line2D([0], [0], color='g', lw=2),  # mid
        Line2D([0], [0], color='r', lw=2)  # bot
    ]
    labels_1 = ['surf', 'therm', 'mid', 'bot']
    
    mix_leg_2 = [  # Second column (3 labels)
        Line2D([0], [0], linestyle='solid', lw=2, color='k'),
        Line2D([0], [0], linestyle='dashed', lw=2, color='k'),
        Line2D([0], [0], linestyle='dotted', lw=2, color='k')
    ]
    labels_2 = power_strings

    mix_leg_3 = [Line2D([0], [0], alpha=0.6, lw=4, color='tab:gray')]

    if co2_scen == 'const' or co2_scen == 'doub':
        labels_3 = ['1pct2xCO\u2082 ctrl']
    elif co2_scen == 'quad':
        labels_3 = ['1pct4xCO\u2082 ctrl']

    power_line_types = ['solid','dashed','dotted']

    plotted_vals = []   # collect values used in this panel

    n_yrs = (end_year - start_year + 1)
    time = np.linspace(start_year,end_year,num=n_yrs)
    
    fig, ax = plt.subplots(figsize=(6,3))

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    if omit_ctrl == False:
        if co2_scen == 'const' or co2_scen == 'doub':
            ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_SLA'
        elif co2_scen == 'quad':
            ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_const_ctrl_SLA'

        ctrl_rad_anom = myVars[ctrl_rad_name][variant].load()
        ax.plot(time,ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)
        plotted_vals.append(np.asarray(ctrl_rad_anom))

    for pow_idx, power in enumerate(power_var_suff):
        ds_root = f'{power}_{start_year}_{end_year}'
        for prof in profiles:
            if verbose:
                print(f'Starting {prof}')

            if co2_scen == 'const':
                ds_name = f'const_{prof}_{ds_root}_SLA'
            elif co2_scen == 'doub':
                ds_name = f'doub_{prof}_{ds_root}_2xctrl_SLA'
            elif co2_scen == 'quad':
                ds_name = f'quad_{prof}_{ds_root}_4xctrl_SLA'

            mix_anom = myVars[ds_name][variant].load()
            ax.plot(time,mix_anom,color=prof_dict[prof],linestyle=power_line_types[pow_idx])
            plotted_vals.append(np.asarray(mix_anom))
                
            if verbose:
                print(f'Ending {prof}')
            
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Sea Level Anomaly (m)")
    ax.set_xlim(start_year-1,end_year)
    
    # --- y-axis grid/ticks every ystep, bounds forced to multiples of ystep ---
    ymin = -ymin_frac * ystep
    
    if ylimits is not None:
        # ymax = np.ceil(ylimits[1] / ystep) * ystep
        ymin, ymax = ylimits
    else:
        all_vals = np.concatenate([arr[np.isfinite(arr)] for arr in plotted_vals])
        ymax = np.ceil(all_vals.max() / ystep) * ystep
    
    if ymax <= 0:
        ymax = ystep
    
    ax.set_ylim(ymin, ymax)

    # major ticks/gridlines at multiples of ystep within the displayed range
    tick_start = ystep * np.ceil(np.round(ymin / ystep, 10))
    tick_end   = ystep * np.floor(np.round(ymax / ystep, 10))
    major_ticks = np.arange(tick_start, tick_end + ystep/2, ystep)
    major_ticks = np.round(major_ticks, 10)
    
    # clean up floating-point noise
    major_ticks = np.round(major_ticks, 10)
    
    ax.yaxis.set_major_locator(FixedLocator(major_ticks))

    legend1 = ax.legend(
    mix_leg_1, labels_1,
    loc=leg_loc,
    fontsize=10, labelspacing=0.1,
    bbox_to_anchor=(0.0, 1.0),  # Adjust position as needed
    frameon=True
    )
    legend2 = ax.legend(
        mix_leg_2, labels_2,
        loc=leg_loc,
        fontsize=10, labelspacing=0.1,
        bbox_to_anchor=(0.2, 1.0),  # Adjust position as needed
        frameon=True
    )
    if omit_ctrl == False:
        legend3 = ax.legend(
            mix_leg_3, labels_3,
            loc=leg_loc,
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(0.415, 1.0),  # Adjust position as needed
            frameon=True
        )
    
    # Add the first legend back to the axis
    ax.add_artist(legend1)
    if omit_ctrl == False:
        ax.add_artist(legend2)
    
    ax.grid("both")
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    
    if co2_scen == 'const':
        fig_name = f"const_mix"
    elif co2_scen == 'doub':
        fig_name = f"2xCO2_mix"
    elif co2_scen == 'quad':
        fig_name = f"4xCO2_mix"

    if omit_title is False:
        title_suff = f"Global Mean {variant.capitalize()} SLA\nComparison: {power_strings[pow_idx]} Cases"
        ax.set_title(title_suff)

    if savefig:
        if fig_pref != None:
            plt.savefig(fig_dir+f'{fig_pref}_SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        # plt.close()


def plot_lin_SLA_ts(co2_scen,fig_dir,start_year,end_year,variant="steric",
                 fig_pref=None,
                 omit_ctrl=False,
                 omit_title=True,
                 ystep_list = [0.1, 0.1, 0.1], ymin_frac_list = [0.2, 0.2, 0.2], ylims_list = [None,None,None],
                 # ylimits = [-0.1,0.5],
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                 profiles = ['surf','therm','mid','bot'],
                 power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                 power_strings = ['0.1 TW', '0.2 TW', '0.3 TW'],
                 savefig=False,
                 verbose=False):
    """
    Function to plot sum of mixing and radiative responses and the realized total response.
    
    Inputs:
    co2_scen (str): one of ['doub','quad']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    """

    prof_dict = {'surf': 'b',
                 'therm': 'm',
                 'mid': 'g',
                 'bot': 'r'}

    # Define custom legend entries
    CO2_lin_leg_1 = [  # First column (4 labels)
        Line2D([0], [0], color='b', lw=2),  # surf
        Line2D([0], [0], color='m', lw=2),  # therm
        Line2D([0], [0], color='g', lw=2),  # mid
        Line2D([0], [0], color='r', lw=2),  # bot
    ]
    labels_1 = ['surf', 'therm', 'mid', 'bot']

    n_yrs = (end_year - start_year + 1)
    time = np.linspace(start_year,end_year,num=n_yrs)

    if omit_ctrl == False:
        CO2_lin_leg_2 = [  # Second column (3 labels)
            Line2D([0], [0], linestyle='solid', lw=2, color='k'),  # realized response
            Line2D([0], [0], linestyle='dashed', lw=2, color='k'),   # CO2 + mixing
            Line2D([0], [0], alpha=0.6, lw=4, color='tab:gray') # 1pct control
        ]

        if co2_scen == 'doub':
            labels_2 = ['realized response', 'CO\u2082 + mixing', '1pct2xCO\u2082 ctrl']
        elif co2_scen == 'quad':
            labels_2 = ['realized response', 'CO\u2082 + mixing', '1pct4xCO\u2082 ctrl']
        
        if co2_scen == 'const' or co2_scen == 'doub':
            ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_SLA'
        elif co2_scen == 'quad':
            ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_const_ctrl_SLA'
            
    else:
        CO2_lin_leg_2 = [  # Second column (2 labels)
            Line2D([0], [0], linestyle='solid', lw=2, color='k'),  # realized response
            Line2D([0], [0], linestyle='dashed', lw=2, color='k')   # CO2 + mixing
        ]
        
        labels_2 = ['realized response', 'CO\u2082 + mixing']

    # separate plot for each power input
    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(6,3))

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        plotted_vals = []   # collect values used in this panel

        if omit_ctrl == False:
            ctrl_rad_anom = myVars[ctrl_rad_name][variant].load()
            ax.plot(time,ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)
            plotted_vals.append(np.asarray(ctrl_rad_anom))
        
        ds_root = f'{power}_{start_year}_{end_year}'
        for prof in profiles:
            if co2_scen == 'doub':
                co2_ds_name = f'doub_{prof}_{ds_root}_1860_SLA'
                mixing_ds_name = f'doub_{prof}_{ds_root}_2xctrl_SLA'
                total_ds_name = f'doub_{prof}_{ds_root}_const_ctrl_SLA'
            elif co2_scen == 'quad':
                co2_ds_name = f'quad_{prof}_{ds_root}_1860_SLA'
                mixing_ds_name = f'quad_{prof}_{ds_root}_4xctrl_SLA'
                total_ds_name = f'quad_{prof}_{ds_root}_const_ctrl_SLA'

            co2_anom = myVars[co2_ds_name][variant].load()
            mixing_anom = myVars[mixing_ds_name][variant].load()
            total_anom = myVars[total_ds_name][variant].load()

            sum_anom = co2_anom + mixing_anom
                
            ax.plot(time,total_anom,color=prof_dict[prof])
            ax.plot(time,sum_anom,linestyle='--',color=prof_dict[prof])
            plotted_vals.append(np.asarray(total_anom))
            plotted_vals.append(np.asarray(sum_anom))

        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Sea Level Anomaly (m)")
        ax.set_xlim(start_year-1,end_year)

        # --- y-axis grid/ticks every ystep, bounds forced to multiples of ystep ---
        ystep = ystep_list[pow_idx]
        ymin = -ymin_frac_list[pow_idx] * ystep
        
        if ylims_list[pow_idx] is not None:
            ymin, ymax = ylims_list[pow_idx]
        else:
            all_vals = np.concatenate([arr[np.isfinite(arr)] for arr in plotted_vals])
            ymax = np.ceil(all_vals.max() / ystep) * ystep
        
        if ymax <= 0:
            ymax = ystep
        
        ax.set_ylim(ymin, ymax)
    
        # major ticks/gridlines at multiples of ystep within the displayed range
        tick_start = ystep * np.ceil(np.round(ymin / ystep, 10))
        tick_end   = ystep * np.floor(np.round(ymax / ystep, 10))
        major_ticks = np.arange(tick_start, tick_end + ystep/2, ystep)
        major_ticks = np.round(major_ticks, 10)
        
        # clean up floating-point noise
        major_ticks = np.round(major_ticks, 10)
        
        ax.yaxis.set_major_locator(FixedLocator(major_ticks))

        if leg_loc == 'upper left':
            if omit_ctrl == False:
                legend1 = ax.legend(
                CO2_lin_leg_1, labels_1,
                loc=leg_loc,
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(0.0, 0.75),  # Adjust position as needed
                frameon=True
                )
                # Second legend (2 labels, positioned above the first)
                legend2 = ax.legend(
                    CO2_lin_leg_2, labels_2,
                    loc=leg_loc,
                    fontsize=10, labelspacing=0.1,
                    bbox_to_anchor=(0.0, 1.0),  # Adjust position as needed
                    frameon=True
                )
            else: # position further down
                legend1 = ax.legend(
                CO2_lin_leg_1, labels_1,
                loc=leg_loc,
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(0.0, 0.82),  # Adjust position as needed
                frameon=True
                )
                # Second legend (2 labels, positioned above the first)
                legend2 = ax.legend(
                    CO2_lin_leg_2, labels_2,
                    loc=leg_loc,
                    fontsize=10, labelspacing=0.1,
                    bbox_to_anchor=(0.0, 1.0),  # Adjust position as needed
                    frameon=True
                )
        # elif leg_loc == 'lower right':
        #     legend1 = ax.legend(
        #     CO2_lin_leg_1, labels_1,
        #     loc=leg_loc,
        #     fontsize=10, labelspacing=0.1,
        #     bbox_to_anchor=(1.0, 0.18),  # Adjust position as needed
        #     frameon=True
        #     )
        #     # Second legend (2 labels, positioned below the first)
        #     legend2 = ax.legend(
        #         CO2_lin_leg_2, labels_2,
        #         loc=leg_loc,
        #         fontsize=10, labelspacing=0.1,
        #         bbox_to_anchor=(1.0, 0.0),  # Adjust position as needed
        #         frameon=True
        #     )
        
        # Add the first legend back to the axis
        ax.add_artist(legend1)
        
        ax.grid("both")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            
        if co2_scen == 'doub':
            fig_name = f"2xCO2_lin_{power}"
            
        elif diff_type == 'quad':
            fig_name = f"4xCO2_lin_{power}"

        if omit_title is False:
            ax.set_title(title_suff)
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if savefig:
            if fig_pref != None:
                plt.savefig(fig_dir+f'{fig_pref}_SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
            else:
                plt.savefig(fig_dir+f'SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
            # plt.close()


def plot_SLA_ts_select(co2_scen,fig_dir,start_year,end_year, power_var_suff, power_strings,
                       variant="steric",
                       profiles=['surf','therm','mid','bot'],
                       doub_ctrl_rescale = 0.1,
                       omit_title=True, omit_ctrl=False,
                       vline_yr=None,
                       fig_pref=None,
                     ylimits = None,
                     xlimits=None,
                     leg_loc = 'upper left',
                     leg_ncols = 1):
    """
    Function to plot mixing response for the select power inputs that were only run for specific profiles.
    
    Inputs:
    co2_scen (str): one of ['const','doub','quad']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    
    """

    prof_dict = {'surf': 'b',
                 'therm': 'm',
                 'mid': 'g',
                 'bot': 'r'}

    mix_leg_options = [  # First column (4 labels)
        Line2D([0], [0], color='b', lw=2),  # surf
        Line2D([0], [0], color='m', lw=2),  # therm
        Line2D([0], [0], color='g', lw=2),  # mid
        Line2D([0], [0], color='r', lw=2)  # bot
    ]
    
    # Define custom legend entries
    labels_1 = []
    mix_leg_1 = []
    for idx, prof in enumerate(profiles):
        labels_1.append(f"{prof} {power_strings[idx]}")
        mix_leg_1.append(mix_leg_options[idx])

    mix_leg_2 = [Line2D([0], [0], alpha=0.6, lw=4, color='tab:gray')]

    if doub_ctrl_rescale != 1:
        labels_2 = [f'{doub_ctrl_rescale} x 1pct2xCO\u2082 ctrl']
    else:
        labels_2 = ['1pct2xCO\u2082 ctrl']

    fig, ax = plt.subplots(figsize=(6,3))

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    if vline_yr:
        ax.axvline(x=70, color='black', linestyle='-', linewidth=1)

    power_line_types = ['solid','dashed','dotted']

    if omit_ctrl == False:
        if co2_scen == 'const' or co2_scen == 'doub':
            ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_SLA'
        elif co2_scen == 'quad':
            ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_const_ctrl_SLA'
            
        ctrl_rad_anom = myVars[ctrl_rad_name][variant]
        time = np.linspace(start_year,end_year,num=len(ctrl_rad_anom))
        ax.plot(time,doub_ctrl_rescale*ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)

    ds_root = f'{start_year}_{end_year}'
    for prof_idx, prof in enumerate(profiles):
        if co2_scen == 'const':
            ds_name = f'const_{prof}_{power_var_suff[prof_idx]}_{ds_root}_SLA'
        elif co2_scen == 'doub':
            ds_name = f'doub_{prof}_{power_var_suff[prof_idx]}_{ds_root}_2xctrl_SLA'
        elif co2_scen == 'quad':
            ds_name = f'quad_{prof}_{power_var_suff[prof_idx]}_{ds_root}_4xctrl_SLA'

        mix_anom = myVars[ds_name][variant]
        time = np.linspace(start_year,end_year,num=len(mix_anom))
        ax.plot(time,mix_anom,color=prof_dict[prof])

    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Sea Level Anomaly (m)")
    ax.set_xlim(start_year-1,end_year)
    if ylimits:
        ax.set_ylim(ylimits)
        
    if xlimits:
        ax.set_xlim(xlimits)
    else:
        ax.set_xlim(start_year-1,end_year)

    legend1 = ax.legend(
    mix_leg_1, labels_1,
    loc=leg_loc,
    fontsize=10, labelspacing=0.1,
    bbox_to_anchor=(0.0, 1.0),  # Adjust position as needed
    frameon=True
    )
    if omit_ctrl == False:
        legend2 = ax.legend(
            mix_leg_2, labels_2,
            loc=leg_loc,
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(0.35, 1.0),  # Adjust position as needed
            frameon=True
        )
    
    # Add the first legend back to the axis
    ax.add_artist(legend1)
    
    ax.grid("both")
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

    if co2_scen == 'const':
        title_str = f"Const CO\u2082 SLA Comparison\n"
        fig_name = f"const_mix_select"
    elif co2_scen == 'doub':
        title_str = f"1pct2xCO\u2082 Mixing SLA Comparison\n"
        fig_name = f"2xCO2_mix_select"
    elif co2_scen == 'quad':
        title_str = f"1pct4xCO\u2082 Mixing SLA Comparison\n"
        fig_name = f"4xCO2_mix_select"

    if omit_title is False:
        ax.set_title(title_str+title_suff)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    if fig_pref != None:
        plt.savefig(fig_dir+f'{fig_pref}_SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(fig_dir+f'SLA_{variant}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')




