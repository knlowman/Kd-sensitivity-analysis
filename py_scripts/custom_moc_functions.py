#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for computing overturning circulations. This is designed to be used with the notebooks read_and_calculate.ipynb and compute_ensemble_means.ipynb.

import numpy as np
import xarray as xr
from xgcm import Grid
from dask.diagnostics import ProgressBar

# modules for plotting datetime data
import matplotlib.dates as mdates
from matplotlib.axis import Axis

# modules for using datetime variables
import datetime
from datetime import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

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

from cmip_basins import generate_basin_codes

import os


# %run /home/Kiera.Lowman/Kd-sensitivity-analysis/notebooks/read_and_calculate.ipynb
# %run /home/Kiera.Lowman/Kd-sensitivity-analysis/notebooks/compute_ensemble_means.ipynb


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


# # Functions for computing psi

def my_define_names(model="mom6",vertical="z"):
    """ modified version of xoverturning function
    define names for coordinates and variables according to model"""

    if model == "mom6":
        names = dict(
            x_center="xh",
            y_center="yh",
            x_corner="xq",
            y_corner="yq",
            lon_t="geolon",
            lat_t="geolat",
            mask_t="wet",
            lon_v="geolon_v",
            lat_v="geolat_v",
            mask_v="wet_v",
            bathy="deptho",
        )
        # names.update(dict(layer=f"{vertical}_l", interface=f"{vertical}_i"))
        if vertical == "z":
            names.update(dict(layer="z_l", interface="z_i"))
        elif vertical == "rho2":
            names.update(dict(layer="zl", interface="zi"))
    return names


def my_merge_grid_dataset(ds, dsgrid, names):
    """ same as original from xoverturning
    merge grid and transports dataset into one"""

    for coord in dsgrid.coords:
        ds[coord] = dsgrid[coord]

    for k, v in names.items():
        if v in dsgrid:
            ds[v] = dsgrid[v]

    return ds


def my_is_symetric(ds, names):
    """ same as original from xoverturning
    check if grid is symetric

    Args:
        ds (xarray.Dataset): dataset containing model's grid
        names (dict): dictionary containing dimensions/coordinates names

    Returns:
        bool: True if grid is symetric
    """

    x_center, y_center = names["x_center"], names["y_center"]
    x_corner, y_corner = names["x_corner"], names["y_corner"]

    if (len(ds[x_corner]) == len(ds[x_center])) and (
        len(ds[y_corner]) == len(ds[y_center])
    ):
        out = False
    elif (len(ds[x_corner]) == len(ds[x_center]) + 1) and (
        len(ds[y_corner]) == len(ds[y_center]) + 1
    ):
        out = True
    else:
        raise ValueError("unsupported combination of coordinates")
    return out


def my_define_grid(ds, names):
    """ same as original from xoverturning
    build a xgcm.Grid object

    Args:
        ds (xarray.Dataset): dataset with model's grid
        names (dict): dictionary containing dimensions/coordinates names

    Returns:
        xgcm.Grid: grid object
    """

    x_center, y_center = names["x_center"], names["y_center"]
    x_corner, y_corner = names["x_corner"], names["y_corner"]

    qcoord = "outer" if my_is_symetric(ds, names) else "right"

    grid = Grid(
        ds,
        coords={
            "X": {"center": x_center, qcoord: x_corner},
            "Y": {"center": y_center, qcoord: y_corner},
        },
        periodic=["X"],
    )
    return grid


def my_substract_hml(ds, umo="umo", vmo="vmo", uhml="uhml", vhml="vhml"):
    """ same as original from xoverturning
    substracting Thickness Flux to Restratify Mixed Layer
    from transports

    Args:
        ds (xarray.Dataset): dataset containing transports
        umo (str, optional): name of zonal transport
        vmo (str, optional): name of meridional transport
        uhml (str, optional): name of zonal Thickness Flux
        vhml (str, optional): name of meriodional Thickness Flux

    Returns:
        xarray.DataArray: corrected transports
    """

    if uhml in ds.variables:
        # substract from meridional transport
        ucorr = ds[umo] - ds[uhml]
    else:
        raise IOError(f"{uhml} not found in dataset")

    if vhml in ds.variables:
        # substract from meridional transport
        vcorr = ds[vmo] - ds[vhml]
    else:
        raise IOError(f"{vhml} not found in dataset")

    return ucorr, vcorr


def my_rotate_velocities_to_geo(ds, da_u, da_v, names):
    """ same as original from xoverturning
    rotate a pair of velocity vectors to the geographical axes

    Args:
        ds (xarray.Dataset): dataset containing velocities to rotate
        da_u (xarray.DataAray): data for u-component of velocity in model coordinates
        da_v (xarray.DataArray): data for v-component of velocity in model coordinates
        names (dict): dictionary containing dimensions/coordinates names

    Returns:
        xarray.DataArray: rotated velocities
    """

    if "cos_rot" in ds.variables and "sin_rot" in ds.variables:
        CS = ds["cos_rot"]
        SN = ds["sin_rot"]
    elif "angle_dx" in ds.variables:
        CS = np.cos(ds["angle_dx"])
        SN = np.sin(ds["angle_dx"])
    else:
        # I would like to have a way to retrieve angle from lon/lat arrays
        raise ValueError("angle or components must be included in dataset")

    # build the xgcm grid object
    grid = my_define_grid(ds, names)
    # interpolate to the cell centers
    u_ctr = grid.interp(da_u, "X", boundary="fill")
    v_ctr = grid.interp(da_v, "Y", boundary="fill")
    # rotation inverse from the model's grid angle
    u_EW = u_ctr * CS - v_ctr * SN
    v_EW = v_ctr * CS + u_ctr * SN

    return u_EW, v_EW


# this function not used anywhere
def my_interp_to_grid_center(ds, da_u, da_v, names):
    """ same as original from xoverturning
    interpolate velocities to cell centers

    Args:
        ds (xarray.Dataset): dataset containing velocities to rotate
        da_u (xarray.DataAray): data for u-component of velocity in model coordinates
        da_v (xarray.DataArray): data for v-component of velocity in model coordinates
        names (dict): dictionary containing dimensions/coordinates names

    Returns:
        xarray.DataArray: interpolated velocities
    """
    # build the xgcm grid object
    grid = my_define_grid(ds, names)
    # interpolate to the cell centers
    u_ctr = grid.interp(da_u, "X", boundary="fill")
    v_ctr = grid.interp(da_v, "Y", boundary="fill")
    return u_ctr, v_ctr


def my_select_basins(
    ds,
    names,
    basin="global",
    lon="geolon",
    lat="geolat",
    mask="wet",
    vertical="z",
    verbose=True,
):
    """ modified version of xoverturning function
    generate a mask for selected basin

    Args:
        ds (xarray.Dataset): dataset contaning model grid
        names (dict): dictionary containing dimensions/coordinates names
        basin (str or list, optional): global/atl-arc/indopac or list of codes. Defaults to "global".
        lon (str, optional): name of geographical lon in dataset. Defaults to "geolon".
        lat (str, optional): name of geographical lat in dataset. Defaults to "geolat".
        mask (str, optional): name of land/sea mask in dataset. Defaults to "wet".
        verbose (bool, optional): Verbose output. Defaults to True.

    Returns:
        xarray.DataArray: mask for selected basin
        xarray.DataArray: mask for MOC streamfunction
    """

    # read or recalculate basin codes
    if "basin" in ds:
        basincodes = ds["basin"]
    else:
        if verbose:
            print("generating basin codes")
        basincodes = generate_basin_codes(ds, lon=lon, lat=lat, mask=mask)

    # expand land sea mask to remove other basins
    if isinstance(basin, str):
        if basin == "global":
            maxcode = basincodes.max()
            assert not np.isnan(maxcode)
            selected_codes = np.arange(1, maxcode + 1)
        elif basin == "atl-arc": # modified to include Southern Ocean
            selected_codes = [1, 2, 4, 6, 7, 8, 9]
            # selected_codes = [2, 4, 6, 7, 8, 9]
        # elif basin == "atl-arc": # I modified to include Southern Ocean only in Atlantic, but it looks weird
        #     selected_codes = [2, 4, 6, 7, 8, 9]
        #     cond1 = ds[lon] < 20.5
        #     cond2 = ds[lon] > -70.5
        #     cond3 = basincodes == 1
        #     maskbin = ds[mask].where((basincodes.isin(selected_codes)) | (cond1 & cond2 & cond3))
        elif basin == "indopac":
            selected_codes = [3, 5, 10, 11]
        elif basin == "pac-only": # for debugging
            selected_codes = [3]
        elif basin == "ind-only": # for debugging
            selected_codes = [5]
        elif basin == "pac-slice": # for debugging
            cond1 = ds[lon] < -100
            cond2 = ds[lon] > -200
            cond3 = basincodes == 3
            maskbin = ds[mask].where((cond1 & cond2 & cond3))
        else:
            raise ValueError("Unknown basin")
    elif isinstance(basin, list):
        for b in basin:
            assert isinstance(b, int)
        selected_codes = basin
    else:
        raise ValueError("basin must be a string or list of int")

    if basin == "pac-slice":
        maskbasin = xr.where(maskbin == 1, True, False)
    else:
        maskbin = ds[mask].where(basincodes.isin(selected_codes))
        maskbasin = xr.where(maskbin == 1, True, False)

    bathy, interface = names["bathy"], names["interface"]
    y_corner, y_center, x_center = (
        names["y_corner"],
        names["y_center"],
        names["x_center"],
    )

    # create a mask for the streamfunction
    if (bathy in ds) and (vertical == "z"):
        if y_corner in maskbasin.dims:
            grid = my_define_grid(ds, names)
            bathy_coloc = grid.interp(ds[bathy], "Y", boundary="fill")
        elif y_center in maskbasin.dims:
            bathy_coloc = ds[bathy]
        else:
            raise ValueError("Unsupported coord")
        bathy_basin = bathy_coloc.where(maskbasin).fillna(0.0)
        max_depth = bathy_basin.max(dim=x_center)
        maskmoc = xr.where(ds[interface] > max_depth, 0, 1)
    else:
        maskmoc = None

    return maskbasin, maskmoc


def my_compute_streamfunction(
    ds,
    names,
    transport="v",
    rho0=1035.0,
    add_offset=False,
    offset=0.1,
    fromtop=False,
):
    """ same as original from xoverturning
    compute the overturning streamfunction from meridional transport

    Args:
        ds (xarray.Dataset): meridional transport in kg.s-1
        names (dict): dictionary containing dimensions/coordinates names
        transport (str, optional): name of transport. Defaults to "v".
        rho0 (float, optional): average density of seawater. Defaults to 1035.0.
        add_offset (bool, optional): add a small number to clean 0 contours. Defaults to False.
        offset (float, optional): offset for contours, should be small. Defaults to 0.1.
        fromtop (bool, optional): integrate from the surface to the bottom. Defaults to False.

    Returns:
        xarray.DataArray: Overturning streamfunction
    """

    x_center = names["x_center"]
    layer, interface = names["layer"], names["interface"]

    # sum over the zonal direction
    zonalsum = ds[transport].sum(dim=x_center)
    
    if fromtop:
        # integrate from surface
        integ_layers_from_surface = zonalsum.cumsum(dim=layer)
        # the result of the integration over layers is evaluated at the interfaces
        # with psi = 0 as the surface boundary condition for the integration
        surface_condition = xr.zeros_like(integ_layers_from_surface.isel({layer: 0}))
        psi_raw = xr.concat([surface_condition, integ_layers_from_surface], dim=layer)
    else:
        # integrate from bottom
        integ_layers_from_bottom = zonalsum.cumsum(dim=layer) - zonalsum.sum(dim=layer)
        # the result of the integration over layers is evaluated at the interfaces
        # with psi = 0 as the bottom boundary condition for the integration
        bottom_condition = xr.zeros_like(integ_layers_from_bottom.isel({layer: -1}))
        psi_raw = xr.concat([integ_layers_from_bottom, bottom_condition], dim=layer)

    psi_raw = psi_raw.chunk({layer: len(psi_raw[layer])})  # need to rechunk to new size

    # rename to correct dimension and add correct vertical coordinate
    psi = psi_raw.rename({layer: interface})
    psi[interface] = ds[interface]
    psi.name = "psi"  # set variable name in dataarray

    # convert kg.s-1 to Sv (1e6 m3.s-1)
    psi_Sv = psi / rho0 / 1.0e6
    # optionally add offset to make plots cleaner
    if add_offset:
        psi_Sv += offset
    return psi_Sv


def my_calcmoc(
    ds,
    dsgrid=None,
    basin="global",
    rotate=False,
    remove_hml=False,
    add_offset=False,
    mask_output=False,
    output_true_lat=False,
    offset=0.1,
    rho0=1035.0,
    vertical="z",
    model="mom6",
    umo="umo",
    vmo="vmo",
    uhml="uhml",
    vhml="vhml",
    verbose=True,
):
    """ modified version of xoverturning function
    Compute Meridional Overturning

    Args:
        ds (xarray.Dataset): input dataset. It should contain at least
                             umo, vmo and some grid information
        dsgrid (xarray.Dataset): grid dataset. It should contain at least
                             lon/lat/mask
        basin (str, optional): Basin to use (global/atl-arc/indopac). Defaults to "global".
        rotate (bool, optional): Rotate velocities to true North. Defaults to False.
        remove_hml (bool, optional): Substract Thickness Flux to Restratify Mixed Layer.
                                     Defaults to False.
        add_offset (bool, optional): Add offset to clean up zero contours in plot. Defaults to False.
        mask_output (bool, optional): mask ocean floor, only for Z-coordinates
        output_true_lat (bool, optional): return the nominal latitude instead of the "yq" index coord.
        offset (float, optional): offset for contours, should be small. Defaults to 0.1.
        rho0 (float, optional): Average density of seawater. Defaults to 1035.0.
        vertical (str, optional): Vertical dimension (z, rho2). Defaults to "z".
        model (str, optional): ocean model used, currently only mom6 is supported.
        umo (str, optional): override for transport name. Defaults to "umo".
        vmo (str, optional): override for transport name. Defaults to "vmo".
        uhml (str, optional): overide for thickness flux. Defaults to "uhml".
        vhml (str, optional): override for thickness flux. Defaults to "vhml".
        verbose (bool, optional): verbose output. Defaults to True.

    Returns:
        xarray.DataArray: meridional overturning
    """

    names = my_define_names(model=model,vertical=vertical)

    if dsgrid is not None:
        ds = my_merge_grid_dataset(ds, dsgrid, names)

    if remove_hml:
        ucorr, vcorr = my_substract_hml(ds, umo=umo, vmo=vmo, uhml=uhml, vhml=vhml)
    else:
        vcorr = ds[vmo]
        if umo in ds.variables: # I edited this
            ucorr = ds[umo]

    if rotate:
        u_ctr, v_ctr = my_rotate_velocities_to_geo(ds, ucorr, vcorr, names)
    else:
        v_ctr = vcorr
        if umo in ds.variables: # I edited this
            u_ctr = ucorr

    # check vertical dimensions are in the dataarray
    layer = names["layer"]
    if layer not in v_ctr.dims:
        raise ValueError(f"{layer} not found in transport array")

    # print("v_ctr.dims")
    # print(v_ctr.dims)
    # print(f"names['x_center'] = {names['x_center']}")
    # print(f"names['y_corner'] = {names['y_corner']}")
    # print(f"names['y_center'] = {names['y_center']}")
    
    # use dimensions of v to know which lon/lat/mask to use
    if (names["y_corner"] in v_ctr.dims) and (names["x_center"] in v_ctr.dims):
        lon, lat, mask = names["lon_v"], names["lat_v"], names["mask_v"]
        # i.e. geolon_v, geolat_v, wet_v
    elif (names["y_center"] in v_ctr.dims) and (names["x_center"] in v_ctr.dims):
        lon, lat, mask = names["lon_t"], names["lat_t"], names["mask_t"]

    maskbasin, maskmoc = my_select_basins(
        ds,
        names,
        basin=basin,
        lon=lon,
        lat=lat,
        mask=mask,
        vertical=vertical,
        verbose=verbose
    )

    ds_v = xr.Dataset()
    ds_v["v"] = v_ctr.where(maskbasin)
    
    for var in [
        names["x_center"],
        names["y_center"],
        names["x_corner"],
        names["y_corner"],
        names["layer"],
        names["interface"]
        ]:
        ds_v[var] = ds[var]

    moc = my_compute_streamfunction(
        ds_v,
        names,
        transport="v", # remember this is ds_v["v"], not ds["v"]
        rho0=rho0,
        add_offset=add_offset,
        offset=offset
        # fromtop=True
    )
    if vmo == "vh_rho" or vmo == "vhGM_rho":
        moc = moc*rho0

    if mask_output:
        moc = moc.where(maskmoc)

    if output_true_lat:
        moc = moc.assign_coords({names["y_corner"]: ds[lat].max(dim=names["x_center"])})
        moc = moc.rename({names["y_corner"]: "lat"})

    return moc


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
        
    zrho = zrho.rename('depth')

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


def compute_zrho_moc(
    static_rho,
    ds_rho,
    basin="global",
    add_offset=False,
    offset=0.1,
    rho0=1035.0,
    verbose=True):

    ds_rho = ds_rho.assign_coords({'geolon_u': static_rho['geolon_u'],
                       'geolat_u': static_rho['geolat_u'],
                       'geolon_v': static_rho['geolon_v'],
                       'geolat_v': static_rho['geolat_v'],
                       'geolon'  : static_rho['geolon'],
                       'geolat'  : static_rho['geolat']
                       })
        
    ds_rho['wet'] = static_rho['wet']

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
    
    # maskbasin, maskmoc = my_select_basins(
    #     ds,
    #     names,
    #     basin=basin,
    #     lon=lon,
    #     lat=lat,
    #     mask=mask,
    #     vertical=vertical,
    #     verbose=verbose
    # )

    maskbasin = selecting_basins(static_rho, basin=basin, lon="geolon", lat = "geolat", mask = "wet")
    maskbasin_v = selecting_basins(static_rho, basin=basin, lon="geolon_v", lat = "geolat_v", mask = "wet_v")
    
    vmo  = ds_rho['vmo'].mean(dim='time').where(maskbasin_v)

    # calculate cell thickness from volcello and areacello (I don't have thkcello saved)
    thk  = (ds_rho['volcello'].mean(dim='time')/static_rho['areacello']).where(maskbasin)
    thk.name = 'thkcello'
    
    vmo  = vmo.where(vmo < 1e14)
    thk  = thk.where(thk < 1e10)
    zrho = thk.mean(dim='xh').cumsum(dim='zl')
    zrho = zrho.rename('depth')
    
    vmo_xsum = vmo.sum(dim='xh')
    psi      = (vmo_xsum.cumsum(dim='zl') - vmo_xsum.sum(dim='zl'))/rho0/1e6
    
    if add_offset:
        psi += offset
        
    psi.name = 'psi'
    
    # add a depth coordinate to psi, with depth 
    # defined by zonal mean of the time mean depth of rho
    depth = grid.interp(zrho, 'Y', boundary='extend')
    psi.coords['depth'] = depth
    
    # psi.load()
    # zrho.load()

    return psi, depth


def compute_moc_max(moc_z,lat_low=None,lat_high=None,zlow=None,zhigh=None,find_min=False,true_lat_dims=True,zrho_method=False,verbose=False):
    """
    Function to return the maximum MOC strength and its depth-latitude coordinate.
    Inputs:
        moc_z: MOC data array returned by my_calcmoc()
        lat_range: optional latitude range
        z_range: optional depth range

    Returns:
        max_value_ds: dataset containing maximum value in Sv ('psi_max'), nominal latitude of maximum MOC 'y_max', 
        and depth of maximum MOC 'z_max'
    """

    if true_lat_dims:
        y_name = 'lat'
    else:
        y_name = 'yq'

    if zrho_method:
        z_name = 'depth'
    else:
        z_name = 'z_i'
        
    # max_value = moc_z.sel(yq=slice(lat_low,lat_high), z_i=slice(zlow,zhigh)).max(dim=['yq', 'z_i'],skipna=True).load()
    
    # Find the indices of the maximum value
    if true_lat_dims:
        moc_selected = moc_z.sel(lat = slice(lat_low, lat_high), z_i=slice(zlow, zhigh))
    elif zrho_method:
        moc_selected = moc_z.sel(yq = slice(lat_low, lat_high), depth=slice(zlow, zhigh))
    else:
        moc_selected = moc_z.sel(yq = slice(lat_low, lat_high), z_i=slice(zlow, zhigh))

    if find_min:
        moc_max = moc_selected.min(dim=[y_name,z_name],keep_attrs='time',skipna=True)
    else:
        moc_max = moc_selected.max(dim=[y_name,z_name],keep_attrs='time',skipna=True)
        
    moc_max = moc_max.chunk({'time': moc_max.sizes['time']})
    max_value_ds = moc_max
    
    # max_indices = moc_selected.argmax(dim=['yq','z_i'],keep_attrs='time',skipna=True)
    
    # # Convert max_indices to NumPy values (if Dask-backed)
    # max_indices_yq = max_indices['yq']#.values
    # max_indices_z_i = max_indices['z_i']#.values
       
    # # Extract the actual coordinates corresponding to the maximum value
    # if len(max_indices_yq) == 1:
    #     yq_max = moc_z['yq'].isel(yq=max_indices_yq.values)
    #     z_i_max = moc_z['z_i'].isel(z_i=max_indices_z_i.values)
    # else:
    #     yq_max = moc_z['yq'].isel(yq=max_indices_yq)
    #     z_i_max = moc_z['z_i'].isel(z_i=max_indices_z_i)
        
    # # Get the maximum value using .sel() for indexing by coordinates
    # psi_max = moc_z.sel(yq=yq_max, z_i=z_i_max)

    # max_value_ds = psi_max.reset_coords(["z_i","yq"]).drop_vars(["z_i","yq"])
    # max_value_ds = max_value_ds.load()
    
    
    return max_value_ds


# # Functions to compute ensemble-mean MOC and MOC anomaly

def calc_ens_mem_MOC(exp_ds_list, grid, basin_list,
                     pp_type='av-annual',
                     zrho_method=False,
                     mask_output=True, verbose=False, debug=False):

    """
    Function to compute ensemble mean MOC and MOC strength. Default latitude bounds of 26-28 N for computing MOC strength.
    """

    # north_atl_lat_range = [26,28]
    # southern_lat_range = [-31,-29]
    # indopac_lat_range = [-30,0]
    # depth_range = [500,None]
    # pac_depths = [1200,None]
    north_atl_lat_range = [26.5,27]
    southern_lat_range = [-30.5,-30]
    indopac_lat_range = [-30,-15]
    depth_range = [None,None]
    pac_depths = [1100,None]
    
    num_ens_mem = len(exp_ds_list)
    
    psi_ens_mean = xr.Dataset()
    max_psi_ens_mean = xr.Dataset()

    for basin in basin_list:
        
        moc_list = [None] * num_ens_mem
        
        if pp_type == 'ts-annual':
            # moc_max_list = [None] * num_ens_mem
            moc_max_list_north = [None] * num_ens_mem
            moc_max_list_south = [None] * num_ens_mem

        if zrho_method:
            depth_list = [None] * num_ens_mem
        
        # calculate MOC for basin
        for i in range(num_ens_mem):
            if zrho_method:
                moc_list[i], depth_list[i] = compute_zrho_moc(grid, exp_ds_list[i], basin=basin, verbose=False)
            else:
                moc_list[i] = my_calcmoc(exp_ds_list[i], dsgrid=grid, basin=basin, mask_output=mask_output, output_true_lat=True, verbose=False)
            if debug:
                print(f"moc_list[{i}] done")
                if i == 0 and basin == basin_list[0]:
                    print("lat values:\n")
                    print(moc_list[i].lat.values)
            if pp_type == 'ts-annual':
                if basin == 'atl-arc':
                    moc_max_list_north[i] = compute_moc_max(moc_list[i],zrho_method=zrho_method,
                                                            lat_low=north_atl_lat_range[0],lat_high=north_atl_lat_range[1],
                                                            zlow=depth_range[0],zhigh=depth_range[1],
                                                            verbose=verbose)
                    # moc_max_list[i] = moc_max_list[i].reset_index(["y_max","z_max"]).drop_vars(names=["y_max","z_max"])
                    moc_max_list_south[i] = compute_moc_max(moc_list[i],zrho_method=zrho_method,
                                                            lat_low=southern_lat_range[0],lat_high=southern_lat_range[1],
                                                            zlow=depth_range[0],zhigh=depth_range[1],
                                                            verbose=verbose)
                elif basin == 'indopac':
                    moc_max_list_south[i] = compute_moc_max(moc_list[i],zrho_method=zrho_method,find_min=True,
                                                            lat_low=indopac_lat_range[0],lat_high=indopac_lat_range[1],
                                                            zlow=pac_depths[0],zhigh=pac_depths[1],
                                                            verbose=verbose)
                elif basin == 'global':
                    moc_max_list_south[i] = compute_moc_max(moc_list[i],zrho_method=zrho_method,
                                                            lat_low=southern_lat_range[0],lat_high=southern_lat_range[1],
                                                            zlow=depth_range[0],zhigh=depth_range[1],
                                                            verbose=verbose)
            
        moc = ensembles.create_ensemble(moc_list).mean("realization")
        if zrho_method:
            depth_field = ensembles.create_ensemble(depth_list).mean("realization")
            depth_field = depth_field['depth']
            print(depth_field)
            
        if pp_type == 'ts-annual':
            if basin == 'atl-arc':
                moc_max_north = ensembles.create_ensemble(moc_max_list_north).mean("realization")
                max_psi_ens_mean['atl-arc_north_psi_max'] = moc_max_north['psi']
                moc_max_south = ensembles.create_ensemble(moc_max_list_south).mean("realization")
                max_psi_ens_mean['atl-arc_south_psi_max'] = moc_max_south['psi']
            elif basin == 'indopac':
                moc_max_south = ensembles.create_ensemble(moc_max_list_south).mean("realization")
                max_psi_ens_mean['indopac_south_psi_max'] = moc_max_south['psi']
            elif basin == 'global':
                moc_max_south = ensembles.create_ensemble(moc_max_list_south).mean("realization")
                max_psi_ens_mean['global_south_psi_max'] = moc_max_south['psi']
                
        # # After computing each basin
        # print(f"Basin {basin} psi shape: {moc['psi'].shape}")
        # print(f"Basin {basin} psi_max shape: {moc_max['psi_max'].shape}")

        psi_ens_mean[basin] = moc['psi']#.load()
        if zrho_method:
            psi_ens_mean[basin].assign_coords({f'{basin}_depth': depth_field.values})

    # return moc, moc_max
    return psi_ens_mean, max_psi_ens_mean #, moc_list, max_moc_list


def calc_ens_mem_MOC_diff(ref_ds_list, exp_ds_list, grid, basin_list, 
                          pp_type='av-annual',
                          zrho_method=False,
                          lat_low=26,lat_high=28,
                          zlow=500,zhigh=None, 
                          mask_output=True, verbose=False, debug=False):

    """
    Function to compute ensemble mean MOC and MOC strength. Default latitude bounds of 26-28 N for computing MOC strength.
    """

    # north_atl_lat_range = [26,28]
    # southern_lat_range = [-31,-29]
    # indopac_lat_range = [-30,0]
    # depth_range = [500,None]
    # pac_depths = [1200,None]
    north_atl_lat_range = [26.5,27]
    southern_lat_range = [-30.5,-30]
    indopac_lat_range = [-30,-15]
    depth_range = [None,None]
    pac_depths = [1100,None]
    
    num_ens_mem = len(ref_ds_list)

    if num_ens_mem != len(exp_ds_list):
        raise IOError(f"ref_ds_list and exp_ds_list have different numbers of ensemble members.")

    diff_ens_mean = xr.Dataset()
    max_diff_ens_mean = xr.Dataset()

    for basin in basin_list:
        moc_diff_list = [None] * num_ens_mem
        if pp_type == 'ts-annual':
            # moc_max_diff_list = [None] * num_ens_mem
            moc_max_diff_list_north = [None] * num_ens_mem
            moc_max_diff_list_south = [None] * num_ens_mem
        
        # calculate MOC for reference and case of interest
        for i in range(num_ens_mem):
            if zrho_method:
                moc_ref = compute_zrho_moc(grid, ref_ds_list[i], basin=basin, verbose=False)
                moc_exp = compute_zrho_moc(grid, exp_ds_list[i], basin=basin, verbose=False)
            else:
                moc_ref = my_calcmoc(ref_ds_list[i], dsgrid=grid, basin=basin, mask_output=mask_output, output_true_lat=True, verbose=False)
                moc_exp = my_calcmoc(exp_ds_list[i], dsgrid=grid, basin=basin, mask_output=mask_output, output_true_lat=True, verbose=False)
            moc_diff_list[i] = moc_exp - moc_ref
            if debug:
                print(f"moc_diff_list[{i}] done")
    
            if pp_type == 'ts-annual':
                if basin == 'atl-arc':
                    moc_ref_max_north = compute_moc_max(moc_ref,zrho_method=zrho_method,
                                                        lat_low=north_atl_lat_range[0],lat_high=north_atl_lat_range[1],
                                                        zlow=depth_range[0],zhigh=depth_range[1],
                                                        verbose=verbose)
                    moc_exp_max_north = compute_moc_max(moc_exp,zrho_method=zrho_method,
                                                        lat_low=north_atl_lat_range[0],lat_high=north_atl_lat_range[1],
                                                        zlow=depth_range[0],zhigh=depth_range[1],
                                                        verbose=verbose)
                    moc_max_diff_list_north[i] = moc_exp_max_north - moc_ref_max_north
                    
                    moc_ref_max_south = compute_moc_max(moc_ref,zrho_method=zrho_method,
                                                        lat_low=southern_lat_range[0],lat_high=southern_lat_range[1],
                                                        zlow=depth_range[0],zhigh=depth_range[1],
                                                        verbose=verbose)
                    moc_exp_max_south = compute_moc_max(moc_exp,zrho_method=zrho_method,
                                                        lat_low=southern_lat_range[0],lat_high=southern_lat_range[1],
                                                        zlow=depth_range[0],zhigh=depth_range[1],
                                                        verbose=verbose)
                    moc_max_diff_list_south[i] = moc_exp_max_south - moc_ref_max_south

                elif basin == 'indopac':
                    moc_ref_max_south = compute_moc_max(moc_ref,zrho_method=zrho_method,find_min=True,
                                                        lat_low=indopac_lat_range[0],lat_high=indopac_lat_range[1],
                                                        zlow=pac_depths[0],zhigh=pac_depths[1],
                                                        verbose=verbose)
                    moc_exp_max_south = compute_moc_max(moc_exp,zrho_method=zrho_method,find_min=True,
                                                        lat_low=indopac_lat_range[0],lat_high=indopac_lat_range[1],
                                                        zlow=pac_depths[0],zhigh=pac_depths[1],
                                                        verbose=verbose)
                    moc_max_diff_list_south[i] = moc_exp_max_south - moc_ref_max_south
                    
                elif basin == 'global':
                    moc_ref_max_south = compute_moc_max(moc_ref,zrho_method=zrho_method,
                                                        lat_low=southern_lat_range[0],lat_high=southern_lat_range[1],
                                                        zlow=depth_range[0],zhigh=depth_range[1],
                                                        verbose=verbose)
                    moc_exp_max_south = compute_moc_max(moc_exp,zrho_method=zrho_method,
                                                        lat_low=southern_lat_range[0],lat_high=southern_lat_range[1],
                                                        zlow=depth_range[0],zhigh=depth_range[1],
                                                        verbose=verbose)
                    moc_max_diff_list_south[i] = moc_exp_max_south - moc_ref_max_south
                    
                # moc_ref_max = compute_moc_max(moc_ref,zrho_method=zrho_method,lat_low=lat_low,lat_high=lat_high,zlow=zlow,zhigh=zhigh,verbose=verbose)
                # # moc_ref_max = moc_ref_max.reset_index(["y_max","z_max"]).drop_vars(names=["y_max","z_max"])
                # moc_exp_max = compute_moc_max(moc_exp,zrho_method=zrho_method,lat_low=lat_low,lat_high=lat_high,zlow=zlow,zhigh=zhigh,verbose=verbose)
                # # moc_exp_max = moc_exp_max.reset_index(["y_max","z_max"]).drop_vars(names=["y_max","z_max"])
                # moc_max_diff_list[i] = moc_exp_max - moc_ref_max
    
                # if moc_max_diff_list[i]['psi_max'].shape != (1,1,1):
                #     print(f"Warning: moc_max_diff_list[i]['psi_max'].shape is {moc_max_diff_list[i]['psi_max'].shape}")
    
        moc_diff = ensembles.create_ensemble(moc_diff_list).mean("realization")
        # if pp_type == 'ts-annual':
        #     moc_max_diff = ensembles.create_ensemble(moc_max_diff_list).mean("realization")

        diff_ens_mean[basin] = moc_diff['psi']#.load()
        
        if pp_type == 'ts-annual':
            if basin == 'atl-arc':
                moc_max_diff_north = ensembles.create_ensemble(moc_max_diff_list_north).mean("realization")
                max_diff_ens_mean['atl-arc_north_diff_max'] = moc_max_diff_north['psi']
                moc_max_diff_south = ensembles.create_ensemble(moc_max_diff_list_south).mean("realization")
                max_diff_ens_mean['atl-arc_south_diff_max'] = moc_max_diff_south['psi']
            elif basin == 'indopac':
                moc_max_diff_south = ensembles.create_ensemble(moc_max_diff_list_south).mean("realization")
                max_diff_ens_mean['indopac_south_diff_max'] = moc_max_diff_south['psi']
            elif basin == 'global':
                moc_max_diff_south = ensembles.create_ensemble(moc_max_diff_list_south).mean("realization")
                max_diff_ens_mean['global_south_diff_max'] = moc_max_diff_south['psi']
            # # max_diff_ens_mean[f'{basin}-y_max'] = moc_max_diff['y_max']
            # # max_diff_ens_mean[f'{basin}-z_max'] = moc_max_diff['z_max']
            # max_diff_ens_mean[f'{basin}_psi_max'] = moc_max_diff['psi']

    
    # max_diff_ens_mean['y_max'] = moc_max_diff['y_max'].load()
    # max_diff_ens_mean['z_max'] = moc_max_diff['z_max'].load()
    # max_diff_ens_mean['psi_max'] = moc_max_diff['psi_max'].load()

    return diff_ens_mean, max_diff_ens_mean
    # return moc_diff, moc_max_diff


# # Main functions

# The function get_ens_MOC_data() computes the MOC for each ensemble member of the reference and perturbation experiments, next takes the difference in MOC for each ensemble member, and then returns both the ensemble-mean and ensemble-mean differences (with respect to various references, depending on the experiment). This approach minimizes the impact of initial condition variability.

def get_ens_MOC_data(co2_scen, avg_period, mem1_start, mem1_end, grid,
                     zrho_method=False,
                var_list=['vmo','e'], #'vh_rho','vhGM_rho',
                     # added variable 'e' (interface height relative to mean sea level) to get z_i
                basin_list = ["global","atl-arc","indopac"],
                pp_type='av-annual',
                diag_file='ocean_monthly_z',
                profiles = ['surf','therm','mid','bot'],
                power_inputs = ['0.1TW', '0.2TW', '0.3TW'],
                power_var_suff = ['0p1TW', '0p2TW', '0p3TW'],
                num_ens_mem = 3,
                omit_mem1=False,
                ramp_exp=False,
                lat_bound=None,
                verbose=False,
                debug=False):

    """
    Returns variables containing the ensemble-mean raw data and the ensemble-mean anomaly data. Anomalies are calculated as the difference relative to 
    the control run during the period corresponding to an ensemble member (i.e. the anomalies for ensemble member 2 for year 201 to 400 are taking as 
    the difference relative to year 201 to 400 of the control run).

        Args:
            co2_scen (str): one of ['const','doub','quad','const+doub','all']; difference datasets will only be created for the co2 scenario specified, 
                            but ensembles + means may be created for control case of other CO2 scenarios
            avg_period (int): number of years for av/ts period
            mem1_start (int): start year of ens. mem. #1
            mem1_end (int): end year of ens. mem. #1
            grid (dataset): grid file to use for psi calculations (not required by my_calcmoc, but needed for my case)
            var_list (str list): list of variables to read
            basin_list (str list): list of basins to compute differences for
            pp_type (str): type of pp data to read
            diag_file (str): name of diag file containing the variables
            profiles (str list): list of profiles to get data for
            power_inputs (str list): list of power inputs to get data for
            power_var_suff (str list): list of variable suffixes for each power input
            verbose: if True, print variable names after declaration
            debug
            
        Returns:
            has no return variables, but creates xarray datasets by using myVars = globals()
            
    """
    allowed_scen = ['const','doub','quad','const+doub','all']
    
    if co2_scen not in allowed_scen:
        raise ValueError(f"'co2_scen' must be one of {allowed_scen}.")

    # start_yrs = [mem1_start,
    #              mem1_start+200,
    #              mem1_start+400]
    # end_yrs = [mem1_end,
    #            mem1_end+200,
    #            mem1_end+400]

    start_yrs = [None] * num_ens_mem
    end_yrs = [None] * num_ens_mem
    for idx in range(num_ens_mem):
        start_yrs[idx] = mem1_start + 200*idx
        end_yrs[idx] = mem1_end + 200*idx

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

    const_ctrl_mem_list, const_ctrl = create_const_doub_ens_mean(const_ctrl_exps,start_yrs,end_yrs,avg_period,var_list,
                                                                 pp_type=pp_type,diag_file=diag_file,omit_mem1=omit_mem1,debug=debug)
    if debug:
        print("Read data for const ctrl")
    # for basin in basin_list:
    moc_z, max_moc = calc_ens_mem_MOC(const_ctrl_mem_list, grid, basin_list, pp_type=pp_type, zrho_method=zrho_method, 
                                      mask_output=True, verbose=verbose, debug=debug)
    myVars.__setitem__(f"const_ctrl_{mem1_start}_{mem1_end}_psi", moc_z)
    print(f'const_ctrl_{mem1_start}_{mem1_end}_psi done')
    if pp_type == 'ts-annual':
        myVars.__setitem__(f"const_ctrl_{mem1_start}_{mem1_end}_max", max_moc)
        print(f'const_ctrl_{mem1_start}_{mem1_end}_max done')

    if co2_scen != 'const':
        ## 2xCO2 control ##
        if num_ens_mem == 3:
            doub_ctrl_exps = ["tune_ctrl_2xCO2_1860IC_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr"]
        elif num_ens_mem == 4:
            doub_ctrl_exps = ["tune_ctrl_2xCO2_1860IC_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr"]
        elif num_ens_mem == 5:
            doub_ctrl_exps = ["tune_ctrl_2xCO2_1860IC_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr",
                              "ens5_tune_ctrl_2xCO2_200yr"]
        elif num_ens_mem == 6:
            doub_ctrl_exps = ["tune_ctrl_2xCO2_1860IC_200yr","ens2_ctrl_2xCO2_200yr",
                              "ens3_ctrl_2xCO2_200yr","ens4_tune_ctrl_2xCO2_200yr",
                              "ens5_tune_ctrl_2xCO2_200yr","ens6_tune_ctrl_2xCO2_200yr"]
        
        doub_ctrl_mem_list, doub_ctrl = create_const_doub_ens_mean(doub_ctrl_exps,start_yrs,end_yrs,avg_period,var_list,
                                                                   pp_type=pp_type,diag_file=diag_file,omit_mem1=omit_mem1,debug=debug)
        if debug:
            print("Read data for 2xCO2 ctrl")
        # for basin in basin_list:
        moc_z, max_moc = calc_ens_mem_MOC(doub_ctrl_mem_list, grid, basin_list, pp_type=pp_type, zrho_method=zrho_method, 
                                          mask_output=True, verbose=verbose, debug=debug)
        myVars.__setitem__(f"doub_ctrl_{mem1_start}_{mem1_end}_psi", moc_z)
        print(f'doub_ctrl_{mem1_start}_{mem1_end}_psi done')
        if pp_type == 'ts-annual':
            myVars.__setitem__(f"doub_ctrl_{mem1_start}_{mem1_end}_max", max_moc) #{basin}_
            print(f'doub_ctrl_{mem1_start}_{mem1_end}_max done')

        if (co2_scen == 'doub' or co2_scen == 'const+doub' or co2_scen == 'all'):
            # differences compared to constant CO2 control #
            # for basin in basin_list:
            diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(const_ctrl_mem_list, doub_ctrl_mem_list, grid, basin_list, 
                                                             pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
            myVars.__setitem__(f"doub_ctrl_{mem1_start}_{mem1_end}_psi_diff", diff_moc)
            print(f'doub_ctrl_{mem1_start}_{mem1_end}_psi_diff done')
            if pp_type == 'ts-annual':
                myVars.__setitem__(f"doub_ctrl_{mem1_start}_{mem1_end}_max_diff", diff_max_moc)
                print(f'doub_ctrl_{mem1_start}_{mem1_end}_max_diff done')

        if (co2_scen == 'quad' or co2_scen == 'all'):
            ## 4xCO2 control ##
            quad_ctrl_exps = ["tune_ctrl_4xCO2_51-201",
                              "ens2_ctrl_4xCO2_51-201",
                              "ens3_ctrl_4xCO2_51-201"]
        
            quad_ctrl_mem_list, quad_ctrl = create_quad_ens_mean(quad_ctrl_exps,doub_ctrl_mem_list,doub_ctrl,start_yrs,end_yrs,
                                                                 avg_period,var_list,pp_type=pp_type,diag_file=diag_file,debug=debug)
            if debug:
                print("Read data for 4xCO2 ctrl")
            moc_z, max_moc = calc_ens_mem_MOC(quad_ctrl_mem_list, grid, basin_list, pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
            myVars.__setitem__(f"quad_ctrl_{mem1_start}_{mem1_end}_psi", moc_z)
            if pp_type == 'ts-annual':
                myVars.__setitem__(f"quad_ctrl_{mem1_start}_{mem1_end}_max", max_moc)
            print(f'quad_ctrl_{mem1_start}_{mem1_end}_psi done')
        
            # differences compared to constant CO2 control #
            diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(const_ctrl_mem_list, quad_ctrl_mem_list, grid, basin_list, 
                                                             pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
            myVars.__setitem__(f"quad_ctrl_{mem1_start}_{mem1_end}_psi_diff_const_ctrl", diff_moc)
            if pp_type == 'ts-annual':
                myVars.__setitem__(f"quad_ctrl_{mem1_start}_{mem1_end}_max_diff_const_ctrl", diff_max_moc)
            
            # differences compared to 2xCO2 control #
            diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(doub_ctrl_mem_list, quad_ctrl_mem_list, grid, basin_list, 
                                                             pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
            myVars.__setitem__(f"quad_ctrl_{mem1_start}_{mem1_end}_psi_diff_2xctrl", diff_moc)
            if pp_type == 'ts-annual':
                myVars.__setitem__(f"quad_ctrl_{mem1_start}_{mem1_end}_max_diff_2xctrl", diff_max_moc)

            print(f'quad_ctrl_{mem1_start}_{mem1_end}_psi_diff_const_ctrl, quad_ctrl_{mem1_start}_{mem1_end}_psi_diff_2xctrl done')
                
    
    ##### PERTURBATION RUNS #####
    
    for prof in profiles:
        for index, power_str in enumerate(power_inputs):
            if verbose:
                print(f"{prof} {power_str} experiments")
                    
            const_exp_name_list, doub_exp_name_list, quad_exp_name_list = create_exp_name_lists(power_str, prof, ramp_exp, lat_bound, num_ens_mem)

            ## ds names for ens means ##
            moc_z_root = f"{prof}_{power_var_suff[index]}_{mem1_start}_{mem1_end}_psi"
            const_moc_name = f"const_{moc_z_root}"
            doub_moc_name = f"doub_{moc_z_root}"
            quad_moc_name = f"quad_{moc_z_root}"

            max_moc_root = f"{prof}_{power_var_suff[index]}_{mem1_start}_{mem1_end}_max"
            const_max_moc_name = f"const_{max_moc_root}"
            doub_max_moc_name = f"doub_{max_moc_root}"
            quad_max_moc_name = f"quad_{max_moc_root}"


            const_ens_mem_list, const_ens_mean = create_const_doub_ens_mean(const_exp_name_list,start_yrs,end_yrs,
                                                                            avg_period,var_list,
                                                                            pp_type=pp_type,diag_file=diag_file,
                                                                            omit_mem1=omit_mem1,debug=debug)
            if debug:
                print(f"Read data for const {prof} {power_str} experiments")
            moc_z, max_moc = calc_ens_mem_MOC(const_ens_mem_list, grid, basin_list, pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
            myVars.__setitem__(const_moc_name, moc_z)
            if pp_type == 'ts-annual':
                myVars.__setitem__(const_max_moc_name, max_moc)
            print(f'{const_moc_name} done')
                
            if co2_scen != 'const':
                doub_ens_mem_list, doub_ens_mean = create_const_doub_ens_mean(doub_exp_name_list,start_yrs,end_yrs,
                                                                              avg_period,var_list,
                                                                              pp_type=pp_type,diag_file=diag_file,
                                                                              omit_mem1=omit_mem1,debug=debug)
                if debug:
                    print(f"Read data for 2xCO2 {prof} {power_str} experiments")
                moc_z, max_moc = calc_ens_mem_MOC(doub_ens_mem_list, grid, basin_list, pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                myVars.__setitem__(doub_moc_name, moc_z)
                if pp_type == 'ts-annual':
                    myVars.__setitem__(doub_max_moc_name, max_moc)
                print(f'{doub_moc_name} done')
            
                if (co2_scen == 'quad' or co2_scen == 'all'):
                    quad_ens_mem_list, quad_ens_mean = create_quad_ens_mean(quad_exp_name_list,doub_ens_mem_list,
                                                                            doub_ens_mean,start_yrs,end_yrs,
                                                                            avg_period,var_list,
                                                                            pp_type=pp_type,diag_file=diag_file,debug=debug)
                    if debug:
                        print(f"Read data for 4xCO2 {prof} {power_str} experiments")
                    moc_z, max_moc = calc_ens_mem_MOC(quad_ens_mem_list, grid, basin_list, pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                    myVars.__setitem__(quad_moc_name, moc_z)
                    if pp_type == 'ts-annual':
                        myVars.__setitem__(quad_max_moc_name, max_moc)
                    print(f'{quad_moc_name} done')

            ### COMPUTE DIFFERENCES ###

            ## Difference in psi ##
            diff_root = f"{prof}_{power_var_suff[index]}_{mem1_start}_{mem1_end}_psi_diff"

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

            ## Difference in location and value of max psi ##
            max_diff_root = f"{prof}_{power_var_suff[index]}_{mem1_start}_{mem1_end}_max_diff"

            # differences wrt 1860 control
            const_max_diff_name = f"const_{max_diff_root}"
            doub_const_ctrl_max_diff_name = f"doub_{max_diff_root}_const_ctrl"
            quad_const_ctrl_max_diff_name = f"quad_{max_diff_root}_const_ctrl"

            # differences wrt 1860 experiment with same diffusivity history
            doub_1860_max_diff_name = f"doub_{max_diff_root}_1860"
            quad_1860_max_diff_name = f"quad_{max_diff_root}_1860"

            # differences wrt control for particular CO2 scenario
            doub_2xctrl_max_diff_name = f"doub_{max_diff_root}_2xctrl"
            quad_4xctrl_max_diff_name = f"quad_{max_diff_root}_4xctrl"

            ## CONST EXPERIMENTS
            if (co2_scen == 'const' or co2_scen == 'const+doub' or co2_scen == 'all'):
                diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(const_ctrl_mem_list, const_ens_mem_list, grid, basin_list, 
                                                         pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                myVars.__setitem__(const_diff_name, diff_moc)
                print(f'{const_diff_name} done')
                if pp_type == 'ts-annual':
                    myVars.__setitem__(const_max_diff_name, diff_max_moc)

            ## 2xCO2 EXPERIMENTS
            if (co2_scen == 'doub' or co2_scen == 'const+doub' or co2_scen == 'all'):
                # differences wrt 1860 control
                diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(const_ctrl_mem_list, doub_ens_mem_list, grid, basin_list, 
                                                                 pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                myVars.__setitem__(doub_const_ctrl_diff_name, diff_moc)
                if pp_type == 'ts-annual':
                    myVars.__setitem__(doub_const_ctrl_max_diff_name, diff_max_moc)
                    
                # differences wrt 1860 experiment with same Kd history
                diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(const_ens_mem_list, doub_ens_mem_list, grid, basin_list, 
                                                         pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                myVars.__setitem__(doub_1860_diff_name, diff_moc)
                if pp_type == 'ts-annual':
                    myVars.__setitem__(doub_1860_max_diff_name, diff_max_moc)
                    
                # differences wrt control for particular CO2 scenario
                diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(doub_ctrl_mem_list, doub_ens_mem_list, grid, basin_list, 
                                                             pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                myVars.__setitem__(doub_2xctrl_diff_name, diff_moc)
                if pp_type == 'ts-annual':
                    myVars.__setitem__(doub_2xctrl_max_diff_name, diff_max_moc)

                print(f'{doub_const_ctrl_diff_name}, {doub_1860_diff_name}, and {doub_2xctrl_diff_name} done')
                
            ## 4xCO2 EXPERIMENTS
            if (co2_scen == 'quad' or co2_scen == 'all'):
                # differences wrt 1860 control
                diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(const_ctrl_mem_list, quad_ens_mem_list, grid, basin_list, 
                                                             pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                myVars.__setitem__(quad_const_ctrl_diff_name, diff_moc)
                if pp_type == 'ts-annual':
                    myVars.__setitem__(quad_const_ctrl_max_diff_name, diff_max_moc)
                    
                # differences wrt 1860 experiment with same Kd history
                diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(const_ens_mem_list, quad_ens_mem_list, grid, basin_list, 
                                                         pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                myVars.__setitem__(quad_1860_diff_name, diff_moc)
                if pp_type == 'ts-annual':
                    myVars.__setitem__(quad_1860_max_diff_name, diff_max_moc)
                
                # differences wrt control for particular CO2 scenario
                diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(quad_ctrl_mem_list, quad_ens_mem_list, grid, basin_list, 
                                                             pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                myVars.__setitem__(quad_4xctrl_diff_name, diff_moc)
                if pp_type == 'ts-annual':
                    myVars.__setitem__(quad_4xctrl_max_diff_name, diff_max_moc)

                print(f'{quad_const_ctrl_diff_name}, {quad_1860_diff_name}, {quad_4xctrl_diff_name} done')

                # additional difference calcs for 4xCO2 cases #
                
                # difference wrt 2xCO2 ctrl
                quad_2xctrl_diff_name = f"quad_{diff_root}_2xctrl"
                quad_2xctrl_max_diff_name = f"quad_{max_diff_root}_2xctrl"
                diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(doub_ctrl_mem_list, quad_ens_mem_list, grid, basin_list, 
                                                             pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                myVars.__setitem__(quad_2xctrl_diff_name, diff_moc)
                if pp_type == 'ts-annual':
                    myVars.__setitem__(quad_2xctrl_max_diff_name, diff_max_moc)
                
                # difference wrt 2xCO2 experiment with same diffusivity history
                quad_2xCO2_diff_name = f"quad_{diff_root}_2xCO2"
                quad_2xCO2_max_diff_name = f"quad_{max_diff_root}_2xCO2"
                diff_moc, diff_max_moc = calc_ens_mem_MOC_diff(doub_ens_mem_list, quad_ens_mem_list, grid, basin_list, 
                                                             pp_type=pp_type, zrho_method=zrho_method, mask_output=True, verbose=verbose, debug=debug)
                myVars.__setitem__(quad_2xCO2_diff_name, diff_moc)
                if pp_type == 'ts-annual':
                    myVars.__setitem__(quad_2xCO2_max_diff_name, diff_max_moc)
                    
                print(f'{quad_2xctrl_diff_name}, {quad_2xCO2_diff_name} done')


# # MOC plotting functions

# ## Mean MOC plotting

def plot_mean_MOC(
    moc_ds, static_ds, basin,
    start_yr, end_yr,
    cb_max=None, icon=None,
    savefig=False, fig_dir=None, prefix=None, verbose=False,
    # === grid-wrapper hooks ===
    ax=None,                     # draw into this axes if provided
    add_colorbar=True,           # wrapper will set False for shared bars
    return_cb_params=False,      # wrapper will set True to build shared bar
    cb_label="MOC (Sv)",
    panel_title=None             # short title when used inside a grid
):
    # ---- data prep ----
    moc_ds = moc_ds.mean("time")

    if basin == "atl-arc":
        basin_mask = selecting_basins(static_ds, basin="atl-arc-south", verbose=False)
    else:
        basin_mask = selecting_basins(static_ds, basin=basin, verbose=False)

    bathy_dat = static_ds['deptho'].where(basin_mask)

    zonal_pct_bathy = xr.apply_ufunc(
        lambda x: np.nanpercentile(x, 75),
        bathy_dat,
        input_core_dims=[["xh"]],
        vectorize=True,
        output_dtypes=[bathy_dat.dtype],
        dask="parallelized"
    )

    correct_lat = zonal_mean(static_ds['geolat'], static_ds)
    zonal_pct_bathy = (zonal_pct_bathy
                       .rename({'yh': 'lat'})
                       .assign_coords({'lat': correct_lat.values}))
    zonal_pct_bathy.values[0] = 0
    zonal_pct_bathy.values = gaussian_filter1d(zonal_pct_bathy.values, sigma=0.5)

    moc_dat = moc_ds[basin]

    # ---- interpolate to fine grid ----
    max_depth = 5500
    lat_res, z_res = 1000, 200
    fine_lat   = np.linspace(moc_dat.lat.min(), moc_dat.lat.max(), lat_res)
    fine_depth = np.linspace(moc_dat.z_i.min(),  moc_dat.z_i.max(),  z_res)
    moc_dat    = moc_dat.interp(lat=fine_lat, z_i=fine_depth)

    # ---- ranges/ticks ----
    per0p5  = float(np.nanpercentile(moc_dat.values, 0.5))
    per99p5 = float(np.nanpercentile(moc_dat.values, 99.5))
    min_val = float(np.nanmin(moc_dat.values))
    max_val = float(np.nanmax(moc_dat.values))

    n_bins = 20
    # if basin == "atl-arc":
    #     plot_min, plot_max, levels_inc = -35, 35, 3.5
    #     # tick_arr = np.linspace(plot_min, plot_max, int((plot_max-plot_min)/5) + 1)
    # elif basin == "indopac":
    #     plot_min, plot_max, levels_inc = -30, 30, 3
    #     # tick_arr = np.linspace(plot_min, plot_max, int((plot_max-plot_min)/5) + 1)
    # else:  # "global"
    #     plot_min, plot_max, levels_inc = -40, 40, 4
    #     # tick_arr = np.linspace(plot_min, plot_max, int((plot_max-plot_min)/10) + 1)
    
    plot_min, plot_max, levels_inc = -25, 25, 2.5

    if   (min_val < plot_min) and (max_val > plot_max): extend = 'both'
    elif (min_val < plot_min):                           extend = 'min'
    elif (max_val > plot_max):                           extend = 'max'
    else:                                                extend = 'neither'

    # ===== axes management =====
    created_fig = None
    if ax is None:
        # created_fig, ax = plt.subplots(figsize=(7.5, 4))
        created_fig, ax = plt.subplots(figsize=(5.5, 2.25))

    n_bins = 20  # << requested number of discrete bins
    boundaries = np.linspace(plot_min, plot_max, n_bins + 1)  # length 31
    tick_arr = np.linspace(plot_min, plot_max, int(n_bins/2) + 1)
    cmap      = cmocean.cm.balance
    disc_cmap = cmap
    disc_norm = mcolors.BoundaryNorm(boundaries, disc_cmap.N, clip=False)
    # print("boundaries:",boundaries)
    # print("tick_arr:",tick_arr)

    # Draw the discrete image
    moc_plot = moc_dat.plot(
        ax=ax, yincrease=False,
        cmap=disc_cmap, norm=disc_norm,
        add_labels=False, add_colorbar=False
    )

    # Contours — include zero explicitly
    contour_levels = tick_arr #np.r_[np.arange(plot_min, 0, levels_inc), 0, np.arange(levels_inc, plot_max + 1e-12, levels_inc)]
    overlay_plot = moc_dat.plot.contour(
        ax=ax, yincrease=False,
        levels=contour_levels, colors='k', linewidths=1
    )
    ax.clabel(overlay_plot, inline=True, fontsize=8, colors='k',levels=contour_levels)

    # cosmetics
    ax.spines['bottom'].set_zorder(30)
    for t in ax.get_xticklines(): t.set_zorder(30)
    for lab in ax.get_xticklabels(): lab.set_zorder(30)
    ax.set_facecolor('grey')
    ax.set_ylim([max_depth, 0])

    if panel_title is not None:
        ax.set_title(panel_title)

    ax.set_xlabel(None)
    ax.set_ylabel('Depth (m)')

    # topo overlay
    zonal_pct_bathy_i = zonal_pct_bathy.interp(lat=fine_lat)
    ax.fill_between(moc_dat['lat'].values, max_depth, zonal_pct_bathy_i, color='grey', zorder=20)

    # x-axis
    xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin, MOC_override=True)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(xlabels_major)
    ax.set_xticks(xticks_minor, minor=True)
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=6)

    if verbose:
        print(f"Min and max strength: {np.nanmin(moc_dat):.2f} and {np.nanmax(moc_dat):.2f}")

    # ---- colorbar with TICKS AT BIN EDGES ----
    cb = None
    if add_colorbar:
        cb = plt.colorbar(
            moc_plot, ax=ax, orientation="vertical", pad=0.02,
            boundaries=boundaries, norm=disc_norm,
            spacing='uniform', extend=extend
        )
        cb.set_ticks(tick_arr)
        cb.set_ticklabels([f"{v:.0f}" for v in tick_arr])
        cb.set_label(cb_label)
        for t in cb.ax.get_yticklabels():
            t.set_horizontalalignment('center')
            t.set_x(2.2 if plot_max >= 10 else 2.0)

    # ---- shared-colorbar spec for wrapper (edge ticks) ----
    cb_params = None
    if return_cb_params:
        cb_params = dict(
            mappable=moc_plot,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,
            extend=extend,
            spacing='uniform',
            ticks=tick_arr,
            ticklabels=[f"{v:.0f}" for v in tick_arr],
            label=cb_label
        )

    # ---- icon ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.065)
        ab = AnnotationBbox(imagebox, (0.95, 1.15), xycoords="axes fraction",
                            frameon=False, zorder=50, box_alignment=(0.5, 1.0))
        ax.add_artist(ab)

    # ---- save if we created the fig ----
    if savefig and created_fig is not None:
        if fig_dir is None: raise ValueError("Must specify 'fig_dir'.")
        if prefix  is None: raise ValueError("Must specify 'prefix'.")
        os.makedirs(fig_dir, exist_ok=True)
        created_fig.savefig(
            fig_dir + f'{prefix}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
            dpi=600, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, moc_plot, cb, cb_params


def plot_mean_MOC_zspace(start_year,end_year,co2_scen,grid,cb_max=None,
                         savefig=False,
                         plot_dir=None,
                         basin_list=["global","atl-arc","indopac"],
                         basin_strings=["Global","Atlantic","Indo-Pacific"],
                         profiles = ['surf','therm','mid','bot'],
                         prof_strings = ["Surf","Therm","Mid","Bot"],
                         power_var_suff = ['0p1TW', '0p2TW', '0p5TW'],
                         power_strings = ['0.1 TW', '0.2 TW', '0.5 TW'],
                         verbose=False):

    ctrl_ds_name = f"{co2_scen}_ctrl_{start_year}_{end_year}_psi"

    if co2_scen == "const":
        co2_str = "Const CO2"
    elif co2_scen == "doub":
        co2_str = "1pct2xCO2"
    elif co2_scen == "quad":
        co2_str = "1pct4xCO2"

    fig_pref = ""

    # control case
    for b_idx, basin in enumerate(basin_list):
        
        # moc_ctrl = my_calcmoc(myVars[ctrl_ds_name], dsgrid=grid, basin=basin, mask_output=True, output_true_lat=True, verbose=False)
        # moc_ctrl_mean = moc_ctrl.mean('time').load()

        plot_title = f"{basin_strings[b_idx]} MOC: Year {start_year} to {end_year}\n{co2_str} Control"

        if savefig:
            fig_pref = f"{basin}_MOC_{co2_scen}_ctrl"

        # print(f"plotting {ctrl_ds_name}")
        psi_dat = myVars[ctrl_ds_name]#.mean("time")

        plot_mean_MOC(psi_dat, grid, basin, plot_title,start_year,end_year,savefig=savefig,fig_dir=plot_dir,prefix=fig_pref,verbose=verbose)

    # perturbation experiments
    for pr_idx, prof in enumerate(profiles):
        for pow_idx, power in enumerate(power_var_suff):
            ens_ds_name = f"{co2_scen}_{prof}_{power}_{start_year}_{end_year}_psi"
            
            for b_idx, basin in enumerate(basin_list):
                
                # moc_exp = my_calcmoc(myVars[ens_ds_name], dsgrid=grid, basin=basin, mask_output=True, output_true_lat=True, verbose=False)
                # moc_exp_mean = moc_exp.mean('time').load()

                plot_title = f"{basin_strings[b_idx]} MOC: Year {start_year} to {end_year}\n{co2_str} {prof_strings[pr_idx]} {power_strings[pow_idx]}"

                if savefig:
                    fig_pref = f"{basin}_MOC_{co2_scen}_{prof}_{power}"

                # print(f"plotting {ens_ds_name}")
                psi_dat = myVars[ens_ds_name]#.mean("time")

                plot_mean_MOC(psi_dat, grid, basin, plot_title,start_year,end_year,icon=prof,savefig=savefig,fig_dir=plot_dir,prefix=fig_pref,verbose=verbose)


# ## MOC anomaly plotting

def plot_diff_MOC(
    moc_diff_ds, static_ds, basin,
    start_yr, end_yr, cb_max=None, icon=None,
    savefig=False, fig_dir=None, prefix=None, verbose=False,
    # === grid-wrapper hooks ===
    ax=None,                     # draw into this axes if provided
    add_colorbar=True,           # wrapper will set False for shared bars
    return_cb_params=False,      # wrapper will set True to build shared bar
    cb_label="MOC Anomaly (Sv)",
    panel_title=None             # short title when used inside a grid
):
    
    # ---- data prep ----
    moc_diff_ds = moc_diff_ds.mean("time")
    
    # ----- bathymetry overlay -----
    if basin == "atl-arc":
        basin_mask = selecting_basins(static_ds, basin="atl-arc-south", verbose=False)
    else:
        basin_mask = selecting_basins(static_ds, basin=basin, verbose=False)

    bathy_dat = static_ds['deptho'].where(basin_mask)

    zonal_pct_bathy = xr.apply_ufunc(
        lambda x: np.nanpercentile(x, 75),
        bathy_dat,
        input_core_dims=[["xh"]],
        vectorize=True,
        output_dtypes=[bathy_dat.dtype],
        dask="parallelized"
    )

    correct_lat = zonal_mean(static_ds['geolat'], static_ds)
    zonal_pct_bathy = (zonal_pct_bathy
                       .rename({'yh': 'lat'})
                       .assign_coords({'lat': correct_lat.values}))
    zonal_pct_bathy.values[0] = 0
    zonal_pct_bathy.values = gaussian_filter1d(zonal_pct_bathy.values, sigma=0.5)

    moc_diff_dat = moc_diff_ds[basin]

    # ----- interpolate to fine grid -----
    max_depth = 5500
    lat_res, z_res = 1000, 200
    fine_lat   = np.linspace(moc_diff_dat.lat.min(),  moc_diff_dat.lat.max(),  lat_res)
    fine_depth = np.linspace(moc_diff_dat.z_i.min(),  moc_diff_dat.z_i.max(),  z_res)
    moc_diff_dat = moc_diff_dat.interp(lat=fine_lat, z_i=fine_depth)

    # ----- bounds/percentiles -----
    per0p5  = float(np.nanpercentile(moc_diff_dat.values, 0.5))
    per99p5 = float(np.nanpercentile(moc_diff_dat.values, 99.5))
    min_val = float(np.nanmin(moc_diff_dat.values))
    max_val = float(np.nanmax(moc_diff_dat.values))

    # choose symmetric bounds
    if cb_max is not None:
        max_mag = float(cb_max)
    else:
        max_mag = max(abs(per0p5), abs(per99p5))

    if cb_max is not None:
        plot_min, plot_max = -max_mag, max_mag
        levels_inc = max_mag / 5
        if max_mag <= 1:
            tick_arr = np.round(np.linspace(plot_min, plot_max, int((plot_max-plot_min)/0.2)+1)/0.1)*0.1
        elif max_mag <= 2.5:
            tick_arr = np.round(np.linspace(plot_min, plot_max, int((plot_max-plot_min)/0.5)+1)/0.1)*0.1
        elif max_mag <= 5:
            tick_arr = np.linspace(plot_min, plot_max, int(plot_max-plot_min)+1)
        elif max_mag <= 10:
            tick_arr = np.linspace(plot_min, plot_max, int((plot_max-plot_min)/2)+1)
        elif max_mag <= 12:
            tick_arr = np.linspace(plot_min, plot_max, int((plot_max-plot_min)/3)+1)
        elif max_mag <= 20:
            tick_arr = np.linspace(plot_min, plot_max, int((plot_max-plot_min)/5)+1)
        else:
            tick_arr = np.linspace(plot_min, plot_max, int((plot_max-plot_min)/10)+1)
            print("Warning: plot bounds more than +/- 20")
    else:
        # buckets similar to your original
        if   max_mag <= 2:   plot_min, plot_max = -2,   2
        elif max_mag <= 3:   plot_min, plot_max = -3,   3
        elif max_mag <= 6:   plot_min, plot_max = -6,   6
        elif max_mag <= 10:  plot_min, plot_max = -10, 10
        elif max_mag <= 12:  plot_min, plot_max = -12, 12
        elif max_mag <= 20:  plot_min, plot_max = -20, 20
        elif max_mag <= 25:  plot_min, plot_max = -25, 25
        else:
            print("Warning: plot bounds more than +/- 25")
            plot_min, plot_max = -30, 30
        # ticks by bucket
        span = plot_max - plot_min
        if   span <= 4:   tick_arr = np.round(np.linspace(plot_min, plot_max, int(span/0.5)+1)/0.1)*0.1
        elif span <= 6:   tick_arr = np.round(np.linspace(plot_min, plot_max, int(span/0.5)+1)/0.1)*0.1
        elif span <= 12:  tick_arr = np.linspace(plot_min, plot_max, int(span)+1)
        elif span <= 20:  tick_arr = np.linspace(plot_min, plot_max, int(span/2)+1)
        elif span <= 24:  tick_arr = np.linspace(plot_min, plot_max, int(span/3)+1)
        elif span <= 40:  tick_arr = np.linspace(plot_min, plot_max, int(span/5)+1)
        else:             tick_arr = np.linspace(plot_min, plot_max, int(span/10)+1)
        levels_inc = plot_max/5

    # colorbar arrows
    if   (min_val < plot_min) and (max_val > plot_max): extend = 'both'
    elif (min_val < plot_min):                           extend = 'min'
    elif (max_val > plot_max):                           extend = 'max'
    else:                                                extend = 'neither'

    # ===== axes management (wrapper-compatible) =====
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7.5, 4))   # consistent with other section plots

    # # === DISCRETIZE THE COLORMAP ===
    # # Pick a bin width you like. Options:
    # # - tie bins to your contour spacing (levels_inc),
    # # - or choose a fixed step (e.g., 1 Sv or 2 Sv).
    # # Using half your contour spacing usually looks nice:
    # step_sv = max(0.1, levels_inc / 2.0)
    
    # # Build boundaries and BoundaryNorm
    # n_bins = max(1, int(np.ceil((plot_max - plot_min) / step_sv)))
    # boundaries = np.linspace(plot_min, plot_max, n_bins + 1)
    
    # cmap = cmocean.cm.balance
    # disc_cmap = cmap
    # disc_norm = mcolors.BoundaryNorm(boundaries, disc_cmap.N, clip=False)
    
    # # ===== draw image + contours =====
    # mappable = moc_diff_dat.plot(
    #     ax=ax, yincrease=False,
    #     cmap=disc_cmap, norm=disc_norm,        # <- use discrete norm
    #     add_labels=False, add_colorbar=False
    # )
    
    # moc_diff_dat.plot.contour(
    #     ax=ax, yincrease=False,
    #     levels=np.concatenate([np.arange(plot_min, 0.1, levels_inc),
    #                            np.arange(levels_inc, plot_max, levels_inc)]),
    #     colors='k', linewidths=1
    # )

    n_bins = 20  # << requested number of discrete bins
    boundaries = np.linspace(plot_min, plot_max, n_bins + 1)  # length 31
    tick_arr = np.linspace(plot_min, plot_max, int(n_bins/2) + 1)
    cmap      = cmocean.cm.balance
    disc_cmap = cmap
    disc_norm = mcolors.BoundaryNorm(boundaries, disc_cmap.N, clip=False)
    # print("boundaries:",boundaries)
    # print("tick_arr:",tick_arr)

    # Draw the discrete image
    mappable = moc_diff_dat.plot(
        ax=ax, yincrease=False,
        cmap=disc_cmap, norm=disc_norm,
        add_labels=False, add_colorbar=False
    )

    # Contours — include zero explicitly
    contour_levels = tick_arr #np.r_[np.arange(plot_min, 0, levels_inc), 0, np.arange(levels_inc, plot_max + 1e-12, levels_inc)]
    overlay_plot = moc_diff_dat.plot.contour(
        ax=ax, yincrease=False,
        levels=contour_levels, colors='k', linewidths=1
    )
    ax.clabel(overlay_plot, inline=True, fontsize=8, colors='k', fmt='%.1f', levels=contour_levels)

    # cosmetics
    ax.spines['bottom'].set_zorder(30)
    for t in ax.get_xticklines(): t.set_zorder(30)
    for lab in ax.get_xticklabels(): lab.set_zorder(30)
    ax.set_facecolor('grey')
    ax.set_ylim([max_depth, 0])

    if panel_title is not None:
        ax.set_title(panel_title)
        
    ax.set_xlabel(None)
    ax.set_ylabel('Depth (m)')

    # bathymetry fill
    zonal_pct_bathy_i = zonal_pct_bathy.interp(lat=fine_lat)
    ax.fill_between(moc_diff_dat['lat'].values, max_depth, zonal_pct_bathy_i, color='grey', zorder=20)

    # # optional discontinuity marker
    # if basin == "atl-arc":
    #     ax.axvspan(-38, -33, color='grey', zorder=10)

    xmin, xmax, xticks_major, xlabels_major, xticks_minor = get_basin_xlims(basin, MOC_override=True)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels(xlabels_major)
    ax.set_xticks(xticks_minor, minor=True)
    # Optional: make minor ticks smaller and unlabeled
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=6)
        
    if verbose:
        print(f"Min and max strength: {np.nanmin(moc_diff_dat):.2f} and {np.nanmax(moc_diff_dat):.2f}")

    # ===== optional per-panel colorbar =====
    cb = None
    if add_colorbar:
        cb = plt.colorbar(
            mappable, ax=ax, orientation="vertical", pad=0.02, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing='proportional'  # <- match plot
        )
        tick_labels = [f"{x:.1f}" if (max_mag <= 2.5 or max_mag == 8) else f"{x:.0f}" for x in tick_arr]
        cb.set_ticks(tick_arr)
        cb.ax.set_yticklabels(tick_labels)
        cb.set_label(cb_label)
        for t in cb.ax.get_yticklabels():
            t.set_horizontalalignment('center')
            t.set_x(2.0 if plot_max < 10 else 2.2)
    
    # ===== shared-colorbar spec for wrapper =====
    cb_params = None
    if return_cb_params:
        cb_params = dict(
            mappable=mappable,
            cmap=disc_cmap,
            norm=disc_norm,
            boundaries=boundaries,              # <- same boundaries
            extend=extend,
            spacing='proportional',
            ticks=tick_arr,
            ticklabels=[f"{x:.1f}" if (max_mag <= 2.5 or max_mag == 8) else f"{x:.0f}" for x in tick_arr],
            label=cb_label
        )

    # ===== icon (consistent placement) =====
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.065)
        ab = AnnotationBbox(imagebox, (0.95, 1.15), xycoords="axes fraction",
                            frameon=False, zorder=50, box_alignment=(0.5, 1.0))
        ax.add_artist(ab)

    # ===== save only if we created the figure =====
    if savefig and created_fig is not None:
        if fig_dir is None: raise ValueError("Must specify 'fig_dir'.")
        if prefix  is None: raise ValueError("Must specify 'prefix'.")
        os.makedirs(fig_dir, exist_ok=True)
        created_fig.savefig(
            fig_dir + f'{prefix}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
            dpi=600, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, mappable, cb, cb_params


def plot_psi_diff_zspace(start_year,end_year,diff_type,grid,cb_max=None,
                         savefig=False,
                         plot_dir=None,
                         basin_list=["global","atl-arc","indopac"],
                         basin_strings=["Global","Atlantic","Indo-Pacific"],
                         profiles = ['surf','therm','mid','bot'],
                         prof_strings = ["Surf","Therm","Mid","Bot"],
                         power_var_suff = ['0p1TW', '0p2TW', '0p5TW'],
                         power_strings = ['0.1 TW', '0.2 TW', '0.5 TW'],
                         verbose=False):

    for i, power_str in enumerate(power_strings):
        for j, prof in enumerate(profiles):
            
            if diff_type == 'const-1860ctrl':
                ds_root = f'const_{prof}_{power_var_suff[i]}_{start_year}_{end_year}_psi_diff'
            elif (diff_type == 'doub-1860exp' or diff_type == 'doub-2xctrl' or diff_type == 'doub-1860ctrl'):
                ds_root = f'doub_{prof}_{power_var_suff[i]}_{start_year}_{end_year}_psi_diff'
            elif (diff_type == 'quad-1860exp' or diff_type == 'quad-4xctrl' or diff_type == 'quad-1860ctrl'):
                ds_root = f'quad_{prof}_{power_var_suff[i]}_{start_year}_{end_year}_psi_diff'
            
            if diff_type == 'const-1860ctrl':
                title_str = f"Const {prof_strings[j]} {power_str}"
                diff_ds_name = ds_root
                fig_name = f"{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'doub-1860exp':
                title_str = f"1pct2xCO\u2082 — Const CO2: {prof_strings[j]} {power_str}"
                diff_ds_name = f'{ds_root}_1860'
                fig_name = f"2xCO2-const_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'doub-2xctrl':
                title_str = f"1pct2xCO\u2082 {prof_strings[j]} {power_str} — 1pct2xCO\u2082 Control"
                diff_ds_name = f'{ds_root}_2xctrl'
                fig_name = f"2xCO2-2xctrl_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'doub-1860ctrl':
                title_str = f"1pct2xCO\u2082 {prof_strings[j]} {power_str} — Const CO\u2082 Control"
                diff_ds_name = f'{ds_root}_const_ctrl'
                fig_name = f"2xCO2-const-ctrl_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'quad-1860exp':
                title_str = f"1pct4xCO\u2082 — Const CO2: {prof_strings[j]} {power_str}"
                diff_ds_name = f'{ds_root}_1860'
                fig_name = f"4xCO2-const_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'quad-4xctrl':
                title_str = f"1pct4xCO\u2082 {prof_strings[j]} {power_str} — 1pct4xCO\u2082 Control"
                diff_ds_name = f'{ds_root}_4xctrl'
                fig_name = f"4xCO2-4xctrl_{prof}_{power_var_suff[i]}"
                
            elif diff_type == 'quad-1860ctrl':
                title_str = f"1pct4xCO\u2082 {prof_strings[j]} {power_str} — Const CO\u2082 Control"
                diff_ds_name = f'{ds_root}_const_ctrl'
                fig_name = f"4xCO2-const-ctrl_{prof}_{power_var_suff[i]}"

            fig_pref = ""
            
            for b_idx, basin in enumerate(basin_list):

                plot_title = f"{basin_strings[b_idx]} MOC Difference: Year {start_year} to {end_year}\n{title_str}"

                if savefig:
                    fig_pref = f"{basin}_MOC_anom_{fig_name}"

                # if power_str == '0.5 TW':
                #     if (basin == 'atl-arc' and start_year == 1):
                #         cb_max = 4
                #     elif basin == 'atl-arc':
                #         cb_max = 15
                #     elif (basin == 'global' and start_year == 1):
                #         if prof == 'surf':
                #             cb_max = 12
                #         elif prof == 'therm':
                #             cb_max = 10
                #         else:
                #             cb_max = 5
                #     elif basin == 'global':
                #         cb_max = 15
                    
                psi_dat = myVars[diff_ds_name].mean("time")
                plot_diff_MOC(psi_dat,grid,basin,plot_title,start_year,end_year,cb_max=cb_max,icon=prof,
                              savefig=savefig,fig_dir=plot_dir,prefix=fig_pref,verbose=verbose)
                    
                # fig, ax = plt.subplots(figsize=(10,5))
                
                # if basin == "global":
                #     myVars[diff_ds_name][basin].plot(ax=ax, yincrease=False,vmin=-10,vmax=10,cmap=cmocean.cm.balance,cbar_kwargs={'ticks': np.arange(-10,12,2)})
                #     myVars[diff_ds_name][basin].plot.contour(ax=ax,yincrease=False, levels=np.concatenate([np.arange(-10,0,1),np.arange(1,11,1)]),
                #                           colors='k', linewidths=1)
                # elif basin == "atl-arc":
                #     myVars[diff_ds_name][basin].plot(ax=ax, yincrease=False,vmin=-5,vmax=5,cmap=cmocean.cm.balance,cbar_kwargs={'ticks': np.arange(-5,6,1)})
                #     myVars[diff_ds_name][basin].plot.contour(ax=ax,yincrease=False, levels=np.concatenate([np.arange(-5,0,0.5),np.arange(0.5,5.5,0.5)]),
                #                           colors='k', linewidths=1)
                # elif basin == "indopac":
                #     myVars[diff_ds_name][basin].plot(ax=ax, yincrease=False,vmin=-5,vmax=5,cmap=cmocean.cm.balance,cbar_kwargs={'ticks': np.arange(-5,6,1)})
                #     myVars[diff_ds_name][basin].plot.contour(ax=ax,yincrease=False, levels=np.concatenate([np.arange(-5,0,0.5),np.arange(0.5,5.5,0.5)]),
                #                           colors='k', linewidths=1)
                # ax.set_ylim([5500,0])
                # ax.set_title(f"{basin_strings[b_idx]} MOC Difference: Year {start_year} to {end_year}\n{title_str}")
                # ax.set_facecolor('grey')
    
                # print(f"Min and max MOC change: {np.nanmin(myVars[diff_ds_name][basin]):.2f} and {np.nanmax(myVars[diff_ds_name][basin]):.2f}")
                
                # plt.show()




