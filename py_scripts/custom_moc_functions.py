#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for computing overturning circulations.

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


# # Define functions

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
        elif basin == "global-no-marg":
            selected_codes = [1, 2, 3, 4, 5] # getting weird AMOC results with inclusion of Med and marginal seas
        elif basin == "atl-arc":
            selected_codes = [2, 4, 6, 7, 8, 9]
        elif basin == "atl-arc-no-marg":
            selected_codes = [2, 4] # getting weird AMOC results with inclusion of Med and marginal seas
        elif basin == "indopac":
            selected_codes = [3, 5, 10, 11]
        else:
            raise ValueError("Unknown basin")
    elif isinstance(basin, list):
        for b in basin:
            assert isinstance(b, int)
        selected_codes = basin
    else:
        raise ValueError("basin must be a string or list of int")

    maskbin = ds[mask].where(basincodes.isin(selected_codes))
    maskbasin = xr.where(maskbin == 1, True, False)

    # bathy, interface = names["bathy"], names["interface"]
    # I edited this
    bathy, interface, layer = names["bathy"], names["interface"], names["layer"]
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
        # maskmoc = xr.where(ds[interface] > max_depth, 0, 1)
        # I edited this
        maskmoc = xr.where(ds[layer] > max_depth, 0, 1)
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
        ucorr, vcorr = ds[umo], ds[vmo]

    if rotate:
        u_ctr, v_ctr = my_rotate_velocities_to_geo(ds, ucorr, vcorr, names)
    else:
        u_ctr, v_ctr = ucorr, vcorr

    # check vertical dimensions are in the dataarray
    layer = names["layer"]
    if layer not in v_ctr.dims:
        raise ValueError(f"{layer} not found in transport array")

    # use dimensions of v to know which lan/lat/mask to use
    if (names["y_corner"] in v_ctr.dims) and (names["x_center"] in v_ctr.dims):
        lon, lat, mask = names["lon_v"], names["lat_v"], names["mask_v"]
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
        verbose=verbose,
    )

    ds_v = xr.Dataset()
    ds_v["v"] = v_ctr.where(maskbasin)
    for var in [
        names["x_center"],
        names["y_center"],
        names["x_corner"],
        names["y_corner"],
        names["layer"],
        names["interface"],
    ]:
        ds_v[var] = ds[var]

    moc = my_compute_streamfunction(
        ds_v,
        names,
        transport="v",
        rho0=rho0,
        add_offset=add_offset,
        offset=offset,
    )

    if mask_output:
        moc = moc.where(maskmoc)

    if output_true_lat:
        moc = moc.assign_coords({names["y_corner"]: ds[lat].max(dim=names["x_center"])})
        moc = moc.rename({names["y_corner"]: "lat"})

    return moc


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

