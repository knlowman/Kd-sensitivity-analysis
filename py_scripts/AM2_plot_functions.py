#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for plotting atmospheric diagnostics.

# # Flux anomaly map

def plot_atmos_flux_diff(
    panel_title, pp_diff_da, flux_var, start_yr, end_yr,
    cb_max=None, hatching=False, icon=None,
    savefig=True, prefix=None, verbose=False,
    fast_preview=False,
    # NEW: wrapper compatibility
    ax=None,                 # if provided, draw into this axes (no new fig)
    add_colorbar=True,       # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,  # if True, return a spec for a figure-level colorbar
    cb_label="Flux Anomaly (W/m$^2$)"  # label reused by wrapper
):
    """
    Plot flux difference on a Robinson projection, compatible with plot_pp_grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
    diff_plot : QuadMesh from xarray
    diff_cb : colorbar instance or None
    cb_params : dict or None
        Spec for a shared colorbar when used with a wrapper.
    """

    diff_da = pp_diff_da[flux_var]

    # ---- Lon normalization ----
    diff_da = diff_da.assign_coords(
        lon=((diff_da.lon + 360) % 360)
    )

    # ---- Target grid for regridding ----
    lat_res = 4 * 90   
    lon_res = 4 * 144  

    target_lat = np.linspace(diff_da.lat.min(), diff_da.lat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res)

    ds_in = xr.Dataset({
        "lat": (["lat"], diff_da.lat.values),
        "lon": (["lon"], diff_da.lon.values),
    })

    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    regridder = xe.Regridder(
        ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False
    )
    diff_da_interp = regridder(diff_da)

    # ---- Diagnostics for color scale ----
    min_val = float(np.nanmin(diff_da.values))
    max_val = float(np.nanmax(diff_da.values))
    per0p5 = float(np.nanpercentile(diff_da.values, 0.5))
    per99p5 = float(np.nanpercentile(diff_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    # ---- Set symmetric bounds and discrete levels ----
    if cb_max is not None:
        max_mag = cb_max
    else:
        max_mag = max(abs(per0p5), abs(per99p5))

    extra_tick_digits = False
    if cb_max is not None:
        if (cb_max == 1 or cb_max == 1.5 or cb_max == 2 or cb_max == 2.5 or cb_max == 3 or cb_max == 4 or cb_max == 5 or cb_max == 6 
        or cb_max == 8 or cb_max == 10 or cb_max == 12 or cb_max == 15 or cb_max == 16 or cb_max == 20):
            chosen_n = 20 # 10
        elif cb_max == 4.5 or cb_max == 7.5:
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

    # ---- Figure / axes management (wrapper-compatible) ----
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={
                'projection': ccrs.Robinson(central_longitude=209.5),
                'facecolor': 'grey'
            }
        )

    # ---- Main plot ----
    diff_plot = diff_da_interp.plot(
        x='lon', y='lat',
        cmap=disc_cmap,
        norm=disc_norm,
        transform=ccrs.PlateCarree(),
        add_labels=False,
        add_colorbar=False,
        ax=ax
    )
    ax = diff_plot.axes  # Get the existing plot axis

    ax.coastlines(resolution='110m', color='black', linewidth=0.8)

    # # print mean value at bottom left
    # area_mean = atmos_horiz_mean(diff_da, pp_diff_da)
    # mean_val = area_mean.isel(time=0).values
    # mean_str = f"{mean_val:.2f}"
    
    # ax.text(
    #         0.2, -0.14, f"{mean_str} W/m$^2$",
    #         ha='left', va='bottom',
    #         fontsize=10,
    #         transform=ax.transAxes,
    #         zorder=10,
    #         color='white', fontweight='bold', alpha=1,
    #         backgroundcolor='grey'       # optional box to improve legibility
        
    #     )

    if hatching:
        hatch_mask = pp_diff_da[f"{flux_var}_hatch"]
    
        # Subsample in lat/lon
        step_y, step_x = 2, 2
        hatch_sub = hatch_mask.isel(
            lat=slice(0, None, step_y),
            lon=slice(0, None, step_x),
        ).squeeze()   # 🔑 drop any length-1 dims (e.g. time=1)
    
        lon_sub = diff_da['lon'].isel(lon=slice(0, None, step_x))
        lat_sub = diff_da['lat'].isel(lat=slice(0, None, step_y))
    
        # Build 2D coordinate grids
        lon2d, lat2d = np.meshgrid(lon_sub.values, lat_sub.values)
    
        # Make sure mask is 2D and boolean
        sel = (hatch_sub.values == 1)   # or >0.5
    
        #### STIPPLING ####
        ax.scatter(
            lon2d[sel],
            lat2d[sel],
            s=2,
            marker='x',
            color='k',
            alpha=0.4,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Colorbar (optional so wrapper can control layout)
    diff_cb = None
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, shrink=0.58, pad=0.04, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing='proportional'
        )
        tick_labels = []
        for val in tick_positions:
            if (np.abs(val) == 0.05 or np.abs(val) == 0.25):
                tick_labels.append(f"{val:.2f}")
            elif extra_tick_digits:
                tick_labels.append(f"{val:.2f}")
            else:
                tick_labels.append(f"{val:.1f}")

        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.set_yticklabels(tick_labels)
        diff_cb.ax.tick_params(labelsize=10)
        diff_cb.set_label("Flux Anomaly (W/m$^2$)", fontdict={'fontsize': 12})
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

    # ---- Icon (same behavior as before) ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(
            imagebox, (0.95, 1.00),
            xycoords="axes fraction", frameon=False
        )
        ax.add_artist(ab)

    if savefig and created_fig is not None:
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")

        image_dpi = 200 if fast_preview else 600
        created_fig.savefig(
            f'{prefix}_{flux_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
            dpi=image_dpi, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


# # Mean flux

def plot_atmos_flux_mean(
    panel_title, pp_mean_da, flux_var, start_yr, end_yr,
    cb_max=100, hatching=False, icon=None,
    savefig=True, prefix=None, verbose=False,
    fast_preview=False,
    # NEW: wrapper compatibility
    ax=None,                 # if provided, draw into this axes (no new fig)
    add_colorbar=True,       # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,  # if True, return a spec for a figure-level colorbar
    cb_label="Flux (W/m$^2$)"  # label reused by wrapper
):
    """
    Plot mean flux on a Robinson projection, compatible with plot_pp_grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
    mean_plot : QuadMesh from xarray
    mean_cb : colorbar instance or None
    cb_params : dict or None
        Spec for a shared colorbar when used with a wrapper.
    """

    mean_da = pp_mean_da[flux_var]

    # ---- Lon normalization ----
    mean_da = mean_da.assign_coords(
        lon=((mean_da.lon + 360) % 360)
    )

    # ---- Target grid for regridding ----
    lat_res = 4 * 90   
    lon_res = 4 * 144  

    target_lat = np.linspace(mean_da.lat.min(), mean_da.lat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res)

    ds_in = xr.Dataset({
        "lat": (["lat"], mean_da.lat.values),
        "lon": (["lon"], mean_da.lon.values),
    })

    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    regridder = xe.Regridder(
        ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False
    )
    mean_da_interp = regridder(mean_da)

    # ---- Diagnostics for color scale ----
    min_val = float(np.nanmin(mean_da.values))
    max_val = float(np.nanmax(mean_da.values))
    per0p5 = float(np.nanpercentile(mean_da.values, 0.5))
    per99p5 = float(np.nanpercentile(mean_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    extra_tick_digits = False
    
    plot_min = -cb_max
    plot_max = cb_max
    if cb_max >= 100:
        tick_spacing = 25
    else:
        tick_spacing = 20
    num = int((plot_max-plot_min)/tick_spacing) + 1
    tick_arr = np.linspace(plot_min,plot_max,num=num)
    num_colors = 4 * (num - 1)
    
    cmap = cmocean.cm.balance  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors
    
    # create the new map
    disc_cmap = mcolors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    norm_bounds = np.linspace(plot_min, plot_max, num_colors + 1)
    disc_norm = mcolors.BoundaryNorm(norm_bounds, cmap.N)

    # ---- Figure / axes management (wrapper-compatible) ----
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={
                'projection': ccrs.Robinson(central_longitude=209.5),
                'facecolor': 'grey'
            }
        )

    # ---- Main plot ----
    mean_plot = mean_da_interp.plot(
        x='lon', y='lat',
        cmap=disc_cmap,
        norm=disc_norm,
        transform=ccrs.PlateCarree(),
        add_labels=False,
        add_colorbar=False,
        ax=ax
    )
    ax = mean_plot.axes  # Get the existing plot axis

    ax.coastlines(resolution='110m', color='black', linewidth=0.8)

    # # print mean value at bottom left
    # area_mean = atmos_horiz_mean(mean_da, pp_mean_da)
    # mean_val = area_mean.isel(time=0).values
    # mean_str = f"{mean_val:.2f}"
    
    # ax.text(
    #         0.1, 0, f"{mean_str} $\\degree$C",
    #         ha='left', va='bottom',
    #         fontsize=12,
    #         transform=ax.transAxes,
    #         zorder=10,
    #         color='black', fontweight='bold', alpha=1
    #     )

    if hatching:
        hatch_mask = pp_mean_da[f"{flux_var}_hatch"]
    
        # Subsample in lat/lon
        step_y, step_x = 2, 2
        hatch_sub = hatch_mask.isel(
            lat=slice(0, None, step_y),
            lon=slice(0, None, step_x),
        ).squeeze()   # 🔑 drop any length-1 dims (e.g. time=1)
    
        lon_sub = mean_da['lon'].isel(lon=slice(0, None, step_x))
        lat_sub = mean_da['lat'].isel(lat=slice(0, None, step_y))
    
        # Build 2D coordinate grids
        lon2d, lat2d = np.meshgrid(lon_sub.values, lat_sub.values)
    
        # Make sure mask is 2D and boolean
        sel = (hatch_sub.values == 1)   # or >0.5
    
        #### STIPPLING ####
        ax.scatter(
            lon2d[sel],
            lat2d[sel],
            s=2,
            marker='x',
            color='k',
            alpha=0.4,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    if min_val < plot_min and max_val > plot_max:
        extend = 'both'
    elif min_val < plot_min:
        extend = 'min'
    elif max_val > plot_max:
        extend = 'max'
    else:
        extend = 'neither'

    # Colorbar (optional so wrapper can control layout)
    mean_cb = None
    if add_colorbar:
        mean_cb = plt.colorbar(mean_plot, ax=ax, ticks=tick_arr, shrink=0.58, pad=0.04, norm=disc_norm, extend=extend)
        mean_cb.ax.tick_params(labelsize=10)
        mean_cb.set_label("Flux (W/m$^2$)", fontdict={'fontsize': 12})
        plt.setp(mean_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.2)

    # --- NEW: package colorbar spec for the wrapper ---
    cb_params = None
    if return_cb_params:

        cb_params = dict(
            mappable=mean_plot,        # carries cmap+norm
            cmap=disc_cmap,
            norm=disc_norm,
            extend=extend,
            ticks=tick_arr,
            label=cb_label
        )
        
    # ---- Icon (same behavior as before) ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(
            imagebox, (0.95, 1.00),
            xycoords="axes fraction", frameon=False
        )
        ax.add_artist(ab)

    if savefig and created_fig is not None:
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")

        image_dpi = 200 if fast_preview else 600
        created_fig.savefig(
            f'{prefix}_{flux_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
            dpi=image_dpi, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, mean_plot, mean_cb, cb_params


# # Temperature anomaly map

def plot_atmos_temp_diff(
    panel_title, pp_diff_da, temp_var, start_yr, end_yr,
    cb_max=None, hatching=False, icon=None,
    savefig=True, prefix=None, verbose=False,
    fast_preview=False,
    # NEW: wrapper compatibility
    ax=None,                 # if provided, draw into this axes (no new fig)
    add_colorbar=True,       # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,  # if True, return a spec for a figure-level colorbar
    cb_label="Temperature Anomaly ($\\degree$C)"  # label reused by wrapper
):
    """
    Plot temperature difference on a Robinson projection, compatible with plot_pp_grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
    diff_plot : QuadMesh from xarray
    diff_cb : colorbar instance or None
    cb_params : dict or None
        Spec for a shared colorbar when used with a wrapper.
    """

    diff_da = pp_diff_da[temp_var]

    # ---- Lon normalization ----
    diff_da = diff_da.assign_coords(
        lon=((diff_da.lon + 360) % 360)
    )

    # ---- Target grid for regridding ----
    lat_res = 4 * 90   
    lon_res = 4 * 144  

    target_lat = np.linspace(diff_da.lat.min(), diff_da.lat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res)

    ds_in = xr.Dataset({
        "lat": (["lat"], diff_da.lat.values),
        "lon": (["lon"], diff_da.lon.values),
    })

    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    regridder = xe.Regridder(
        ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False
    )
    diff_da_interp = regridder(diff_da)

    # ---- Diagnostics for color scale ----
    min_val = float(np.nanmin(diff_da.values))
    max_val = float(np.nanmax(diff_da.values))
    per0p5 = float(np.nanpercentile(diff_da.values, 0.5))
    per99p5 = float(np.nanpercentile(diff_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    # ---- Set symmetric bounds and discrete levels ----
    if cb_max is not None:
        max_mag = cb_max
    else:
        max_mag = max(abs(per0p5), abs(per99p5))

    extra_tick_digits = False
    if cb_max is not None:
        if cb_max == 1 or cb_max == 1.5 or cb_max == 2 or cb_max == 2.5 or cb_max == 3 or cb_max == 4 or cb_max == 5 or cb_max == 6 or cb_max == 8:
            chosen_n = 20 # 10
        elif cb_max == 4.5 or cb_max == 7.5:
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

    # ---- Figure / axes management (wrapper-compatible) ----
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={
                'projection': ccrs.Robinson(central_longitude=209.5),
                'facecolor': 'grey'
            }
        )

    # ---- Main plot ----
    diff_plot = diff_da_interp.plot(
        x='lon', y='lat',
        cmap=disc_cmap,
        norm=disc_norm,
        transform=ccrs.PlateCarree(),
        add_labels=False,
        add_colorbar=False,
        ax=ax
    )
    ax = diff_plot.axes  # Get the existing plot axis

    ax.coastlines(resolution='110m', color='black', linewidth=0.8)

    # # print mean value at bottom left
    # area_mean = atmos_horiz_mean(diff_da, pp_diff_da)
    # mean_val = area_mean.isel(time=0).values
    # mean_str = f"{mean_val:.2f}"
    
    # ax.text(
    #         0.1, 0, f"{mean_str} $\\degree$C",
    #         ha='left', va='bottom',
    #         fontsize=12,
    #         transform=ax.transAxes,
    #         zorder=10,
    #         color='black', fontweight='bold', alpha=1
    #     )

    if hatching:
        hatch_mask = pp_diff_da[f"{temp_var}_hatch"]
    
        # Subsample in lat/lon
        step_y, step_x = 2, 2
        hatch_sub = hatch_mask.isel(
            lat=slice(0, None, step_y),
            lon=slice(0, None, step_x),
        ).squeeze()   # 🔑 drop any length-1 dims (e.g. time=1)
    
        lon_sub = diff_da['lon'].isel(lon=slice(0, None, step_x))
        lat_sub = diff_da['lat'].isel(lat=slice(0, None, step_y))
    
        # Build 2D coordinate grids
        lon2d, lat2d = np.meshgrid(lon_sub.values, lat_sub.values)
    
        # Make sure mask is 2D and boolean
        sel = (hatch_sub.values == 1)   # or >0.5
    
        #### STIPPLING ####
        ax.scatter(
            lon2d[sel],
            lat2d[sel],
            s=2,
            marker='x',
            color='k',
            alpha=0.4,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Colorbar (optional so wrapper can control layout)
    diff_cb = None
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, shrink=0.58, pad=0.04, extend=extend,
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
        
    # ---- Icon (same behavior as before) ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(
            imagebox, (0.95, 1.00),
            xycoords="axes fraction", frameon=False
        )
        ax.add_artist(ab)

    # ---- Saving (same pattern as plot_pp_temp_diff) ----
    if savefig and created_fig is not None:
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")

        image_dpi = 200 if fast_preview else 600
        created_fig.savefig(
            f'{prefix}_{temp_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
            dpi=image_dpi, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


# # Cloud anomaly map

def plot_cloud_diff(
    panel_title, pp_diff_da, cloud_var, start_yr, end_yr,
    cb_max=None, hatching=False, icon=None,
    savefig=True, prefix=None, verbose=False,
    fast_preview=False,
    # NEW: wrapper compatibility
    ax=None,                 # if provided, draw into this axes (no new fig)
    add_colorbar=True,       # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,  # if True, return a spec for a figure-level colorbar
    cb_label="Cloud Amount Anomaly ($\%$)"  # label reused by wrapper
):
    """
    Plot cloud fraction difference on a Robinson projection, compatible with plot_pp_grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
    diff_plot : QuadMesh from xarray
    diff_cb : colorbar instance or None
    cb_params : dict or None
        Spec for a shared colorbar when used with a wrapper.
    """

    diff_da = pp_diff_da[cloud_var]

    # ---- Lon normalization ----
    diff_da = diff_da.assign_coords(
        lon=((diff_da.lon + 360) % 360)
    )

    # ---- Target grid for regridding ----
    lat_res = 4 * 90   
    lon_res = 4 * 144  

    target_lat = np.linspace(diff_da.lat.min(), diff_da.lat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res)

    ds_in = xr.Dataset({
        "lat": (["lat"], diff_da.lat.values),
        "lon": (["lon"], diff_da.lon.values),
    })

    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    regridder = xe.Regridder(
        ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False
    )
    diff_da_interp = regridder(diff_da)

    # ---- Diagnostics for color scale ----
    min_val = float(np.nanmin(diff_da.values))
    max_val = float(np.nanmax(diff_da.values))
    per0p5 = float(np.nanpercentile(diff_da.values, 0.5))
    per99p5 = float(np.nanpercentile(diff_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    # ---- Set symmetric bounds and discrete levels ----
    if cb_max is not None:
        max_mag = cb_max
    else:
        max_mag = max(abs(per0p5), abs(per99p5))

    extra_tick_digits = False
    if cb_max is not None:
        chosen_n = 20
        data_max = cb_max
        chosen_step = 2 * data_max / chosen_n
    else:
        chosen_n, chosen_step = get_cb_spacing(
            per0p5, per99p5, min_bnd=5.0, min_spacing=0.5, min_n=10, max_n=20, verbose=verbose
        )

    max_mag = 0.5 * chosen_n * chosen_step  # final ± range

    zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
        max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
    )

    # ---- Figure / axes management (wrapper-compatible) ----
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={
                'projection': ccrs.Robinson(central_longitude=209.5),
                'facecolor': 'grey'
            }
        )

    # ---- Main plot ----
    diff_plot = diff_da_interp.plot(
        x='lon', y='lat',
        cmap=disc_cmap,
        norm=disc_norm,
        transform=ccrs.PlateCarree(),
        add_labels=False,
        add_colorbar=False,
        ax=ax
    )
    ax = diff_plot.axes  # Get the existing plot axis

    ax.coastlines(resolution='110m', color='black', linewidth=0.8)

    # # print mean value at bottom left
    # area_mean = atmos_horiz_mean(diff_da, pp_diff_da)
    # mean_val = area_mean.isel(time=0).values
    # mean_str = f"{mean_val:.2f}"
    
    # ax.text(
    #         0.1, 0, f"{mean_str} $\\degree$C",
    #         ha='left', va='bottom',
    #         fontsize=12,
    #         transform=ax.transAxes,
    #         zorder=10,
    #         color='black', fontweight='bold', alpha=1
    #     )

    if hatching:
        hatch_mask = pp_diff_da[f"{cloud_var}_hatch"]
    
        # Subsample in lat/lon
        step_y, step_x = 2, 2
        hatch_sub = hatch_mask.isel(
            lat=slice(0, None, step_y),
            lon=slice(0, None, step_x),
        ).squeeze()   # 🔑 drop any length-1 dims (e.g. time=1)
    
        lon_sub = diff_da['lon'].isel(lon=slice(0, None, step_x))
        lat_sub = diff_da['lat'].isel(lat=slice(0, None, step_y))
    
        # Build 2D coordinate grids
        lon2d, lat2d = np.meshgrid(lon_sub.values, lat_sub.values)
    
        # Make sure mask is 2D and boolean
        sel = (hatch_sub.values == 1)   # or >0.5
    
        #### STIPPLING ####
        ax.scatter(
            lon2d[sel],
            lat2d[sel],
            s=2,
            marker='x',
            color='k',
            alpha=0.4,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Colorbar (optional so wrapper can control layout)
    diff_cb = None
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, shrink=0.58, pad=0.04, extend=extend,
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
        diff_cb.set_label("Cloud Amount Anomaly ($\%$)", fontdict={'fontsize': 12})
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
        
    # ---- Icon (same behavior as before) ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(
            imagebox, (0.95, 1.00),
            xycoords="axes fraction", frameon=False
        )
        ax.add_artist(ab)

    if savefig and created_fig is not None:
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")

        image_dpi = 200 if fast_preview else 600
        created_fig.savefig(
            f'{prefix}_{cloud_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
            dpi=image_dpi, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


# # Mean cloud amount

def plot_cloud_mean(
    panel_title, pp_mean_da, cloud_var, start_yr, end_yr,
    hatching=False, icon=None,
    savefig=True, prefix=None, verbose=False,
    fast_preview=False,
    # NEW: wrapper compatibility
    ax=None,                 # if provided, draw into this axes (no new fig)
    add_colorbar=True,       # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,  # if True, return a spec for a figure-level colorbar
    cb_label="Cloud Amount ($\%$)"  # label reused by wrapper
):
    """
    Plot cloud fraction mean on a Robinson projection, compatible with plot_pp_grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
    mean_plot : QuadMesh from xarray
    mean_cb : colorbar instance or None
    cb_params : dict or None
        Spec for a shared colorbar when used with a wrapper.
    """

    mean_da = pp_mean_da[cloud_var]

    # ---- Lon normalization ----
    mean_da = mean_da.assign_coords(
        lon=((mean_da.lon + 360) % 360)
    )

    # ---- Target grid for regridding ----
    lat_res = 4 * 90   
    lon_res = 4 * 144  

    target_lat = np.linspace(mean_da.lat.min(), mean_da.lat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res)

    ds_in = xr.Dataset({
        "lat": (["lat"], mean_da.lat.values),
        "lon": (["lon"], mean_da.lon.values),
    })

    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    regridder = xe.Regridder(
        ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False
    )
    mean_da_interp = regridder(mean_da)

    # ---- Diagnostics for color scale ----
    min_val = float(np.nanmin(mean_da.values))
    max_val = float(np.nanmax(mean_da.values))
    per0p5 = float(np.nanpercentile(mean_da.values, 0.5))
    per99p5 = float(np.nanpercentile(mean_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    extra_tick_digits = False
    
    plot_min = 0
    plot_max = 100
    num = int((plot_max-plot_min)/20) + 1
    tick_arr = np.linspace(plot_min,plot_max,num=num)
    num_colors = 4 * (num - 1)
    
    cmap = cmocean.cm.tempo  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors
    
    # create the new map
    disc_cmap = mcolors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    norm_bounds = np.linspace(plot_min, plot_max, num_colors + 1)
    disc_norm = mcolors.BoundaryNorm(norm_bounds, cmap.N)

    # ---- Figure / axes management (wrapper-compatible) ----
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={
                'projection': ccrs.Robinson(central_longitude=209.5),
                'facecolor': 'grey'
            }
        )

    # ---- Main plot ----
    mean_plot = mean_da_interp.plot(
        x='lon', y='lat',
        cmap=disc_cmap,
        norm=disc_norm,
        transform=ccrs.PlateCarree(),
        add_labels=False,
        add_colorbar=False,
        ax=ax
    )
    ax = mean_plot.axes  # Get the existing plot axis

    ax.coastlines(resolution='110m', color='black', linewidth=0.8)

    # # print mean value at bottom left
    # area_mean = atmos_horiz_mean(mean_da, pp_mean_da)
    # mean_val = area_mean.isel(time=0).values
    # mean_str = f"{mean_val:.2f}"
    
    # ax.text(
    #         0.1, 0, f"{mean_str} $\\degree$C",
    #         ha='left', va='bottom',
    #         fontsize=12,
    #         transform=ax.transAxes,
    #         zorder=10,
    #         color='black', fontweight='bold', alpha=1
    #     )

    if hatching:
        hatch_mask = pp_mean_da[f"{cloud_var}_hatch"]
    
        # Subsample in lat/lon
        step_y, step_x = 2, 2
        hatch_sub = hatch_mask.isel(
            lat=slice(0, None, step_y),
            lon=slice(0, None, step_x),
        ).squeeze()   # 🔑 drop any length-1 dims (e.g. time=1)
    
        lon_sub = mean_da['lon'].isel(lon=slice(0, None, step_x))
        lat_sub = mean_da['lat'].isel(lat=slice(0, None, step_y))
    
        # Build 2D coordinate grids
        lon2d, lat2d = np.meshgrid(lon_sub.values, lat_sub.values)
    
        # Make sure mask is 2D and boolean
        sel = (hatch_sub.values == 1)   # or >0.5
    
        #### STIPPLING ####
        ax.scatter(
            lon2d[sel],
            lat2d[sel],
            s=2,
            marker='x',
            color='k',
            alpha=0.4,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # for clouds, extend should always be none unless something weird happens
    if min_val < plot_min and max_val > plot_max:
        extend = 'both'
    elif min_val < plot_min:
        extend = 'min'
    elif max_val > plot_max:
        extend = 'max'
    else:
        extend = 'neither'

    # Colorbar (optional so wrapper can control layout)
    mean_cb = None
    if add_colorbar:
        mean_cb = plt.colorbar(mean_plot, ax=ax, ticks=tick_arr, shrink=0.58, pad=0.04, norm=disc_norm, extend=extend)
        mean_cb.ax.tick_params(labelsize=10)
        mean_cb.set_label("Cloud Amount ($\%$)", fontdict={'fontsize': 12})
        plt.setp(mean_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.2)

    # --- NEW: package colorbar spec for the wrapper ---
    cb_params = None
    if return_cb_params:

        cb_params = dict(
            mappable=mean_plot,        # carries cmap+norm
            cmap=disc_cmap,
            norm=disc_norm,
            extend=extend,
            ticks=tick_arr,
            label=cb_label
        )
        
    # ---- Icon (same behavior as before) ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(
            imagebox, (0.95, 1.00),
            xycoords="axes fraction", frameon=False
        )
        ax.add_artist(ab)

    if savefig and created_fig is not None:
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")

        image_dpi = 200 if fast_preview else 600
        created_fig.savefig(
            f'{prefix}_{cloud_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
            dpi=image_dpi, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, mean_plot, mean_cb, cb_params


# # Wind stress anomaly map

def plot_tau_diff(
    panel_title, pp_diff_da, tau_var, start_yr, end_yr,
    cb_max=None, hatching=False, icon=None,
    savefig=True, prefix=None, verbose=False,
    fast_preview=False,
    # NEW: wrapper compatibility
    ax=None,                 # if provided, draw into this axes (no new fig)
    add_colorbar=True,       # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,  # if True, return a spec for a figure-level colorbar
    cb_label="Wind Stress Anomaly (Pa)"  # label reused by wrapper
):
    """
    Plot flux difference on a Robinson projection, compatible with plot_pp_grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
    diff_plot : QuadMesh from xarray
    diff_cb : colorbar instance or None
    cb_params : dict or None
        Spec for a shared colorbar when used with a wrapper.
    """

    diff_da = pp_diff_da[tau_var]

    # ---- Lon normalization ----
    diff_da = diff_da.assign_coords(
        lon=((diff_da.lon + 360) % 360)
    )

    # ---- Target grid for regridding ----
    lat_res = 4 * 90
    lon_res = 4 * 144

    target_lat = np.linspace(diff_da.lat.min(), diff_da.lat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res)

    ds_in = xr.Dataset({
        "lat": (["lat"], diff_da.lat.values),
        "lon": (["lon"], diff_da.lon.values),
    })

    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    regridder = xe.Regridder(
        ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False
    )
    diff_da_interp = regridder(diff_da)

    # ---- Diagnostics for color scale ----
    min_val = float(np.nanmin(diff_da.values))
    max_val = float(np.nanmax(diff_da.values))
    per0p5 = float(np.nanpercentile(diff_da.values, 0.5))
    per99p5 = float(np.nanpercentile(diff_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    # ---- Set symmetric bounds and discrete levels ----
    if cb_max is not None:
        max_mag = cb_max
    else:
        max_mag = max(abs(per0p5), abs(per99p5))

    extra_tick_digits = False
    if cb_max is not None:
        # if (cb_max == 1 or cb_max == 1.5 or cb_max == 2 or cb_max == 2.5 or cb_max == 3 or cb_max == 4 or cb_max == 5 or cb_max == 6 
        # or cb_max == 8 or cb_max == 10 or cb_max == 12 or cb_max == 15 or cb_max == 16 or cb_max == 20):
        #     chosen_n = 20 # 10
        # elif cb_max == 4.5 or cb_max == 7.5:
        #     chosen_n = 12
        # elif cb_max == 2.222:
        #     extra_tick_digits = True
        #     cb_max = 2
        #     chosen_n = 12
        # elif cb_max == 3.333:
        #     cb_max = 3
        #     chosen_n = 12
        # elif cb_max == 4.444:
        #     extra_tick_digits = True
        #     cb_max = 4
        #     chosen_n = 12
        # else:
        #     raise ValueError("cb_max is not an acceptable value.")
        chosen_n = 20
        data_max = cb_max
        chosen_step = 2 * data_max / chosen_n
    else:
        chosen_n, chosen_step = get_cb_spacing(
            per0p5, per99p5, min_bnd=0.05, min_spacing=0.005, min_n=10, max_n=20, verbose=verbose
        )

    max_mag = 0.5 * chosen_n * chosen_step  # final ± range

    zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
        max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
    )

    # ---- Figure / axes management (wrapper-compatible) ----
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={
                'projection': ccrs.Robinson(central_longitude=209.5),
                'facecolor': 'grey'
            }
        )

    # ---- Main plot ----
    diff_plot = diff_da_interp.plot(
        x='lon', y='lat',
        cmap=disc_cmap,
        norm=disc_norm,
        transform=ccrs.PlateCarree(),
        add_labels=False,
        add_colorbar=False,
        ax=ax
    )
    ax = diff_plot.axes  # Get the existing plot axis

    ax.coastlines(resolution='110m', color='black', linewidth=0.8)

    # # print mean value at bottom left
    # area_mean = atmos_horiz_mean(diff_da, pp_diff_da)
    # mean_val = area_mean.isel(time=0).values
    # mean_str = f"{mean_val:.2f}"
    
    # ax.text(
    #         0.2, -0.14, f"{mean_str} W/m$^2$",
    #         ha='left', va='bottom',
    #         fontsize=10,
    #         transform=ax.transAxes,
    #         zorder=10,
    #         color='white', fontweight='bold', alpha=1,
    #         backgroundcolor='grey'       # optional box to improve legibility
        
    #     )

    if hatching:
        hatch_mask = pp_diff_da[f"{tau_var}_hatch"]
    
        # Subsample in lat/lon
        step_y, step_x = 2, 2
        hatch_sub = hatch_mask.isel(
            lat=slice(0, None, step_y),
            lon=slice(0, None, step_x),
        ).squeeze()   # 🔑 drop any length-1 dims (e.g. time=1)
    
        lon_sub = diff_da['lon'].isel(lon=slice(0, None, step_x))
        lat_sub = diff_da['lat'].isel(lat=slice(0, None, step_y))
    
        # Build 2D coordinate grids
        lon2d, lat2d = np.meshgrid(lon_sub.values, lat_sub.values)
    
        # Make sure mask is 2D and boolean
        sel = (hatch_sub.values == 1)   # or >0.5
    
        #### STIPPLING ####
        ax.scatter(
            lon2d[sel],
            lat2d[sel],
            s=2,
            marker='x',
            color='k',
            alpha=0.4,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    # Colorbar (optional so wrapper can control layout)
    diff_cb = None
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, shrink=0.58, pad=0.04, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing='proportional'
        )
        tick_labels = []
        for val in tick_positions:
            if (np.abs(val) == 0.05 or np.abs(val) == 0.25):
                tick_labels.append(f"{val:.1e}")
            elif extra_tick_digits:
                tick_labels.append(f"{val:.2e}")
            else:
                tick_labels.append(f"{val:.1e}")

        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.set_yticklabels(tick_labels)
        diff_cb.ax.tick_params(labelsize=10)
        diff_cb.set_label("Wind Stress Anomaly (Pa)", fontdict={'fontsize': 12})
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
                tick_labels.append(f"{val:.1e}")
            elif extra_tick_digits:
                tick_labels.append(f"{val:.2e}")
            else:
                tick_labels.append(f"{val:.1e}")

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

    # ---- Icon (same behavior as before) ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(
            imagebox, (0.95, 1.00),
            xycoords="axes fraction", frameon=False
        )
        ax.add_artist(ab)

    if savefig and created_fig is not None:
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")

        image_dpi = 200 if fast_preview else 600
        created_fig.savefig(
            f'{prefix}_{tau_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
            dpi=image_dpi, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


# # Mean wind stress

def plot_tau_mean(
    panel_title, pp_mean_da, tau_var, start_yr, end_yr,
    cb_max=0.3, hatching=False, icon=None,
    savefig=True, prefix=None, verbose=False,
    fast_preview=False,
    # NEW: wrapper compatibility
    ax=None,                 # if provided, draw into this axes (no new fig)
    add_colorbar=True,       # allow wrapper to suppress per-panel colorbars
    return_cb_params=False,  # if True, return a spec for a figure-level colorbar
    cb_label="Wind Stress (Pa)"  # label reused by wrapper
):
    """
    Plot wind stress on a Robinson projection, compatible with plot_pp_grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
    mean_plot : QuadMesh from xarray
    mean_cb : colorbar instance or None
    cb_params : dict or None
        Spec for a shared colorbar when used with a wrapper.
    """

    mean_da = pp_mean_da[tau_var]

    # ---- Lon normalization ----
    mean_da = mean_da.assign_coords(
        lon=((mean_da.lon + 360) % 360)
    )

    # ---- Target grid for regridding ----
    lat_res = 4 * 90
    lon_res = 4 * 144

    target_lat = np.linspace(mean_da.lat.min(), mean_da.lat.max(), lat_res)
    target_lon = np.linspace(0, 360, lon_res)

    ds_in = xr.Dataset({
        "lat": (["lat"], mean_da.lat.values),
        "lon": (["lon"], mean_da.lon.values),
    })

    ds_out = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    regridder = xe.Regridder(
        ds_in, ds_out, method="bilinear", periodic=True, reuse_weights=False
    )
    mean_da_interp = regridder(mean_da)

    # ---- Diagnostics for color scale ----
    min_val = float(np.nanmin(mean_da.values))
    max_val = float(np.nanmax(mean_da.values))
    per0p5 = float(np.nanpercentile(mean_da.values, 0.5))
    per99p5 = float(np.nanpercentile(mean_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    extra_tick_digits = False

    if tau_var == "tau_x":
        tick_spacing = 0.1
    elif tau_var == "tau_y":
        tick_spacing = 0.05
        
    plot_min = -cb_max
    plot_max = cb_max
    num = int(np.round((plot_max-plot_min)/tick_spacing)) + 1
    tick_arr = np.linspace(plot_min,plot_max,num=num)
    num_colors = 4 * (num - 1)
    
    cmap = cmocean.cm.balance  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors
    
    # create the new map
    disc_cmap = mcolors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    norm_bounds = np.linspace(plot_min, plot_max, num_colors + 1)
    disc_norm = mcolors.BoundaryNorm(norm_bounds, cmap.N)

    # ---- Figure / axes management (wrapper-compatible) ----
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={
                'projection': ccrs.Robinson(central_longitude=209.5),
                'facecolor': 'grey'
            }
        )

    # ---- Main plot ----
    mean_plot = mean_da_interp.plot(
        x='lon', y='lat',
        cmap=disc_cmap,
        norm=disc_norm,
        transform=ccrs.PlateCarree(),
        add_labels=False,
        add_colorbar=False,
        ax=ax
    )
    ax = mean_plot.axes  # Get the existing plot axis

    ax.coastlines(resolution='110m', color='black', linewidth=0.8)

    if hatching:
        hatch_mask = pp_mean_da[f"{tau_var}_hatch"]
    
        # Subsample in lat/lon
        step_y, step_x = 2, 2
        hatch_sub = hatch_mask.isel(
            lat=slice(0, None, step_y),
            lon=slice(0, None, step_x),
        ).squeeze()   # 🔑 drop any length-1 dims (e.g. time=1)
    
        lon_sub = mean_da['lon'].isel(lon=slice(0, None, step_x))
        lat_sub = mean_da['lat'].isel(lat=slice(0, None, step_y))
    
        # Build 2D coordinate grids
        lon2d, lat2d = np.meshgrid(lon_sub.values, lat_sub.values)
    
        # Make sure mask is 2D and boolean
        sel = (hatch_sub.values == 1)   # or >0.5
    
        #### STIPPLING ####
        ax.scatter(
            lon2d[sel],
            lat2d[sel],
            s=2,
            marker='x',
            color='k',
            alpha=0.4,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    if created_fig is None:
        ax.set_title(f"{panel_title}")
    else:
        ax.set_title(f"{panel_title}\nYear {start_yr}–{end_yr}")

    if min_val < plot_min and max_val > plot_max:
        extend = 'both'
    elif min_val < plot_min:
        extend = 'min'
    elif max_val > plot_max:
        extend = 'max'
    else:
        extend = 'neither'
            
    # Colorbar (optional so wrapper can control layout)
    mean_cb = None

    if add_colorbar:
        mean_cb = plt.colorbar(mean_plot, ax=ax, ticks=tick_arr, shrink=0.58, pad=0.04, norm=disc_norm, extend=extend)
        mean_cb.ax.tick_params(labelsize=10)
        mean_cb.set_label("Wind Stress (Pa)", fontdict={'fontsize': 12})
        plt.setp(mean_cb.ax.get_yticklabels(), horizontalalignment='center', x=2.2)

    # --- NEW: package colorbar spec for the wrapper ---
    cb_params = None
    if return_cb_params:

        cb_params = dict(
            mappable=mean_plot,        # carries cmap+norm
            cmap=disc_cmap,
            norm=disc_norm,
            extend=extend,
            ticks=tick_arr,
            label=cb_label
        )

    # ---- Icon (same behavior as before) ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(
            imagebox, (0.95, 1.00),
            xycoords="axes fraction", frameon=False
        )
        ax.add_artist(ab)

    if savefig and created_fig is not None:
        if prefix is None:
            raise ValueError("Must specify prefix for figure file name.")

        image_dpi = 200 if fast_preview else 600
        created_fig.savefig(
            f'{prefix}_{tau_var}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png',
            dpi=image_dpi, bbox_inches='tight'
        )
        plt.close(created_fig)

    return ax, mean_plot, mean_cb, cb_params




