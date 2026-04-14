#!/usr/bin/env python
# coding: utf-8

# # Calculations

def calc_sea_ice_extent(input_ds, threshold = 0.15):
    extent = xr.where(input_ds["siconc"].isnull(), np.nan, (input_ds["siconc"] >= threshold).astype(int))
    input_ds["extent"] = extent
    
    north_conc = xr.where(input_ds["yT"] < 0, np.nan, input_ds["siconc"])
    north_area = float((north_conc * input_ds["CELL_AREA"]).sum(dim=["xT","yT"], skipna=True))
    
    south_conc = xr.where(input_ds["yT"] > 0, np.nan, input_ds["siconc"])
    south_area = float((south_conc * input_ds["CELL_AREA"]).sum(dim=["xT","yT"], skipna=True))
    
    north_extent = xr.where(input_ds["yT"] < 0, np.nan, extent)
    south_extent = xr.where(input_ds["yT"] > 0, np.nan, extent)
    
    return extent, north_area, south_area #, south_extent, north_extent


def calc_ice_zonal_anom(ctrl_ds, input_ds, var, static):
    anomaly = input_ds[var] - ctrl_ds[var]
    zonal_anom = ice_zonal_mean(anomaly, static)
    
    correct_lat = ice_zonal_mean(static.GEOLAT, static)
    zonal_anom = zonal_anom.rename({'yT': 'true_lat'})
    zonal_anom = zonal_anom.assign_coords({'true_lat': correct_lat.values})
    
    return zonal_anom


# # Sea ice concentration & mass anomaly maps

# def plot_sea_ice_diff(
#     panel_title=None,
#     diff_da=None,
#     var='siconc',
#     start_yr=None, end_yr=None,
#     cb_max=None, icon=None,
#     savefig=False, fig_dir=None, prefix=None,
#     fast_preview=True,
#     verbose=False,
#     hemisphere="N",                         # "N" or "S"
#     ax=None, add_colorbar=True, return_cb_params=False,
#     cb_label="Sea Ice Concentration Anomaly"
# ):

#     # ---- Colorbar scaling ----
#     min_val = float(np.nanmin(diff_da.values))
#     max_val = float(np.nanmax(diff_da.values))
#     per0p5  = float(np.nanpercentile(diff_da.values, 0.5))
#     per99p5 = float(np.nanpercentile(diff_da.values, 99.5))

#     if verbose:
#         print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
#         print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")
    
#     # Normalize longitudes to [0, 360)
#     diff_da = diff_da.assign_coords(GEOLON=((diff_da.GEOLON + 360) % 360))

#     # Target grid
#     lat_res = 3 * 210
#     lon_res = 3 * 360
#     target_lat = np.linspace(diff_da.GEOLAT.min(), diff_da.GEOLAT.max(), lat_res)
#     target_lon = np.linspace(0, 360, lon_res, endpoint=False)

#     ds_in = xr.Dataset({
#         "lat": (["yT", "xT"], diff_da.GEOLAT.values),
#         "lon": (["yT", "xT"], diff_da.GEOLON.values),
#     })
#     ds_out = xr.Dataset({
#         "lat": (["lat"], target_lat),
#         "lon": (["lon"], target_lon),
#     })

#     regridder = xe.Regridder(ds_in, ds_out, 
#                              method="bilinear", periodic=True, 
#                              filename="siconc_3xregrid_weights.nc",
#                              reuse_weights=False)
#     diff_da_interp = regridder(diff_da)
    
#     # if use_xesmf:
#     #     import xesmf as xe
    
#     #     # rename dims on source to (y, x) for xESMF
#     #     src = diff_da#.rename({"yT": "y", "xT": "x"})  # adjust as needed
    
#     #     src_grid = xr.Dataset(
#     #         {
#     #             "lon": (("yT", "xT"), src["GEOLON"].values),
#     #             "lat": (("yT", "xT"), src["GEOLAT"].values),
#     #         }
#     #     )
    
#     #     # build target grid
#     #     # Example: 4× refinement between 45–90N, -180–180E
#     #     factor = 4
        
#     #     lat_target = np.linspace(-76.75, 89.75, int(4*166.5))
#     #     lon_target = np.linspace(-279.5, 79.5, int(4 * 359))
    
#     #     dst_grid = xr.Dataset(
#     #         {
#     #             "lon": (("xT",), lon_target),
#     #             "lat": (("yT",), lat_target),
#     #         }
#     #     )
    
#     #     if regridder is None:
#     #         # regridder = xe.Regridder(
#     #         #     src_grid,
#     #         #     dst_grid,
#     #         #     method="bilinear",  # or "conservative", "nearest_s2d", etc.
#     #         #     reuse_weights=False  # set True if weights already exist on disk
#     #         # )
#     #         regridder = xe.Regridder(
#     #             src_grid,
#     #             dst_grid,
#     #             method="bilinear",
#     #             filename="/home/Kiera.Lowman/siconc_weights.nc",
#     #             reuse_weights=False
#     #         )
    
#     #     diff_regrid = regridder(src)
    
#     #     # For plotting with x='GEOLON', y='GEOLAT', you can just alias:
#     #     diff_to_plot = diff_regrid.rename({"lon": "GEOLON", "lat": "GEOLAT"})
    

#     extra_tick_digits = False
#     if cb_max is not None:
#         chosen_n = 10
#         data_max = cb_max
#         chosen_step = 2 * data_max / chosen_n
#     else:
#         chosen_n, chosen_step = get_cb_spacing(
#             per0p5, per99p5, min_bnd=0.1, min_spacing=0.01,
#             min_n=10, max_n=20, verbose=verbose
#         )

#     max_mag = 0.5 * chosen_n * chosen_step
#     zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
#         max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
#     )

#     # ---- Figure/axes ----
#     created_fig = None
#     if ax is None:
#         proj = ccrs.NorthPolarStereo(central_longitude=-30.0) if hemisphere.upper() == "N" \
#                else ccrs.SouthPolarStereo(central_longitude=-120.0)
#         created_fig, ax = plt.subplots(
#             figsize=(7.5, 5),
#             subplot_kw={'projection': proj, 'facecolor': 'grey'}
#         )

#     # ---- Plot ----
#     # diff_plot = diff_da.plot(
#     #     x='GEOLON', y='GEOLAT',
#     #     cmap=disc_cmap, norm=disc_norm,
#     #     transform=ccrs.PlateCarree(),
#     #     add_labels=False, add_colorbar=False, ax=ax
#     # )
#     diff_plot = diff_da_interp.plot(
#         x='lon', y='lat',
#         cmap=disc_cmap, norm=disc_norm,
#         transform=ccrs.PlateCarree(),
#         add_labels=False, add_colorbar=False, ax=ax
#     )

#     # # ---- Gridlines and labels ----
#     # gl = ax.gridlines(
#     #     draw_labels=True, color='black', alpha=0.4, linestyle='--',
#     #     # x_inline=False, y_inline=True  # inline latitudes only
#     # )

#     # # Choose label locations and appearance
#     # gl.top_labels = gl.right_labels = False
#     # gl.left_labels = True
#     # gl.bottom_labels = True
#     # gl.xlabel_style = {'size': 9, 'color': 'black'}
#     # gl.ylabel_style = {'size': 9, 'color': 'black'}

#     # Set tick intervals (deg)
#     # gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
#     if hemisphere.upper() == "N":
#         # gl.ylocator = mticker.FixedLocator(np.arange(40, 91, 10))
#         ax.set_extent([-300, 60, 45, 90], ccrs.PlateCarree())
#     else:
#         # gl.ylocator = mticker.FixedLocator(np.arange(-90, -39, 10))
#         ax.set_extent([-300, 60, -60, -90], ccrs.PlateCarree())

#     # # Format tick labels
#     # gl.xformatter = LONGITUDE_FORMATTER
#     # gl.yformatter = LATITUDE_FORMATTER

#     # ---- Title ----
#     hemi_name = "North" if hemisphere.upper() == "N" else "South"
#     if created_fig and panel_title != None:
#         ax.set_title(f"{panel_title} — {hemi_name}")
#     # else:
#     #     ax.set_title(f"Sea Ice Concentration Anomaly ({hemi_name} Pole): {start_yr}–{end_yr}")

#     # ---- Colorbar ----
#     diff_cb = None
#     if add_colorbar:
#         diff_cb = plt.colorbar(
#             diff_plot, ax=ax, shrink=0.75, pad=0.04, extend=extend,
#             boundaries=boundaries, norm=disc_norm, spacing='proportional'
#         )
#         tick_labels = []
#         for val in tick_positions:
#             if chosen_step < 0.01:
#                 tick_labels.append(f"{val:.3f}")
#             elif chosen_step == 0.05:
#                 tick_labels.append(f"{val:.1f}")
#             elif extra_tick_digits:
#                 tick_labels.append(f"{val:.3f}")
#             else:
#                 tick_labels.append(f"{val:.2f}")
#         diff_cb.set_ticks(tick_positions)
#         diff_cb.ax.set_yticklabels(tick_labels)
#         diff_cb.ax.tick_params(labelsize=10)
#         diff_cb.set_label(cb_label)

#     # ---- Package colorbar parameters ----
#     cb_params = None
#     if return_cb_params:
#         # Build tick labels the same way as panel bars
#         tick_labels = []
#         for val in tick_positions:
#             if chosen_step < 0.01:
#                 tick_labels.append(f"{val:.3f}")
#             elif chosen_step == 0.05:
#                 tick_labels.append(f"{val:.1f}")
#             elif extra_tick_digits:
#                 tick_labels.append(f"{val:.3f}")
#             else:
#                 tick_labels.append(f"{val:.2f}")
                
#         cb_params = dict(
#             mappable=diff_plot, cmap=disc_cmap, norm=disc_norm,
#             boundaries=boundaries, extend=extend, spacing='proportional',
#             ticks=tick_positions, ticklabels=tick_labels, label=cb_label
#         )

#     # ---- Optional icon ----
#     if icon is not None:
#         image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
#         img = mpimg.imread(image_path)
#         imagebox = OffsetImage(img, zoom=0.09)
#         ab = AnnotationBbox(imagebox, (1.08, 0.95),
#                             xycoords="axes fraction", frameon=False)
#         ax.add_artist(ab)

#     # ---- Save ----
#     if savefig and created_fig is not None:
#         dpi_out = 200 if fast_preview else 600
#         out = fig_dir + f'{prefix}_{var}_{hemisphere.upper()}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png'
#         created_fig.savefig(out, dpi=dpi_out, bbox_inches='tight')
#         plt.close(created_fig)

#     return ax, diff_plot, diff_cb, cb_params


def plot_sea_ice_diff(
    panel_title=None,
    diff_da=None,
    var='siconc',
    start_yr=None, end_yr=None,
    cb_max=None, icon=None,
    savefig=False, fig_dir=None, prefix=None,
    fast_preview=True,
    verbose=False,
    hemisphere="N",                         # "N" or "S"
    ax=None, add_colorbar=True, return_cb_params=False,
    cb_label="Sea Ice Concentration Anomaly"
):
    import xesmf as xe
    from cartopy.util import add_cyclic_point
    from scipy.ndimage import gaussian_filter

    # ---- Colorbar scaling on original field ----
    min_val = float(np.nanmin(diff_da.values))
    max_val = float(np.nanmax(diff_da.values))
    per0p5  = float(np.nanpercentile(diff_da.values, 0.5))
    per99p5 = float(np.nanpercentile(diff_da.values, 99.5))

    if verbose:
        print(f"Full data min/max: {min_val:.3f}/{max_val:.3f}")
        print(f"Percentile-based max magnitude: {max(abs(per0p5), abs(per99p5)):.3f}")

    # ------------------------------------------------------------------
    # Regrid with xESMF: curvilinear (GEOLON/GEOLAT, yT/xT) -> regular lat/lon
    # ------------------------------------------------------------------
    # diff_da = diff_da.copy()

    # # Normalize longitudes to [0, 360)
    # diff_da = diff_da.assign_coords(GEOLON=((diff_da.GEOLON + 360) % 360))

    # geolat = diff_da.GEOLAT.values  # 2D (yT, xT)
    # geolon = diff_da.GEOLON.values  # 2D (yT, xT), now 0..360

    # # Derive native spacing and refine by factor
    # refine_factor = 3  # change to 4 if you want finer

    # dlat_native = np.nanmean(np.abs(geolat[1:, :] - geolat[:-1, :]))
    # dlon_raw = geolon[:, 1:] - geolon[:, :-1]
    # dlon_wrapped = (dlon_raw + 540.0) % 360.0 - 180.0
    # dlon_native = np.nanmean(np.abs(dlon_wrapped))

    # dlat_new = dlat_native / refine_factor
    # dlon_new = dlon_native / refine_factor

    # lat_min = float(np.nanmin(geolat))
    # lat_max = float(np.nanmax(geolat))

    # target_lat = np.arange(lat_min, lat_max + 0.5 * dlat_new, dlat_new)
    # target_lon = np.arange(0.0, 360.0, dlon_new)   # 0, dlon, ..., <360

    # ds_in = xr.Dataset(
    #     {
    #         "lat": (["yT", "xT"], geolat),
    #         "lon": (["yT", "xT"], geolon),
    #     }
    # )
    # ds_out = xr.Dataset(
    #     {
    #         "lat": (["lat"], target_lat),
    #         "lon": (["lon"], target_lon),
    #     }
    # )

    # regridder = xe.Regridder(
    #     ds_in,
    #     ds_out,
    #     method="bilinear",
    #     periodic=True,
    #     filename="siconc_3xregrid_weights.nc",
    #     reuse_weights=True,
    #     ignore_degenerate=True,
    # )

    # diff_da_interp = regridder(diff_da, keep_attrs=True)  # dims: (lat, lon)

    # # ------------------------------------------------------------------
    # # Optional smoothing to reduce jagged appearance
    # # ------------------------------------------------------------------
    # data = diff_da_interp.values
    # nan_mask = np.isnan(data)
    # data_filled = np.where(nan_mask, 0.0, data)

    # # tweak sigma as desired (lat, lon)
    # data_smooth = gaussian_filter(data_filled, sigma=(0.5, 0.5))
    # data_smooth[nan_mask] = np.nan

    # diff_da_smooth = diff_da_interp.copy(data=data_smooth)

    # # ------------------------------------------------------------------
    # # Add cyclic point in longitude to remove seam at 0/360
    # # ------------------------------------------------------------------
    # data_cyc, lon_cyc = add_cyclic_point(
    #     diff_da_smooth.values,
    #     coord=diff_da_smooth.lon.values
    # )

    # diff_da_plot = xr.DataArray(
    #     data_cyc,
    #     coords={
    #         "lat": diff_da_smooth.lat.values,
    #         "lon": lon_cyc,
    #     },
    #     dims=("lat", "lon"),
    #     name=diff_da_smooth.name,
    #     attrs=diff_da_smooth.attrs,
    # )

    # ---- Colorbar params (unchanged) ----
    extra_tick_digits = False
    if cb_max is not None:
        chosen_n = 10
        data_max = cb_max
        chosen_step = 2 * data_max / chosen_n
    else:
        if var == 'siconc':
            chosen_n, chosen_step = get_cb_spacing(
                per0p5, per99p5, min_bnd=0.1, min_spacing=0.01,
                min_n=10, max_n=20, verbose=verbose
            )
        elif var == 'simass':
            chosen_n, chosen_step = get_cb_spacing(
                per0p5, per99p5, min_bnd=25, min_spacing=2.5,
                min_n=10, max_n=20, verbose=verbose
            )

    max_mag = 0.5 * chosen_n * chosen_step
    zero_step, disc_cmap, disc_norm, boundaries, extend, tick_positions = create_cb_params(
        max_mag, min_val, max_val, chosen_n, chosen_step, verbose=verbose
    )

    # ---- Figure/axes ----
    created_fig = None
    if ax is None:
        proj = ccrs.NorthPolarStereo(central_longitude=-30.0) if hemisphere.upper() == "N" \
               else ccrs.SouthPolarStereo(central_longitude=-120.0)
        created_fig, ax = plt.subplots(
            figsize=(7.5, 5),
            subplot_kw={'projection': proj, 'facecolor': 'grey'}
        )

    # # ---- Plot on regular lat/lon ----
    # diff_plot = diff_da_plot.plot(
    #     x='lon', y='lat',
    #     cmap=disc_cmap, norm=disc_norm,
    #     transform=ccrs.PlateCarree(),
    #     add_labels=False, add_colorbar=False, ax=ax
    # )
    # ---- Plot on regular lat/lon ----
    diff_plot = diff_da.plot(
        x='GEOLON', y='GEOLAT',
        cmap=disc_cmap, norm=disc_norm,
        transform=ccrs.PlateCarree(),
        add_labels=False, add_colorbar=False, ax=ax
    )

    # ---- Map extent (use actual max lat of regridded grid) ----
    # lat_max_plot = float(diff_da_plot['lat'].max())
    lat_max_plot = 90
    if hemisphere.upper() == "N":
        ax.set_extent([-300, 60, 45, lat_max_plot], ccrs.PlateCarree())
    else:
        ax.set_extent([-300, 60, -lat_max_plot, -45], ccrs.PlateCarree())

    # ---- Title ----
    hemi_name = "North" if hemisphere.upper() == "N" else "South"
    if created_fig and panel_title is not None:
        ax.set_title(f"{panel_title} — {hemi_name}")

    # ---- Colorbar ----
    diff_cb = None
    if add_colorbar:
        diff_cb = plt.colorbar(
            diff_plot, ax=ax, shrink=0.75, pad=0.04, extend=extend,
            boundaries=boundaries, norm=disc_norm, spacing='proportional'
        )
        tick_labels = []
        for val in tick_positions:
            if chosen_step < 0.01:
                tick_labels.append(f"{val:.3f}")
            elif chosen_step == 0.05:
                tick_labels.append(f"{val:.1f}")
            elif extra_tick_digits:
                tick_labels.append(f"{val:.3f}")
            else:
                tick_labels.append(f"{val:.2f}")
        diff_cb.set_ticks(tick_positions)
        diff_cb.ax.set_yticklabels(tick_labels)
        diff_cb.ax.tick_params(labelsize=10)
        diff_cb.set_label(cb_label)

    # ---- Package colorbar parameters ----
    cb_params = None
    if return_cb_params:
        tick_labels = []
        for val in tick_positions:
            if chosen_step < 0.01:
                tick_labels.append(f"{val:.3f}")
            elif chosen_step == 0.05:
                tick_labels.append(f"{val:.1f}")
            elif extra_tick_digits:
                tick_labels.append(f"{val:.3f}")
            else:
                tick_labels.append(f"{val:.2f}")
        cb_params = dict(
            mappable=diff_plot, cmap=disc_cmap, norm=disc_norm,
            boundaries=boundaries, extend=extend, spacing='proportional',
            ticks=tick_positions, ticklabels=tick_labels, label=cb_label
        )

    # ---- Optional icon ----
    if icon is not None:
        image_path = f"/home/Kiera.Lowman/profile_icons/{icon}_icon.png"
        img = mpimg.imread(image_path)
        imagebox = OffsetImage(img, zoom=0.09)
        ab = AnnotationBbox(imagebox, (1.08, 0.95),
                            xycoords="axes fraction", frameon=False)
        ax.add_artist(ab)

    # ---- Save ----
    if savefig and created_fig is not None:
        dpi_out = 200 if fast_preview else 600
        out = fig_dir + f'{prefix}_{var}_{hemisphere.upper()}_{str(start_yr).zfill(4)}_{str(end_yr).zfill(4)}.png'
        created_fig.savefig(out, dpi=dpi_out, bbox_inches='tight')
        plt.close(created_fig)

    return ax, diff_plot, diff_cb, cb_params


# # Grid figure function

def plot_sea_ice_diff_grid(
    panels,
    suptitle=None,
    outfile=None,
    figsize=None,
    cbar_label="Sea Ice Concentration Anomaly",
    fast_preview=True,
    shared_cbar=True,
    cb_max=None,
):

    n_exp = len(panels)
    if n_exp not in (1, 2, 3, 4):
        raise ValueError("panels must have length 1, 2, 3, or 4.")

    # Figure size (w × h): scale by columns; leave room on bottom for colorbar
    if figsize is None:
        base_w = 2
        base_h = 4
        figsize = (base_w * n_exp, base_h + 1.0)

    # NOTE: use manual spacing (constrained_layout=False) for tighter packing
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, n_exp, figure=fig, wspace=0.0, hspace=0.0) #0.2

    # Manually trim margins; leaves space for colorbar and labels
    plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08, wspace=0.0, hspace=0.0)

    # Build axes (two per experiment)
    ax_matrix = [[None for _ in range(n_exp)] for _ in range(2)]
    for j in range(n_exp):
        axN = fig.add_subplot(gs[0, j], projection=ccrs.NorthPolarStereo(central_longitude=-30.0))
        axN.set_facecolor('grey')
        ax_matrix[0][j] = axN

        axS = fig.add_subplot(gs[1, j], projection=ccrs.SouthPolarStereo(central_longitude=-120.0))
        axS.set_facecolor('grey')
        ax_matrix[1][j] = axS

    # after you create the axes, and before/after set_extent (both are fine)
    for row in ax_matrix:
        for ax in row:
            ax.set_aspect('equal', adjustable='datalim')

    # Draw panels (suppress per-panel titles/icons for grid use)
    cb_spec = None
    extension_param = 'neither'
    for j, panel_kwargs in enumerate(panels):
        base_kw = dict(panel_kwargs)
        if cb_max is not None and 'cb_max' not in base_kw:
            base_kw['cb_max'] = cb_max

        # ---- NORTH (no subplot title/icon) ----
        kwN = dict(base_kw)
        kwN.update({
            'ax': ax_matrix[0][j],
            'hemisphere': 'N',
            'savefig': False,
            'add_colorbar': (not shared_cbar),
            'return_cb_params': shared_cbar,
            'panel_title': None,      # suppress per-subplot titles
            'icon': None,             # suppress per-subplot icons
        })
        _, _mN, _, cbN = plot_sea_ice_diff(**kwN)

        # ---- SOUTH (no subplot title/icon) ----
        kwS = dict(base_kw)
        kwS.update({
            'ax': ax_matrix[1][j],
            'hemisphere': 'S',
            'savefig': False,
            'add_colorbar': (not shared_cbar),
            'return_cb_params': shared_cbar,
            'panel_title': None,
            'icon': None,
        })
        _, _mS, _, cbS = plot_sea_ice_diff(**kwS)

        # Merge shared-colorbar spec
        if shared_cbar:
            for cb_params in (cbN, cbS):
                if cb_params is None:
                    continue
                cb_spec = cb_params
                ext = cb_params.get('extend', 'neither')
                if extension_param == 'neither':
                    extension_param = ext
                elif extension_param == 'min' and ext in ('both', 'max'):
                    extension_param = 'both'
                elif extension_param == 'max' and ext in ('both', 'min'):
                    extension_param = 'both'

    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import matplotlib.image as mpimg
    
    # ---- Suptitle ----
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.97)
    
    # ---- Shared colorbar ----
    if shared_cbar and cb_spec is not None:
        cbar = fig.colorbar(
            cb_spec['mappable'],
            ax=[ax for row in ax_matrix for ax in row],
            orientation='horizontal',
            pad=0.02, shrink=0.6,
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
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.tick_params(labelrotation=45)
        cbar.set_label(cbar_label or cb_spec.get('label', ''), fontsize=12)

    # ---- Column labels ABOVE each column (text + optional icon) ----
    fig.canvas.draw()  # finalize positions after suptitle + colorbar
    
    LABEL_FONTSIZE   = 12
    GAP_Y_POINTS     = 6   # vertical gap above top axes, in points
    GAP_X_POINTS     = 24   # horizontal gap between text and icon, in points
    ICON_ZOOM        = 0.05
    
    # convert gaps in points to figure-fraction
    fig_w, fig_h = fig.get_size_inches()
    gap_y_frac = (GAP_Y_POINTS / 72.0) / fig_h
    gap_x_frac = (GAP_X_POINTS / 72.0) / fig_w
    
    for j, panel_kwargs in enumerate(panels):
        title = panel_kwargs.get('panel_title')
        icon_name = panel_kwargs.get('icon')
    
        if not title:
            continue
    
        # --- position just above the top-row axis for this column ---
        ax_top = ax_matrix[0][j]
        p = ax_top.get_position()  # Bbox in figure fraction
        x_mid = p.x0 + 0.5 * p.width
        y_label = p.y1 + gap_y_frac
        y_label = min(y_label, 0.99)
    
        # --- text label ---
        fig.text(
            x_mid, y_label, title,
            ha='center', va='bottom',
            fontsize=LABEL_FONTSIZE,
            transform=fig.transFigure,
            zorder=10,
        )
    
        # --- optional icon to the RIGHT of the text ---
        if icon_name:
            try:
                img = mpimg.imread(f"/home/Kiera.Lowman/profile_icons/{icon_name}_icon.png")
            except FileNotFoundError:
                continue  # just skip icon if not found
    
            ico = OffsetImage(img, zoom=ICON_ZOOM)
            # place icon a bit to the right of text
            x_icon = x_mid + gap_x_frac
    
            ab = AnnotationBbox(
                ico,
                (x_icon, y_label),
                xycoords='figure fraction',
                frameon=False,
                box_alignment=(0.0, 0.0),  # left/bottom of icon at (x_icon, y_label)
                zorder=10,
            )
            fig.add_artist(ab)


    # ---- Save / return ----
    dpi_out = 200 if fast_preview else 600
    if outfile:
        fig.savefig(outfile, dpi=dpi_out)# bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, ax_matrix




