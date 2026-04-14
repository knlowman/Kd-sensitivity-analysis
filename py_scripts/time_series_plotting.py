#!/usr/bin/env python
# coding: utf-8

# Notebook with functions for plotting anomalies and raw time series.

from matplotlib.ticker import MultipleLocator, FixedLocator, AutoMinorLocator


# # Ocean scalar variables

# ## Regular anomaly and mean

def plot_ts_diff(diff_type,fig_dir,start_year,end_year,anom_var="thetaoga",title_suff="Volume Mean Ocean Temperature Anomaly",
                 fig_pref=None,
                 omit_title=True,
                 roll_mean = True,
                 roll_mean_window = 10,
                 ylimits = None,
                 # ylimits = [-0.1,1.2],
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                       profiles = ['surf','therm','mid','bot'], 
                       power_inputs = ['0.1TW', '0.2TW', '0.3TW'], 
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):
    """
    Function to plot anomaly over time from time series data for a particular CO2 scenario (each power input on separate plot).
    
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
        fig, ax = plt.subplots(figsize=(6,3))

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ds_root = f'{power}_{start_year}_{end_year}_diff'
        for prof in profiles:
            if diff_type == 'const-1860ctrl':
                ds_name = f'const_{prof}_{ds_root}'
            elif diff_type == 'doub-1860exp':
                ds_name = f'doub_{prof}_{ds_root}_1860'
            elif diff_type == 'doub-2xctrl':
                ds_name = f'doub_{prof}_{ds_root}_2xctrl'
            elif diff_type == 'doub-1860ctrl':
                ds_name = f'doub_{prof}_{ds_root}_const_ctrl'
            elif diff_type == 'quad-1860exp':
                ds_name = f'quad_{prof}_{ds_root}_1860'
            elif diff_type == 'quad-4xctrl':
                ds_name = f'quad_{prof}_{ds_root}_4xctrl'
            elif diff_type == 'quad-1860ctrl':
                ds_name = f'quad_{prof}_{ds_root}_const_ctrl'
            
            anom = myVars[ds_name][anom_var]
            time = np.linspace(start_year,end_year,num=len(anom) )
            if roll_mean:
                anom = anom.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,anom,label=prof,color=prof_dict[prof])

        if diff_type == 'doub-1860exp' or diff_type == 'doub-1860ctrl':
            ctrl_diff_name = f'doub_ctrl_{start_year}_{end_year}_diff'
            anom = myVars[ctrl_diff_name][anom_var]
            time = np.linspace(start_year,end_year,num=len(anom) )
            if roll_mean:
                anom = anom.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,anom,label='control',color='k')

        ax.set_xlabel("Time (Years)")
        if anom_var == "ave_hfds":
            ax.set_ylabel("Flux Anomaly (W/m$^2$)")
        elif anom_var == "OHC":
            ax.set_ylabel("Heat Content Anomaly (ZJ)")
        else:
            ax.set_ylabel("Temperature Anomaly ($\degree$C)")
            
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
            title_str = f"1pct2xCO\u2082 Radiative {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-const_{power}"
            
        elif diff_type == 'doub-2xctrl':
            title_str = f"1pct2xCO\u2082 Mixing {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-2xctrl_{power}"
            
        elif diff_type == 'doub-1860ctrl':
            title_str = f"1pct2xCO\u2082 Total {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-const-ctrl_{power}"
            
        elif diff_type == 'quad-1860exp':
            title_str = f"1pct4xCO\u2082 Radiative {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-const_{power}"
            
        elif diff_type == 'quad-4xctrl':
            title_str = f"1pct4xCO\u2082 Mixing {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-4xctrl_{power}"
            
        elif diff_type == 'quad-1860ctrl':
            title_str = f"1pct4xCO\u2082 Total {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-const-ctrl_{power}"

        if omit_title is False:
            ax.set_title(title_str+title_suff)
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if fig_pref != None:
            plt.savefig(fig_dir+f'{fig_pref}_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


def plot_ts_mean(co2_scen,fig_dir,start_year,end_year,var="thetaoga",title_suff="Volume Mean Ocean Temperature",
                 fig_pref=None,
                 omit_title=True,
                 roll_mean = True,
                 roll_mean_window = 10,
                 ylimits = None,
                 # ylimits = [3.0,5.0],
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                       profiles = ['surf','therm','mid','bot'], 
                       power_inputs = ['0.1TW', '0.2TW', '0.3TW'], 
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):
    """
    Function to plot raw time series variable (not anomaly).
    
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

    # separate plot for each power input
    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(6,3))
        
        ds_root = f'{power}_{start_year}_{end_year}'
        for prof in profiles:
            ds_name = f'{co2_scen}_{prof}_{ds_root}'
            mean_ts = myVars[ds_name][var]
            time = np.linspace(start_year,end_year,num=len(mean_ts) )
            if roll_mean:
                mean_ts = mean_ts.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,mean_ts,label=prof,color=prof_dict[prof])

        ctrl_ds_name = f'{co2_scen}_ctrl_{start_year}_{end_year}'
        mean_ts = myVars[ctrl_ds_name][var]
        time = np.linspace(start_year,end_year,num=len(mean_ts) )
        if roll_mean:
            mean_ts = mean_ts.rolling(time=roll_mean_window, center=True).mean()
        ax.plot(time,mean_ts,label='control',color='k')

        ax.set_xlabel("Time (Years)")
        if var == "ave_hfds":
            ax.set_ylabel("Flux (W/m$^2$)")
        elif var == "OHC":
            ax.set_ylabel("Ocean Heat Content (ZJ)")
        else:
            ax.set_ylabel("Temperature ($\degree$C)")
            
        ax.set_xlim(start_year-1,end_year)
        if ylimits:
            ax.set_ylim(ylimits)
        ax.legend(loc=leg_loc,ncols=leg_ncols)
        ax.grid("both")
    
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

        if co2_scen == 'const':
            title_str = f"Const CO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"const_{power}"
            
        elif co2_scen == 'doub':
            title_str = f"1pct2xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2_{power}"
            
        elif co2_scen == 'quad':
            title_str = f"1pct4xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2_{power}"

        if omit_title is False:
            ax.set_title(title_str+title_suff)
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        
        if fig_pref != None:
            plt.savefig(fig_dir+f'{fig_pref}_{var}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'{var}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


# ## Mixing and radiative anomaly components

def plot_ts_mixing_diff(co2_scen,fig_dir,start_year,end_year,
                        omit_title=True, roll_mean = True, roll_mean_window = 10,
                        anom_var="thetaoga",title_suff="Volume Mean Ocean Temperature Anomaly",fig_pref=None,
                        ystep_list = [0.1, 0.2, 0.2], ymin_frac_list = [0.2, 0.2, 0.2], ylims_list = [None,None,None],
                     leg_loc = 'upper left',
                     leg_ncols = 1,
                       profiles = ['surf','therm','mid','bot'], 
                       power_inputs = ['0.1TW', '0.2TW', '0.3TW'], 
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):
    """
    Function to plot mixing response for multiple CO2 scenarios.
    
    Inputs:
    co2_scen (str): one of ['doub','quad','all']
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

    # separate plot for each power input
    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(6,3))

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ds_root = f'{power}_{start_year}_{end_year}_diff'

        plotted_vals = []   # collect values used in this panel
        
        for prof in profiles:
            const_ds_name = f'const_{prof}_{ds_root}'
            doub_ds_name = f'doub_{prof}_{ds_root}_2xctrl'
            quad_ds_name = f'quad_{prof}_{ds_root}_4xctrl'

            const_anom = myVars[const_ds_name][anom_var]
            time = np.linspace(start_year,end_year,num=len(const_anom))
            if roll_mean:
                const_anom = const_anom.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,const_anom,color=prof_dict[prof])
            plotted_vals.append(np.asarray(const_anom))
            
            if co2_scen == 'doub' or co2_scen == 'all':
                doub_anom = myVars[doub_ds_name][anom_var]
                if roll_mean:
                    doub_anom = doub_anom.rolling(time=roll_mean_window, center=True).mean()
                ax.plot(time,doub_anom,linestyle='dashed',color=prof_dict[prof])
                plotted_vals.append(np.asarray(doub_anom))
                
            elif co2_scen == 'quad' or co2_scen == 'all':
                quad_anom = myVars[quad_ds_name][anom_var]
                if roll_mean:
                    quad_anom = quad_anom.rolling(time=roll_mean_window, center=True).mean()
                ax.plot(time,quad_anom,linestyle='dotted',color=prof_dict[prof])
                plotted_vals.append(np.asarray(quad_anom))

        ax.set_xlabel("Time (Years)")
        if anom_var == "OHC":
            ax.set_ylabel("Heat Content Anomaly (ZJ)")
        else:
            ax.set_ylabel("Temperature Anomaly ($\degree$C)")
        ax.set_xlim(start_year-1,end_year)

        # # --- y-axis grid/ticks every ystep, bounds forced to multiples of ystep ---
        # ystep = ystep_list[pow_idx]
        # ymin = -ymin_frac_list[pow_idx] * ystep
        
        # # if ylimits is not None:
        # #     ymax = np.ceil(ylimits[1] / ystep) * ystep
        # # else:
        # all_vals = np.concatenate([arr[np.isfinite(arr)] for arr in plotted_vals])
        # ymax = np.ceil(all_vals.max() / ystep) * ystep
        
        # if ymax <= 0:
        #     ymax = ystep
        
        # ax.set_ylim(ymin, ymax)
        
        # major_ticks = np.arange(0, ymax + ystep, ystep)
        # ax.yaxis.set_major_locator(FixedLocator(major_ticks))

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

        ax.legend(mixing_leg, mixing_leg_labels, loc=leg_loc, fontsize=10, ncol = leg_ncols, labelspacing=0.1)
        
        ax.grid("both")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

        title_str = f"Mixing $\Delta T$ Comparison: {power_strings[pow_idx]} Cases\n"
        
        if co2_scen == 'doub':
            fig_name = f"const_2xCO2_mixing_{power}"
        elif co2_scen == 'quad':
            fig_name = f"const_4xCO2_mixing_{power}"
        elif co2_scen == 'all':
            fig_name = f"all_mixing_{power}"

        if omit_title is False:
            ax.set_title(title_str+title_suff)
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if fig_pref != None:
            plt.savefig(fig_dir+f'{fig_pref}_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


def plot_ts_rad_pow_diff(co2_scen,fig_dir,start_year,end_year,
                         omit_title=True, roll_mean = True, roll_mean_window = 10,
                     anom_var="thetaoga",title_suff="Volume Mean Ocean Temperature Anomaly",fig_pref=None,
                     ylimits = None, ystep = 0.2, ymin_frac = 0.2,
                     leg_loc = 'upper left',
                     leg_ncols = 1,
                         power_inputs = ['0.1TW', '0.2TW', '0.3TW'],
                         power_var_suff = ['0p1TW', '0p2TW', '0p3TW'],
                         power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):
    """
    Function to plot radiative response for multiple power inputs.
    
    Inputs:
    co2_scen (str): one of ['doub','quad']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    """

    profiles = ['surf','therm','mid','bot']

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
    
    fig, ax = plt.subplots(figsize=(6,3))

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    power_line_types = ['solid','dashed','dotted']

    plotted_vals = []   # collect values used in this panel

    if co2_scen == 'doub':
        ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_diff'
    elif co2_scen == 'quad':
        ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_diff_const_ctrl'
        
    ctrl_rad_anom = myVars[ctrl_rad_name][anom_var]
    time = np.linspace(start_year,end_year,num=len(ctrl_rad_anom))
    if roll_mean:
        ctrl_rad_anom = ctrl_rad_anom.rolling(time=roll_mean_window, center=True).mean()
    ax.plot(time,ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)
    plotted_vals.append(np.asarray(ctrl_rad_anom))
    
    for pow_idx, power in enumerate(power_var_suff):
        ds_root = f'{power}_{start_year}_{end_year}_diff'
        
        for prof in profiles:
            if co2_scen == 'doub':
                ds_name = f'doub_{prof}_{ds_root}_1860'
            elif co2_scen == 'quad':
                ds_name = f'quad_{prof}_{ds_root}_1860'
    
            rad_anom = myVars[ds_name][anom_var]
            # time = np.linspace(start_year,end_year,num=len(rad_anom))
            if roll_mean:
                rad_anom = rad_anom.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,rad_anom,color=prof_dict[prof],linestyle=power_line_types[pow_idx])
            plotted_vals.append(np.asarray(rad_anom))

    ax.set_xlabel("Time (Years)")
    if anom_var == "OHC":
        ax.set_ylabel("Heat Content Anomaly (ZJ)")
    else:
        ax.set_ylabel("Temperature Anomaly ($\degree$C)")
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
    
    # major_ticks = np.arange(0, ymax + ystep, ystep)
    # ax.yaxis.set_major_locator(FixedLocator(major_ticks))

    ax.legend(rad_leg, rad_leg_labels, loc=leg_loc, fontsize=10, ncol = leg_ncols, labelspacing=0.1)
    
    ax.grid("both")
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    
    if co2_scen == 'doub':
        title_str = f"1pct2xCO\u2082 Radiative $\Delta T$ Comparison\n"
        fig_name = f"2xCO2_rad"
    elif diff_type == 'quad':
        title_str = f"1pct4xCO\u2082 Radiative $\Delta T$ Comparison\n"
        fig_name = f"4xCO2_rad"

    if omit_title is False:
        ax.set_title(title_str+title_suff)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    if fig_pref != None:
        plt.savefig(fig_dir+f'{fig_pref}_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(fig_dir+f'{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


def plot_ts_mix_pow_diff(co2_scen,fig_dir,start_year,end_year,
                         omit_title=True, roll_mean = True, roll_mean_window = 10,
                         omit_ctrl=False,
                     anom_var="thetaoga",title_suff="Volume Mean Ocean Temperature Anomaly",fig_pref=None,
                     ylimits = None, ystep = 0.2, ymin_frac = 0.2,
                     leg_loc = 'upper left',
                     leg_ncols = 1,
                         profiles = ['surf','therm','mid','bot'],
                         power_var_suff = ['0p1TW', '0p2TW', '0p3TW'],
                         power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):
    """
    Function to plot mixing response for multiple power inputs.
    
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

    fig, ax = plt.subplots(figsize=(6,3))

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    power_line_types = ['solid','dashed','dotted']

    plotted_vals = []   # collect values used in this panel

    if omit_ctrl == False:
        if co2_scen == 'const' or co2_scen == 'doub':
            ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_diff'
        elif co2_scen == 'quad':
            ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_diff_const_ctrl'
            
        ctrl_rad_anom = myVars[ctrl_rad_name][anom_var]
        time = np.linspace(start_year,end_year,num=len(ctrl_rad_anom))
        if roll_mean:
            ctrl_rad_anom = ctrl_rad_anom.rolling(time=roll_mean_window, center=True).mean()
        ax.plot(time,ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)
        plotted_vals.append(np.asarray(ctrl_rad_anom))
    
    for pow_idx, power in enumerate(power_var_suff):
        ds_root = f'{power}_{start_year}_{end_year}_diff'
        
        for prof in profiles:
            if co2_scen == 'const':
                ds_name = f'const_{prof}_{ds_root}'
            elif co2_scen == 'doub':
                ds_name = f'doub_{prof}_{ds_root}_2xctrl'
            elif co2_scen == 'quad':
                ds_name = f'quad_{prof}_{ds_root}_4xctrl'
    
            mix_anom = myVars[ds_name][anom_var]
            time = np.linspace(start_year,end_year,num=len(mix_anom))
            if roll_mean:
                mix_anom = mix_anom.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,mix_anom,color=prof_dict[prof],linestyle=power_line_types[pow_idx])
            plotted_vals.append(np.asarray(mix_anom))

    ax.set_xlabel("Time (Years)")
    if anom_var == "OHC":
        ax.set_ylabel("Heat Content Anomaly (ZJ)")
    else:
        ax.set_ylabel("Temperature Anomaly ($\degree$C)")
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
    
    # major_ticks = np.arange(0, ymax + ystep, ystep)
    # ax.yaxis.set_major_locator(FixedLocator(major_ticks))

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
        title_str = f"Const CO\u2082 $\Delta T$ Comparison\n"
        fig_name = f"const_mix"
    elif co2_scen == 'doub':
        title_str = f"1pct2xCO\u2082 Mixing $\Delta T$ Comparison\n"
        fig_name = f"2xCO2_mix"
    elif co2_scen == 'quad':
        title_str = f"1pct4xCO\u2082 Mixing $\Delta T$ Comparison\n"
        fig_name = f"4xCO2_mix"

    if omit_title is False:
        ax.set_title(title_str+title_suff)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    if fig_pref != None:
        plt.savefig(fig_dir+f'{fig_pref}_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(fig_dir+f'{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


def plot_ts_lin_pow_diff(co2_scen,fig_dir,start_year,end_year,
                         omit_title=True, roll_mean = True, roll_mean_window = 10,
                         omit_ctrl=False,
                     anom_var="thetaoga",title_suff="Volume Mean Ocean Temperature Anomaly",fig_pref=None,
                     ylimits = None, ystep = 0.2, ymin_frac = 0.2,
                     leg_loc = 'upper left',
                     leg_ncols = 1,
                         power_floats = [0.1, 0.2, 0.3],
                         ref_power = 0.3,
                         power_var_suff = ['0p1TW', '0p2TW', '0p3TW']):
    """
    Function to plot mixing response for multiple power inputs scaled by power.
    
    Inputs:
    co2_scen (str): one of ['const','doub','quad']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    """

    profiles = ['surf','therm','mid','bot']

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
    labels_2 = []

    power_floats = np.array(power_floats)
    factors = np.zeros(len(power_floats))
    
    for idx, power in enumerate(power_floats):
        factors[idx]= ref_power/power
        if power == ref_power:
            labels_2.append(f"{power} TW")
        else:
            labels_2.append(f"{factors[idx]:.1f} x {power} TW")

    mix_leg_3 = [Line2D([0], [0], alpha=0.6, lw=4, color='tab:gray')]

    if co2_scen == 'const' or co2_scen == 'doub':
        labels_3 = ['1pct2xCO\u2082 ctrl']
    elif co2_scen == 'quad':
        labels_3 = ['1pct4xCO\u2082 ctrl']

    fig, ax = plt.subplots(figsize=(6,3))

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    power_line_types = ['solid','dashed','dotted']

    plotted_vals = []   # collect values used in this panel

    if omit_ctrl == False:
        if co2_scen == 'const' or co2_scen == 'doub':
            ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_diff'
        elif co2_scen == 'quad':
            ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_diff_const_ctrl'
            
        ctrl_rad_anom = myVars[ctrl_rad_name][anom_var]
        time = np.linspace(start_year,end_year,num=len(ctrl_rad_anom))
        if roll_mean:
            ctrl_rad_anom = ctrl_rad_anom.rolling(time=roll_mean_window, center=True).mean()
        ax.plot(time,ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)
        plotted_vals.append(np.asarray(ctrl_rad_anom))
    
    for pow_idx, power in enumerate(power_var_suff):
        ds_root = f'{power}_{start_year}_{end_year}_diff'
        
        for prof in profiles:
            if co2_scen == 'const':
                ds_name = f'const_{prof}_{ds_root}'
            elif co2_scen == 'doub':
                ds_name = f'doub_{prof}_{ds_root}_2xctrl'
            elif co2_scen == 'quad':
                ds_name = f'quad_{prof}_{ds_root}_4xctrl'
    
            mix_anom = myVars[ds_name][anom_var]
            time = np.linspace(start_year,end_year,num=len(mix_anom))
            if roll_mean:
                mix_anom = mix_anom.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,factors[pow_idx]*mix_anom,color=prof_dict[prof],linestyle=power_line_types[pow_idx])
            plotted_vals.append(np.asarray(factors[pow_idx]*mix_anom))

    ax.set_xlabel("Time (Years)")
    if anom_var == "OHC":
        ax.set_ylabel("Heat Content Anomaly (ZJ)")
    else:
        ax.set_ylabel("Temperature Anomaly ($\degree$C)")
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
    
    # major_ticks = np.arange(0, ymax + ystep, ystep)
    # ax.yaxis.set_major_locator(FixedLocator(major_ticks))

    if leg_loc == 'upper left':
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
                bbox_to_anchor=(0.49, 1.0),  # Adjust position as needed
                frameon=True
            )
    elif leg_loc == 'lower left':
        legend1 = ax.legend(
        mix_leg_1, labels_1,
        loc=leg_loc,
        fontsize=10, labelspacing=0.1,
        bbox_to_anchor=(0.0, 0.0),  # Adjust position as needed
        frameon=True
        )
        legend2 = ax.legend(
            mix_leg_2, labels_2,
            loc=leg_loc,
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(0.2, 0.0),  # Adjust position as needed
            frameon=True
        )
        if omit_ctrl == False:
            legend3 = ax.legend(
                mix_leg_3, labels_3,
                loc=leg_loc,
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(0.49, 0.0),  # Adjust position as needed
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
        title_str = f"Const CO\u2082 $\Delta T$ Comparison\n"
        fig_name = f"const_mix_lin_pow"
    elif co2_scen == 'doub':
        title_str = f"1pct2xCO\u2082 Mixing $\Delta T$ Comparison\n"
        fig_name = f"2xCO2_mix_lin_pow"
    elif co2_scen == 'quad':
        title_str = f"1pct4xCO\u2082 Mixing $\Delta T$ Comparison\n"
        fig_name = f"4xCO2_mix_lin_pow"

    if omit_title is False:
        ax.set_title(title_str+title_suff)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    if fig_pref != None:
        plt.savefig(fig_dir+f'{fig_pref}_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(fig_dir+f'{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


def plot_ts_diff_select(co2_scen,fig_dir,start_year,end_year, power_var_suff, power_strings,
                        profiles=['surf','therm','mid','bot'],
                        doub_ctrl_rescale = 0.1,
                         omit_title=True, roll_mean = True, roll_mean_window = 10,
                         omit_ctrl=False,
                                vline_yr=None,
                     anom_var="thetaoga",title_suff="Volume Mean Ocean Temperature Anomaly",fig_pref=None,
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
            ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_diff'
        elif co2_scen == 'quad':
            ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_diff_const_ctrl'
            
        ctrl_rad_anom = myVars[ctrl_rad_name][anom_var]
        time = np.linspace(start_year,end_year,num=len(ctrl_rad_anom))
        if roll_mean:
            ctrl_rad_anom = ctrl_rad_anom.rolling(time=roll_mean_window, center=True).mean()
        ax.plot(time,doub_ctrl_rescale*ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)

    ds_root = f'{start_year}_{end_year}_diff'
    for prof_idx, prof in enumerate(profiles):
        if co2_scen == 'const':
            ds_name = f'const_{prof}_{power_var_suff[prof_idx]}_{ds_root}'
        elif co2_scen == 'doub':
            ds_name = f'doub_{prof}_{power_var_suff[prof_idx]}_{ds_root}_2xctrl'
        elif co2_scen == 'quad':
            ds_name = f'quad_{prof}_{power_var_suff[prof_idx]}_{ds_root}_4xctrl'

        mix_anom = myVars[ds_name][anom_var]
        time = np.linspace(start_year,end_year,num=len(mix_anom))
        if roll_mean:
            mix_anom = mix_anom.rolling(time=roll_mean_window, center=True).mean()
        ax.plot(time,mix_anom,color=prof_dict[prof])

    ax.set_xlabel("Time (Years)")
    if anom_var == "OHC":
        ax.set_ylabel("Heat Content Anomaly (ZJ)")
    else:
        ax.set_ylabel("Temperature Anomaly ($\degree$C)")
        
    if xlimits:
        ax.set_xlim(xlimits)
    else:
        ax.set_xlim(start_year-1,end_year)
        
    if ylimits:
        ax.set_ylim(ylimits)

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
        title_str = f"Const CO\u2082 $\Delta T$ Comparison\n"
        fig_name = f"const_mix_select"
    elif co2_scen == 'doub':
        title_str = f"1pct2xCO\u2082 Mixing $\Delta T$ Comparison\n"
        fig_name = f"2xCO2_mix_select"
    elif co2_scen == 'quad':
        title_str = f"1pct4xCO\u2082 Mixing $\Delta T$ Comparison\n"
        fig_name = f"4xCO2_mix_select"

    if omit_title is False:
        ax.set_title(title_str+title_suff)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    if fig_pref != None:
        plt.savefig(fig_dir+f'{fig_pref}_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(fig_dir+f'{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


def plot_ts_diff_custom(ds_name,fig_dir,start_year,end_year, da_label,
                        doub_ctrl_rescale = 0.1,
                         omit_title=True, roll_mean = True, roll_mean_window = 10,
                                vline_yr=None,
                     anom_var="thetaoga",title_suff="Volume Mean Ocean Temperature Anomaly",fig_pref=None,
                     ylimits = None,
                                xlimits=None,
                     leg_loc = 'upper left',
                     leg_ncols = 1):
    """
    Function to plot mixing response for the select power inputs that were only run for specific profiles.
    
    Inputs:
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    
    """

    custom_leg = [
        Line2D([0], [0], color='k', lw=2),
        Line2D([0], [0], alpha=0.6, lw=4, color='tab:gray')
    ]
    
    custom_labels = [da_label]
    
    if doub_ctrl_rescale != 1:
        custom_labels.append(f'{doub_ctrl_rescale} x 1pct2xCO\u2082 ctrl')
    else:
        custom_labels.append('1pct2xCO\u2082 ctrl')

    fig, ax = plt.subplots(figsize=(6,3))

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    if vline_yr:
        ax.axvline(x=70, color='black', linestyle='-', linewidth=1)

    power_line_types = ['solid','dashed','dotted']

    # if omit_ctrl == False:
    ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_diff'
        
    ctrl_rad_anom = myVars[ctrl_rad_name][anom_var]
    time = np.linspace(start_year,end_year,num=len(ctrl_rad_anom))
    if roll_mean:
        ctrl_rad_anom = ctrl_rad_anom.rolling(time=roll_mean_window, center=True).mean()
    ax.plot(time,doub_ctrl_rescale*ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)

    mix_anom = myVars[ds_name][anom_var]
    time = np.linspace(start_year,end_year,num=len(mix_anom))
    if roll_mean:
        mix_anom = mix_anom.rolling(time=roll_mean_window, center=True).mean()
    ax.plot(time,mix_anom,color='k')

    ax.set_xlabel("Time (Years)")
    if anom_var == "OHC":
        ax.set_ylabel("Heat Content Anomaly (ZJ)")
    else:
        ax.set_ylabel("Temperature Anomaly ($\degree$C)")
        
    if xlimits:
        ax.set_xlim(xlimits)
    else:
        ax.set_xlim(start_year-1,end_year)
        
    if ylimits:
        ax.set_ylim(ylimits)

    legend1 = ax.legend(
    custom_leg, custom_labels,
    loc=leg_loc,
    fontsize=10, labelspacing=0.1,
    bbox_to_anchor=(0.0, 1.0),  # Adjust position as needed
    frameon=True
    )
    
    # # Add the first legend back to the axis
    # ax.add_artist(legend1)
    
    ax.grid("both")
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

    fig_name = f"{ds_name}"
    
    if omit_title is False:
        ax.set_title(title_suff)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    if fig_pref != None:
        plt.savefig(fig_dir+f'{fig_pref}_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(fig_dir+f'{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


def plot_ts_diff_rescale(co2_scen,fig_dir,start_year,end_year, factors, pow_labels,
                         omit_title=True, roll_mean = True, roll_mean_window = 10,
                         omit_ctrl=False,
                                vline_yr=None,
                     anom_var="thetaoga",title_suff="Volume Mean Ocean Temperature Anomaly",fig_pref=None,
                     ylimits = None,
                                xlimits=None,
                     leg_loc = 'upper left',
                     leg_ncols = 1,
                         power_var_suff = ['0p1TW','0p1TW','0p3TW','0p3TW'],
                         power_strings = ['0.1 TW','0.1 TW','0.3 TW','0.3 TW']):
    """
    Function to plot mixing response rescaled by different factors for different profiles (to estimate a mixing response for some power input).
    power_var_suff and power_strings corresponds to the experiment data to rescale for each profile.
    
    Inputs:
    co2_scen (str): one of ['const','doub','quad']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    
    """

    profiles = ['surf','therm','mid','bot']

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
    labels_1 = []
    for idx, factor in enumerate(factors):
        # labels_1.append(f"{factor:.3g} x {power_strings[idx]} {profiles[idx]}")
        labels_1.append(f"{profiles[idx]} {pow_labels[idx]} TW")
    
    # mix_leg_2 = [  # Second column (3 labels)
    #     Line2D([0], [0], linestyle='solid', lw=2, color='k'),
    #     Line2D([0], [0], linestyle='dashed', lw=2, color='k'),
    #     Line2D([0], [0], linestyle='dotted', lw=2, color='k')
    # ]
    # labels_2 = []

    mix_leg_3 = [Line2D([0], [0], alpha=0.6, lw=4, color='tab:gray')]

    if co2_scen == 'const' or co2_scen == 'doub':
        labels_3 = ['0.1 x 1pct2xCO\u2082 ctrl']
    elif co2_scen == 'quad':
        labels_3 = ['0.1 x 1pct4xCO\u2082 ctrl']

    fig, ax = plt.subplots(figsize=(6,3))

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    if vline_yr:
        ax.axvline(x=70, color='black', linestyle='-', linewidth=1)

    power_line_types = ['solid','dashed','dotted']

    if omit_ctrl == False:
        if co2_scen == 'const' or co2_scen == 'doub':
            ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_diff'
        elif co2_scen == 'quad':
            ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_diff_const_ctrl'
            
        ctrl_rad_anom = myVars[ctrl_rad_name][anom_var]
        time = np.linspace(start_year,end_year,num=len(ctrl_rad_anom))
        if roll_mean:
            ctrl_rad_anom = ctrl_rad_anom.rolling(time=roll_mean_window, center=True).mean()
        ax.plot(time,0.1*ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)

    ds_root = f'{start_year}_{end_year}_diff'
    for prof_idx, prof in enumerate(profiles):
        if co2_scen == 'const':
            ds_name = f'const_{prof}_{power_var_suff[prof_idx]}_{ds_root}'
        elif co2_scen == 'doub':
            ds_name = f'doub_{prof}_{power_var_suff[prof_idx]}_{ds_root}_2xctrl'
        elif co2_scen == 'quad':
            ds_name = f'quad_{prof}_{power_var_suff[prof_idx]}_{ds_root}_4xctrl'

        mix_anom = myVars[ds_name][anom_var]
        time = np.linspace(start_year,end_year,num=len(mix_anom))
        if roll_mean:
            mix_anom = mix_anom.rolling(time=roll_mean_window, center=True).mean()
        ax.plot(time,factors[prof_idx]*mix_anom,color=prof_dict[prof])

    ax.set_xlabel("Time (Years)")
    if anom_var == "OHC":
        ax.set_ylabel("Heat Content Anomaly (ZJ)")
    else:
        ax.set_ylabel("Temperature Anomaly ($\degree$C)")
        
    if xlimits:
        ax.set_xlim(xlimits)
    else:
        ax.set_xlim(start_year-1,end_year)
        
    if ylimits:
        ax.set_ylim(ylimits)

    legend1 = ax.legend(
    mix_leg_1, labels_1,
    loc=leg_loc,
    fontsize=10, labelspacing=0.1,
    bbox_to_anchor=(0.0, 1.0),  # Adjust position as needed
    frameon=True
    )
    # legend2 = ax.legend(
    #     mix_leg_2, labels_2,
    #     loc=leg_loc,
    #     fontsize=10, labelspacing=0.1,
    #     bbox_to_anchor=(0.2, 1.0),  # Adjust position as needed
    #     frameon=True
    # )
    if omit_ctrl == False:
        legend3 = ax.legend(
            mix_leg_3, labels_3,
            loc=leg_loc,
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(0.4, 1.0),  # Adjust position as needed
            frameon=True
        )
    
    # Add the first legend back to the axis
    ax.add_artist(legend1)
    
    ax.grid("both")
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

    if co2_scen == 'const':
        title_str = f"Const CO\u2082 $\Delta T$ Comparison\n"
        fig_name = f"const_mix_rescale"
    elif co2_scen == 'doub':
        title_str = f"1pct2xCO\u2082 Mixing $\Delta T$ Comparison\n"
        fig_name = f"2xCO2_mix_rescale"
    elif co2_scen == 'quad':
        title_str = f"1pct4xCO\u2082 Mixing $\Delta T$ Comparison\n"
        fig_name = f"4xCO2_mix_rescale"

    if omit_title is False:
        ax.set_title(title_str+title_suff)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    if fig_pref != None:
        plt.savefig(fig_dir+f'{fig_pref}_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(fig_dir+f'{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


# ## Linearity of mixing and radiative anomaly components

def plot_lin_ts_diff(co2_scen,fig_dir,start_year,end_year,
                     omit_ctrl = False,
                     omit_title=True, roll_mean = True, roll_mean_window = 10,
                     anom_var="thetaoga",title_suff="Volume Mean Ocean Temperature Anomaly",fig_pref=None,
                     ystep_list = [0.1, 0.2, 0.2], ymin_frac_list = [0.2, 0.2, 0.2], ylims_list = [None,None,None],
                     leg_loc = 'upper left',
                       profiles = ['surf','therm','mid','bot'], 
                       power_inputs = ['0.1TW', '0.2TW', '0.3TW'], 
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):
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
    
    # CO2_lin_leg_2 = [  # Second column (2 labels)
    #     Line2D([0], [0], linestyle='solid', lw=2, color='k'),  # realized response
    #     Line2D([0], [0], linestyle='dashed', lw=2, color='k')   # CO2 + mixing
    # ]
    # labels_2 = ['realized response', 'CO\u2082 + mixing']

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
            ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_diff'
        elif co2_scen == 'quad':
            ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_diff_const_ctrl'
            
    else:
        CO2_lin_leg_2 = [  # Second column (2 labels)
            Line2D([0], [0], linestyle='solid', lw=2, color='k'),  # realized response
            Line2D([0], [0], linestyle='dashed', lw=2, color='k')   # CO2 + mixing
        ]
        
        labels_2 = ['realized response', 'CO\u2082 + mixing']

    n_yrs = (end_year - start_year + 1)
    time = np.linspace(start_year,end_year,num=n_yrs)

    # separate plot for each power input
    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(6,3))

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        plotted_vals = []   # collect values used in this panel

        if omit_ctrl == False:
            if co2_scen == 'const' or co2_scen == 'doub':
                ctrl_rad_name = f'doub_ctrl_{start_year}_{end_year}_diff'
            elif co2_scen == 'quad':
                ctrl_rad_name = f'quad_ctrl_{start_year}_{end_year}_diff_const_ctrl'
                
            ctrl_rad_anom = myVars[ctrl_rad_name][anom_var]
            # time = np.linspace(start_year,end_year,num=len(ctrl_rad_anom))
            if roll_mean:
                ctrl_rad_anom = ctrl_rad_anom.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,ctrl_rad_anom,color='tab:gray',alpha=0.6,lw=4)
            plotted_vals.append(np.asarray(ctrl_rad_anom))
        
        ds_root = f'{power}_{start_year}_{end_year}_diff'
        for prof in profiles:
            if co2_scen == 'doub':
                co2_ds_name = f'doub_{prof}_{ds_root}_1860'
                mixing_ds_name = f'doub_{prof}_{ds_root}_2xctrl'
                total_ds_name = f'doub_{prof}_{ds_root}_const_ctrl'
            elif co2_scen == 'quad':
                co2_ds_name = f'quad_{prof}_{ds_root}_1860'
                mixing_ds_name = f'quad_{prof}_{ds_root}_4xctrl'
                total_ds_name = f'quad_{prof}_{ds_root}_const_ctrl'

            co2_anom = myVars[co2_ds_name][anom_var]
            mixing_anom = myVars[mixing_ds_name][anom_var]
            total_anom = myVars[total_ds_name][anom_var]

            sum_anom = co2_anom + mixing_anom
            
            # time = np.linspace(start_year,end_year,num=len(total_anom) )
            if roll_mean:
                total_anom = total_anom.rolling(time=roll_mean_window, center=True).mean()
                sum_anom = sum_anom.rolling(time=roll_mean_window, center=True).mean()
                
            ax.plot(time,total_anom,color=prof_dict[prof])
            ax.plot(time,sum_anom,linestyle='--',color=prof_dict[prof])
            plotted_vals.append(np.asarray(total_anom))
            plotted_vals.append(np.asarray(sum_anom))

        ax.set_xlabel("Time (Years)")
        if anom_var == "OHC":
            ax.set_ylabel("Heat Content Anomaly (ZJ)")
        else:
            ax.set_ylabel("Temperature Anomaly ($\degree$C)")
        ax.set_xlim(start_year-1,end_year)

        # # --- y-axis grid/ticks every ystep, bounds forced to multiples of ystep ---
        # ystep = ystep_list[pow_idx]
        # ymin = -ymin_frac_list[pow_idx] * ystep
        
        # # if ylimits is not None:
        # #     ymax = np.ceil(ylimits[1] / ystep) * ystep
        # # else:
        # all_vals = np.concatenate([arr[np.isfinite(arr)] for arr in plotted_vals])
        # ymax = np.ceil(all_vals.max() / ystep) * ystep
        
        # if ymax <= 0:
        #     ymax = ystep
        
        # ax.set_ylim(ymin, ymax)
        
        # major_ticks = np.arange(0, ymax + ystep, ystep)
        # ax.yaxis.set_major_locator(FixedLocator(major_ticks))

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
            else:
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
        elif leg_loc == 'lower right':
            if omit_ctrl == False:
                legend1 = ax.legend(
                CO2_lin_leg_1, labels_1,
                loc=leg_loc,
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(1.0, 0.0),  # Adjust position as needed
                frameon=True
                )
                # Second legend (2 labels, positioned below the first)
                legend2 = ax.legend(
                    CO2_lin_leg_2, labels_2,
                    loc=leg_loc,
                    fontsize=10, labelspacing=0.1,
                    bbox_to_anchor=(0.8, 0.0),  # Adjust position as needed
                    frameon=True
                )
            else:
                legend1 = ax.legend(
                CO2_lin_leg_1, labels_1,
                loc=leg_loc,
                fontsize=10, labelspacing=0.1,
                bbox_to_anchor=(1.0, 0.18),  # Adjust position as needed
                frameon=True
                )
                # Second legend (2 labels, positioned below the first)
                legend2 = ax.legend(
                    CO2_lin_leg_2, labels_2,
                    loc=leg_loc,
                    fontsize=10, labelspacing=0.1,
                    bbox_to_anchor=(1.0, 0.0),  # Adjust position as needed
                    frameon=True
                )
        
        # Add the first legend back to the axis
        ax.add_artist(legend1)
        
        ax.grid("both")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            
        if co2_scen == 'doub':
            title_str = f"1pct2xCO\u2082 Total $\Delta T$ Comparison: {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2_lin_{power}"
            
        elif diff_type == 'quad':
            title_str = f"1pct4xCO\u2082 Total $\Delta T$ Comparison: {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2_lin_{power}"

        if omit_title is False:
            ax.set_title(title_str+title_suff)
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        
        if fig_pref != None:
            plt.savefig(fig_dir+f'{fig_pref}_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


# ## MOC strength anomalies and means

def plot_moc_anom_ts(diff_type,moc_type,fig_dir,start_year,end_year,
                     fig_pref=None,
                      omit_title=True,roll_mean = True,roll_mean_window = 10,
                     ylimits = None,
                 # ylimits = [-5,15],
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                       profiles = ['surf','therm','mid','bot'], 
                       power_inputs = ['0.1TW', '0.2TW', '0.3TW'], 
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):
    """
    Inputs:
    diff_type (str): one of
                    ['const-1860ctrl',
                    'doub-1860exp','doub-2xctrl','doub-1860ctrl',
                    'quad-1860exp','quad-4xctrl','quad-1860ctrl']
    moc_type (str): one of
                    ['north-atl','south-atl','south-global','south-indopac']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    """

    anom_dict = {'north-atl':'atl-arc_north_diff_max',
                 'south-atl':'atl-arc_south_diff_max',
                 'south-global':'global_south_diff_max',
                 'south-indopac': 'indopac_south_diff_max'
               }
    diff_name = anom_dict[moc_type]

    prof_dict = {'surf': 'b',
                 'therm': 'm',
                 'mid': 'g',
                 'bot': 'r'}

    # separate plot for each power input
    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(6,3))

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ds_root = f'{power}_{start_year}_{end_year}_max_diff'
        for prof in profiles:
            if diff_type == 'const-1860ctrl':
                ds_name = f'const_{prof}_{ds_root}'
            elif diff_type == 'doub-1860exp':
                ds_name = f'doub_{prof}_{ds_root}_1860'
            elif diff_type == 'doub-2xctrl':
                ds_name = f'doub_{prof}_{ds_root}_2xctrl'
            elif diff_type == 'doub-1860ctrl':
                ds_name = f'doub_{prof}_{ds_root}_const_ctrl'
            elif diff_type == 'quad-1860exp':
                ds_name = f'quad_{prof}_{ds_root}_1860'
            elif diff_type == 'quad-4xctrl':
                ds_name = f'quad_{prof}_{ds_root}_4xctrl'
            elif diff_type == 'quad-1860ctrl':
                ds_name = f'quad_{prof}_{ds_root}_const_ctrl'

            max_dat = myVars[ds_name][diff_name].compute()
            # max_dat = myVars[ds_name]["atl-arc_psi_max"].values
            time = np.linspace(start_year,end_year,num=len(max_dat))
            if roll_mean:
                max_dat = max_dat.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,max_dat,label=prof,color=prof_dict[prof])

        if diff_type == 'doub-1860exp' or diff_type == 'doub-1860ctrl':
            ctrl_diff_name = f'doub_ctrl_{start_year}_{end_year}_max_diff'
            ctrl_max_dat = myVars[ctrl_diff_name][diff_name].compute()
            # ctrl_max_dat = myVars[ctrl_diff_name]["atl-arc_psi_max"].values
            time = np.linspace(start_year,end_year,num=len(ctrl_max_dat))
            if roll_mean:
                ctrl_max_dat = ctrl_max_dat.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,ctrl_max_dat,label='control',color='k')

        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Strength Anomaly (Sv)")
        ax.set_xlim(start_year-1,end_year)
        if ylimits:
            ax.set_ylim(ylimits)
        ax.legend(loc=leg_loc,ncols=leg_ncols)
        ax.grid("both")
    
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        
        if diff_type == 'const-1860ctrl':
            title_str = f"Const CO\u2082 Response: {power_strings[pow_idx]} Cases\n"
            fig_name = f"const_{power}"
            
        elif diff_type == 'doub-1860exp':
            title_str = f"1pct2xCO\u2082 Radiative Response: {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-const_{power}"
            
        elif diff_type == 'doub-2xctrl':
            title_str = f"1pct2xCO\u2082 Mixing Response: {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-2xctrl_{power}"
            
        elif diff_type == 'doub-1860ctrl':
            title_str = f"1pct2xCO\u2082 Total Response: {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-const-ctrl_{power}"
            
        elif diff_type == 'quad-1860exp':
            title_str = f"1pct4xCO\u2082 Radiative Response: {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-const_{power}"
            
        elif diff_type == 'quad-4xctrl':
            title_str = f"1pct4xCO\u2082 Mixing Response: {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-4xctrl_{power}"
            
        elif diff_type == 'quad-1860ctrl':
            title_str = f"1pct4xCO\u2082 Total Response: {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-const-ctrl_{power}"

        if omit_title is False:
            ax.set_title(title_str+"MOC Strength Anomaly")
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if fig_pref != None:
            plt.savefig(fig_dir+f'{fig_pref}_{moc_type}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'{moc_type}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


def plot_moc_anom_mix_pow_ts(co2_scen,moc_type,fig_dir,start_year,end_year,
                             fig_pref=None,
                             omit_title=True,roll_mean = True,roll_mean_window = 10,
                             ylimits = None,
                             # ylimits = [-5,15],
                             profiles = ['surf','therm','mid','bot'],
                             power_inputs = ['0.1TW', '0.2TW', '0.3TW'],
                             power_var_suff = ['0p1TW', '0p2TW', '0p3TW'],
                             power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):

    """
    Inputs:
    co2_scen (str): one of ['const','doub','quad']
    moc_type (str): one of
                    ['north-atl','south-atl','south-global','south-indopac']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    """

    anom_dict = {'north-atl':'atl-arc_north_diff_max',
                 'south-atl':'atl-arc_south_diff_max',
                 'south-global':'global_south_diff_max',
                 'south-indopac': 'indopac_south_diff_max'
               }
    
    diff_name = anom_dict[moc_type]
    
    profiles = ['surf','therm','mid','bot']

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

    # if co2_scen == 'const' or co2_scen == 'doub':
    if co2_scen == 'doub':
        labels_3 = ['1pct2xCO\u2082 ctrl']
    elif co2_scen == 'quad':
        labels_3 = ['1pct4xCO\u2082 ctrl']

    fig, ax = plt.subplots(figsize=(6,3))

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    power_line_types = ['solid','dashed','dotted']

    if co2_scen == 'doub' or co2_scen == 'quad':
        if co2_scen == 'doub':
            ctrl_diff_name = f'doub_ctrl_{start_year}_{end_year}_max_diff'
        elif co2_scen == 'quad':
            ctrl_diff_name = f'quad_ctrl_{start_year}_{end_year}_max_diff'
        ctrl_max_dat = myVars[ctrl_diff_name][diff_name].compute()
        time = np.linspace(start_year,end_year,num=len(ctrl_max_dat))
        if roll_mean:
            ctrl_max_dat = ctrl_max_dat.rolling(time=roll_mean_window, center=True).mean()
        ax.plot(time,ctrl_max_dat,label='control', alpha=0.6, lw=4, color='tab:gray')
    
    # add each power input
    for pow_idx, power in enumerate(power_var_suff):
        ds_root = f'{power}_{start_year}_{end_year}_max_diff'
        for prof in profiles:
            if co2_scen == 'const':
                ds_name = f'const_{prof}_{ds_root}'
            elif co2_scen == 'doub':
                ds_name = f'doub_{prof}_{ds_root}_2xctrl'
            elif co2_scen == 'quad':
                ds_name = f'quad_{prof}_{ds_root}_4xctrl'

            max_dat = myVars[ds_name][diff_name].compute()
            time = np.linspace(start_year,end_year,num=len(max_dat))
            if roll_mean:
                max_dat = max_dat.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,max_dat,color=prof_dict[prof],linestyle=power_line_types[pow_idx])

    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Strength Anomaly (Sv)")
    ax.set_xlim(start_year-1,end_year)
    if ylimits:
        ax.set_ylim(ylimits)

    leg_loc = 'upper left'
    
    legend1 = ax.legend(
    mix_leg_1, labels_1,
    loc=leg_loc,
    fontsize=10, labelspacing=0.1,
    bbox_to_anchor=(0.0, 1.0),  # Adjust position as needed
    frameon=True
    )
    # Second legend (2 labels, positioned below the first)
    legend2 = ax.legend(
        mix_leg_2, labels_2,
        loc=leg_loc,
        fontsize=10, labelspacing=0.1,
        bbox_to_anchor=(0.2, 1.0),  # Adjust position as needed
        frameon=True
    )
    if co2_scen == 'doub' or co2_scen == 'quad':
        legend3 = ax.legend(
            mix_leg_3, labels_3,
            loc=leg_loc,
            fontsize=10, labelspacing=0.1,
            bbox_to_anchor=(0.415, 1.0),  # Adjust position as needed
            frameon=True
        )
    
    # Add the other legends back to the axis
    ax.add_artist(legend1)
    if co2_scen == 'doub' or co2_scen == 'quad':
        ax.add_artist(legend2)
    
    ax.grid("both")
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    
    if co2_scen == 'const':
        title_str = f"Const CO\u2082 Response\n"
        fig_name = f"const_mix"
        
    elif co2_scen == 'doub':
        title_str = f"1pct2xCO\u2082 Mixing Response\n"
        fig_name = f"2xCO2_mix"
        
    elif co2_scen == 'quad':
        title_str = f"1pct4xCO\u2082 Mixing Response\n"
        fig_name = f"4xCO2_mix"

    if omit_title is False:
        ax.set_title(title_str+"MOC Strength Anomaly")
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    if fig_pref != None:
        plt.savefig(fig_dir+f'{fig_pref}_{moc_type}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(fig_dir+f'{moc_type}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


def plot_moc_mean_ts(co2_scen,moc_type,fig_dir,start_year,end_year,
                     fig_pref=None,
                      omit_title=True,roll_mean = True,roll_mean_window = 10,
                     ylimits = None,
                 # ylimits = [15,40],
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                       profiles = ['surf','therm','mid','bot'], 
                       power_inputs = ['0.1TW', '0.2TW', '0.3TW'], 
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):
    """
    Inputs:
    co2_scen (str): one of ['const','doub','quad']
    moc_type (str): one of
                    ['north-atl','south-atl','south-global','south-indopac']
    fig_dir (str): directory path to save figure
    start_year (int): start year of avg period
    end_year (int): end year of avg period
    """

    prof_dict = {'surf': 'b',
                 'therm': 'm',
                 'mid': 'g',
                 'bot': 'r'}

    moc_dict = {'north-atl':'atl-arc_north_psi_max',
                'south-atl':'atl-arc_south_psi_max',
                'south-global':'global_south_psi_max',
                'south-indopac': 'indopac_south_psi_max'
               }
    moc_name = moc_dict[moc_type]

    # separate plot for each power input
    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(6,3))

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ds_root = f'{power}_{start_year}_{end_year}_max'
        for prof in profiles:
            ds_name = f'{co2_scen}_{prof}_{ds_root}'
            max_dat = myVars[ds_name][moc_name].compute()
            time = np.linspace(start_year,end_year,num=len(max_dat))
            if roll_mean:
                max_dat = max_dat.rolling(time=roll_mean_window, center=True).mean()
            ax.plot(time,max_dat,label=prof,color=prof_dict[prof])

        ctrl_ds_name = f'{co2_scen}_ctrl_{start_year}_{end_year}_max'
        ctrl_max_dat = myVars[ctrl_ds_name][moc_name].compute()
        time = np.linspace(start_year,end_year,num=len(ctrl_max_dat))
        if roll_mean:
            ctrl_max_dat = ctrl_max_dat.rolling(time=roll_mean_window, center=True).mean()
        ax.plot(time,ctrl_max_dat,label='control',color='k')

        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Strength (Sv)")
        ax.set_xlim(start_year-1,end_year)
        if ylimits:
            ax.set_ylim(ylimits)
        ax.legend(loc=leg_loc,ncols=leg_ncols)
        ax.grid("both")

        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

        if co2_scen == 'const':
            title_str = f"Const CO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"const_{power}"
            
        elif co2_scen == 'doub':
            title_str = f"1pct2xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2_{power}"
            
        elif co2_scen == 'quad':
            title_str = f"1pct4xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2_{power}"

        if omit_title is False:
            ax.set_title(title_str+"MOC Strength")
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if fig_pref != None:
            plt.savefig(fig_dir+f'{fig_pref}_{moc_type}_mean_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'{moc_type}_mean_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


# # Atmos monthly variables (requires horizontal averaging)

# ## Regular anomaly and mean

def plot_atmos_ts_diff_2d(diff_type,fig_dir,start_year,end_year,
                          omit_title=True, roll_mean = True, roll_mean_window = 10,
                          anom_var="olr",title_suff="Mean OLR Anomaly",fig_pref=None,
                          cb_label = "Flux Anomaly (W/m$^2$)",
                          offline_mean=False,
                          ylimits = None,
                 # ylimits = [-0.5,0.5],
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                       profiles = ['surf','therm','mid','bot'], 
                       power_inputs = ['0.1TW', '0.2TW', '0.3TW'], 
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):
    """
    Function to plot anomaly over time from time series data for a particular CO2 scenario (each power input on separate plot).
    
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
        fig, ax = plt.subplots(figsize=(6,3))

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ds_root = f'{power}_{start_year}_{end_year}_diff'
        for prof in profiles:
            if diff_type == 'const-1860ctrl':
                ds_name = f'const_{prof}_{ds_root}'
            elif diff_type == 'doub-1860exp':
                ds_name = f'doub_{prof}_{ds_root}_1860'
            elif diff_type == 'doub-2xctrl':
                ds_name = f'doub_{prof}_{ds_root}_2xctrl'
            elif diff_type == 'doub-1860ctrl':
                ds_name = f'doub_{prof}_{ds_root}_const_ctrl'
            elif diff_type == 'quad-1860exp':
                ds_name = f'quad_{prof}_{ds_root}_1860'
            elif diff_type == 'quad-4xctrl':
                ds_name = f'quad_{prof}_{ds_root}_4xctrl'
            elif diff_type == 'quad-1860ctrl':
                ds_name = f'quad_{prof}_{ds_root}_const_ctrl'

            if offline_mean:
                ds_name = ds_name + "_mean"
                global_mean = myVars[ds_name][anom_var]
            else:
                if anom_var == 'EEI':
                    myVars[ds_name]['EEI'] = myVars[ds_name]['swdn_toa'] - myVars[ds_name]['swup_toa'] - myVars[ds_name]['olr']
                    global_mean = atmos_horiz_mean(myVars[ds_name]['EEI'],myVars[ds_name])
                elif anom_var == 'temp':
                    myVars[ds_name]['temp'] = myVars[ds_name]['temp'].isel(pfull = -1)
                else:
                    global_mean = atmos_horiz_mean(myVars[ds_name][anom_var],myVars[ds_name])

            time = np.linspace(start_year,end_year,num=len(global_mean) )
            
            if roll_mean:
                global_mean = global_mean.rolling(time=roll_mean_window, center=True).mean()
                
            ax.plot(time,global_mean,label=prof,color=prof_dict[prof])

        if diff_type == 'doub-1860exp' or diff_type == 'doub-1860ctrl':
            # plotting control difference wrt 1860 ctrl
            ds_name = f'doub_ctrl_{start_year}_{end_year}_diff'
            if offline_mean:
                ds_name = ds_name + "_mean"
                global_mean = myVars[ds_name][anom_var]
            else:
                if anom_var == 'EEI':
                    myVars[ds_name]['EEI'] = myVars[ds_name]['swdn_toa'] - myVars[ds_name]['swup_toa'] - myVars[ds_name]['olr']
                    global_mean = atmos_horiz_mean(myVars[ds_name]['EEI'],myVars[ds_name])
                else:
                    global_mean = atmos_horiz_mean(myVars[ds_name][anom_var],myVars[ds_name])

            time = np.linspace(start_year,end_year,num=len(global_mean) )
            
            if roll_mean:
                global_mean = global_mean.rolling(time=roll_mean_window, center=True).mean()
                
            ax.plot(time,global_mean,label='control',color='k')

        ax.set_xlabel("Time (Years)")
        ax.set_ylabel(cb_label)
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
            title_str = f"1pct2xCO\u2082 Radiative $\Delta Q$: {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-const_{power}"
            
        elif diff_type == 'doub-2xctrl':
            title_str = f"1pct2xCO\u2082 Mixing $\Delta Q$: {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-2xctrl_{power}"
            
        elif diff_type == 'doub-1860ctrl':
            title_str = f"1pct2xCO\u2082 Total $\Delta Q$: {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2-const-ctrl_{power}"
            
        elif diff_type == 'quad-1860exp':
            title_str = f"1pct4xCO\u2082 Radiative $\Delta Q$: {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-const_{power}"
            
        elif diff_type == 'quad-4xctrl':
            title_str = f"1pct4xCO\u2082 Mixing $\Delta Q$: {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-4xctrl_{power}"
            
        elif diff_type == 'quad-1860ctrl':
            title_str = f"1pct4xCO\u2082 Total $\Delta Q$: {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2-const-ctrl_{power}"

        if omit_title is False:
            ax.set_title(title_str+title_suff)
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if fig_pref != None:
            plt.savefig(fig_dir+f'{fig_pref}_{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'{anom_var}_anom_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')


def plot_atmos_ts_2d(co2_scen,fig_dir,start_year,end_year,
                     omit_title=True, roll_mean = True, roll_mean_window = 10,
                     var="olr",title_suff="Mean OLR",fig_pref=None,
                     cb_label = "Flux Anomaly (W/m$^2$)",
                     offline_mean=False,
                     ylimits = None,
                 # ylimits = [3.0,5.0],
                 leg_loc = 'upper left',
                 leg_ncols = 1,
                       profiles = ['surf','therm','mid','bot'], 
                       power_inputs = ['0.1TW', '0.2TW', '0.3TW'], 
                       power_var_suff = ['0p1TW', '0p2TW', '0p3TW'], 
                       power_strings = ['0.1 TW', '0.2 TW', '0.3 TW']):
    """
    Function to plot raw time series variable (not anomaly).
    
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

    # separate plot for each power input
    for pow_idx, power in enumerate(power_var_suff):
        fig, ax = plt.subplots(figsize=(6,3))

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ds_root = f'{power}_{start_year}_{end_year}'
        for prof in profiles:
            ds_name = f'{co2_scen}_{prof}_{ds_root}'
            if offline_mean:
                ds_name = ds_name + "_mean"
                global_mean = myVars[ds_name][var]
            else:
                if var == 'EEI':
                    myVars[ds_name]['EEI'] = myVars[ds_name]['swdn_toa'] - myVars[ds_name]['swup_toa'] - myVars[ds_name]['olr']
                    global_mean = atmos_horiz_mean(myVars[ds_name]['EEI'],myVars[ds_name])
                else:
                    global_mean = atmos_horiz_mean(myVars[ds_name][var],myVars[ds_name])
                
            time = np.linspace(start_year,end_year,num=len(global_mean) )

            if roll_mean:
                global_mean = global_mean.rolling(time=roll_mean_window, center=True).mean()
                
            ax.plot(time,global_mean,label=prof,color=prof_dict[prof])

        # control
        ds_name = f'{co2_scen}_ctrl_{start_year}_{end_year}'
        if offline_mean:
            ds_name = ds_name + "_mean"
            global_mean = myVars[ds_name][var]
        else:
            if var == 'EEI':
                myVars[ds_name]['EEI'] = myVars[ds_name]['swdn_toa'] - myVars[ds_name]['swup_toa'] - myVars[ds_name]['olr']
                global_mean = atmos_horiz_mean(myVars[ds_name]['EEI'],myVars[ds_name])
            else:
                global_mean = atmos_horiz_mean(myVars[ds_name][var],myVars[ds_name])
        
        time = np.linspace(start_year,end_year,num=len(global_mean) )
        
        if roll_mean:
            global_mean = global_mean.rolling(time=roll_mean_window, center=True).mean()
            
        ax.plot(time,global_mean,label='control',color='k')
        ax.set_xlabel("Time (Years)")
        ax.set_ylabel(cb_label)
        ax.set_xlim(start_year-1,end_year)
        if ylimits:
            ax.set_ylim(ylimits)
        ax.legend(loc=leg_loc,ncols=leg_ncols)
        ax.grid("both")
    
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

        if co2_scen == 'const':
            title_str = f"Const CO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"const_{power}"
            
        elif co2_scen == 'doub':
            title_str = f"1pct2xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"2xCO2_{power}"
            
        elif co2_scen == 'quad':
            title_str = f"1pct4xCO\u2082 {power_strings[pow_idx]} Cases\n"
            fig_name = f"4xCO2_{power}"

        if omit_title is False:
            ax.set_title(title_str+title_suff)
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        
        if fig_pref != None:
            plt.savefig(fig_dir+f'{fig_pref}_{var}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')
        else:
            plt.savefig(fig_dir+f'{var}_{fig_name}_{start_year}_{end_year}.pdf', dpi=600, bbox_inches='tight')




