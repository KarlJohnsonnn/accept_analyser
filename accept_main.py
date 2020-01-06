#!/usr/bin/env python3
"""
Short description:
   Analyser for ACCEPT field campaign data, for detailed evaluation of Ed Luke et at. 2010 ANN approach for prediction of supercooled liquid layers in clouds
   cloud radar and polarization lidar observations.

"""

import os
import sys

from itertools import combinations

# disable the OpenMP warnings

os.environ['KMP_WARNINGS'] = 'off'
sys.path.append('../larda/')

import pyLARDA.helpers as h
import pyLARDA.Transformations as tr

from utility import *

import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2020, ACCEPT Field Campaign Analyser"
__credits__ = ["Willi Schimmel", "Heike Kalesse"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

########################################################################################################################################################
########################################################################################################################################################
#
#
#               _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#               |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#               |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
########################################################################################################################################################
########################################################################################################################################################


oct_cases = 'accept_cases_oct.csv'
nov_cases = 'accept_cases_nov.csv'

#BASE_DIR = '/media/sdig/LACROS/cloudnet/data/accept/matfiles/timeseries/'      # workstation path
BASE_DIR = '/Users/willi/data/MeteoData/ACCEPT/timeseries/'     # local path
PLOTS_PATH = f'{__file__[:__file__.rfind("/") + 1]}plots/'


# choose which files to load
do_read_cloudnet_class = True
do_read_cloudnet_categ = True  # contains MWR LWP
do_read_polly          = False  #
do_read_lipred         = True  # lipred = lidar prediction from NN
do_read_mira_spec_mom  = False  # vertically pointing MIRA radar spectra
do_read_sounding       = False
do_read_GDAS1          = False

span_smoo_NNout  = 5    # number of range gates for smoothing (1 = 30m)
span_smoo_MWRlwp = 10   # number of time steps for smoothing (1 = 30sec)

# {'disdro', 'anydrizzle', 'rg0drizzle'} ... exclude all profiles, where the disdrometer or cloudnet classifies any or at range gate zero a pixel as drizzle
exclude_drizzle = 'rg0drizzle'     

# exclude all pixels below the first cloud base detection by cloudnet
exclude_below_ceilo_cb = True

# remove x seconds after last rain flag
exclude_wet_radome_lwp = 600      

# {'wrm', 'scl', 'both'} ... exclude all pixel except those including {warm droplets, 'super cooled droplets', 'both warm and super cooled droplets'}
exclude_all_but = 'both'

# {one element of isotherm_list} ... exclude all pixels below this temperature
exclude_below_temperature = -38

plot_size_2D = [10, 8]
plot_target_classification    = True
plot_liquid_pixel_masks       = True
plot_cloudnet_radar_moments   = True
plot_cloudnet_lidar_variables = True
plot_cloudnet_mwr_lwp         = True
plot_scatter_depol_bsc        = True
plot_FoO                      = True

if __name__ == '__main__':

    start_time = time.time()
    case_list  = cases_from_csv(oct_cases)

    # loading mat files
    classification_tot = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_classification.mat', 'classification') if do_read_cloudnet_class   else None
    categorization_tot = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_categorization.mat', 'categorization') if do_read_cloudnet_categ   else None
    pollyXT_tot        = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_polly.mat', 'PollyXT') if do_read_polly else  None
    hsrl_pred_tot      = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_hsrl_prediction_30s.mat', 'HSRL prediction') if do_read_lipred else None
    mira_moments_tot   = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_moments_30s.mat', 'MIRA moments') if do_read_mira_spec_mom else None
    radiosondes_tot    = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_radiosondes_mean.mat', 'radiosondes') if do_read_sounding else None

    # load range and timesteps and reshape to an 1D array
    n_tot_ts_class = classification_tot['ts_class_time'].shape[0]
    n_tot_rg_class = classification_tot['h_class'].shape[0]
    cloudnet_dn    = classification_tot['ts_class_time'].reshape((n_tot_ts_class,))
    cloudnet_dt    = np.array([datenum2datetime(ts) for ts in cloudnet_dn])
    cloudnet_rg    = classification_tot['h_class'].reshape((n_tot_rg_class,))

    for case in case_list[15:16]:
        # if case['notes'] == 'ex': continue  # exclude this case and check the next one

        begin_dt, end_dt = case['begin_dt'], case['end_dt']

        # create directory for plots
        h.change_dir(f'{PLOTS_PATH}case_study_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}/')
        logging.basicConfig(filename=f'case_study_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}.log', level=logging.INFO)

        # find indices for slicing
        rg0val, rgNval = case['plot_range']
        rg0idx, rgNidx = h.argnearest(cloudnet_rg, rg0val), h.argnearest(cloudnet_rg, rgNval)
        rgN = rgNidx - rg0idx

        ts0val, tsNval = datetime2datenum(begin_dt), datetime2datenum(end_dt)
        ts0idx, tsNidx = h.argnearest(cloudnet_dn, ts0val), h.argnearest(cloudnet_dn, tsNval)
        tsN = tsNidx - ts0idx

        assert tsN * rgN > 0, ValueError('Error occurred! Number of time steps or range bins invalid, check rgN and tsN!')

        ts_smth = '_smoothed' if span_smoo_MWRlwp > 0 else ''

        logger.info(f'\n*********BEGIN CASE STUDY*********\n')
        logger.info(f'Slicing time from {begin_dt:%Y-%m-%d %H:%M:%S} (UTC) to {end_dt:%Y-%m-%d %H:%M:%S} (UTC)')
        logger.info(f'Slicing range from {rg0val:.2f} (km) to {rgNval:.2f} (km)\n')
        logger.info(f'Global parameter settings:')
        logger.info(f'  Smoothing:')
        logger.info(f'      - smooth NN output = {span_smoo_NNout>0 },    span_smoo_NNout  = {span_smoo_NNout}')
        logger.info(f'      - smooth MWR LWP   = {span_smoo_MWRlwp>0},    span_smoo_MWRlwp = {span_smoo_MWRlwp}')
        logger.info(f'  Excluding data:')
        logger.info(f'      - exclude_drizzle           = {exclude_drizzle}')
        logger.info(f'      - exclude_below_ceilo_cb    = {exclude_below_ceilo_cb}')
        logger.info(f'      - exclude_wet_radome_lwp    = {exclude_wet_radome_lwp} [sec] after last rain flag')
        logger.info(f'      - exclude_all_but           = {exclude_all_but} [liquid pixel]')
        logger.info(f'      - exclude_below_temperature = {exclude_below_temperature} [degC]\n')

        classification = time_height_slicer(classification_tot, [ts0idx, tsNidx], [rg0idx, rgNidx]) if do_read_cloudnet_class else None
        categorization = time_height_slicer(categorization_tot, [ts0idx, tsNidx], [rg0idx, rgNidx]) if do_read_cloudnet_categ else None
        pollyXT        = time_height_slicer(pollyXT_tot,        [ts0idx, tsNidx], [rg0idx, rgNidx]) if do_read_polly          else None
        hsrl_pred      = time_height_slicer(hsrl_pred_tot,      [ts0idx, tsNidx], [rg0idx, rgNidx]) if do_read_lipred         else None
        mira_moments   = time_height_slicer(mira_moments_tot,   [ts0idx, tsNidx], [rg0idx, rgNidx]) if do_read_mira_spec_mom  else None
        radiosondes    = time_height_slicer(radiosondes_tot,    [ts0idx, tsNidx], [rg0idx, rgNidx]) if do_read_sounding       else None

        T = tr.combine(toC, [wrapper(categorization, var_name='T_mod_ts', var_unit='C', var_lims=[-50, 20])], {'var_unit': "C"})
        contour = {'data': T, 'levels': isotherm_list}

        # add indices of first cloud base to classification dict
        classification.update({'first_cb_idx': get_1st_cloud_base_idx(classification['cb_first_ts'], cloudnet_rg)})

        # extract isotherm lines, indices, temperatures at index and range at index
        isotherm_lines = {f'{iTemp}degC': get_temp_lines(categorization['T_mod_ts'], cloudnet_rg, iTemp) for iTemp in isotherm_list}

        # assuming this works with nan values
        # span = number of data points used for smoothing of NN BSC + depol before liquid mask creation; here: 5 profiles at 30s resolution = 2.5min
        if rgN > span_smoo_NNout:
            for iH in range(rgN):
                hsrl_pred['bsc_NN_ts'][iH, :]  = h.smooth(hsrl_pred['bsc_NN_ts'][iH, :], span_smoo_NNout)
                hsrl_pred['CDR_NN_ts'][iH, :]  = h.smooth(hsrl_pred['CDR_NN_ts'][iH, :], span_smoo_NNout)
                hsrl_pred['dpol_NN_ts'][iH, :] = h.smooth(hsrl_pred['dpol_NN_ts'][iH, :], span_smoo_NNout)

        # number of data points (profiles) used for smoothing time series data of MWR-LWP and predicted liquid layer thickness;
        # here: 4h*60min*2profiles/min % E.Luke: 4hrs
        if tsN > span_smoo_MWRlwp:
            categorization.update({'lwp_ts_smoothed': h.smooth(categorization['lwp_ts'], span_smoo_MWRlwp)})

        # loop over the different box thresholds
        for ithresh_name, ithresh_val in lidar_thresh_dict.items():
            """ _  _ ____ _  _ ____ ____ _       _  _ ____ ___ _ _ _ ____ ____ _  _    _    _ ____ _  _ _ ___     _  _ ____ ____ _  _ 
                |\ | |___ |  | |__/ |__| |       |\ | |___  |  | | | |  | |__/ |_/     |    | |  | |  | | |  \    |\/| |__| [__  |_/  
                | \| |___ |__| |  \ |  | |___    | \| |___  |  |_|_| |__| |  \ | \_    |___ | |_\| |__| | |__/    |  | |  | ___] | \_ 
                                                                                                                      
                ann_liq_mask    % (range, time) liquid mask where True == liquid, False == non-liquid, depends on the selected threshold box/linfcn

                Caution! Warnings for invalid values disabled!
            """
            
            # derive liquid water mask using different box-thresholds
            logger.info('\n' + '*' * 80 + '\n')
            ann_liquid_mask = np.full(hsrl_pred['bsc_NN_ts'].shape, False)
            if ithresh_name == 'linear':
                dpl_xaxis = np.linspace(0, 0.3, 120)
                bsc_yaxis = ithresh_val['slope'] * dpl_xaxis + ithresh_val['intersection']
                # nested loop avoidable?
                for irg in range(rgN):
                    for its in range(tsN):
                        if np.isnan(hsrl_pred['dpol_NN_ts'][irg, its]):
                            continue
                        idx_nearest = h.argnearest(bsc_yaxis, hsrl_pred['bsc_NN_ts'][irg, its])

                        if hsrl_pred['bsc_NN_ts'][irg, its] > bsc_yaxis[idx_nearest] and hsrl_pred['dpol_NN_ts'][irg, its] < dpl_xaxis[idx_nearest]:
                            ann_liquid_mask[irg, its] = True

                logger.info(f'Threshold for liquid classification: {ithresh_name} '
                            f'with slope = {ithresh_val["slope"]:.2f} and intersection = '
                            f'{ithresh_val["intersection"]:.2f}\n')
            else:
                with np.errstate(invalid='ignore'):
                    ann_liquid_mask[np.logical_and((hsrl_pred['bsc_NN_ts'] > ithresh_val['bsc']),
                                                   (hsrl_pred['CDR_NN_ts'] < ithresh_val['dpl']))] = True
                logger.info(f'Threshold for liquid classification: {ithresh_name} '
                            f'with bsc > {ithresh_val["bsc"]:.2e} and '
                            f'depol > {ithresh_val["dpl"]:.2e}\n')


            """ ____ _    ____ _  _ ___  _  _ ____ ___    _    _ ____ _  _ _ ___     _  _ ____ ____ _  _ 
                |    |    |  | |  | |  \ |\ | |___  |     |    | |  | |  | | |  \    |\/| |__| [__  |_/  
                |___ |___ |__| |__| |__/ | \| |___  |     |___ | |_\| |__| | |__/    |  | |  | ___] | \_ 
                       
                categories contains:
                                                                                         
                % 0 = "not-a-category"
                % 1 = cloud droplets only
                % 2 = drizzle/rain
                % 3 = drizzle/rain + cloud droplets
                % 4 = ice
                % 5 = ice + supercooled liquid
                % 6 = melting ice
                % 7 = melting ice + cloud liquid droplets
                % 8 = aerosol
                % 9 = insects
                % 10 = aerosol + insects
                cloudnet_liq_mask % combination of all Cloudnet classes containing only scl (excluding rain and scl below -38°C)
                combi_liq_mask    % masks filtered below isotherms where 0=no liquid, 1=ann and cloudnet detecting liquid, 2=ann only, 3=cloudnet only
                
                Caution! Warnings for invalid values disabled!
                
            """
            categories = [classification['target_class_ts'] == iclass for iclass in range(1, 11)]
            categories.insert(0, 'not-a-category')

            # initialize the (range, time) liquid mask for cloudnet
            cloudnet_liq_mask = np.full((rgN, tsN), False)
            cloudnet_liq_mask[get_indices(categories, liq_mask_flags[exclude_all_but])] = True

            """ ____ ____ _  _ ___  _ _  _ ____ ___     _    _ ____ _  _ _ ___     _  _ ____ ____ _  _ 
                |    |  | |\/| |__] | |\ | |___ |  \    |    | |  | |  | | |  \    |\/| |__| [__  |_/  
                |___ |__| |  | |__] | | \| |___ |__/    |___ | |_\| |__| | |__/    |  | |  | ___] | \_ 
                
            """
            # initialize combined (range, time) liquid mask, containing all detected liquid pixels
            combi_liq_mask = {'tot': get_combined_liquid_mask(ann_liquid_mask, cloudnet_liq_mask)}

            # add liquid masks for different isotherm thresholds
            combi_liq_mask.update({key: mask_below_temperature(combi_liq_mask['tot'].copy(), val) for key, val in isotherm_lines.items()})

            tot_nnz = np.count_nonzero(combi_liq_mask['tot'])
            logger.info(f'nr. of pxl where Cloudnet and/or NN classify liquid after  aerosol/insect removal  : {tot_nnz}')

            """ ____ _  _ ____ _    _  _ ___  ____    _  _ _  _ _ _ _ ____ _  _ ___ ____ ___     ___  _ _  _ ____ _    
                |___  \/  |    |    |  | |  \ |___    |  | |\ | | | | |__| |\ |  |  |___ |  \    |__] |  \/  |___ |    
                |___ _/\_ |___ |___ |__| |__/ |___    |__| | \| |_|_| |  | | \|  |  |___ |__/    |    | _/\_ |___ |___ 
                                                                                                        
            """
            # exctract different rain flags, ignore invalid value warnings
            with np.errstate(invalid='ignore'):
                rain_flag = {'disdro': categorization['rainrate_ts'] > 0.0,
                             'anydrizzle': categories[2].any(axis=0),
                             'rg0drizzle': categories[2][0, :].copy()}

            # remove drizzle/rain profiles from all masks 
            if exclude_drizzle in ['disdro', 'anydrizzle', 'rg0drizzle']:
                ann_liquid_mask[:, rain_flag[exclude_drizzle]] = False
                cloudnet_liq_mask[:, rain_flag[exclude_drizzle]] = False
                for key in combi_liq_mask.keys():
                    combi_liq_mask[key][:, rain_flag[exclude_drizzle]] = 0

            # remove all pxl below ceilo base (rain!)
            if exclude_below_ceilo_cb:
                for iT in range(tsN):
                    idx = classification['first_cb_idx'][iT] if not np.isnan(classification['first_cb_idx'][iT]) else 0
                    ann_liquid_mask[:int(idx), iT] = False
                    cloudnet_liq_mask[:int(idx), iT] = False
                    for key in combi_liq_mask.keys():
                        combi_liq_mask[key][:int(idx), iT] = 0

            # remove all pxl below a certain temperature
            if exclude_below_temperature in isotherm_list:
                ann_liquid_mask[combi_liq_mask[f'{exclude_below_temperature}degC'] == 0] = False
                cloudnet_liq_mask[combi_liq_mask[f'{exclude_below_temperature}degC'] == 0] = False

            """ ____ ___  ___  _ ___ _ ____ _  _ ____ _       _ _  _ ____ ____ ____ _  _ ____ ___ _ ____ _  _ 
                |__| |  \ |  \ |  |  | |  | |\ | |__| |       | |\ | |___ |  | |__/ |\/| |__|  |  | |  | |\ | 
                |  | |__/ |__/ |  |  | |__| | \| |  | |___    | | \| |    |__| |  \ |  | |  |  |  | |__| | \| 
                                                                                              
            """
            # calculate layer thickness for all different categories
            sum_ll_thickness = sum_liquid_layer_thickness_per_category(cloudnet_liq_mask, ann_liquid_mask, combi_liq_mask, rg_res=30.0)

            if span_smoo_MWRlwp > 0:
                sum_ll_thickness.update({f'{var}_smoothed': h.smooth(sum_ll_thickness[var], span_smoo_MWRlwp) for var in sum_ll_thickness.keys()})

            # count amount of liquid pixels for each combination
            key, val = np.unique(combi_liq_mask['tot'], return_counts=True)
            combi_mask_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            combi_mask_counts.update(zip(key, val))

            logger.info('\n----- (comparison of amount of pxl classified as liquid by Cloudnet or NN output) --------')
            logger.info(f'overlapping pxl where NN + Cloudnet detected liquid : {combi_mask_counts[1] * 100 / tot_nnz:.2f} %')
            logger.info(f'overlapping pxl where ONLY NN predicts liquid       : {combi_mask_counts[2] * 100 / tot_nnz:.2f} %')
            logger.info(f'overlapping pxl where ONLY Cloudnet detected liquid : {combi_mask_counts[3] * 100 / tot_nnz:.2f} %\n')

            categorization[f"lwp_ts{ts_smth}"] = np.ma.masked_invalid(categorization[f"lwp_ts{ts_smth}"])
            corr_coefs = {f'lwp-cn{ts_smth}': np.ma.corrcoef(categorization[f"lwp_ts{ts_smth}"], sum_ll_thickness[f"cloudnet{ts_smth}"]),
                          f'lwp-nn{ts_smth}': np.ma.corrcoef(categorization[f"lwp_ts{ts_smth}"], sum_ll_thickness[f"neuralnet{ts_smth}"])}
            logger.info(f'correlation between MWR-LWP{ts_smth} and Cloudnet ll-thickness{ts_smth}  : {corr_coefs[f"lwp-cn{ts_smth}"][0,1]:.2f}')
            logger.info(f'correlation between MWR-LWP{ts_smth} and neuralnet ll-thickness{ts_smth} : {corr_coefs[f"lwp-nn{ts_smth}"][0,1]:.2f}')


            # plotting inside the lidar threshold loop
            if plot_liquid_pixel_masks:
                fig_name = f'LIQ-MASK_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT_{ithresh_name}.png'
                classification.update({'combi_liq_mask': combi_liq_mask[f'{exclude_below_temperature}degC']})
                fig, _ = tr.plot_timeheight(wrapper(classification, var_name='combi_liq_mask', var_unit='-'), title=fig_name,
                                            range_interval=case['plot_range'], fig_size=plot_size_2D, contour=contour)
                fig.savefig(fig_name)
                logger.info(f'Figure saved :: {fig_name}')
            if plot_FoO:
                # plot every combination of variabels against each other, except var1 == var2
                for (x_varname, x_info), (y_varname, y_info) in combinations(variable_dict.items(), 2):
                    if x_varname != y_varname:
                        titlestring = f'FoO  {x_varname}  VS  {y_varname}  --  ACCEPT\n{begin_dt:%Y-%m-%d %H:%M:%S} - {end_dt:%Y-%m-%d %H:%M:%S} [UTC]'
                        x_var = wrapper({**hsrl_pred, **categorization}, var_name=x_varname, var_unit=x_info[0])
                        y_var = wrapper({**hsrl_pred, **categorization}, var_name=y_varname, var_unit=y_info[0])

                        if exclude_all_but in liq_mask_flags:
                            x_var['mask'][cloudnet_liq_mask.T == False] = True
                            y_var['mask'][cloudnet_liq_mask.T == False] = True

                        fig, ax = tr.plot_scatter(x_var, y_var, fig_size=[8, 8], x_lim=x_info[1], y_lim=y_info[1], title=titlestring, colorbar=True)
                        fig_name = f'FoO-{ithresh_name}-{x_varname}-vs-{y_varname}_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
                        fig.savefig(fig_name)
                        logger.info(f'Figure saved :: {fig_name}')
            if plot_cloudnet_mwr_lwp:
                fig_name = f'CN-mwr-lwp-{ithresh_name}_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
                titlestring = f'MWR-LWP versus Cloudnet or ANN liquid layer thickness -- ACCEPT \n'\
                              f' date: {begin_dt:%Y-%m-%d %H:%M:%S} till {end_dt:%Y-%m-%d %H:%M:%S} -- threshold: {ithresh_name}'
                fig, ax = tr.plot_timeseries(wrapper(categorization, var_name=f'lwp_ts{ts_smth}', var_unit='g m-2', var_lims=[0, 500]),
                                             fig_size=plot_size_2D, linewidth=3.5, alpha=0.9)
                ax.legend(['MWR-LWP'], loc='upper left', prop={'size': 15, 'weight': 'semibold'})
                ax1 = add_ll_thichkness(ax, cloudnet_dt[ts0idx:tsNidx], sum_ll_thickness, smooth=True)
                ax1.legend(['cloudnet', 'neural net'], loc='upper right', prop={'size': 15, 'weight': 'semibold'})
                ax1.set_title(titlestring, fontsize=15, fontweight='semibold')
                corr_coef_str = r'$\rho_{MWR, CN}=$' + f'{corr_coefs[f"lwp-cn{ts_smth}"][0, 1]:.2f}\n' \
                                                       r'$\rho_{MWR, NN}=$' + f'{corr_coefs[f"lwp-nn{ts_smth}"][0, 1]:.2f}'
                ax.text(0.4, .85, corr_coef_str, horizontalalignment='left', transform=ax.transAxes, fontsize=15,
                        bbox=dict(facecolor='white', alpha=0.75))
                plt.show()
                fig.savefig(fig_name)
                logger.info(f'Figure saved :: {fig_name}')


        """
        ************************************************************************************************************************************
             _____          _____  _______ _______ _____ __   _  ______      _______ _______ _______ _______ _____  _____  __   _
            |_____] |      |     |    |       |      |   | \  | |  ____      |______ |______ |          |      |   |     | | \  |
            |       |_____ |_____|    |       |    __|__ |  \_| |_____|      ______| |______ |_____     |    __|__ |_____| |  \_|

        ************************************************************************************************************************************
        """
        # outside of the lidar threshold loop
        logger.info('')
        logger.info('Plotting radar moments, lidar variables, lwp, etc.:')
        if plot_cloudnet_radar_moments:
            fig_name = f'CN-radar-0-Ze_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
            fig, _ = tr.plot_timeheight(wrapper(categorization, var_name='Ze_cc_ts', var_unit='dBZ', var_lims=[-60, 20]),
                                        range_interval=case['plot_range'], fig_size=plot_size_2D, contour=contour)
            fig.savefig(fig_name)
            logger.info(f'Figure saved :: {fig_name}')

            fig_name = f'CN-radar-1-VEL_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
            fig, _ = tr.plot_timeheight(wrapper(categorization, var_name='Vd_cc_ts', var_unit='m s-1', var_lims=[-4, 2]),
                                        range_interval=case['plot_range'], fig_size=plot_size_2D, contour=contour)
            fig.savefig(fig_name)
            logger.info(f'Figure saved :: {fig_name}')

            fig_name = f'CN-radar-2-sw_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
            fig, _ = tr.plot_timeheight(wrapper(categorization, var_name='width_cc_ts', var_unit='m s-1', var_lims=[0, 2]),
                                        range_interval=case['plot_range'], fig_size=plot_size_2D, contour=contour)
            fig.savefig(fig_name)
            logger.info(f'Figure saved :: {fig_name}')

            fig_name = f'CN-radar-X-ldr_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
            fig, _ = tr.plot_timeheight(wrapper(categorization, var_name='ldr_cc_ts', var_unit='dB', var_lims=[-30, 0]),
                                        range_interval=case['plot_range'], fig_size=plot_size_2D, contour=contour)
            fig.savefig(fig_name)
            logger.info(f'Figure saved :: {fig_name}')
        if plot_cloudnet_lidar_variables:
            fig_name = f'CN-lidar-bsc_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
            fig, _ = tr.plot_timeheight(wrapper(categorization, var_name='att_bscatt_ts', var_unit='m-1 sr-1', var_lims=[1.e-7, 1.e-4]),
                                        range_interval=case['plot_range'], fig_size=plot_size_2D, var_converter='log', contour=contour)
            fig.savefig(fig_name)
            logger.info(f'Figure saved :: {fig_name}')
        if plot_target_classification:
            fig_name = f'CLASS_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
            fig, _ = tr.plot_timeheight(wrapper(classification, var_name='target_class_ts', var_unit='-'),
                                        range_interval=case['plot_range'], fig_size=plot_size_2D, contour=contour)
            fig.savefig(fig_name)
            logger.info(f'Figure saved :: {fig_name}')
        if plot_scatter_depol_bsc:
            titlestring = f'FoO predicted depol vs. attbsc -- ACCEPT\ndate: {begin_dt:%Y-%m-%d};   {begin_dt:%H:%M:%S} - {end_dt:%H:%M:%S} [UTC]'
            pred_bsc = wrapper(hsrl_pred, var_name='bsc_NN_ts', var_unit='log(sr-1 m-1)')
            pred_dpl = wrapper(hsrl_pred, var_name='dpol_NN_ts', var_unit='1')

            if exclude_all_but in liq_mask_flags:
                pred_dpl['mask'][cloudnet_liq_mask.T == False] = True
                pred_bsc['mask'][cloudnet_liq_mask.T == False] = True

            fig, ax = tr.plot_scatter(pred_dpl, pred_bsc, fig_size=[8, 8], x_lim=[0, 0.3], y_lim=[-6, -2.5], title=titlestring, colorbar=True)
            ax      = add_boxes(ax, lidar_thresh_dict, **{'size': 15, 'weight': 'semibold'})
            fig_name = f'FoO-ANN-dpl-vs-bsc_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
            fig.savefig(fig_name)
            logger.info(f'Figure saved :: {fig_name}')

    dummy = 5
