#!/usr/bin/env python3
"""
Short description:
   Analyser for ACCEPT field campaign data, for detailed evaluation of Ed Luke et at. 2010 ANN approach for prediction of supercooled liquid layers in clouds
   cloud radar and polarization lidar observations.

"""

import os
import sys
import logging
import time

import numpy as np

import matplotlib.pyplot as plt

# disable the OpenMP warnings
from utility import get_1st_cloud_base_idx

os.environ['KMP_WARNINGS'] = 'off'
sys.path.append('../larda/')

import pyLARDA.helpers as h
import pyLARDA.Transformations as tr

from utility import *

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

logger = logging.getLogger('accept_analyser')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

oct_cases = 'accept_cases_oct.csv'
nov_cases = 'accept_cases_nov.csv'

#BASE_DIR = '/media/sdig/LACROS/cloudnet/data/accept/matfiles/timeseries/'      # workstation path
BASE_DIR = '/Users/willi/data/MeteoData/ACCEPT/timeseries/'     # local path
PLOTS_PATH = f'{__file__[:__file__.rfind("/") + 1]}plots/'

lidar_box_thresh = {'Luke': {'bsc': -4.45, 'dpl': -7.422612476299660e-01},
                    'deBoer': {'bsc': -4.3, 'dpl': -1.208620483882601e+00},
                    'Shupe': {'bsc': -4.5, 'dpl': -6.532125137753436e-01},
                    'Cloudnet': {'bsc': -4.7, 'dpl': 20.000000000000000e+00},
                    #'Willi': {'bsc': -5.2, 'dpl': -6.532125137753436e-01}
                    }

lidar_lin_thresh = {'linear': {'slope': 10, 'intersection': -6}}

isotherm_list = [-38, -25, -10, 0]  # list must be increasing

# choose which files to load
do_read_cloudnet_class = True
do_read_cloudnet_categ = True  # contains MWR LWP
do_read_polly          = False  #
do_read_lipred         = True  # lipred = lidar prediction from NN
do_read_mira_spec_mom  = False  # vertically pointing MIRA radar spectra
do_read_sounding       = False
do_read_GDAS1          = False

do_smooth_NNoutput = True
span_smoo_NNout = 5

do_smooth_MWRlwp = True
span_smoo_MWRlwp = 480

remove_drizzle = 'rg0drizz'

plot_size_2D = [8, 5]
plot_target_classification      = True
plot_liquid_pixel_masks         = True
plot_cloudnet_radar_moments     = True
plot_cloudnet_lidar_variables   = True
plot_cloudnet_mwr_lwp           = True
plot_scatter_depol_bsc          = True

if __name__ == '__main__':

    start_time = time.time()

    #    log = logging.getLogger('pyLARDA')
    #    log.setLevel(logging.CRITICAL)
    #    log.addHandler(logging.StreamHandler())
    #
    #    larda = pyLARDA.LARDA().connect('lacros_accept_gpu', build_lists=False)

    logger.info('\n-------------- ACCEPT ANALYSER --------------\n')

    case_list = cases_from_csv(oct_cases)
    # pprint.plogger.info(case_list)

    # loading mat files
    classification_tot = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_classification.mat', 'classification') if do_read_cloudnet_class   else None
    categorization_tot = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_categorization.mat', 'categorization') if do_read_cloudnet_categ   else None
    pollyXT_tot        = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_polly.mat', 'PollyXT') if do_read_polly else  None
    hsrl_pred_tot      = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_hsrl_prediction_30s.mat', 'HSRL prediction') if do_read_lipred else None
    #mira_moments_tot   = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_moments_from_spectra_30s.mat', 'MIRA moments') if do_read_mira_spec_mom else
    # None
    radiosondes_tot    = load_dot_mat_file(f'{BASE_DIR}ACCEPT_20141005-20141118_radiosondes_mean.mat', 'radiosondes') if do_read_sounding else None

    # load range and timesteps and reshape to an 1D array
    n_tot_ts_class = classification_tot['ts_class_time'].shape[0]
    n_tot_rg_class = classification_tot['h_class'].shape[0]
    cloudnet_dn    = classification_tot['ts_class_time'].reshape((n_tot_ts_class,))
    cloudnet_dt    = np.array([h.ts_to_dt(ts) for ts in cloudnet_dn])
    cloudnet_rg    = classification_tot['h_class'].reshape((n_tot_rg_class,))

    for case in case_list[25:26]:
        # if case['notes'] == 'ex': continue  # exclude this case and check the next one

        begin_dt, end_dt = case['begin_dt'], case['end_dt']

        # create directory for plots
        h.change_dir(f'{PLOTS_PATH}case_study_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}/')

        # find indices for slicing
        rg0val, rgNval = case['plot_range']
        rg0idx, rgNidx = h.argnearest(cloudnet_rg, rg0val), h.argnearest(cloudnet_rg, rgNval)
        rgN = rgNidx - rg0idx

        ts0val, tsNval = datetime2datenum(begin_dt), datetime2datenum(end_dt)
        ts0idx, tsNidx = h.argnearest(cloudnet_dn, ts0val), h.argnearest(cloudnet_dn, tsNval)
        tsN = tsNidx - ts0idx

        assert tsN * rgN > 0, ValueError('Error occurred! Number of time steps or range bins invalid, check rgN and tsN!')

        logger.info(f'\n*********BEGIN CASE STUDY*********\n')
        logger.info(f'Slicing time from {begin_dt:%Y-%m-%d %H:%M:%S} (UTC) to {end_dt:%Y-%m-%d %H:%M:%S} (UTC)')
        logger.info(f'Slicing range from {rg0val:.2f} (km) to {rgNval:.2f} (km)')

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
        if do_smooth_NNoutput and rgN > span_smoo_NNout:
            for iH in range(rgN):
                hsrl_pred['bsc_NN_ts'][iH, :]  = h.smooth(hsrl_pred['bsc_NN_ts'][iH, :], span_smoo_NNout)
                hsrl_pred['CDR_NN_ts'][iH, :]  = h.smooth(hsrl_pred['CDR_NN_ts'][iH, :], span_smoo_NNout)
                hsrl_pred['dpol_NN_ts'][iH, :] = h.smooth(hsrl_pred['dpol_NN_ts'][iH, :], span_smoo_NNout)

        # number of data points (profiles) used for smoothing time series data of MWR-LWP and predicted liquid layer thickness;
        # here: 4h*60min*2profiles/min % E.Luke: 4hrs
        if do_smooth_MWRlwp and tsN > span_smoo_MWRlwp:
            categorization.update({'lwp_ts_smoothed': h.smooth(categorization['lwp_ts'], span_smoo_MWRlwp)})

        # loop over the different box thresholds
        for ithresh_name, ithresh_val in lidar_box_thresh.items():

            logger.info(f'threshold for liquid classification: {ithresh_name} with bsc > {ithresh_val["bsc"]:.2e} and depol > {ithresh_val["dpl"]:.2e}\n')

            # derive liquid water mask using different box-thresholds
            ann_liquid_mask = np.full(hsrl_pred['bsc_NN_ts'].shape, fill_value=False)
            if ithresh_name == 'linear':
                raise ValueError('linear threshold not implemented jet')
            else:
                with np.errstate(invalid='ignore'):
                    ann_liquid_mask[np.logical_and((hsrl_pred['bsc_NN_ts'] > ithresh_val['bsc']), (hsrl_pred['CDR_NN_ts'] < ithresh_val['dpl']))] = True

            # remove all pxl below ceilo base (rain!)
            for iT in range(tsN):
                idx = classification['first_cb_idx'][iT] if not np.isnan(classification['first_cb_idx'][iT]) else 0
                ann_liquid_mask[:int(idx), iT] = False

            """CLOUTNET TARGET CATEGORIES + EXTRAS
                % "not-a-category"
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
                cloudnet_liq_mask % combination of all Cloudnet classes containing warm liquid and scl (excluding rain and scl below -38°C)
                cloudnet_scl_mask % combination of all Cloudnet classes containing only scl (excluding rain and scl below -38°C)
                combi_mask_liq    % masks filtered below isotherms where 0=no liquid, 1=ann and cloudnet detecting liquid, 2=ann only, 3=cloudnet only
                
                Caution! Warnings for invalid values disabled!
            """
            categories = [classification['target_class_ts'] == iclass for iclass in range(1, 11)]
            categories.insert(0, ['not-a-category'])  # there is no category[0]

            cloudnet_scl_mask = np.full((rgN, tsN), False)
            cloudnet_scl_mask[get_indices(categories, [5, 7])] = True    # mask containing only supercooled liquid droplets

            cloudnet_liq_mask = np.full((rgN, tsN), False)
            cloudnet_liq_mask[get_indices(categories, [1, 3, 5, 7])] = True    # mask containing all liquid cloud droplets

            combi_mask_liq = {'tot': get_combined_liquid_mask(ann_liquid_mask, cloudnet_liq_mask)}
            # add liquid masks for different isotherm thresholds
            combi_mask_liq.update({key: mask_below_temperature(combi_mask_liq['tot'].copy(), val) for key, val in isotherm_lines.items()})

            tot_nnz = np.count_nonzero(combi_mask_liq['tot'])
            logger.info(f'nr. of pxl where Cloudnet and/or NN classify liquid after  aerosol/insect removal  : {tot_nnz}')

            # set pxl for which NN predicted liquid but which Cloudnet classifies as insects / aerosols to NaN
            combi_mask_liq['tot'][get_indices(categories, [8, 9, 10])] = False

            # count amount of liquid pixels for each combination
            key, val = np.unique(combi_mask_liq['tot'], return_counts=True)
            combi_mask_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            combi_mask_counts.update(zip(key, val))

            logger.info('\n----- (comparison of amount of pxl classified as liquid by Cloudnet or NN output) --------')
            logger.info(f'overlapping pxl where NN + Cloudnet detect liquid      : {combi_mask_counts[1] * 100 / tot_nnz:.2f} %')
            logger.info(f'overlapping pxl where ONLY NN predicts liquid          : {combi_mask_counts[2] * 100 / tot_nnz:.2f} %')
            logger.info(f'overlapping pxl where ONLY Cloudnet determines liquid  : {combi_mask_counts[3] * 100 / tot_nnz:.2f} %\n')

            # exctract different rain flags, ignore invalid value warnings
            with np.errstate(invalid='ignore'):
                rain_flag = {'disdro': categorization['rainrate_ts'] > 0.0,
                             'anydrizzle': categories[2].any(axis=0),
                             'rg0drizzle': categories[2][0, :].copy()}

            # remove drizzle/rain profiles from all masks 
            if remove_drizzle in ['disdro', 'anydrizzle', 'rg0drizzle']:
                cloudnet_liq_mask[:, rain_flag[remove_drizzle]] = False
                cloudnet_scl_mask[:, rain_flag[remove_drizzle]] = False
                for key in combi_mask_liq.keys():
                    combi_mask_liq[key][:, rain_flag[remove_drizzle]] = 0

            # calculate layer thickness for all different categories
            sum_ll_thickness = sum_liquid_layer_thickness_per_category(categories, cloudnet_liq_mask, ann_liquid_mask, combi_mask_liq, rg_res=30.0)

            """
            ************************************************************************************************************************************
                 _____          _____  _______ _______ _____ __   _  ______      _______ _______ _______ _______ _____  _____  __   _
                |_____] |      |     |    |       |      |   | \  | |  ____      |______ |______ |          |      |   |     | | \  |
                |       |_____ |_____|    |       |    __|__ |  \_| |_____|      ______| |______ |_____     |    __|__ |_____| |  \_|
                
            ************************************************************************************************************************************
            """
            # inside the lidar threshold lopp
            if plot_liquid_pixel_masks:
                fig_name = f'LIQ-MASK_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT_{ithresh_name}.png'
                classification.update({'combi_mask_liq': combi_mask_liq['-38degC']})
                fig, ax = tr.plot_timeheight(wrapper(classification, var_name='combi_mask_liq', var_unit='-'),
                                             range_interval=case['plot_range'], fig_size=plot_size_2D, contour=contour)
                fig.savefig(fig_name)
                logger.info(f'Figure saved :: {fig_name}')

        # outside of the lidar threshold loop
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

        if plot_cloudnet_mwr_lwp:
            fig_name = f'CN-mwr-lwp_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
            fig, _ = tr.plot_timeseries(wrapper(categorization, var_name='lwp_ts', var_unit='g m-2', var_lims=[0, 500]), fig_size=plot_size_2D)
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

            #pred_bsc['mask'][cloudnet_scl_mask.T==0] = True
            #pred_dpl['mask'][cloudnet_scl_mask.T==0] = True

            fig, ax = tr.plot_scatter(pred_dpl, pred_bsc, fig_size=[8, 8], x_lim=[0, 0.3], y_lim=[-6, -2.5], title=titlestring, colorbar=True)
            ax      = add_boxes(ax, lidar_box_thresh, **{'size': 15, 'weight': 'semibold'})
            fig_name = f'FoO-ANN-dpl-vs-bsc_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}_ACCEPT.png'
            fig.savefig(fig_name)
            logger.info(f'Figure saved :: {fig_name}')
            dummy=4

    plt.show()
    dummy = 5
