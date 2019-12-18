#!/usr/bin/env python3
"""
Short description:
   Analyser for ACCEPT field campaign data, for detailed evaluation of Ed Luke et at. 2010 ANN approach for prediction of supercooled liquid layers in clouds
   cloud radar and polarization lidar observations.

"""

import os
import pprint
import sys

import time
import numpy as np

# disable the OpenMP warnings
from accept_analyser.utility import get_1st_cloud_base_idx

os.environ['KMP_WARNINGS'] = 'off'
sys.path.append('../larda/')

#import pyLARDA
import pyLARDA.helpers as h

from utility import *

from scipy.io import loadmat

__author__      = "Willi Schimmel"
__copyright__   = "Copyright 2019, ACCEPT Field Campaign Analyser"
__credits__     = ["Willi Schimmel", "Heike Kalesse"]
__license__     = "MIT"
__version__     = "0.0.1"
__maintainer__  = "Willi Schimmel"
__email__       = "willi.schimmel@uni-leipzig.de"
__status__      = "Prototype"

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

PLOTS_PATH = './plots/'
BASE_DIR   = '/media/sdig/LACROS/cloudnet/data/accept/matfiles/timeseries/'

lidar_box_thresh = {'Luke':     {'bsc': -4.45, 'dpl': -7.422612476299660e-01},
                    'deBoer':   {'bsc': -4.3,  'dpl': -1.208620483882601e+00},
                    'Shupe':    {'bsc': -4.3,  'dpl': -6.532125137753436e-01},
                    'Cloudnet': {'bsc': -4.3,  'dpl': 20.000000000000000e+00},
                    'Willi':    {'bsc': -5.2,  'dpl': -6.532125137753436e-01},
                    'linear':   {'slope': 10,  'intersection': -6}}

isotherm_list = [0, -10, -25, -38]

# choose which files to load, 1 = yes, 0 = no
do_read_cloudnet_class = 1
do_read_cloudnet_categ = 1  # contains MWR LWP
do_read_polly          = 0  #
do_read_lipred         = 1  # lipred = lidar prediction from NN
do_read_mira_spec_mom  = 0  # vertically pointing MIRA radar spectra
do_read_sounding       = 0
do_read_GDAS1          = 0

do_smooth_NNoutput     = 1
span_smoo_NNout        = 5

if __name__ == '__main__':

    start_time = time.time()

#    log = logging.getLogger('pyLARDA')
#    log.setLevel(logging.CRITICAL)
#    log.addHandler(logging.StreamHandler())
#
#    larda = pyLARDA.LARDA().connect('lacros_accept_gpu', build_lists=False)

    print('\n-------------- ACCEPT ANALYSER --------------\n')

    case_list = cases_from_csv(oct_cases)
    #pprint.pprint(case_list)

    # loading mat files
    classification  = loadmat(f'{BASE_DIR}ACCEPT_20141005-20141118_classification.mat')           if do_read_cloudnet_class else None
    categorization  = loadmat(f'{BASE_DIR}ACCEPT_20141005-20141118_categorization.mat')           if do_read_cloudnet_categ else None
    pollyXT         = loadmat(f'{BASE_DIR}ACCEPT_20141005-20141118_polly.mat')                    if do_read_polly          else None
    hsrl_pred       = loadmat(f'{BASE_DIR}ACCEPT_20141005-20141118_hsrl_prediction_30s.mat')      if do_read_lipred         else None
    mira_moments    = loadmat(f'{BASE_DIR}ACCEPT_20141005-20141118_moments_from_spectra_30s.mat') if do_read_mira_spec_mom  else None
    radiosondes     = loadmat(f'{BASE_DIR}ACCEPT_20141005-20141118_radiosondes_mean.mat')         if do_read_sounding       else None

    # load range and timesteps and reshape to an 1D array
    n_tot_ts_class  = classification['ts_class_time'].shape[0]
    n_tot_rg_class  = classification['h_class'].shape[0]
    cloudnet_ts = classification['ts_class_time'].reshape((n_tot_ts_class,))
    cloudnet_dt = np.array([h.ts_to_dt(ts) for ts in cloudnet_ts])
    cloudnet_rg = classification['h_class'].reshape((n_tot_rg_class,))

    for case in case_list[:1]:
        for ithresh_name, ithresh_val in lidar_box_thresh.items():
            #if case['notes'] == 'ex': continue  # exclude this case and check the next one

            begin_dt, end_dt = case['begin_dt'], case['end_dt']

            # create directory for plots
            h.change_dir(os.path.join(PLOTS_PATH + f'case_study_{begin_dt:%Y%m%d%H%M%S}-{end_dt:%Y%m%d%H%M%S}/'))

            # find indices for slicing
            rg_start_val, rg_end_val = case['plot_range']
            rg_start_idx, rg_end_idx = h.argnearest(cloudnet_rg, rg_start_val), h.argnearest(cloudnet_rg, rg_end_val)
            n_rg_class = rg_end_idx - rg_start_idx

            ts_start_val, ts_end_val = datetime2datenum(begin_dt), datetime2datenum(end_dt)
            ts_start_idx, ts_end_idx = h.argnearest(cloudnet_ts, ts_start_val), h.argnearest(cloudnet_ts, ts_end_val)
            n_ts_class = ts_end_idx - ts_start_idx

            print(f'Slicing time from {begin_dt:%Y-%m-%d %H:%M:%S} (UTC) to {end_dt:%Y-%m-%d %H:%M:%S} (UTC)')
            print(f'Slicing range from {rg_start_val:.2f} (km) to {rg_end_val:.2f} (km)')
            print(f'threshold for liquid classification: {ithresh_name} with bsc > {ithresh_val["bsc"]:.2e} and depol > {ithresh_val["dpl"]:.2e}\n')

            classification  = time_height_slicer(classification,   [ts_start_idx, ts_end_idx], [rg_start_idx, rg_end_idx]) if do_read_cloudnet_class else None
            categorization  = time_height_slicer(categorization,   [ts_start_idx, ts_end_idx], [rg_start_idx, rg_end_idx]) if do_read_cloudnet_categ else None
            pollyXT         = time_height_slicer(pollyXT,          [ts_start_idx, ts_end_idx], [rg_start_idx, rg_end_idx]) if do_read_polly          else None
            hsrl_pred       = time_height_slicer(hsrl_pred,        [ts_start_idx, ts_end_idx], [rg_start_idx, rg_end_idx]) if do_read_lipred         else None
            mira_moments    = time_height_slicer(mira_moments,     [ts_start_idx, ts_end_idx], [rg_start_idx, rg_end_idx]) if do_read_mira_spec_mom  else None
            radiosondes     = time_height_slicer(radiosondes,      [ts_start_idx, ts_end_idx], [rg_start_idx, rg_end_idx]) if do_read_sounding       else None

            # add indices of first cloud base to classification dict
            if do_read_cloudnet_class:
                classification.update({'first_cb_idx': get_1st_cloud_base_idx(classification['cb_first_ts'], cloudnet_rg)})

            # extract isotherm lines, indices, temperatures at index and range at index
            if do_read_cloudnet_categ:
                isotherm_lines = {f'{iTemp}degC': get_temp_lines(categorization['T_mod_ts'], cloudnet_rg, iTemp) for iTemp in isotherm_list}

            # prepare ANN output: smoothing, removing pixels below 1st cloud base, ...
            if do_read_lipred:

                # assuming this works with nan values
                if do_smooth_NNoutput:
                    for iH in range(n_rg_class):
                        hsrl_pred['bsc_NN_ts'][iH, :]  = h.smooth(hsrl_pred['bsc_NN_ts'][iH, :], span_smoo_NNout)
                        hsrl_pred['CDR_NN_ts'][iH, :]  = h.smooth(hsrl_pred['CDR_NN_ts'][iH, :], span_smoo_NNout)
                        hsrl_pred['dpol_NN_ts'][iH, :] = h.smooth(hsrl_pred['dpol_NN_ts'][iH, :], span_smoo_NNout)

                # derive liquid water mask using different box-thresholds
                ann_liquid_mask = np.zeros(hsrl_pred['bsc_NN_ts'].shape, dtype=np.int)
                if ithresh_name == 'linear':
                    pass
                else:
                    ann_liquid_mask[(hsrl_pred['bsc_NN_ts'] > ithresh_val['bsc'])*(hsrl_pred['CDR_NN_ts'] < ithresh_val['dpl'])] = 1

                # remove all pxl below ceilo base (rain!)
                for iT in range(n_ts_class):
                    idx = classification['first_cb_idx'][iT] if not np.isnan(classification['first_cb_idx'][iT]) else None
                    ann_liquid_mask[:idx, iT] = 0

            if do_read_cloudnet_class:
                """
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
                    cloudnet_liq_mask % combination of all Cloudnet classes containing liquid (excluding rain!)
                    cloudnet_scl_mask % combination of all Cloudnet classes containing scl
                """
                categories = [classification['target_class_ts'] == iclass for iclass in range(1, 11)]
                categories.insert(0, ['not-a-category'])    # there is no category[0]

                cloudnet_scl_mask = np.zeros(classification['target_class_ts'].shape, dtype=np.int)
                cloudnet_scl_mask[np.logical_or(categories[5], categories[7])] = 1

                cloudnet_liq_mask = np.zeros(classification['target_class_ts'].shape, dtype=np.int)
                cloudnet_liq_mask[np.logical_or.reduce((categories[1], categories[3], categories[5], categories[7]))] = 1
                cloudnet_liq_mask[np.logical_or.reduce((categories[8], categories[9], categories[10]))] = 0    # set aerosol & insects to nan

                combi_mask_liq = get_combined_liquid_mask(classification, categories, ann_liquid_mask, cloudnet_liq_mask, isotherm_lines)

                # exctract different rain flags
                rain_flag = {'disdro':     categorization['rainrate_ts'] > 0.0,
                             'anydrizzle': categories[2].any(axis=0),
                             'rg0drizzle': categories[2][0, :].copy()}

                sum_ll_thickness = sum_liquid_layer_thickness_per_category(categories, cloudnet_liq_mask, ann_liquid_mask, combi_mask_liq, rg_res=30.0)


                dummy=5
