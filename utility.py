import pandas
import datetime
import numpy as np
import sys

sys.path.append('../larda/')
import pyLARDA.helpers as h


def cases_from_csv(data_loc, **kwargs):
    """This function extracts information from an excel sheet. It can be used for different scenarios.
    The first row of the excel sheet contains the headline, defined as follows:

    """

    df = pandas.read_csv(data_loc)
    case_list = []
    # exclude header from data
    for icase in range(df.shape[0]):
        begin_dt = datetime.datetime.strptime(str(df['begin_dt'][icase]), '%Y%m%d%H%M%S')
        end_dt = datetime.datetime.strptime(str(df['end_dt'][icase]), '%Y%m%d%H%M%S')

        if begin_dt < end_dt:
            case_list.append({
                'begin_dt': begin_dt,
                'end_dt': end_dt,
                'plot_range': [float(df['h0'][icase]), float(df['hend'][icase])],
                'cloudtype': df['cloudtype'][icase],
                'notes': df['notes'][icase],
                'extra': df['extra'][icase]
            })

    return case_list


def datenum2datetime(dn):
    """Converting Matlab datenum to Python datetime format.
    """
    return datetime.datetime.fromordinal(int(dn)) - datetime.timedelta(days=366)


def datetime2datenum(dt):
    """Converting Python datetime to Matlab datenum to format.
    """
    mdn = dt + datetime.timedelta(days=366)
    frac = (dt - datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac

def time_height_slicer(container, ts_bound, rg_bound):
    """Routine for slicing along time and range dimension for case studies.

    Args:
        container (dict) : complete .mat file data
        ts_bound (list) : list of indices for slicing along time axsis
        rg_bound (list) : list of indices for slicing along rane axsis

    Return:
        container (dict) : sliced container
    """

    assert ts_bound[0] != ts_bound[1], 'slicing failed, time boundaries wrong'
    assert rg_bound[0] != rg_bound[1], 'slicing failed, range boundaries wrong'

    var1D_time_list, var1D_range_list, var2D_list = [], [], []
    if 'ts_class_time' in container.keys():     # set variable lists for cloudnet classification dict
        var1D_time_list  = ['cb_first_ts', 'ct_last_ts', 'ts_class_time']
        var1D_range_list = ['h_class']
        var2D_list       = ['detect_status_ts', 'target_class_ts']
    elif 'ts_cat_time' in container.keys():     # set variable lists for cloudnet categorization dict
        var1D_time_list  = ['lwp_ts', 'rainrate_ts', 'ts_cat_time']
        var1D_range_list = ['h_cat']
        var2D_list       = ['T_mod_ts', 'Vd_cc_ts', 'Ze_cc_ts', 'att_bscatt_ts', 'ldr_cc_ts', 'p_mod_ts', 'rh_mod_ts', 'theta_mod_ts',
                            'wdir_mod_ts', 'width_cc_ts', 'winddirshear_mod_ts', 'windspdshear_mod_ts', 'wspd_mod_ts']
    elif 'ts_polly_time' in container.keys():     # set variable lists for PollyXT dict
        var1D_time_list  = ['ts_polly_time']
        var1D_range_list = ['h_class']
        var2D_list       = ['bsc_polly_ts', 'bsc_polly_woAI_ts', 'dpol_polly_ts', 'dpol_polly_woAI_ts']
    elif 'ts_NN_time' in container.keys():     # set variable lists for predicted Ed Luke ANN prediction dict
        var1D_time_list  = ['ts_NN_time']
        var1D_range_list = ['h_class']
        var2D_list       = ['CDR_NN_ts', 'bsc_NN_ts', 'dpol_NN_ts']
    elif 'ts_sp_time' in container.keys():     # set variable lists for Mira Doppler radar dict
        var1D_time_list  = ['ts_sp_time']
        var1D_range_list = ['h_class']
        var2D_list       = ['Vd_sp_ts', 'Ze_cc_ts', 'kurt_smoo_ts', 'ldr_cor_sp_ts', 'le_peak1_sp_ts', 'left_edge_sp_ts', 'ls_peak1_smoo_ts',
                            'peak1_loc_sp_ts',  're_peak1_sp_ts', 'right_edge_sp_ts', 'rs_peak1_smoo_ts', 'skew_smoo_ts',  'width_sp_ts']
    elif 'ts_rs_time' in container.keys():     # set variable lists for radiosonde dict
        var1D_time_list  = ['ts_rs_time']
        var1D_range_list = ['h_rs']
        var2D_list       = ['rh_rs_ts', 'T_rs_ts', 'TD_rs_ts']

    assert len(var1D_range_list) > 0, 'slicing failed, unknown container'

    for ivar in var1D_time_list:
        container[ivar] = container[ivar].reshape(container[ivar].size)
        container[ivar] = container[ivar][ts_bound[0]:ts_bound[1]]

    for ivar in var1D_range_list:
        container[ivar] = container[ivar].reshape(container[ivar].size)
        container[ivar] = container[ivar][rg_bound[0]:rg_bound[1]]

    for ivar in var2D_list:
        container[ivar] = container[ivar][rg_bound[0]:rg_bound[1], ts_bound[0]:ts_bound[1]]

    return container

def get_1st_cloud_base_idx(cb_first_ts, range_list):
    """Extract the indices of the first cloud base.

    Args:
        cb_first_ts (list) : list or np.array of fist cloud base occurrence, fill_value=nan
        range_list (list) : range bins

    Return:
        idx_1st_cloud_base (np.array) : sliced container
    """
    idx_1st_cloud_base = []
    for i_cb_ts in cb_first_ts:
        if np.isnan(i_cb_ts):
            idx_1st_cloud_base.append(np.nan)
        else:
            idx_1st_cloud_base.append(h.argnearest(range_list, i_cb_ts))

    return np.array(idx_1st_cloud_base)

def get_temp_lines(temperatures, rg_list, isoTemp):
    idx_list, temp_list, range_list = [], [], []
    for iT in range(temperatures.shape[1]):
        idx, val = next(((idx, temp) for idx, temp in enumerate(temperatures[:, iT]) if temp < isoTemp), np.nan)
        idx_list.append(idx)
        temp_list.append(val)
        range_list.append(rg_list[idx])

    return {'idx': np.array(idx_list), 'temp': np.array(temp_list), 'rg': np.array(range_list)}

def get_combined_liquid_mask(classification, categories, liq_pred_mask, cloudnet_liq_mask, isotherm_lines):
    combi_mask_liq = {'tot': np.zeros(classification['target_class_ts'].shape, dtype=np.int)}
    combi_mask_liq['tot'][liq_pred_mask * cloudnet_liq_mask] = 1  # pixel classified as liquid in both
    combi_mask_liq['tot'][liq_pred_mask * ~cloudnet_liq_mask] = 2  # pixel classified as liquid by NN only
    combi_mask_liq['tot'][~liq_pred_mask * cloudnet_liq_mask] = 3  # pixel classified as liquid by Cloudnet only

    # set pxl for which NN predicted liquid but which Cloudnet classifies as insects / aerosols to NaN
    tot_nnz = np.count_nonzero(combi_mask_liq['tot'])
    print(f'nr. of pxl where Cloudnet and/or NN classify liquid before aerosol/insect removal  : {tot_nnz}')

    combi_mask_liq['tot'][np.logical_or.reduce((categories[8], categories[9], categories[10]))] = 0
    tot_nnz = np.count_nonzero(combi_mask_liq['tot'])
    print(f'nr. of pxl where Cloudnet and/or NN classify liquid after aerosol removal          : {tot_nnz}')

    key, val = np.unique(combi_mask_liq['tot'], return_counts=True)
    combi_mask_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    combi_mask_counts.update(zip(key, val))

    print('\n----- (comparison of amount of pxl classified as liquid by Cloudnet or NN output) --------')
    print(f'percentage of overlapping pxl where NN + Cloudnet detect liquid      : {combi_mask_counts[1] * 100 / tot_nnz:.2f}')
    print(f'percentage of overlapping pxl where ONLY NN predicts liquid          : {combi_mask_counts[2] * 100 / tot_nnz:.2f}')
    print(f'percentage of overlapping pxl where ONLY Cloudnet determines liquid  : {combi_mask_counts[3] * 100 / tot_nnz:.2f}')

    # remove liquid pixel below certain temperatures
    for key, val in isotherm_lines.items():
        if not np.isnan(val['idx']).any():
            tmp_mask = combi_mask_liq['tot'].copy()
            for iT in range(tmp_mask.shape[1]):
                tmp_mask[val['idx'][iT]:, iT] = 0
            combi_mask_liq.update({key: tmp_mask})

    return combi_mask_liq


def sum_liquid_layer_thickness_per_category(categories, cloudnet_liq_mask, ann_liquid_mask, combi_mask_liq, rg_res=30.0):
    sum_ts = {i+1: np.nansum(imask*1.0, axis=0)*rg_res for i, imask in enumerate(categories[1:])}
    sum_ts.update({itemp: np.count_nonzero(ival, axis=0)*rg_res for itemp, ival in combi_mask_liq.items()})
    sum_ts.update({'cloudnet': np.nansum(cloudnet_liq_mask*1.0, axis=0)*rg_res})
    sum_ts.update({'neuralnet': np.nansum(ann_liquid_mask*1.0, axis=0)*rg_res})
    return sum_ts

#function [idx, hgt]= extract_temp_lines(ia_T_mod_ts, height, isoTemp)
#tmp = find(ia_T_mod_ts < isoTemp, 1, 'first');
#idx = length(height); hgt = height(end);
#if ~isempty(tmp)
#    idx  = tmp;                               % lowest height index at which T is < isoTemp
#    hgt  = height(find(ia_T_mod_ts < isoTemp, 1));   % height of the approximate 0degC isotherm
#end
#end
