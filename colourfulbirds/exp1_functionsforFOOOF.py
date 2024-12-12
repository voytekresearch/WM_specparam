'''This module is used to replicate the frindings from Adam et al. (2018), but with a 
FOOOF (specparam) twist. It imports exp1_EEGperCondition and performs a FOOOF analyses
on the power spectra (PSD's) to extract the periodic and aperiodic components. '''

import numpy as np
import pandas as pd
from neurodsp import spectral  # probably not necessary anymore if I'm using mne
# from mne.time_frequency import psd_array_multitaper
from fooof.utils import trim_spectrum
from fooof.analysis import get_band_peak_fg
from fooof import FOOOF
from fooof import FOOOFGroup
# Data import helper for EEG data
# from datahelper import loadmat
from colourfulbirds.exp1_settings import (
    NUM_CONDITIONS, TRIAL_MINIMUM, LEFT, RIGHT, ALPHA, THETA, IPSI, CONTRA, SETSIZES, BASELINE,
    RETENTION, LEFT_CHANNELS, RIGHT_CHANNELS, FRONTAL_CHANNELS, START_BAS, END_BAS, START_RET, END_RET,
    START_RET_CONTROL, END_RET_CONTROL, USE_UN_BASELINED, SETS, FREQ_RANGE, SUB_THETA, SUB_NO_THETA
)
from colourfulbirds.exp1_EEGperCondition import (
    load_data, get_behavior_param, get_condition, get_trial_conditions,
    reject_bad_trials, get_eeg_data, reject_subject
)


def get_band_pow(freqs, spectra, frequency):
    """get average power in frequency band in spectrum
    Parameters
    ----------
    freqs : 1d array
        Frequency values
    spectra : 1d array
        Power spectrum power values
    frequncy : list of [float, float]
        Frequency range of interest
    Returns
    -------
    band_pow : float
        average power in band
    """

    trim_freqs, trim_pows = trim_spectrum(freqs, spectra, f_range=frequency)
    band_pow = np.mean(trim_pows, axis=1)
    
    return band_pow

def get_band_auc(freqs, spectra, frequency):
    """get average power in frequency band in spectrum
    Parameters
    ----------
    freqs : 1d array
        Frequency values
    spectra : 1d array
        Power spectrum power values
    frequncy : list of [float, float]
        Frequency range of interest
    Returns
    -------
    band_pow : float
        average power in band
    """

    trim_freqs, trim_pows = trim_spectrum(freqs, spectra, f_range=frequency)
    band_pow = np.trapz(trim_pows, axis=1)
    
    return band_pow


def del_bad_psd(df, psd_bas, psd_ret, frequency=None):
    '''Delete bad fits from PSD data

    Parameters
    ----------
    df : dataframe
        Contains data of 1 ppn, and 1 conditions (if alpha, also 1 side (?))
    psd_bas : array or list of array if ALPHA
        Contains power spectra of 1 ppn for 1 condition, and 1 time period (baseline)
        If frequency==ALPHA --> first in list is contra, second is ipsi
    psd_ret : array or list of array if ALPHA
        Contains power spectra of 1 ppn for 1 condition, and 1 time period (retention/delay)
        If frequency==ALPHA --> first in list is contra, second is ipsi
    frequncy : (int, int)
        either 'ALPHA' or 'THETA'

    Returns
    ----------
    psd_bas_clean : array or list of array if ALPHA
        Power spectra without bad fits for baseline
    psd_ret_clean : array or list of array if ALPHA
        Power spectra without bad fits for retention/delay
    '''

    # Get the trial indices to delete from psd's from the dataframes to_exclude column
    n_trials = df['trial_n'].max() + 1 # get the max trial count including 0 (+1)
    to_delete_bool = df['to_exclude'][:n_trials]

    if frequency == ALPHA:
        # delete: indices should be corresponding for both bas and retention: In other words, 
        # if a bad fit was detected during baseline, the respective retention is also deleted
        # Delete for both ipsi and contra and baseline and retention because ALPHA
        psd_bas_clean_c = np.delete(psd_bas[0], to_delete_bool, axis=0)
        psd_ret_clean_c = np.delete(psd_ret[0], to_delete_bool, axis=0)
        psd_bas_clean_i = np.delete(psd_bas[1], to_delete_bool, axis=0)
        psd_ret_clean_i = np.delete(psd_ret[1], to_delete_bool, axis=0)

        psd_bas_clean = [psd_bas_clean_c, psd_bas_clean_i]
        psd_ret_clean = [psd_ret_clean_c, psd_ret_clean_i]


    else: # theta
        # delete: indices should be corresponding for both bas and retention: In other words, 
        # if a bad fit was detected during baseline, the respective retention is also deleted
        psd_bas_clean = np.delete(psd_bas, to_delete_bool, axis=0)
        psd_ret_clean = np.delete(psd_ret, to_delete_bool, axis=0)

    return psd_bas_clean, psd_ret_clean


def get_bad_fits(df, frequency):
    '''Create panda dataframes for parameters

    Parameters
    ----------
    df : dataframe
        Contains data of 1 ppn, and 1 conditions (if alpha, also 1 side)
    frequncy : string
        either 'ALPHA' or 'THETA' 


    Returns
    ----------
    df : pandas DataFrame
        Contains data of 1 ppn, and 1 conditions (if alpha, also 1 side); with column indicating model fit
    
    '''

    if frequency == ALPHA:
        # Extract r2s for baseline and retention from df
        r2s_bas_contra = df.query('bas_ret_diff == "bas" & lateralization == "contra"')['r2s'].values
        r2s_bas_ipsi = df.query('bas_ret_diff == "bas" & lateralization == "ipsi"')['r2s'].values
        r2s_ret_contra = df.query('bas_ret_diff == "ret" & lateralization == "contra"')['r2s'].values
        r2s_ret_ipsi = df.query('bas_ret_diff == "ret" & lateralization == "ipsi"')['r2s'].values

        # Check model fits (r_squared) and determine threshold (2*std from mean)
        r2s_concat = [*r2s_bas_contra, *r2s_bas_ipsi, *r2s_ret_contra, *r2s_ret_ipsi]
        thresh_r2s = np.mean(r2s_concat) - (2 * np.std(r2s_concat))

        # Create boolean array of length(trials) with True to exclude based on trials_exclude indices
        exclude_bas_contra = r2s_bas_contra < thresh_r2s
        exclude_ret_contra = r2s_bas_ipsi < thresh_r2s
        exclude_bas_ipsi = r2s_ret_contra < thresh_r2s
        exclude_ret_ipsi = r2s_ret_ipsi < thresh_r2s 

        # Boolean of trials that need to be excluded
        to_exclude = exclude_bas_contra + exclude_ret_contra + exclude_bas_ipsi + exclude_ret_ipsi
        to_exclude_list = np.tile(to_exclude, 2*3) # <-- 2 sides (contra+ipsi) and 3 time frames
        # Add column to df
        df['to_exclude'] = to_exclude_list


    else: # Frequency is THETA
        # Extract r2s for baseline and retention from df
        r2s_bas = df.query('bas_ret_diff == "bas"')['r2s'].values
        r2s_ret = df.query('bas_ret_diff == "ret"')['r2s'].values
        # Check model fits (r_squared) and determine threshold (2*std from mean)
        r2s_concat = [r2s_bas, r2s_ret]
        thresh_r2s = np.mean(r2s_concat) - (2 * np.std(r2s_concat))
        
        # Create boolean array of length(trials) with True to exclude based on trials_exclude indices
        exclude_bas = r2s_bas < thresh_r2s
        exclude_ret = r2s_ret < thresh_r2s

        # Boolean of trials that need to be excluded
        to_exclude_list = np.tile((exclude_bas + exclude_ret), 3) # TO_CHECK
        # Add column to df
        df['to_exclude'] = to_exclude_list


    return df 


def get_dfs(sub_n, size, frequency, periodic_all, aperiodic_all, bandpower_all, auc_all, auc_log_all, auc_lin_all, percent_peak_all, r2s, lateralization=None):
    '''Create panda dataframes for parameters

    Parameters
    ----------
        subject id
    size : string
        either set-size 1,3 or 6
    periodic_power_all : list of arrays
        Containing the power values of baseline, retention, and difference
    periodic_cf_all : list of arrays
        Containing the center frequency values of baseline, retention, and difference
    aperiodic_all : list of 2D arrays
        Containing the exponent and offsets values of baseline, retention, and difference
    r2s : list of 2D arrays
        Containing the model fits of baseline, retention


    Returns
    ----------
    df : pandas DataFrame
        Containing all the parameters for a single subject and condition
    psds : numpy 2D array
        Containing the power spectra, including the ones with bad fits
    '''
    
    # first caluclate mean and 2xsd of r2s for removing those trials <-- MOVED TO GET_BAD_FITS
    
    # # Check model fits (r_squared) and determine threshold (2*std from mean)
    # r2s_concat = [*r2s[0], *r2s[1]]
    # thresh_r2s = np.mean(r2s_concat) - (2 * np.std(r2s_concat))
    
    # # Create boolean array of length(trials) with True to exclude based on trials_exclude indices
    # exclude_bas = r2s[0] < thresh_r2s
    # exclude_ret = r2s[1] < thresh_r2s    
        
    # If I want to create the dataframe now I need to:
    # 1. Create header names
    col_names = ['sub_id', 'trial_n', 'set_size', 'bas_ret_diff', 'exponent', 'offset', 'power', 
                'cf', 'bandpower', 'auc', 'auc_log_osc', 'auc_lin_osc', 'peak_perc', 'r2s']
    # 2. Create lists for sub_id, trials, and conditions...
    n_trials = len(r2s[0])
    trial_list = np.tile(np.arange(0, n_trials), 3) # 3 time points; bas, ret and diff
    sub_id_list = np.repeat(sub_n, 3*n_trials) 
    ss_list = np.repeat(size, 3*n_trials)
    bas_ret_diff_list = np.repeat(['bas', 'ret', 'diff'], n_trials)
    # to_exclude_list = np.tile((exclude_bas + exclude_ret), 3) # Boolean of trials that need to be excluded

    # 3. Unpack the parameters into lists 
    # index [bas/ret/diff][:,0:offset or 1:exponent]
    exponent_list = np.concatenate((aperiodic_all[0][:,1], aperiodic_all[1][:,1], aperiodic_all[2][:,1])) 
    offset_list = np.concatenate((aperiodic_all[0][:,0], aperiodic_all[1][:,0], aperiodic_all[2][:,0])) 
    power_list = np.concatenate((periodic_all[0][:,1], periodic_all[1][:,1], periodic_all[2][:,1]))
    cf_list = np.concatenate((periodic_all[0][:,0], periodic_all[1][:,0], periodic_all[2][:,0]))
    bandpower_list = np.concatenate((bandpower_all[0], bandpower_all[1], bandpower_all[2]))
    auc_list = np.concatenate((auc_all[0], auc_all[1], auc_all[2]))
    auc_log_list = np.concatenate((auc_log_all[0], auc_log_all[1], auc_log_all[2]))
    auc_lin_list = np.concatenate((auc_lin_all[0], auc_lin_all[1], auc_lin_all[2]))
    r2s_list = np.concatenate((r2s[0], r2s[1], np.full([1,n_trials], np.nan)[0]))
    # For percentage of peaks over trials within a condition, it's kinda weird, because I'm just
        # repeating the same value to fit it in the dataframe. I just want all data to be in 1
    peak_perc_list = np.concatenate((np.repeat(percent_peak_all[0], n_trials), 
            np.repeat(percent_peak_all[1], n_trials), np.repeat(percent_peak_all[2], n_trials)))
    
    # Now create the dataframe
    list_of_tuples = list(zip(sub_id_list, trial_list, ss_list, bas_ret_diff_list,
                              exponent_list, offset_list, power_list, cf_list, bandpower_list, 
                              auc_list, auc_log_list, auc_lin_list, peak_perc_list, r2s_list))                               
    
    df = pd.DataFrame(list_of_tuples, columns=col_names)

    # If we're doing alpha stuff, we need to add a column indicating lateralization
    if frequency == ALPHA:
        df['lateralization'] = lateralization

    return df


def get_spectrums(sub_eeg, timepoints, sfreq, timeframe, analysis_type): #, start_time=None, end_time=None, electrodes=None
    '''Creates the power spectra for either the baseline period and retention period for all trials
    and all selected electrodes in sub_eeg.
    I want to change the start_time and end_time to timeframe, similar to exp1

    Parameters
    ----------
    sub_eeg : np.array
        The EEG data [n_trials, n_electrodes, 988 timepoints]
    sfreq : int
        sampling frequency
    timeframe : str
        Either BASELINE or RETENTION

    Returns
    ----------
    freq : np.array 1D
        Frequency axis corresponding to the PSD (psd_av)
    psd_av : np.array 2D
        Power spectra of all trials from the EEG data without nan's [n_trials, 988 timepoints]

    '''

    # Find exact idx of timepoints
    sb_idx = np.where(timepoints == START_BAS)[0][0]
    eb_idx = np.where(timepoints == END_BAS)[0][0]
    sr_idx = np.where(timepoints == START_RET)[0][0]
    er_idx = np.where(timepoints == END_RET)[0][0]

    print(sb_idx)

    # Get the correct period & whether it is the main or control analysis
    if timeframe == BASELINE:
        if analysis_type == 'main_analysis':
            # eeg_bas = pad_baseline(sub_eeg[:, :, sb_idx:eb_idx]) # needs zero-padding
            freq, psd = spectral.compute_spectrum(sub_eeg[:, :, sb_idx:eb_idx], sfreq, method = 'welch', avg_type = 'mean',
                                            nperseg = eb_idx - sb_idx)
        # else: # control analysis
        #     eeg_bas = pad_baseline(sub_eeg[:, :, s_b:e_b]) # zero-pad so frequency range is same as main analysis
        #     freq, psd = spectral.compute_spectrum(eeg_bas, 
        #                                     sfreq, method = 'welch', avg_type = 'mean',
        #                                     nperseg = eeg_bas.shape[2])  # eeg_bas.shape[2] should be 275
    else: # retention period
        if analysis_type == 'main_analysis':
            # retention period doesn't need zero padding
            freq, psd = spectral.compute_spectrum(sub_eeg[:, :, sr_idx:er_idx], sfreq,
                method = 'welch', avg_type = 'mean', nperseg = er_idx - sr_idx)
                # end_ret-start_ret should be 275
        # else: # control analysis 
        #         eeg_ret = pad_baseline(sub_eeg[:, :, START_RET_CONTROL:END_RET_CONTROL]) # zero-pad so frequency range is same as main analysis
        #         freq, psd = spectral.compute_spectrum(eeg_ret, sfreq,
        #         method = 'welch', avg_type = 'mean', nperseg = eeg_ret.shape[2])

    # First average psd over electrodes ipsi and contra (without smushing trials)
    # This is where thigns might go wrong with theta ??
    psd_av = np.nanmean(psd, axis = 1)  # average over electrodes
    psd_av = psd_av[~np.isnan(psd_av).any(axis=1)]

    return freq, psd_av


def get_fooof_params(sub_eeg, timepoints, timeframe, sfreq, frequency, sets, analysis_type):
    '''Function that will give you the FOOOF parameterers for
    either the baseline period or the retention period
    Baselining will be done in another function.
    If nan in trials for the amplitude, because there is no peak detected,
    replace it with 0 power.

        Parameters
    ----------
    eeg_data : np.masked.array (n_trials, n_channels, 988 timepoints)
        The eeg data
    timeframe : str
        either BASELINE or RETENTION
    sfreq : int
        sampling frequency
    frequency : (int, int)
        Frequency range of interest, either Alpha or Theta
    sets : dict
        Settings for creating the FOOOF object

    Returns
    ----------
    periodic : np.array 2D (n_trials, 3)
        Periodic components, power, cf, and bandwidth
    aperiodic : np.array 2D (n_trials, 2)
        Aperiodic components, offset and exponent
    freq : np.array 1D
        Frequencies on the x-axis, 43 points
    psd_av : np.array 2D (n_trials, 43 frequencies)
        Averaged PSD over channels

    '''

    # Compute PSD using traditional welch's approach: method = 'mean'
    freq, psd_av = get_spectrums(sub_eeg, timepoints, sfreq, timeframe, analysis_type)

    # initialize fooof + set settings
    fg = FOOOFGroup(**sets)

    # Fooof it
    fg.fit(freq, psd_av, FREQ_RANGE, n_jobs=-1)

    periodic = get_band_peak_fg(fg, frequency)  # periodic params (peak, cf, bandwidth)
    aperiodic = fg.get_params('aperiodic_params')  # get the aperiodic 1/f properties (offset, exp)
    # find_peaks = fg.get_params('peak_params')  # last col is trial number
    r2s = fg.get_params('r_squared')

    # Get bandpower + AUC
    bandpower = get_band_pow(freq, psd_av, frequency)
    bandpower_auc = get_band_auc(freq, psd_av, frequency)

    # Take linear and log (addiive vs multiplicative) baselining of aperiodic-adjusted osc power
    adj_auc_log = []
    adj_auc_lin = []
    for trial in np.arange(0, len(fg)):
        # take 1 trial
        adj_temp = fg.get_fooof(trial)
        # 1. Take difference of power spectrum and aperiodic fit
        aper_diff_log = adj_temp.power_spectrum - adj_temp._ap_fit
        aper_diff_lin = 10**adj_temp.power_spectrum - 10**adj_temp._ap_fit
        # 2. Trim spectrum to specific frequency range of interest (alpha or theta)
        trim_freqs, trim_pows_log = trim_spectrum(adj_temp.freqs, aper_diff_log, f_range=frequency)
        trim_freqs, trim_pows_lin = trim_spectrum(adj_temp.freqs, aper_diff_lin, f_range=frequency)
        # 3. AUC
        adj_auc_log.append(np.trapz(trim_pows_log, trim_freqs))
        adj_auc_lin.append(np.trapz(trim_pows_lin, trim_freqs))
    
    adj_auc_log = np.array(adj_auc_log)
    adj_auc_lin = np.array(adj_auc_lin)

    # Replace nan --> 0: only for power                     
    # In the variable "periodic" the 2nd row (power values) already doesn't have nan's in it
    periodic_power = periodic[:, 1] # shape = (n_peaks ,3)
    nan_periodic_power = np.isnan(periodic_power)
    # replace nan's as 0 power
    periodic_power[nan_periodic_power] = 0
    # now add cf column to periodic array with power
    periodic_altered = np.vstack([periodic[:,0], periodic_power]).T

    return periodic_altered, aperiodic, bandpower, bandpower_auc, adj_auc_log, adj_auc_lin, freq, psd_av, r2s



def get_fooof(sub_n, size, sub_eeg, timepoints, frequency, sfreq, lateralization=None, analysis_type=None):
    '''Function that creates PSD's, and executes FOOOF fits on
    the baseline period and the retention period.
    Furthermore, it extracts the useful information (peak, cf, offset and exponent).
    And automatically baselines these parameters.

    Parameters
    ----------
    lateralization : str
        either contra or ipsi
    eeg : dict
        eeg data matrix
    size : int
        setsize 1, 3 or 6
    sfreq : int
        sampling frequency

    Returns
    ----------
    periodic_dif_power : np.array 1D
        baselined periodic power per trial. Length is the total trials of this condition
    periodic_dif_cf : np.array 1D
        baselined periodic cf per trial. Length is the number of trials a peak was found in
        both time periods
    aperiodic_dif : np.array 2D
        baselined aperiodic parameters (offset and exponent) per trial. 
        Length is the total trials of this condition
    psd_av_bas : np.array 1D
        Average psd of baseline period
    psd_av_ret : np.array 1D
        Average psd of retention period
    freq_bas : np.array 1D
        Frequency resolution axis for the psd

    ###### Add another output parameter for cf percentages


    '''
    if frequency == ALPHA: # probably need to define this differently, since ALPHA is an array? 
        eeg_data_left = get_eeg_data(sub_eeg, size, LEFT, lateralization, eeg_channels=None)
        eeg_data_right = get_eeg_data(sub_eeg, size, RIGHT, lateralization, eeg_channels=None)
        eeg_data = np.concatenate((eeg_data_left, eeg_data_right))
    else:  # frequency == THETA
        eeg_data_left = get_eeg_data(sub_eeg, size, LEFT, eeg_channels=FRONTAL_CHANNELS)
        eeg_data_right = get_eeg_data(sub_eeg, size, RIGHT, eeg_channels=FRONTAL_CHANNELS)
        eeg_data = np.concatenate((eeg_data_left, eeg_data_right))


    # This function will create PSD and perform fooof on those; Get periodic, aperiodic and psd's
    periodic_bas, aperiodic_bas, bp_bas, auc_bas, auc_log_bas, auc_lin_bas, freq_bas, psd_bas, r2s_bas = get_fooof_params(eeg_data, timepoints, BASELINE, 
        sfreq, frequency, SETS, analysis_type)
    periodic_ret, aperiodic_ret, bp_ret, auc_ret, auc_log_ret, auc_lin_ret, _, psd_ret, r2s_ret= get_fooof_params(eeg_data, timepoints, RETENTION, 
        sfreq, frequency, SETS, analysis_type)


    # Difference of periodic component
    # --> first col is center frequency, second col is power                     
    periodic_dif = periodic_ret - periodic_bas

    # Take the difference of the aperiodic components
    # --> first col is offset, second col is exponent
    aperiodic_dif = aperiodic_ret - aperiodic_bas

    # Take difference in bandpower and auc
    bp_dif = bp_ret - bp_bas
    auc_dif = auc_ret - auc_bas

    # baseline the log and linear aperiodic adjusted osc power AUC
    auc_log_dif = auc_log_ret - auc_log_bas
    auc_lin_dif = auc_lin_ret - auc_lin_bas

    # periodic_bas/ret <--- contains cf data either first column. nan when no peak is detected
    percent_bas_has_peak = np.isnan(periodic_bas[:,0]).sum() / len(periodic_bas[:,0]) * 100
    percent_ret_has_peak = np.isnan(periodic_ret[:,0]).sum() / len(periodic_ret[:,0]) * 100
    percent_peak_dif = percent_ret_has_peak - percent_bas_has_peak
    
    # create list of arrays so I can save both baseline, retention, and difference between params
    periodic_all = [periodic_bas, periodic_ret, periodic_dif]
    aperiodic_all = [aperiodic_bas, aperiodic_ret, aperiodic_dif]
    perc_peak_all = [percent_bas_has_peak, percent_ret_has_peak, percent_peak_dif]
    bp_all = [bp_bas, bp_ret, bp_dif]
    auc_all = [auc_bas, auc_ret, auc_dif]
    auc_log_all = [auc_log_bas, auc_log_ret, auc_log_dif]
    auc_lin_all = [auc_lin_bas, auc_lin_ret, auc_lin_dif]
    r2s = [r2s_bas, r2s_ret]
    
    df = get_dfs(sub_n, size, frequency, periodic_all, aperiodic_all, bp_all, auc_all, auc_log_all, auc_lin_all, perc_peak_all, r2s, lateralization)

    return df, psd_bas, psd_ret, freq_bas


def check_sub_fooof(eeg_data, frequency, sfreq, sets):
    '''Function to check if a subject has ALPHA oscillations at all. (Theta was done manually,
    so that is not included in this function)
    More specific; does the subject exhibit alpha power over the entire trial?
    If yes, then use this subject in further analyses.
    if no, reject this subject from further analyses
    
    Parameters
    ----------
    eeg_data : array or something
        eeg data of all conditions

    Returns
    ----------
    FOOOfresults
        The results of the spectral parameterization

    '''

    # initialize fooof + set settings
    fm = FOOOF(**sets)

    # concatenate eeg_data based on setsize. Add new condition on new rows (eeg[n_trials, 22, 988])
    eeg = np.concatenate((eeg_data[0], eeg_data[1]), axis = 0)
    eeg = np.concatenate((eeg, eeg_data[2]), axis = 0)
    eeg = np.concatenate((eeg, eeg_data[3]), axis = 0)
    eeg = np.concatenate((eeg, eeg_data[4]), axis = 0)
    eeg = np.concatenate((eeg, eeg_data[5]), axis = 0)

    # remove nan if present in the data
    if np.isnan(eeg).any():
        nan = np.argwhere(np.isnan(eeg))
        eeg = np.delete(eeg, nan[0], axis = 0)

    if frequency == ALPHA:
        # select electrodes on occipital lobe for alpha power
        electrodes = np.concatenate((LEFT_CHANNELS, RIGHT_CHANNELS))
    else:
        electrodes = FRONTAL_CHANNELS

    # Create PSD
    freq, psd = spectral.compute_spectrum(eeg[:, electrodes, :], sfreq, \
        method = 'welch', avg_type = 'mean', nperseg = sfreq*2)
    # First average psd over electrodes ipsi and contra
    psd_av = np.average(np.nanmean(psd, axis = 0), axis = 0)
    # Chedk what ryan said about np.average. Check shape of psd
    # Average over trials and electrodes

    # Fooof it
    fm.fit(freq, psd_av, FREQ_RANGE)

    return fm.get_results(), fm


def check_sub_peak(eeg_data, frequency, sfreq, sets):
    '''Check wether the subject has oscillations in the desired frequency (ALPHA)
    More specific; does the subject exhibit alpha power over the entire trial?
    If yes, then use this subject in further analyses.
    if no, reject this subject from further analyses

    Parameters
    ----------
    eeg_data : list
        Eeg data of all conditions
    frequency : list (int, int) 
        Frequency range of interest. ALPHA

    Returns
    ----------
    peak_present : str
        There is a/no peak. Either "no peak" or "Yay peak!"

    '''

    # eeg_data should be of shape 1D
    fooof_data, model = check_sub_fooof(eeg_data, frequency, sfreq, sets)
    peak_data = fooof_data[1]

    # Check wether this indexes the right parameter from the fooofresults
    # after updating to the newest fooof version :D
    peak_detect = peak_data[(peak_data[:,0] >= frequency[0]) & (peak_data[:,0] <= frequency[1])]

    if not peak_detect.shape[0]:  # if empty --> this is FALSE
        peak_present = "No peak"
    else:
        peak_present = "Yay peak!"

    return peak_present, model


def pad_baseline(eeg):
    '''The baseline period needs some zero-padding to make sure the window sizes
    for computing the PSD is the same between the baseline period
    and the retention period.
    
    Parameters
    ----------
    eeg : np.masked.array 3D (n_trials, n_chanels, timepoints)
        eeg signal from the baseline period (200 timepoints)

    Returns
    ----------
    padded : np.array 3D (n_trials, n_channels, 275 timepoints)
        The zero-padded eeg signal for all the trials and electrodes

    '''

    # baseline_period is eeg data from the baseline period
    # 100 ms --> 25 timepoints 0's before and after the baseline period + baseline the 500 mseconds
    n_trial = eeg.shape[0]
    n_electrodes = eeg.shape[1]
    n_time = eeg.shape[2]

    # eeg_baseline = np.zeros((n_trial, n_electrodes, n_time))
    padded = np.zeros((n_trial, n_electrodes, n_time + 75))

    for trial in range(n_trial):
        for electrode in range(n_electrodes):
            # baseline / demean
            average = np.average(eeg[trial,electrode,:])
            baseline_period = eeg[trial, electrode, :] - average

            # add zero padding, replace with amount of timepoints corresponding to 250 msec
            padded[trial, electrode, :] = \
                np.pad(baseline_period, (38,37), 'constant', constant_values = 0)

    return padded
