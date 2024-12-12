'''This module is used to replicate the original frindings from Adam et al. (2018).
It imports exp1_eegperCondition and performs a Hilbert transform
to extract alpha and theta power over time. '''

################ Imports ################


import numpy as np
from neurodsp.timefrequency import amp_by_time

# Data import helper for EEG data
from colourfulbirds.exp2_settings import (
    NUM_CONDITIONS, TRIAL_MINIMUM, THRESHOLD, LEFT, RIGHT, IPSI, CONTRA, GOOD, POOR,
    BASELINE, RETENTION, ALPHA, THETA, LEFT_CHANNELS, RIGHT_CHANNELS, FRONTAL_CHANNELS,
    START_BAS, END_BAS, START_RET, END_RET, USE_UN_BASELINED
)
from colourfulbirds.exp2_EEGperCondition import (
    load_data, get_behavior_param, get_condition, get_trial_conditions,
    reject_bad_trials, get_eeg_data, reject_subject, get_channels, extract_correct_eeg
)


def get_hilbert(eeg, timepoints, sub_acc, performance, sfreq, frequency, side=None,
    lateralization=None, eeg_channels=None):
    """Gets the Hilbert transform for each trial and electrode of the eeg data and baselines it.

    Parameters
    ----------
    eeg : list
        EEG dataframe
    timepoints : np.Array
        Array of timepoints, actual ms stamps
    sub_acc : list
        Behavioral dataframe of performance on trials
    performance : str
        Condition selected. Either GOOD or POOR
    sfreq : int
        sampling frequency
    frequency : (int, int)
        Frequency band to select. Either ALPHA OR THETA
    side : str
        Behavioral paradigm, either attent to the left or the right. Either LEFT or RIGHT
    lateralization : str
        Occipital lateralization. Either IPSI or CONTRA
    eeg_channels : list
        List of eeg_channels to extract data from. Selected from settings (left_electrodes,
        right_electrodes, frontal_electrodes)

    Returns
    ----------
    hilbert : np.Array
        Hilbert transformed data. Contains alpha power over time [n_trials, n_electrodes, n_time]

    """
    eeg_data = get_eeg_data(eeg, sub_acc, performance, side=side, lateralization=lateralization,
        eeg_channels=eeg_channels)

    n_trials = eeg_data.shape[0]
    n_electrode = eeg_data.shape[1]
    n_time = len(timepoints)
    hilbert = np.zeros((n_trials, n_electrode, n_time))

    for trial in np.arange(n_trials):
        for electrode in np.arange(n_electrode):
            if(len(np.where(~np.isnan(eeg_data[trial, electrode, :]))[0]) == 0) :
                continue

            var = amp_by_time(eeg_data[trial, electrode, :], sfreq, frequency)

            # baseline --> convert to percentage change
            base = np.average(var[100:200])
            hilbert[trial, electrode, :] = (var - base)

    return hilbert

def get_average_hilbert(lateralization, eeg, timepoints, performance, sub_acc, sfreq, frequency):
    '''Averages the hilbert of the LEFT and RIGHT sides.

    Parameters
    ----------
    lateralization : str
        IPSI or CONTRA
    eeg : list
        Matrix of eeg data [condition, trials, channel, timepoints]
    timepoints : np.array
        the timespoints to do the analysis over
    performance : str
        Either GOOD or POOR
    sub_acc : list
        Behavioral dataframe of performance on trials
    sfreq : int
        the sampling frequency = 250
    frequency : (int, int)
        Frequency band of interest. Indicates lower and upper range

    Returns
    ----------
    average_hilbert : np.array (988 timepoints)
        Average over left and right hilbert time series
    '''

    left_hilbert = get_hilbert(eeg, timepoints, sub_acc, performance, sfreq, frequency, side=LEFT,
        lateralization=lateralization)
    right_hilbert = get_hilbert(eeg, timepoints, sub_acc, performance, sfreq, frequency, side=RIGHT,
        lateralization=lateralization)

    average_hilbert = np.mean(np.mean(np.concatenate((left_hilbert, right_hilbert), axis = 0),
        axis = 0), axis = 0)

    return average_hilbert
