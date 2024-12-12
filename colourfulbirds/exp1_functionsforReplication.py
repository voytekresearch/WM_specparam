'''This module is used to replicate the original frindings from Adam et al. (2018).
It imports exp1_eegperCondition and performs a Hilbert transform
to extract alpha power over time. '''

import numpy as np
from neurodsp.timefrequency import amp_by_time

# Data import helper for eeg data
from colourfulbirds.exp1_settings import (
    NUM_CONDITIONS, TRIAL_MINIMUM, LEFT, RIGHT, ALPHA, THETA, IPSI, CONTRA, SETSIZES, BASELINE,
        RETENTION, LEFT_CHANNELS, RIGHT_CHANNELS, FRONTAL_CHANNELS, START_BAS, END_BAS, START_RET, END_RET,
            USE_UN_BASELINED, SETS
)
from colourfulbirds.exp1_EEGperCondition import (
    load_data, get_behavior_param, get_condition, get_trial_conditions,
    reject_bad_trials, get_eeg_data, reject_subject
)


# def get_hilbert(side, size, lateralization, eeg, timepoints, sfreq):
def get_hilbert(eeg, timepoints, size, sfreq, frequency, side=None, lateralization=None, eeg_channels=None):
    '''Gets the Hilbert transform for each trial and electrode of the eeg data and baselines it.

    Parameters
    ----------
    side : str
        RIGHT or LEFT
    size : int
        1, 3, or 6
    lateralization : str
        IPSI or CONTRA
    eeg : list
        Matrix of eeg data [condition, trials, channel, timepoints]
    timepoints : np.array
        the timespoints to do the analysis over
    sfreq : int
        the sampling frequency = 250

    Returns
    ----------
    hilbert : np.array (n_trials, channels, timepoints) = (x, 5, 988)
        contains baselined hilbert transformed data per trial and electrode

    '''

#     eeg_data = get_eeg_data(eeg, side, size, lateralization)
    eeg_data = get_eeg_data(eeg, size, side=side, lateralization=lateralization,
        eeg_channels=eeg_channels)

    n_trials = eeg_data.shape[0]
    n_electrode = eeg_data.shape[1]
    n_time = len(timepoints)
    hilbert = np.zeros((n_trials, n_electrode, n_time))

    for trial in np.arange(n_trials):
        for electrode in np.arange(n_electrode):

            # Check if there are corrupt trials at the end and skip those
            if len(np.where(~np.isnan(eeg_data[trial, electrode, :]))[0]) == 0:
                continue

            var = amp_by_time(eeg_data[trial, electrode, :], sfreq, frequency)
            # baseline: we do absolute baseline, original paper uses percentage
            base = np.average(var[100:200])  # -1500 until -1100 ms, which is before the cue onset
            hilbert[trial, electrode, :] = (var - base)

    # delete trials with only 0's at the end (something from initialization)
    to_delete = hilbert[:,0,0] == 0
    hilbert = np.delete(hilbert, to_delete, axis=0)

    return hilbert


def get_average_hilbert(size, lateralization, eeg, timepoints, sfreq):
    '''Averages the hilbert of the LEFT and RIGHT sides.

    Parameters
    ----------
    size : int
        1, 3, or 6
    lateralization : st
        IPSI or CONTRA
    eeg : list
        Matrix of eeg data [condition, trials, channel, timepoints]
    timepoints : np.array
        the timespoints to do the analysis over
    sfreq : int
        the sampling frequency = 250

    Returns
    ----------
    average_hilbert : np.array (988 timepoints)
        Average over left and right hilbert time series

    '''
    left_hilbert = get_hilbert(eeg, timepoints, size, sfreq, ALPHA, side=LEFT, lateralization=lateralization)
    right_hilbert = get_hilbert(eeg, timepoints, size, sfreq, ALPHA, side=RIGHT, lateralization=lateralization)

    # Average over the trials: concatenate, add left and right as rows under each other
    # Take the mean of the trials and electrodes
    average_hilbert = np.mean(np.mean(np.concatenate((left_hilbert, right_hilbert)),
        axis = 0), axis = 0)  # runtime warning can be ignored

    return average_hilbert
