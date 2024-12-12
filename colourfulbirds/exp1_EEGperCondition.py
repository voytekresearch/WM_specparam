"""This module uses general functions to load and process EEG data. Both modules 
exp1_functionsforReplication and exp1_functionsforFOOOF use this module."""


import numpy as np
import warnings
from colourfulbirds.datahelper import loadmat  # Data import helper for EEG data
from colourfulbirds.exp1_settings import (
    NUM_CONDITIONS, TRIAL_MINIMUM, LEFT, RIGHT, IPSI, CONTRA, LEFT_CHANNELS, RIGHT_CHANNELS,
    ALPHA, THETA
)

# Set path for EEG data -- Quirine
eeg_dat_path = '../Experiment_1/Versioned_EEG_Files'
eeg_dat_suffix = '_EEG_timeLockMem.mat'
# Set path for Behavior data
beh_dat_path = '../Experiment_1/Behavior'
beh_dat_suffix = '_discreteWR_biLat.mat'

# Set path for EEG data -- Aaron
# eeg_dat_path = '/mnt/d/adam_wm_eeg/Experiment_1/Versioned_EEG_Files'
# eeg_dat_suffix = '_EEG_timeLockMem.mat'
# # Set path for Behavior data
# beh_dat_path = '/mnt/d/adam_wm_eeg/Experiment_1/Behavior'
# beh_dat_suffix = '_discreteWR_biLat.mat'


def load_data(sub_number):
    '''Loads the subject's data.

    Parameters
    ----------
    sub_numberf int
        The subject number (for experiement 1, this is an integer [1, 31])

    Returns
    ----------
    eeg_dat : dict
        The eeg data cleaned by datahelper so it has the correct key values (13).
        Each key contains arrays with either data or more info.
    beh_dat : dict
        The behavioral data cleaned by datahelper so it has the correct key values (6).
        There are dictionaries inside the dictionary

    Raises
    ----------
    FileNotFoundError
        If subject data cannot be found

    '''

    # Sometimes the first 9 participants don't start with 0 --> 01, 02 etc.
    # So this corrects for it and makes sure the ppt's are run in the same order on every device.
    sub_number = str(sub_number).zfill(2)

    sub_eeg_file = sub_number + eeg_dat_suffix
    sub_beh_file = sub_number + beh_dat_suffix

    eeg_dat, beh_dat = loadmat(eeg_dat_path + "/" + sub_eeg_file), \
        loadmat(beh_dat_path + "/" + sub_beh_file)

    return eeg_dat, beh_dat


def get_condition(side, size):
    '''Returns the condition code given a side (LEFT or RIGHT) and a size (1, 3, or 6).

    Parameters
    ----------
    side : str
        LEFT or RIGHT
    size : int
        1, 3, or 6 (default 1, if out of bounds)

    Returns
    ----------
    cond : int
        condition in a number ranging from 1 to 6, because there are 2 sides * 3 set-sizes

    '''

    # Get the offset by size.
    cond = 0
    if size == 3:
        cond = 1
    elif size == 6:
        cond = 2

    # Add 3 if right side.
    if side == RIGHT:
        cond += 3

    return cond


def get_behavior_param(beh_data, argument):
    '''Extracts the behavioral argument from the from the behavioral data.
    Former functions: get_set_sizes, get_set_sides, get_trial_accuracies.

    Parameters
    ----------
    beh_data : dict
        The behaivioral data returned by load_data()

    Returns
    ----------
    parameter : np.array
        List of all the trial's parameter specified by argument and
        flattens it to a 1D array with nans removed.

    '''

    # Get Accuracy for each trial
    parameter = beh_data['data'][argument] # --> shape is 30x30 (trials by blocks)

    # turn the 30 trials x 30 blocks into one array
    parameter = parameter.flatten('F') # --> size (900,)

    # remove nan
    parameter = parameter[~np.isnan(parameter)]

    return parameter


def get_trial_conditions(beh_data):
    '''Gets the trial condition code corresponding to each valid trial.

    Parameters
    ----------
    beh_data : dict 
        The behaivioral data returned by load_data()

    Returns
    ----------
    trial_conditions : 
        List of all the trial's condition codes that correspond to the trial accuracies.

    '''

    # Get Side and Set Size information for each trial (Use new combined function)
    # names of arguments should correspond with the names from the behavioral file
    setsizes = get_behavior_param(beh_data, 'setSize')
    sides = get_behavior_param(beh_data, 'screenSide')
    trial_accs = get_behavior_param(beh_data, 'trialAcc')

    # n_trials = len(sides) <-- old, should be removed
    n_trials = len(trial_accs)
    trial_conditions = np.zeros(n_trials)

    # Label conditions given the side and setsizes
    # making one array with condition [0 - 5] left 1, 3, 6 and right 1, 3, 6
    # 1 is left. 2 is right.
    for trial in np.arange(n_trials):
        if sides[trial] == 1:
            trial_conditions[trial] = get_condition(LEFT, setsizes[trial])
        elif sides[trial] == 2:
            trial_conditions[trial] = get_condition(RIGHT, setsizes[trial])

    return trial_conditions


def reject_bad_trials(eeg, beh_data, artifacts):
    '''Returns accs and eeg data by Condition, aligned by increasing trial number, with
    trials that have artifacts rejected.

    Parameters
    ----------
    eeg : np.array of shape (6, x, 22, 988)
        Matrix of eeg data [condition, trials, channel, timepoints]
        x for trials, because that can differ per ppn
    beh_data : dict
        behaivioral data loaded from behaivor file
    artifacts : dict
        original artifacts matrix from eeg_data with 'artifactInd' separated on condition.

    Returns
    ----------
    accs : list 
        Accuracies per trial, rows represent the conditions [condition][trials]
    eegs : list 
        EEG signal per trial, rows represent the conditions. Same shape as accs but with time
        and electrodes [condition](trials, 22, 988)

    '''

    # Initialize empty accuracy and eeg variables to return
    accs = [0]*NUM_CONDITIONS
    eegs = [0]*NUM_CONDITIONS

    # Label the conditions and accuracies with a trial number by putting into a matrix.
    n_trials = len(get_behavior_param(beh_data, 'trialAcc'))
    behavior_mat = np.zeros((n_trials, 3))
    behavior_mat[:,0] = np.arange(n_trials)
    behavior_mat[:,1] = get_trial_conditions(beh_data)
    behavior_mat[:,2] = get_behavior_param(beh_data, 'trialAcc')

    for condition in range(NUM_CONDITIONS):
        # Find trials to reject for this condition.
        to_reject = np.nonzero(artifacts['artifactInd'][condition])

        # Extract trial number and associated accuracies.
        trial_nums = np.extract(behavior_mat[:,1] == condition, behavior_mat[:,0])
        accuracies = np.extract(behavior_mat[:,1] == condition, behavior_mat[:,2])

        # Label accuracies with their trial numbers.
        labeled_accs = np.zeros((len(accuracies),2))
        labeled_accs[:,0] = trial_nums
        labeled_accs[:,1] = accuracies

        # Reject on accuracies/reject bad trials from accuracies matrix
        # (n, 2) array: rows are n_trials, first col = trial_num, second col = accuracies
        cleaned_acc = np.delete(labeled_accs, to_reject, axis = 0)

        # Reject on EEG. / reject bad trials from EEG matrix
        # shape should be (n_trials, 22, 988)
        cleaned_eeg = np.delete(eeg[condition,:,:,:], to_reject, axis = 0)

        # Account for potential extra trial or recording at the end + lengths have to be the same
        extra_trials = cleaned_acc.shape[0] - cleaned_eeg.shape[0]

        if extra_trials > 0: # Behaivior has more than eeg data
            # Delete the last trials from accuracy matrix
            # cleaned_acc = np.delete(cleaned_acc, np.s_[cleaned_eeg.shape[0]:], axis = 0)
            cleaned_acc = cleaned_acc[:-1]
        elif extra_trials < 0: # EEG has more than behaivior
            # delete the last trial from the EEG matrix
            # cleaned_eeg = np.delete(cleaned_eeg, np.s_[cleaned_acc.shape[0]:], axis = 0)
            cleaned_eeg = cleaned_eeg[:-1]

        # Save data to our accuracy and baselined data arrays (sorted on condition)
        # These are the right formats and correspond to the trial sort in the EEG data
        accs[condition] = cleaned_acc
        eegs[condition] = cleaned_eeg

    # extra check to make sure the eeg and behavioral data have the same number of trials
    if accs[condition].shape[0] != eegs[condition].shape[0]:
        print("eeg and behavior not aligned!")

    return accs, eegs


def get_channels(side=None, lateralization=None, eeg_channels=None, check_sub=False):
    """Function that retrieves the correct EEG channels based on the frequency of interst and
    whether you're checking if a subject has a peak or generating results. For ALPHA, specify
    side and lateralization. For THETA, specify eeg_channels. For checking whether a subject
    has ALPHA oscillatory activity, specify check_sub as True. Checking whether a subject has a peak
    is done in check_sub_peak(). Whereas producing actual results for further analyses is done with
    get_fooof().

    Parameters
    ----------
    side : str
        The screen side which was cued to attend to. Default=None
    lateralization : str
        The hemisphere of interest. Default=None
    eeg_channels : np.array 1D
        EEG_channels to be analysed. Default=None. Or specify FRONTAL_CHANNELS for THETA analysis
    check_sub : bool
        Default=False to use for generating results. Specify as True to check subject oscillations.

    Returns
    ----------
    eeg_channels : np.array 1D
        Returns correct eeg_channels based on analysis to perform
    """

    # If eeg_channels is already specified, this means that it's the frontal ones for theta
    if eeg_channels is not None:
        return eeg_channels

    elif check_sub:
        # If you want to check the subject peak or not? If yes add left and right up
        eeg_channels = np.concatenate((LEFT_CHANNELS, RIGHT_CHANNELS))

    elif side is not None and lateralization is not None:
        # Otherwise we have to account for side and lateralization for alpha analasys.
        if side == LEFT:
            if lateralization == IPSI:
                eeg_channels = LEFT_CHANNELS
            elif lateralization == CONTRA:
                eeg_channels = RIGHT_CHANNELS
        elif side == RIGHT:
            if lateralization == IPSI:
                eeg_channels = RIGHT_CHANNELS
            elif lateralization == CONTRA:
                eeg_channels = LEFT_CHANNELS
    else:  # basically a warning ?
        warnings.warn("Must specify side and lateralization OR eeg_channels!")
        return -1

    return eeg_channels



def get_eeg_data(sub_eeg, size, side, lateralization=None, eeg_channels=None):
    '''Returns eeg data corresponding to the side, size, and lateralization of the trials
    of interest.

    Parameters
    ----------
    sub_eeg : list
        Matrix of eeg data [condition, trials, channel, timepoints]
    side : str
        RIGHT or LEFT
    size : int
        1, 3, or 6
    lateralization : str
        IPSI or CONTRA

    Returns
    ----------
    to_return_eeg_data : np masked array
        EEG data selected based on condition and channel
        [condition][trials, channels_of_interest, timepoints]

    '''
    # old
    # eeg_channels = 0
    # to_return_eeg_data = 0

    # if side == LEFT and lateralization == IPSI:
    #     eeg_channels = LEFT_CHANNELS
    # elif side == LEFT and lateralization == CONTRA:
    #     eeg_channels = RIGHT_CHANNELS
    # elif side == RIGHT and lateralization == IPSI:
    #     eeg_channels = RIGHT_CHANNELS
    # else:  # side == RIGHT and lateralization == CONTRA
    #     eeg_channels = LEFT_CHANNELS

    # to_return_eeg_data = np.ma.masked_array(sub_eeg[cond][:, eeg_channels, :], \
    #     np.isnan(sub_eeg[cond][:, eeg_channels, :]))

    # here start new function to accomodate for theta too
    eeg_channels = get_channels(side, lateralization, eeg_channels)
    cond = get_condition(side, size)

    to_return_eeg_data = sub_eeg[cond][:, eeg_channels, :]

    return to_return_eeg_data


def reject_subject(sub_accs=None, dfs=None, part='before', frequency=None):
    '''Decides whether or not the subject should be rejected
    with the trial minimum requirement for each condition.

    Parameters
    ----------
    sub_accs : 
        The subject's accuracies by condition.
    trials_after_r2 : list of list
        Some parameters after deleting trials with bad R^2. list of 3 set-sizes
    part : str
        determining enough trials before or after fooofing and deleting trials based on R^2

    Returns
    ----------
        boolean : 
            1 if should reject. 0 if not.

    '''

    if part == 'before':
        for condition in range(int(NUM_CONDITIONS/2)):
            if (sub_accs[condition].shape[0] + sub_accs[condition+3].shape[0]) < TRIAL_MINIMUM:
                return 1
        return 0
    else:
        for condition in range(int(NUM_CONDITIONS/2)):
            # Get total number of trials
            total_trial = dfs[condition]['trial_n'].values[-1] + 1 # +1 because python starts at 0
            
            # Extract sum of trials to be deleted based on 'to_exclude' from df
            if frequency == ALPHA:
                n_to_delete = dfs[condition].query('bas_ret_diff == "bas" & lateralization == "contra"')['to_exclude'].values.sum()
            else:
                n_to_delete = dfs[condition].query('bas_ret_diff == "bas"')['to_exclude'].values.sum()
            # Count how many trials are left after exlcusion: trials = total trials - exclude
            if (total_trial - n_to_delete) < TRIAL_MINIMUM:
                return 1
        return 0