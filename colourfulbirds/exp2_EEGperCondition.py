'''This module is used to create the functions necessary for extracting and cleaning the
EEG data from Adam et al. (2018). The functions in here can be used for both the replication of 
the results, and for extracting the parameters from the PSD using FOOOF/SpecParam'''


import numpy as np
import warnings
from colourfulbirds import datahelper
from colourfulbirds.exp2_settings import (
    NUM_CONDITIONS, TRIAL_MINIMUM, THRESHOLD, LEFT, RIGHT, IPSI, CONTRA, GOOD, POOR,
    BASELINE, RETENTION, ALPHA, THETA, LEFT_CHANNELS, RIGHT_CHANNELS, FRONTAL_CHANNELS,
    START_BAS, END_BAS, START_RET, END_RET, USE_UN_BASELINED, EEG_DAT_SUFFIX, BEH_DAT_SUFFIX
)

# Set path for EEG data -- Quirine
eeg_dat_path = '../Experiment_2/Versioned_EEG_Files'
eeg_dat_suffix = '_EEG_timeLockMem.mat'
# Set path for Behavior data
beh_dat_path = '../Experiment_2/Behavior'
beh_dat_suffix = '_discreteWR_biLat.mat'


def load_data(sub_number):
    '''Loads the subject's data. Not that you can change the folder where the data is stored
    when changing the argument eeg_dat_path and/or beg_dat_path.

    Parameters
    ----------
    sub_number : int
        The subject number (for experiement 2, this is an integer [1, 48])

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

    sub_eeg_file = sub_number + EEG_DAT_SUFFIX
    sub_beh_file = sub_number + BEH_DAT_SUFFIX



    eeg_data, beh_data = datahelper.loadmat(eeg_dat_path + "/" + sub_eeg_file), \
        datahelper.loadmat(beh_dat_path + "/" + sub_beh_file)

    return eeg_data, beh_data


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


def get_condition(side):
    '''Returns the condition code given a side (LEFT or RIGHT).

    Parameters
    ----------
    side: str
        LEFT or RIGHT

    Returns
    ----------
    cond : int
        condition in a number ranging from 1 to 2, because there are 2 sides
    '''

    if side == LEFT:
        cond = 0
    elif side == RIGHT:
        cond = 1

    return cond


def get_trial_conditions(beh_data) :
    '''Gets the trial condition code corresponding to each valid trial.

    Parameters
    ----------
    beh_data : dict
        The behaivioral data returned by load_data()

    Returns
    ----------
    trial_conditions : np.array
        List of all the trial's condition; 0 = left, 1 = right
    '''

    # Get Side and Set Size information for each trial
    sides = get_behavior_param(beh_data, 'screenSide')
    trial_accs = get_behavior_param(beh_data, 'trialAcc')

    n_trials = len(trial_accs)
    trial_conditions = np.zeros(n_trials)

    # Label conditions given the side, 0 = left, 1 = right
    for trial in np.arange(n_trials):
        if sides[trial] == 1:
            trial_conditions[trial] = 0
        elif sides[trial] == 2:
            trial_conditions[trial] = 1

    return trial_conditions


def get_channels(side=None, lateralization=None, eeg_channels=None, check_sub=False, frequency=None):
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
        if frequency == ALPHA: # If you want to check the subject peak or not? If yes add left and right up
            eeg_channels = np.concatenate((LEFT_CHANNELS, RIGHT_CHANNELS))
        else: # THETA
            eeg_channels = FRONTAL_CHANNELS

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
    else:  # basically a warning
        warnings.warn("Must specify side and lateralization OR eeg_channels!")
        return -1

    return eeg_channels


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
        Original artifacts matrix from eeg_data with 'artifactInd' separated on condition.
        0's means keep trial, 1's mean remove trial

    Returns
    ----------
    accs : list
        Accuracies per trial, rows represent the conditions [condition][trials]
    eegs : list
        EEG signal per trial, rows represent the conditions. Same shape as accs but with time
        and electrodes [condition](trials, 22, 988)
    '''

    accs = [0] * NUM_CONDITIONS
    eegs = [0] * NUM_CONDITIONS

    trial_accs = get_behavior_param(beh_data, 'trialAcc')
    n_trials = len(trial_accs)

    # Label the conditions and accuracies with a trial number by putting into a table.
    behavior_mat = np.zeros((n_trials, 3))
    behavior_mat[:,0] = np.arange(n_trials)
    behavior_mat[:,1] = get_trial_conditions(beh_data)
    behavior_mat[:,2] = trial_accs

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

        # Reject on accuracies; --> condition is already determined. Con is a different shape
        cleaned_acc = np.delete(labeled_accs, to_reject, axis = 0)

        # Reject on EEG.
        cleaned_eeg = np.delete(eeg[condition], to_reject, axis = 0)

        # if the behavioral and eeg trials are not equal. Then delete trials from either
        # behavioral dataset or eeg dataset
        max_len = min((len(cleaned_eeg), len(cleaned_acc)))                                                                                        

        accs[condition] = cleaned_acc[:max_len]                                                                                                             
        eegs[condition] = cleaned_eeg[:max_len]

    if accs[condition].shape[0] != eegs[condition].shape[0]:
        print(accs[condition].shape)
        print(eegs[condition].shape)
        print("eeg and beh not aligned!")

    return accs, eegs


def extract_correct_eeg(accs, eegs):
    """Organizes the eeg data into [left_good, left_poor, right_good, right_poor]. Because the data
    needs to be separated based on the performance and whether the condition was left or right
    (for lateralization effect).

    Parameters
    ----------
    accs : list
        sub_accs from rejectBadTrials() containing all non-rejected trials for both conditions
    eegs : list
        sub_eegs from rejectBadTrials() containing all non-rejected trials for both conditions

    Returns
    ----------
    eeg_performance : list
        The eeg_data separated on trials with good or poor performance, and whether they were on
        the left or right side. Formated [left_good, left_poor, right_good, right_poor]
    """

    eeg_performance = [0] * NUM_CONDITIONS * 2

    for condition in range(NUM_CONDITIONS):
        if condition == 0:  # representing attending to the left side of the screen
            # THRESHOLD is set at 3 items
            # np.squeeze is necessary because we only need a 1D array of trial_numbers/index
            left_good = np.squeeze(np.argwhere(accs[condition][:,1] > THRESHOLD), axis=1)
            left_poor = np.squeeze(np.argwhere(accs[condition][:,1] < THRESHOLD), axis=1)
        else:  #  condition == 1:  # representing attending to the right side of the screen
            right_good = np.squeeze(np.argwhere(accs[condition][:,1] > THRESHOLD), axis=1)
            right_poor = np.squeeze(np.argwhere(accs[condition][:,1] < THRESHOLD), axis=1)

    # dim 1 = trials, dim 2 = channels, dim 3 = timepoints
    eeg_performance[0] = eegs[0][left_good,:,:]
    eeg_performance[1] = eegs[0][left_poor,:,:]
    eeg_performance[2] = eegs[1][right_good,:,:]
    eeg_performance[3] = eegs[1][right_poor,:,:]

    return eeg_performance


def get_eeg_data(sub_eeg, sub_acc, performance, side, lateralization=None, eeg_channels=None):
    '''Returns eeg data corresponding to the side, size, and lateralization of the trials
    of interest.

    Parameters
    ----------
    sub_eeg : list
        Matrix of eeg data [condition, trials, channel, timepoints]
    side : str
        RIGHT or LEFT, default is None
    lateralization : str
        IPSI or CONTRA, default is None
    eeg_channels : str
        Either left or right occipital channels, or frontal channels. Default is None
    sub_acc :
        Not sure why this is default None, because there should always be sub_acc specified
        Unless it's for checking the subject peak ??
    performance : str
        Either good or bad. Default is all, but this should be removed as well

    Returns
    ----------
    to_return_eeg_data : np.array 3D 
        EEG data selected based on condition and channel
        [condition][trials, channels_of_interest, timepoints]
    '''

    eeg_channels = get_channels(side, lateralization, eeg_channels)
    cond = get_condition(side)
    trial_of_interest = slice(sub_eeg[cond].shape[0])

    if performance == GOOD:
        trial_of_interest = np.squeeze(np.argwhere(sub_acc[cond][:,1] > THRESHOLD))
    else:  # performance == POOR:
        trial_of_interest = np.squeeze(np.argwhere(sub_acc[cond][:,1] < THRESHOLD))

    trials_eeg = sub_eeg[cond][trial_of_interest, :, :]
    to_return_eeg_data = trials_eeg[:, eeg_channels, :]  

    return to_return_eeg_data


def reject_subject(eeg_perf=None, dfs=None, part='before', frequency=None):
    """Determines whether or not the subject with the sub_acc data provided has enough trials to be
    considered for analysis. Rejected if the subject has < 50 trials for a condition, where
    condition is either GOOD or POOR.

    Parameters
    ----------
    eeg_perf: np.array
        The eeg data outputted by extract_correct_eeg(), thus only containing EEG trials with
        poor/good performance
    trials_after_r2 : list of list
        Some parameters after deleting trials with bad R^2. list of 3 set-sizes
        --> This should be the dataframe with to_exclude column of both conditions
    part : str
        determining enough trials before or after fooofing and deleting trials based on R^2

    Returns
    ----------
    0 or 1 : int
        1 if subject needs to be rejected. 0, means subject is included
    """

    if part == 'before':
        for condition in range(NUM_CONDITIONS):
            # There are 2 conditions (good and poor), but we need to combine left an right trials
            if (eeg_perf[condition].shape[0] + eeg_perf[condition+2].shape[0]) < TRIAL_MINIMUM:
                return 1
        return 0
    else:
        for condition in range(NUM_CONDITIONS):
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
