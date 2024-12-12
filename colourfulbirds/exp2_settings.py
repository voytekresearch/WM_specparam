'''Settings for Experiment 2'''

import numpy as np

# Path settings
EEG_DAT_SUFFIX = '_EEG_timeLockMem.mat'
BEH_DAT_SUFFIX = '_discreteWR_biLat_ss6.mat'

NUM_CONDITIONS = 2  # only good and poor performance as conditions
TRIAL_MINIMUM = 40  # ppts need at least 40 trials in each condition
THRESHOLD = 3  # behavioral threshold; good > threshold; poor < threshold

# Attention sides
LEFT = 'left'
RIGHT = 'right'

# Brain hemispheres
IPSI = 'ipsi'
CONTRA = 'contra'

# Performance
GOOD = 'good'
POOR = 'poor'

BASELINE = 'baseline'
RETENTION = 'retention'

ALPHA = [8, 12]
THETA = [4, 7]

# Electrodes
LEFT_CHANNELS = [0, 6, 8, 10, 14]  # PO3, P3, O1, OL, T5
RIGHT_CHANNELS = [1, 7, 9, 11, 15]  # PO4, P4, O2, OR, T6
FRONTAL_CHANNELS = [2, 3, 18]  # F3, F4, Fz

# Timepoints, Selected timepoints to extract the baseline and retention period (not in ms)
START_BAS = -1100  # baseline
END_BAS = 0
START_RET = 400  # Retention
END_RET = 1500

# Timpoints to extract the baseline and retention period (in ms) - CONTROL ANALYSIS
# to see whether having the baseline and retention period be the same length changes outcome
START_RET_CONTROL = 600  # Retention
END_RET_CONTROL = 800

# Unfiltered, unbaselined data with extra timepoints
USE_UN_BASELINED = ('data', 'time_long')

# FOOOF settings
SETS = {'peak_width_limits': [2, 8], 'min_peak_height': 0.2, 'verbose': False}
FREQ_RANGE = [2, 40]

# theta groups are manually inspected
# Rejected subjects in exp2 should be 7 in total: [6, 9, 22, 23, 33, 35, 42]
# Old FOOOF version based and manually checked for theta bumps lists:
# SUB_THETA = np.array([1, 2, 3, 5, 8, 9, 11, 15, 23, 27, 28, 30, 31, 34, 38, 39, 40, 41, 42, 47])
# SUB_NO_THETA = np.array([4, 6, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 29, 32,
#     33, 35, 36, 37, 44, 45, 46]) # skip ppn 48, because it throws and annoying IndexError.

# New FOOOF version based and manually checked for theta bumps list:
SUB_THETA = np.array([1, 2, 3, 4, 8, 9, 10, 11, 12, 27, 30, 31, 34, 38, 39, 40, 41, 42, 47])
SUB_NO_THETA = np.array([5, 6, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 
                         25, 26, 28, 29, 32, 33, 35, 36, 44, 46, 46]) # skip 48 IndexError

# Alpha subjects devided by whether they have an alpha peak or not
SUB_ALPHA = np.array([1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 15, 16, 17, 18, 20, 22, 23, 24, 25, 27, 
    28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 42, 44, 45, 46, 47, 48])
SUB_NO_ALPHA = np.array([10, 11, 14, 21, 26, 30, 41])

