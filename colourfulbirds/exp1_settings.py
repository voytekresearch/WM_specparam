""" Settings for experiment 1 """ 

import numpy as np

# 6 conditions (3 set-sizes * 2 sides) in experiment 1. Experiment 2 has 2 conditions
NUM_CONDITIONS = 6
TRIAL_MINIMUM = 75  # ppts need at least 75 trials in each condition

LEFT = 'left'
RIGHT = 'right'

ALPHA = [8, 12]
THETA = [4, 7]

IPSI = 'ipsi'
CONTRA = 'contra'

SETSIZES = {1, 3, 6}

BASELINE = 'baseline'
RETENTION = 'retention'

LEFT_CHANNELS = [0, 6, 8, 10, 14]  # PO3, P3, O1, OL, T5
RIGHT_CHANNELS = [1, 7, 9, 11, 15]  # PO4, P4, O2, OR, T6
FRONTAL_CHANNELS = [2, 3, 18]  # F3, F4, Fz


# Timepoints to extract the baseline and retention period (in ms) - MAIN ANALYSIS
START_BAS = -1100  # baseline
END_BAS = 0
START_RET = 400  # Retention
END_RET = 1500

# Timpoints to extract the baseline and retention period (in ms) - CONTROL ANALYSIS
# to see whether having the baseline and retention period be the same length changes outcome
START_RET_CONTROL = 600  # Retention
END_RET_CONTROL = 800

USE_UN_BASELINED = ('data', 'time_long')  # Unfiltered, unbaselined data with extra timepoints

# Settings for fooof
SETS = {'peak_width_limits': [2, 8], 'min_peak_height': 0.2, 'verbose': False}
FREQ_RANGE = [2, 40]

SUB_THETA = np.array([1, 5, 7, 10, 11, 16, 17, 18, 19, 22, 23, 28])
SUB_NO_THETA = np.array([2, 3,4, 6, 8, 9, 12, 13, 14, 15, 20, 21, 24, 25, 26, 29, 30, 31])
