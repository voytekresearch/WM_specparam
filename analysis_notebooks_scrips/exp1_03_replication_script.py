
################ Imports ################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import seaborn as sns
import warnings
sns.set_context('paper')

sys.path.append('../code')
from colourfulbirds import exp1_functionsforReplication as dp

# Default figure size larger! 
figure = {'figsize': (10,6)}
plt.rc('figure', **figure)

ss1_contra_lap = []
ss3_contra_lap = []
ss6_contra_lap = []
ss1_ipsi_lap = []
ss3_ipsi_lap = []
ss6_ipsi_lap = []

fig_1_lap = []
fig_3_lap = []
fig_6_lap = []

timepoints = []
timepoint_start = 400
timepoint_end = 1500

num_subjects = 31; 
# num_subs_included = 27; # --> this variable is actually not used
for sub in range(1, num_subjects + 1):    
    
    eeg_dat = None
    beh_dat = None
    try:
        eeg_dat, beh_dat = dp.load_data(sub)
    except FileNotFoundError:
        warnings.warn('File not found')
        continue
 
    # Ppt 24 was giving an Index out of bound error. So, skip this one for now
    if sub == 24:
        continue
        
    # Extract helpful data from the EEG file
    channels = eeg_dat['chanLabels']
    artifacts = eeg_dat['arf']
    timepoints = eeg_dat[dp.USE_UN_BASELINED[1]] # --> 'time_long' = unbaselined, 'time' = baselined
    settings = eeg_dat['settings']
    sfreq = settings['srate']
    
    try: 
        # Get subject accuracies and eeg with artifacts rejected
        sub_accs, sub_eeg = dp.reject_bad_trials(eeg_dat[dp.USE_UN_BASELINED[0]], beh_dat, artifacts)
        print("Subject is included " + str(sub))
    except:
        print("Subject not included " + str(sub))
        continue
    
    if dp.reject_subject(sub_accs):
        print("Subject rejected " + str(sub))
        continue
    
    averaging_start = np.where(timepoints==timepoint_start)[0][0] 
    averaging_end = np.where(timepoints==timepoint_end)[0][0] 
    
    ss1_contra = dp.get_average_hilbert(1, dp.CONTRA, sub_eeg, timepoints, sfreq)
    ss3_contra = dp.get_average_hilbert(3, dp.CONTRA, sub_eeg, timepoints, sfreq)
    ss6_contra = dp.get_average_hilbert(6, dp.CONTRA, sub_eeg, timepoints, sfreq)
    ss1_ipsi = dp.get_average_hilbert(1, dp.IPSI, sub_eeg, timepoints, sfreq)
    ss3_ipsi = dp.get_average_hilbert(3, dp.IPSI, sub_eeg, timepoints, sfreq)
    ss6_ipsi = dp.get_average_hilbert(6, dp.IPSI, sub_eeg, timepoints, sfreq)
    
    ss1_contra_lap.append(np.nanmean(ss1_contra[averaging_start:averaging_end]))
    ss3_contra_lap.append(np.nanmean(ss3_contra[averaging_start:averaging_end]))
    ss6_contra_lap.append(np.nanmean(ss6_contra[averaging_start:averaging_end]))
    ss1_ipsi_lap.append(np.nanmean(ss1_ipsi[averaging_start:averaging_end]))
    ss3_ipsi_lap.append(np.nanmean(ss3_ipsi[averaging_start:averaging_end]))
    ss6_ipsi_lap.append(np.nanmean(ss6_ipsi[averaging_start:averaging_end]))
  
    # for extra fancy figure; lap = lateralized alpha power
    fig_1_lap.append((ss1_contra - ss1_ipsi))
    fig_3_lap.append((ss3_contra - ss3_ipsi))
    fig_6_lap.append((ss6_contra - ss6_ipsi))
    
# Average over all ppt; there are nan's in the beginning and end
# this is to ensure the plotting corresponds to the time points    
av_1 = np.mean(fig_1_lap, axis=0)
av_3 = np.mean(fig_3_lap, axis=0)
av_6 = np.mean(fig_6_lap, axis=0)
sem_1 = sp.stats.sem(fig_1_lap,axis=0) # compute standard error of the mean
sem_3 = sp.stats.sem(fig_3_lap,axis=0)
sem_6 = sp.stats.sem(fig_6_lap,axis=0)


################ Save ################

# Convert data into pandas dataframe + save for analysis in JASP
subs = np.arange(1, len(ss1_contra_lap) +1, 1)
df = pd.DataFrame({"sub_id": subs, "size1_contra": ss1_contra_lap, "size3_contra": ss3_contra_lap,
                  "size6_contra": ss6_contra_lap, "size1_ipsi": ss1_ipsi_lap, "size3_ipsi": ss3_ipsi_lap,
                  "size6_ipsi": ss6_ipsi_lap})

# save the dataframes to csv and excel types for further analysis
df.to_csv('saved_files/replicated_results/exp1_alpha_replicated.csv')


################ Figure ################

plt.plot(timepoints, av_1, 'g', label = 'set-size 1', linewidth = 2)
plt.fill_between(timepoints, av_1-sem_1, av_1+sem_1, color = 'g', alpha = 0.2)
plt.plot(timepoints, av_3, 'r', label = 'set-size 3', linewidth = 2)
plt.fill_between(timepoints, av_3-sem_3, av_3+sem_3, color = 'r', alpha = 0.2)
plt.plot(timepoints, av_6, 'b', label = 'set-size 6', linewidth = 2)
plt.fill_between(timepoints, av_6-sem_6, av_6+sem_6, color = 'b', alpha = 0.2)

plt.plot(timepoints, np.zeros(len(timepoints)), 'k--')
plt.xlabel("time (ms)", fontsize=13)
plt.ylabel("Lateralized Alpha Power (dB)", fontsize=13)
plt.xlim([-1500, 1500])

# plt.axvline(x=250, ymin = -0.5, color = 'k')
plt.axvline(x=0, ymin = -0.5, color = 'k')
plt.axvline(x=-1100, ymin = -0.5, color = 'k')
plt.axvline(x=1550, ymin = -0.5, color = 'k')

plt.text(-1000, -0.5, 'cue', fontsize = 10)
plt.text(100, -0.5, 'retention', fontsize = 10)

# plt.legend(prop = {'size': 22}, loc=1)

plt.savefig('saved_files/replicated_results/replicated_exp1_alpha.pdf')
plt.show()


############# Frontal theta power #############


## mean frontal theta power over 3 electrodes " frontal_electrodes"
theta_1 = 0
theta_3 = 0
theta_6 = 0
figure_1 = []
figure_3 = []
figure_6 = []
size1 = []
size3 = []
size6 = []
# sfreq = 250
timepoints = []
timepoint_start = 400
timepoint_end = 1500

num_subjects = 31

for sub in range(1, num_subjects + 1) :
    eeg_dat = None
    beh_dat = None

    try:
        eeg_dat, beh_dat = dp.load_data(sub)
    except FileNotFoundError:
        warnings.warn('File not found')
        continue

    # Extract helpful data from the EEG file
    channels = eeg_dat['chanLabels']
    artifacts = eeg_dat['arf']
    timepoints = eeg_dat[dp.USE_UN_BASELINED[1]] # --> 'time_long' = unbaselined, 'time' = baselined
    settings = eeg_dat['settings']
    sfreq = settings['srate']

    try: 
        # Get subject accuracies and eeg with artifacts rejected
        sub_accs, sub_eeg = dp.reject_bad_trials(eeg_dat[dp.USE_UN_BASELINED[0]], beh_dat, artifacts)
        print("Subject is included " + str(sub))
    except:
        print("Subject not included " + str(sub))
        continue

    if dp.reject_subject(sub_accs):
        print("Subject rejected " + str(sub))
        continue

    start_avg = np.where(timepoints==timepoint_start)[0][0]
    end_avg = np.where(timepoints==timepoint_end)[0][0]
    
    # Get Hilbert transform
    theta_1_left = np.mean(dp.get_hilbert(sub_eeg, timepoints, 1, sfreq, dp.THETA, 
                                              side=dp.LEFT, eeg_channels=dp.FRONTAL_CHANNELS), axis=1)
    theta_1_right = np.mean(dp.get_hilbert(sub_eeg, timepoints, 1, sfreq, dp.THETA, 
                                              side=dp.RIGHT, eeg_channels=dp.FRONTAL_CHANNELS), axis=1)
    theta_3_left = np.mean(dp.get_hilbert(sub_eeg, timepoints, 3, sfreq, dp.THETA, 
                                              side=dp.LEFT, eeg_channels=dp.FRONTAL_CHANNELS), axis=1)
    theta_3_right = np.mean(dp.get_hilbert(sub_eeg, timepoints, 3, sfreq, dp.THETA, 
                                              side=dp.RIGHT, eeg_channels=dp.FRONTAL_CHANNELS), axis=1)
    theta_6_left = np.mean(dp.get_hilbert(sub_eeg, timepoints, 6, sfreq, dp.THETA, 
                                              side=dp.LEFT, eeg_channels=dp.FRONTAL_CHANNELS), axis=1)
    theta_6_right = np.mean(dp.get_hilbert(sub_eeg, timepoints, 6, sfreq, dp.THETA, 
                                              side=dp.RIGHT, eeg_channels=dp.FRONTAL_CHANNELS), axis=1)

    # for calculating mean
    size1_left = (np.mean(theta_1_left[:,start_avg:end_avg]))
    size1_right = (np.mean(theta_1_right[:, start_avg:end_avg]))
    size3_left = (np.mean(theta_3_left[:,start_avg:end_avg]))
    size3_right = (np.mean(theta_3_right[:, start_avg:end_avg]))
    size6_left = (np.mean(theta_6_left[:,start_avg:end_avg]))
    size6_right = (np.mean(theta_6_right[:, start_avg:end_avg]))

    size1.append((size1_left + size1_right) / 2)
    size3.append((size3_left + size3_right) / 2)
    size6.append((size6_left + size6_right) / 2)

    # For figure
    theta_1 = np.stack((np.nanmean(theta_1_left, axis = 0),
        np.nanmean(theta_1_right, axis = 0)), axis = 1)
    theta_3 = np.stack((np.nanmean(theta_3_left, axis = 0),
        np.nanmean(theta_3_right, axis = 0)), axis = 1)
    theta_6 = np.stack((np.nanmean(theta_6_left, axis = 0),
        np.nanmean(theta_6_right, axis = 0)), axis = 1)

    figure_1.append(np.mean(theta_1, axis = 1))
    figure_3.append(np.mean(theta_3, axis = 1))
    figure_6.append(np.mean(theta_6, axis = 1))


av_1 = np.mean(figure_1, axis=0)
av_3 = np.mean(figure_3, axis=0)
av_6 = np.mean(figure_6, axis=0)
sem_1 = sp.stats.sem(figure_1,axis=0)
sem_3 = sp.stats.sem(figure_3,axis=0)
sem_6 = sp.stats.sem(figure_6,axis=0)


############# Frontal theta power : save #############

subs = np.arange(1, len(size1) +1, 1)
df_theta_replicated = pd.DataFrame({'sub_id': subs, 'size_1': size1, 'size_3': size3, 'size_6': size6})

df_theta_replicated.to_csv('saved_files/replicated_results/exp1_theta_replicated.csv')

############# Frontal theta power : figure #############
plt.clf()

# plt.axvline(x=250, ymin = -0.5, color = 'k')
plt.axvline(x=0, ymin = -0.5, color = 'k')
plt.axvline(x=-1100, ymin = -0.5, color = 'k')
plt.axvline(x=1550, ymin = -0.5, color = 'k')

plt.plot(timepoints, av_1, 'g', label = 'size 1', linewidth = 2)
plt.fill_between(timepoints, av_1-sem_1, av_1+sem_1, color = 'g', alpha = 0.2)
plt.plot(timepoints, av_3, 'r', label = 'size 3', linewidth = 2)
plt.fill_between(timepoints, av_3-sem_3, av_3+sem_3, color = 'r', alpha = 0.2)
plt.plot(timepoints, av_6, 'b', label = 'size 6', linewidth = 2)
plt.fill_between(timepoints, av_6-sem_6, av_6+sem_6, color = 'b', alpha = 0.2)
plt.plot(timepoints, np.zeros(len(timepoints)), 'k--')
plt.xlim([-1500, 1500])


# plt.title("Frontal Theta Power -- Good vs. Poor performance", fontsize = 30)
plt.ylabel("Theta power (dB)", fontsize = 13)
plt.xlabel("time (ms)", fontsize = 13)
plt.text(-1000, -0.6, 'cue', fontsize = 10)
plt.text(100, -0.6, 'retention', fontsize = 10)
# plt.legend(prop = {'size': 22})

plt.savefig('saved_files/replicated_results/replicate_exp1_theta.pdf')

plt.show()
