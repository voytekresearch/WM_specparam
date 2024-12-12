import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import warnings
# from statsmodels.stats.anova import AnovaRM
import seaborn as sns
sns.set_context('paper')

from colourfulbirds import exp2_functionsforReplication as dp

# Default figure size larger!
figure = {'figsize': (10,6)}
plt.rc('figure', **figure)

############# Lateralized Alpha power #############

good_contra_lap = []
good_ipsi_lap = []
poor_contra_lap = []
poor_ipsi_lap = []

fig_good_lap = []
fig_poor_lap = []

timepoints = []
timepoint_start = 400  # Displayed in ms. In settings these are in timepoints (200)
timepoint_end = 1500  # corresponds to timepoint ... ??

num_subjects = 48
num_included_subjects = 38

for sub in range(1, num_subjects + 1):
    print('Running participant: ', sub)
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

    # Get subject accuracies and eeg with artifacts rejected
    sub_accs, sub_eeg = dp.reject_bad_trials(eeg_dat[dp.USE_UN_BASELINED[0]], beh_dat, artifacts)
    correct_eeg = dp.extract_correct_eeg(sub_accs, sub_eeg)

    if dp.reject_subject(correct_eeg):
        print("Subject rejected " + str(sub))
        continue

    good_contra = dp.get_average_hilbert(dp.CONTRA, sub_eeg, timepoints, dp.GOOD, sub_accs,
        sfreq, dp.ALPHA)
    poor_contra = dp.get_average_hilbert(dp.CONTRA, sub_eeg, timepoints, dp.POOR, sub_accs,
        sfreq, dp.ALPHA)
    good_ipsi = dp.get_average_hilbert(dp.IPSI, sub_eeg, timepoints, dp.GOOD, sub_accs,
        sfreq, dp.ALPHA)
    poor_ipsi = dp.get_average_hilbert(dp.IPSI, sub_eeg, timepoints, dp.POOR, sub_accs,
        sfreq, dp.ALPHA)

    # Time points corresponding to retention time (400 - 1500 ms) in timepoints: 600 - 875
    # These time points are already in settings. However, this is a better(?) non-hard coded version
    start_avg = np.where(timepoints==timepoint_start)[0][0]
    end_avg = np.where(timepoints==timepoint_end)[0][0]

    good_contra_lap.append(np.nanmean(good_contra[start_avg:end_avg]))
    good_ipsi_lap.append(np.nanmean(good_ipsi[start_avg:end_avg]))
    poor_contra_lap.append(np.nanmean(poor_contra[start_avg:end_avg]))
    poor_ipsi_lap.append(np.nanmean(poor_ipsi[start_avg:end_avg]))

    # For figure
    fig_good_lap.append(good_contra - good_ipsi)
    fig_poor_lap.append(poor_contra - poor_ipsi)


av_good = np.mean(fig_good_lap, axis=0)
av_poor = np.mean(fig_poor_lap, axis=0)
sem_good = sp.stats.sem(fig_good_lap,axis=0)
sem_poor = sp.stats.sem(fig_poor_lap,axis=0)


############# Lateralized Alpha power : save #############
subs = np.arange(1, len(good_contra_lap) +1, 1)
df_alpha_replicated = pd.DataFrame({'sub_id': subs, 'poor_contra': poor_contra_lap,
    'poor_ipsi': poor_ipsi_lap, 'good_contra': good_contra_lap, 'good_ipsi': good_ipsi_lap})

df_alpha_replicated.to_csv('saved_files/replicated_results/exp2_alpha_replicated.csv')

############# Lateralized Alpha power : figure #############

# plt.axvline(x=250, ymin = -0.5, color = 'k')
plt.axvline(x=0, ymin = -0.5, color = 'k')
plt.axvline(x=-1100, ymin = -0.5, color = 'k')
plt.axvline(x=1550, ymin = -0.5, color = 'k')

plt.plot(timepoints, av_good, 'g', label = 'good', linewidth = 2)
plt.fill_between(timepoints, av_good-sem_good, av_good+sem_good, color = 'g', alpha = 0.2)
plt.plot(timepoints, av_poor, 'r', label = 'poor', linewidth = 2)
plt.fill_between(timepoints, av_poor-sem_poor, av_poor+sem_poor, color = 'r', alpha = 0.2)
plt.plot(timepoints, np.zeros(len(timepoints)), 'k--')

# plt.title("Lateralized Alpha Power -- Good vs. poor performance", fontsize = 30)
plt.xlabel("time (ms)", fontsize = 13)
plt.ylabel("Lateralized Alpha power (dB)", fontsize = 13)
# plt.legend(prop = {'size': 22}, loc = 1)
plt.text(-1000, -0.5, 'cue', fontsize = 10)
plt.text(100, -0.5, 'retention', fontsize = 10)
plt.xlim([-1500, 1500])

plt.savefig('saved_files/replicated_results/replicate_exp2_alpha.pdf')

plt.show()


############# Frontal theta power #############


## mean frontal theta power over 3 electrodes " frontal_electrodes"
theta_good = 0
theta_poor = 0
figure_good = []
figure_poor = []
good = []
poor = []
# sfreq = 250
timepoints = []
timepoint_start = 400
timepoint_end = 1500

num_subjects = 48
num_included_subjects = 38

for sub in range(1, num_subjects + 1) :
    eeg_dat = None
    beh_dat = None
    print('Running participant: ', sub)
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

    # Get subject accuracies and eeg with artifacts rejected
    sub_accs, sub_eeg = dp.reject_bad_trials(eeg_dat[dp.USE_UN_BASELINED[0]], beh_dat, artifacts)
    correct_eeg = dp.extract_correct_eeg(sub_accs, sub_eeg)

    if dp.reject_subject(correct_eeg):
        print("Subject rejected " + str(sub))
        continue

    start_avg = np.where(timepoints==timepoint_start)[0][0]
    end_avg = np.where(timepoints==timepoint_end)[0][0]

    sub_theta_good_left = np.mean(dp.get_hilbert(sub_eeg, timepoints, sub_accs, dp.GOOD, sfreq,
        dp.THETA, side=dp.LEFT, eeg_channels=dp.FRONTAL_CHANNELS), axis = 1)
    sub_theta_poor_left = np.mean(dp.get_hilbert(sub_eeg, timepoints, sub_accs, dp.POOR, sfreq,
        dp.THETA, side=dp.LEFT, eeg_channels=dp.FRONTAL_CHANNELS), axis = 1)
    sub_theta_good_right = np.mean(dp.get_hilbert(sub_eeg, timepoints, sub_accs, dp.GOOD, sfreq,
        dp.THETA, side=dp.RIGHT, eeg_channels=dp.FRONTAL_CHANNELS), axis = 1)
    sub_theta_poor_right = np.mean(dp.get_hilbert(sub_eeg, timepoints, sub_accs, dp.POOR, sfreq,
        dp.THETA, side=dp.RIGHT, eeg_channels=dp.FRONTAL_CHANNELS), axis = 1)

    # for calculating mean
    good_left = (np.mean(sub_theta_good_left[:,start_avg:end_avg]))
    good_right = (np.mean(sub_theta_good_right[:, start_avg:end_avg]))
    poor_left = (np.mean(sub_theta_poor_left[:, start_avg:end_avg]))
    poor_right = (np.mean(sub_theta_poor_right[:, start_avg:end_avg]))

    good.append((good_left + good_right) / 2)
    poor.append((poor_left + poor_right) / 2)

    # For figure
    theta_good = np.stack((np.nanmean(sub_theta_good_left, axis = 0),
        np.nanmean(sub_theta_good_right, axis = 0)), axis = 1)
    theta_poor = np.stack((np.nanmean(sub_theta_poor_left, axis = 0),
        np.nanmean(sub_theta_poor_right, axis = 0)), axis = 1)

    figure_good.append(np.mean(theta_good, axis = 1))
    figure_poor.append(np.mean(theta_poor, axis = 1))


av_good = np.mean(figure_good, axis=0)
av_poor = np.mean(figure_poor, axis=0)
sem_good = sp.stats.sem(figure_good,axis=0)
sem_poor = sp.stats.sem(figure_poor,axis=0)


############# Frontal theta power : save #############

subs = np.arange(1, len(good) +1, 1)
df_theta_replicated = pd.DataFrame({'sub_id': subs, 'poor': poor, 'good': good})

df_theta_replicated.to_csv('saved_files/replicated_results/exp2_theta_replicated.csv')

############# Frontal theta power : figure #############
plt.clf()

# plt.axvline(x=250, ymin = -0.5, color = 'k')
plt.axvline(x=0, ymin = -0.5, color = 'k')
plt.axvline(x=-1100, ymin = -0.5, color = 'k')
plt.axvline(x=1550, ymin = -0.5, color = 'k')

plt.plot(timepoints, av_good, 'g', label = 'good', linewidth = 2)
plt.fill_between(timepoints, av_good-sem_good, av_good+sem_good, color = 'g', alpha = 0.2)
plt.plot(timepoints, av_poor, 'r', label = 'poor', linewidth = 2)
plt.fill_between(timepoints, av_poor-sem_poor, av_poor+sem_poor, color = 'r', alpha = 0.2)
plt.plot(timepoints, np.zeros(len(timepoints)), 'k--')
plt.xlim([-1500, 1500])
plt.ylim([-0.5, 0.9])

# plt.title("Frontal Theta Power -- Good vs. Poor performance", fontsize = 30)
plt.ylabel("Theta power (dB)", fontsize = 13)
plt.xlabel("time (ms)", fontsize = 13)
plt.text(-1000, -0.4, 'cue', fontsize = 10)
plt.text(100, -0.4, 'retention', fontsize = 10)
# plt.legend(prop = {'size': 22})

plt.savefig('saved_files/replicated_results/replicate_exp2_theta.pdf')

plt.show()
