'''Script to run over participants for experiment 1. Using spec_param (previously FOOOF) to
extract the different parameters from a Power Spectrum'''


import sys
import warnings
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle as pkl
sys.path.append('../code')
from colourfulbirds import exp1_functionsforFOOOF as dp

random.seed(0)

'''Alpha: run specparam model and generate parameters'''

############### Initialize ################


# LOOP OVER MAIN ANALYSIS AND CONTROL ANALYSIS FOR WINDOW SIZE
for analysis_type in ['main_analysis']:  # , 'control_analysis' FOR LOOP OBSOLETE since not padding the baseline anymore
    print(analysis_type)

    # settings per main or control analysis
    if analysis_type == 'main_analysis':
        save_path = 'saved_files/fooof_results/exp1_output/main_analysis/'
    else:
        save_path = 'saved_files/fooof_results/exp1_output/control_analysis/'
    
    print(save_path)


    num_subjects = 31  # total subjects
    subs = 27  # subjeds included in final analysis
    # Initialize empty array structure to save psd's
    psd_exp1_alpha = np.full((subs, 138, 3, 2, 2), np.nan)
    psd_exp1_no_alpha = np.full((subs, 138, 3, 2, 2), np.nan)
    power_detected, no_power_detected = 0, 0  # counter to fill in above array

    # Initialize empty lists for data storage
    alpha_avg_perform = []
    no_alpha_avg_perform = []
    df_all_alpha = []
    df_all_no_alpha = []

    # timepoints = []
    sub_no_alpha_peak = []
    sub_with_alpha_peak = []
    sub_included = []
    sub_rejected = []
    sub_rejected_after_r2 = []
    sub_rejected_without = []


    # Start looping over participants
    for sub in range(1, num_subjects + 1):

        if sub == 24:
            print('skip ppn 24 due to bad behavioral file')
            continue
        print(sub)
        eeg_dat = None
        beh_dat = None

        # Load in data
        try:
            eeg_dat, beh_dat = dp.load_data(sub)
        except FileNotFoundError:
            warnings.warn('File not found')
            continue

        # Extract helpful data from the EEG file
        channels = eeg_dat['chanLabels']
        artifacts = eeg_dat['arf']
        timepoints = eeg_dat[dp.USE_UN_BASELINED[1]] # --> 'time_long' = unbaselined
        settings = eeg_dat['settings']
        sfreq = settings['srate']

        # Get subject accuracies and eeg with artifacts rejected
        sub_accs, sub_eeg = dp.reject_bad_trials(eeg_dat[dp.USE_UN_BASELINED[0]], beh_dat, artifacts)

        # Calculate ppt average performance to average list
        alpha_temp = np.nanmean(beh_dat['data']['trialAcc'])

        # check if participant has alpha power
        peak, model = dp.check_sub_peak(sub_eeg, dp.ALPHA, sfreq, dp.SETS)
        print("Subject number " + str(sub) + ": " + peak)
        if peak == "Yay peak!":
            sub_with_alpha_peak.append(sub)

            # # Check if subject has enough trials per performance condition: if not --> skip/continue
            # if dp.reject_subject(correct_eeg):
            #     print("Subject rejected " + str(sub))
            #     sub_rejected.append(sub)
            #     continue

            # FOOOF the data per lateralization and performance
            df_1c, psd_bas_1c, psd_ret_1c, freq_x_axis = dp.get_fooof(sub, 1,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.CONTRA, analysis_type)
            df_1i, psd_bas_1i, psd_ret_1i, _ = dp.get_fooof(sub, 1,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.IPSI, analysis_type)
            df_3c, psd_bas_3c, psd_ret_3c, _ = dp.get_fooof(sub, 3,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.CONTRA, analysis_type)
            df_3i, psd_bas_3i, psd_ret_3i, _ = dp.get_fooof(sub, 3,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.IPSI, analysis_type)
            df_6c, psd_bas_6c, psd_ret_6c, _ = dp.get_fooof(sub, 6,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.CONTRA, analysis_type)
            df_6i, psd_bas_6i, psd_ret_6i, _ = dp.get_fooof(sub, 6,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.IPSI, analysis_type)
            

            # concatenate dataframes over lateralization --> so per performance
            df_1 = pd.concat([df_1c, df_1i])
            df_3 = pd.concat([df_3c, df_3i])
            df_6 = pd.concat([df_6c, df_6i])

            # Also, check the model fits and add column to dataframes per condition/performance
            df_ext_1 = dp.get_bad_fits(df_1, dp.ALPHA)
            df_ext_3 = dp.get_bad_fits(df_3, dp.ALPHA)
            df_ext_6 = dp.get_bad_fits(df_6, dp.ALPHA)

            # Check if subject still has enough trials
            if dp.reject_subject(dfs=[df_ext_1, df_ext_3, df_ext_6], part='after', frequency=dp.ALPHA):
                print("Subject rejected " + str(sub))
                sub_rejected_after_r2.append(sub)
                continue
            
            sub_included.append(sub) # subject has a peak and enough trials

            # Concatenate conditions for this subject
            sub_df = pd.concat([df_1, df_3, df_6])

            # Add aditional data to dataframes for each subject
            # 1. Average working memory performance over all trials
            sub_df['wm_capacity'] = alpha_temp
            # 2. Whether this subject at theta oscillations or not
            sub_df['osc_presence'] = True # < -- needs to be if/else statement if I'm doing this in a single loop

            # Append to list of dataframes where all subjects are saved
            df_all_alpha.append(sub_df)

            # Redudant, but also add alpha average performance to this list:
            alpha_avg_perform.append(alpha_temp)


            ### Now we also need to clean the power spectra arrays from bad fits
            psd_bas_clean_1, psd_ret_clean_1 = dp.del_bad_psd(df_ext_1, [psd_bas_1c, psd_bas_1i], 
                                                            [psd_ret_1c, psd_ret_1i], frequency=dp.ALPHA)
            psd_bas_clean_3, psd_ret_clean_3 = dp.del_bad_psd(df_ext_3, [psd_bas_3c, psd_bas_3i], 
                                                            [psd_ret_3c, psd_ret_3i], frequency=dp.ALPHA)
            psd_bas_clean_6, psd_ret_clean_6 = dp.del_bad_psd(df_ext_6, [psd_bas_6c, psd_bas_6i], 
                                                            [psd_ret_6c, psd_ret_6i], frequency=dp.ALPHA)
            
            # psd_exp1_alpha[ppn, freq_axis, setsize, lateralization, timeperiod]
                    # freq_axis: 138 stamps of frequencies
                    # setsize: 0 = 1, 1 = 3, 2 = 6                
                    # lateralization: 0 = contra, 1 = ipsi
                    # timeperiod: 0 = baseline, 1 = retention
            # for set-size 1
            psd_exp1_alpha[power_detected, :, 0, 0, 0] = np.mean(psd_bas_clean_1[0], axis=0)
            psd_exp1_alpha[power_detected, :, 0, 0, 1] = np.mean(psd_ret_clean_1[0], axis=0)
            psd_exp1_alpha[power_detected, :, 0, 1, 0] = np.mean(psd_bas_clean_1[1], axis=0)
            psd_exp1_alpha[power_detected, :, 0, 1, 1] = np.mean(psd_ret_clean_1[1], axis=0)
            # for set-size 3
            psd_exp1_alpha[power_detected, :, 1, 0, 0] = np.mean(psd_bas_clean_3[0], axis=0)
            psd_exp1_alpha[power_detected, :, 1, 0, 1] = np.mean(psd_ret_clean_3[0], axis=0)
            psd_exp1_alpha[power_detected, :, 1, 1, 0] = np.mean(psd_bas_clean_3[1], axis=0)
            psd_exp1_alpha[power_detected, :, 1, 1, 1] = np.mean(psd_ret_clean_3[1], axis=0)
            # for set-size 6
            psd_exp1_alpha[power_detected, :, 2, 0, 0] = np.mean(psd_bas_clean_6[0], axis=0)
            psd_exp1_alpha[power_detected, :, 2, 0, 1] = np.mean(psd_ret_clean_6[0], axis=0)
            psd_exp1_alpha[power_detected, :, 2, 1, 0] = np.mean(psd_bas_clean_6[1], axis=0)
            psd_exp1_alpha[power_detected, :, 2, 1, 1] = np.mean(psd_ret_clean_6[1], axis=0)
            
            power_detected += 1

        # else --> no alpha oscillation detected in this participant, and save psd only
        else:
            sub_no_alpha_peak.append(sub)
            
            df_no_alpha_1c, no_alpha_psd_bas_1c, no_alpha_psd_ret_1c, _ = dp.get_fooof(sub, 1,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.CONTRA, analysis_type)
            df_no_alpha_1i, no_alpha_psd_bas_1i, no_alpha_psd_ret_1i, _ = dp.get_fooof(sub, 1,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.IPSI, analysis_type)
            df_no_alpha_3c, no_alpha_psd_bas_3c, no_alpha_psd_ret_3c, _ = dp.get_fooof(sub, 3,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.CONTRA, analysis_type)
            df_no_alpha_3i, no_alpha_psd_bas_3i, no_alpha_psd_ret_3i, _ = dp.get_fooof(sub, 3,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.IPSI, analysis_type)
            df_no_alpha_6c, no_alpha_psd_bas_6c, no_alpha_psd_ret_6c, _ = dp.get_fooof(sub, 6,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.CONTRA, analysis_type)
            df_no_alpha_6i, no_alpha_psd_bas_6i, no_alpha_psd_ret_6i, _ = dp.get_fooof(sub, 6,
                                                sub_eeg, timepoints, dp.ALPHA, sfreq, dp.IPSI, analysis_type)

            ## Save no alpha details
            # concatenate dataframes over lateralization --> so per performance
            df_1 = pd.concat([df_no_alpha_1c, df_no_alpha_1i])
            df_3 = pd.concat([df_no_alpha_3c, df_no_alpha_3i])
            df_6 = pd.concat([df_no_alpha_6c, df_no_alpha_6i])

            # Also, check the model fits and add column to dataframes per condition/performance
            df_ext_1 = dp.get_bad_fits(df_1, dp.ALPHA)
            df_ext_3 = dp.get_bad_fits(df_3, dp.ALPHA)
            df_ext_6 = dp.get_bad_fits(df_6, dp.ALPHA)

            sub_df = pd.concat([df_ext_1, df_ext_3, df_ext_6])

            # Add aditional data to dataframes for each subject
            # 1. Average working memory performance over all trials
            sub_df['wm_capacity'] = alpha_temp
            # 2. Whether this subject at theta oscillations or not
            sub_df['osc_presence'] = False # < -- needs to be if/else statement if I'm doing this in a single loop

            # Append to list of dataframes where all subjects are saved
            df_all_no_alpha.append(sub_df)

            # Redudant, but also add alpha average performance to this list:
            no_alpha_avg_perform.append(alpha_temp)


            ### Now we also need to clean the power spectra arrays from bad fits
            psd_bas_clean_1, psd_ret_clean_1 = dp.del_bad_psd(df_ext_1, [psd_bas_1c, psd_bas_1i], 
                                                            [psd_ret_1c, psd_ret_1i], frequency=dp.ALPHA)
            psd_bas_clean_3, psd_ret_clean_3 = dp.del_bad_psd(df_ext_3, [psd_bas_3c, psd_bas_3i], 
                                                            [psd_ret_3c, psd_ret_3i], frequency=dp.ALPHA)
            psd_bas_clean_6, psd_ret_clean_6 = dp.del_bad_psd(df_ext_6, [psd_bas_6c, psd_bas_6i], 
                                                            [psd_ret_6c, psd_ret_6i], frequency=dp.ALPHA)
            
            # psd_exp1_alpha[ppn, freq_axis, setsize, lateralization, timeperiod]
                    # freq_axis: 138 stamps of frequencies
                    # setsize: 0 = 1, 1 = 3, 2 = 6                
                    # lateralization: 0 = contra, 1 = ipsi
                    # timeperiod: 0 = baseline, 1 = retention
            # for set-size 1
            psd_exp1_no_alpha[power_detected, :, 0, 0, 0] = np.mean(psd_bas_clean_1[0], axis=0)
            psd_exp1_no_alpha[power_detected, :, 0, 0, 1] = np.mean(psd_ret_clean_1[0], axis=0)
            psd_exp1_no_alpha[power_detected, :, 0, 1, 0] = np.mean(psd_bas_clean_1[1], axis=0)
            psd_exp1_no_alpha[power_detected, :, 0, 1, 1] = np.mean(psd_ret_clean_1[1], axis=0)
            # for set-size 3
            psd_exp1_no_alpha[power_detected, :, 1, 0, 0] = np.mean(psd_bas_clean_3[0], axis=0)
            psd_exp1_no_alpha[power_detected, :, 1, 0, 1] = np.mean(psd_ret_clean_3[0], axis=0)
            psd_exp1_no_alpha[power_detected, :, 1, 1, 0] = np.mean(psd_bas_clean_3[1], axis=0)
            psd_exp1_no_alpha[power_detected, :, 1, 1, 1] = np.mean(psd_ret_clean_3[1], axis=0)
            # for set-size 6
            psd_exp1_no_alpha[power_detected, :, 2, 0, 0] = np.mean(psd_bas_clean_6[0], axis=0)
            psd_exp1_no_alpha[power_detected, :, 2, 0, 1] = np.mean(psd_ret_clean_6[0], axis=0)
            psd_exp1_no_alpha[power_detected, :, 2, 1, 0] = np.mean(psd_bas_clean_6[1], axis=0)
            psd_exp1_no_alpha[power_detected, :, 2, 1, 1] = np.mean(psd_ret_clean_6[1], axis=0)
            
            no_power_detected += 1


    ############### Clean and Save ################

    # Delete rows (ppns) with only nan: np.array with psd's for plotting
    to_delete_psd_alpha = np.where(np.isnan(psd_exp1_alpha[:,0,0,0,0]))
    psd_exp1_alpha = np.delete(psd_exp1_alpha, to_delete_psd_alpha, axis=0)

    # Delete rows (ppns) with only nan: np.array with psd's for plotting
    to_delete_psd_no_alpha = np.where(np.isnan(psd_exp1_no_alpha[:,0,0,0,0]))
    psd_exp1_no_alpha = np.delete(psd_exp1_no_alpha, to_delete_psd_no_alpha, axis=0)

    # Create dictionary out of included/rejected subjects
    alpha_exp1_inc = {'included': sub_included,
                    'with_peak': sub_with_alpha_peak,
                    'wo_peak': sub_no_alpha_peak,
                    'rejected': sub_rejected,
                    'rejected_r2': sub_rejected_after_r2}


    # Use pickle to save list of dataframes;save pkl files 
    file_name = 'alpha_all_dfs.pkl'

    # for ppn with alpha peak
    with open (save_path + file_name, 'wb') as f:
        pkl.dump(df_all_alpha, f)

    # for ppn without alpha peak
    file_name = 'no_alpha_all_dfs.pkl'

    with open (save_path + file_name, 'wb') as f:
        pkl.dump(df_all_no_alpha, f)

    np.save(save_path + 'alpha_exp1_psd', psd_exp1_alpha)
    np.save(save_path + 'no_alpha_exp1_psd', psd_exp1_no_alpha)
    np.save(save_path + 'freq_x_axis', freq_x_axis)
    np.save(save_path + 'alpha_exp1_ppns', alpha_exp1_inc)


    print('ALPHA part is done and saved! :D')


    '''Theta: run specparam model and generate parameters'''

    ############### Initialize ################

    # Premade variable with subjects who exhibit oscillatory theta power (manually inspected)
    num_subjects = 31  # total subjects
    subs = len(dp.SUB_THETA) # either sub_theta or sub_no_theta

    # Initialize empty array structure to save psd's
    psd_exp1_theta = np.full((subs, 138, 3, 2), np.nan)
    power_detected = 0  # counter to fill in above array

    # Initialize empty lists for data storage
    theta_avg_perform = []
    df_all_theta = []

    # Other variables that have to be set at 0 before the loop
    sub_included = []
    sub_rejected = []
    sub_rejected_without = []
    sub_rejected_after_r2 = []

    ############### Start for-loop ################

    for sub in dp.SUB_THETA:  # either dp.SUB_THETA or dp.SUB_NO_THETA to iterate over ppns
    # for sub in range(1, num_subjects + 1):  # to create average PSD for all participants
        print(sub)

        if sub == 24:
            print('skip ppn 24 due to bad behavioral file')
            continue

        eeg_dat = None
        beh_dat = None

        # Load in data
        try:
            eeg_dat, beh_dat = dp.load_data(sub)
        except FileNotFoundError:
            warnings.warn('File not found')
            continue

        # Add each ppt average performance to average list
        theta_temp = np.nanmean(beh_dat['data']['trialAcc'])
        theta_avg_perform.append(theta_temp)

        # Extract helpful data from the EEG file
        channels = eeg_dat['chanLabels']
        artifacts = eeg_dat['arf']
        timepoints = eeg_dat[dp.USE_UN_BASELINED[1]] # --> 'time_long' = unbaselined
        settings = eeg_dat['settings']
        sfreq = settings['srate']

        # Get subject accuracies and eeg with artifacts rejected
        sub_accs, sub_eeg = dp.reject_bad_trials(eeg_dat[dp.USE_UN_BASELINED[0]], beh_dat, artifacts)
        
        # check if participant has theta power <-- manually done by looping over specific ppn list
        # make a figure and save it if running over all ppn's
        # model_fit.plot()
        # plt.title('ppn: ' + str(sub) + " " + peak)
        # plt.axvspan(dp.THETA[0], dp.THETA[1], alpha = 0.2, color = 'red')
        # file_path = 'saved_files/exp1_check_peak/'
        # file_name = 'ppn_' + str(sub) + '_psd_model.jpeg'
        # plt.savefig(file_path + file_name, format='jpeg')

        # Check if subject has enough trials per setsize condition
        if dp.reject_subject(sub_eeg):
            print("Subject rejected " + str(sub))
            sub_rejected.append(sub)
            continue

        # FOOOF the data per performance
        df_1, psd_bas_1, psd_ret_1, freq_x_axis = dp.get_fooof(sub, 1,
                                                sub_eeg, timepoints, dp.THETA, sfreq, analysis_type=analysis_type)
        df_3, psd_bas_3, psd_ret_3, _ = dp.get_fooof(sub, 3,
                                                sub_eeg, timepoints, dp.THETA, sfreq, analysis_type=analysis_type)
        df_6, psd_bas_6, psd_ret_6, _ = dp.get_fooof(sub, 6,
                                                sub_eeg, timepoints, dp.THETA, sfreq, analysis_type=analysis_type)
        

        # Also, check the model fits and add column to dataframes per condition/performance
        df_ext_1 = dp.get_bad_fits(df_1, dp.THETA)
        df_ext_3 = dp.get_bad_fits(df_3, dp.THETA)
        df_ext_6 = dp.get_bad_fits(df_6, dp.THETA)

        # Check if subject still has enough trials
        if dp.reject_subject(dfs=[df_ext_1, df_ext_3, df_ext_6], part='after', frequency=dp.THETA):
            print("Subject rejected " + str(sub))
            sub_rejected_after_r2.append(sub)
            continue

        sub_included.append(sub)

        # Concatenate conditions for this subject
        sub_df = pd.concat([df_ext_1, df_ext_3, df_ext_6])

        # Add aditional data to dataframes for each subject
        # 1. Average working memory performance over all trials
        sub_df['wm_capacity'] = theta_temp
        # 2. Whether this subject at theta oscillations or not
        sub_df['osc_presence'] = True # < -- needs to be if/else statement if I'm doing this in a single loop

        # Append to list of dataframes where all subjects are saved
        df_all_theta.append(sub_df)


        ### Now we also need to clean the power spectra arrays from bad fits
        psd_bas_clean_1, psd_ret_clean_1 = dp.del_bad_psd(df_ext_1, psd_bas_1, psd_ret_1, frequency=dp.THETA)
        psd_bas_clean_3, psd_ret_clean_3 = dp.del_bad_psd(df_ext_3, psd_bas_3, psd_ret_3, frequency=dp.THETA)
        psd_bas_clean_6, psd_ret_clean_6 = dp.del_bad_psd(df_ext_6, psd_bas_6, psd_ret_6, frequency=dp.THETA)

        # Insert into an array we can save; and take average over trials
                # psd_exp2_no_theta[ppn, freq_axis, setsize, timeperiod]
                # freq_axis: 138 stamps of frequencies
                # setsize: 0 = 1, 1 = 3, 2 = 6                
                # timeperiod: 0 = baseline, 1 = retention
        psd_exp1_theta[power_detected, :, 0, 0] = np.mean(psd_bas_clean_1, axis=0)
        psd_exp1_theta[power_detected, :, 0, 1] = np.mean(psd_ret_clean_1, axis=0)
        psd_exp1_theta[power_detected, :, 1, 0] = np.mean(psd_bas_clean_3, axis=0)
        psd_exp1_theta[power_detected, :, 1, 1] = np.mean(psd_ret_clean_3, axis=0)
        psd_exp1_theta[power_detected, :, 2, 0] = np.mean(psd_bas_clean_6, axis=0)
        psd_exp1_theta[power_detected, :, 2, 1] = np.mean(psd_ret_clean_6, axis=0)

        power_detected += 1 # add to counter

    print('We did it! Participants that have THETA oscillatory activity! :D')

    ############### Save data ################

    # Delete rows (ppns) with only nan
    to_delete = np.where(np.isnan(psd_exp1_theta[:,0,0,0]))
    psd_exp1_theta = np.delete(psd_exp1_theta, to_delete, axis=0)

    # Save subject lists as dictionary for later
    theta_exp1_inc = {'considered': dp.SUB_THETA,
                    'included': sub_included,
                    'rejected': sub_rejected,
                    'rejected_r2': sub_rejected_after_r2}


    # Use pickle to save list of dataframes;save pkl files
    file_name = 'theta_all_dfs.pkl'

    with open (save_path + file_name, 'wb') as f:
        pkl.dump(df_all_theta, f)

    np.save(save_path + 'theta_exp1_psd', psd_exp1_theta)
    np.save(save_path + 'theta_exp1_avg_perf', theta_avg_perform)
    np.save(save_path + 'freq_x_axis', freq_x_axis)
    np.save(save_path + 'theta_exp1_ppns.npy', theta_exp1_inc)

    print('THETA part is done and saved! :D')



    ''' NO Theta: run specparam model and generate parameters'''

    ############### Initialize ################

    # Premade variable with subjects who exhibit oscillatory theta power (manually inspected)
    num_subjects = 31  # total subjects
    subs = len(dp.SUB_NO_THETA) # either sub_theta or sub_no_theta

    # Initialize empty array structure to save psd's
    psd_exp1_no_theta = np.full((subs, 138, 3, 2), np.nan)
    counter = 0  # counter to fill in above array

    # Initialize empty lists for data storage
    no_theta_avg_perform = []
    df_no_theta = []

    # Other variables that have to be set at 0 before the loop
    sub_included = []
    sub_rejected = []
    sub_rejected_without = []
    sub_rejected_after_r2 = []

    ############### Start for-loop ################

    for sub in dp.SUB_NO_THETA:  # either dp.SUB_THETA or dp.SUB_NO_THETA to iterate over ppns
    # for sub in range(1, num_subjects + 1):  # to create average PSD for all participants
        print(sub)

        if sub == 24:
            print('skip ppn 24 due to bad behavioral file')
            continue

        eeg_dat = None
        beh_dat = None

        # Load in data
        try:
            eeg_dat, beh_dat = dp.load_data(sub)
        except FileNotFoundError:
            warnings.warn('File not found')
            continue

        # Add each ppt average performance to average list
        theta_temp = np.nanmean(beh_dat['data']['trialAcc'])
        no_theta_avg_perform.append(theta_temp)

        # Extract helpful data from the EEG file
        channels = eeg_dat['chanLabels']
        artifacts = eeg_dat['arf']
        timepoints = eeg_dat[dp.USE_UN_BASELINED[1]] # --> 'time_long' = unbaselined
        settings = eeg_dat['settings']
        sfreq = settings['srate']

        # Get subject accuracies and eeg with artifacts rejected
        sub_accs, sub_eeg = dp.reject_bad_trials(eeg_dat[dp.USE_UN_BASELINED[0]], beh_dat, artifacts)
        
        # check if participant has theta power <-- manually done by looping over specific ppn list
        # make a figure and save it if running over all ppn's
        # model_fit.plot()
        # plt.title('ppn: ' + str(sub) + " " + peak)
        # plt.axvspan(dp.THETA[0], dp.THETA[1], alpha = 0.2, color = 'red')
        # file_path = 'saved_files/exp1_check_peak/'
        # file_name = 'ppn_' + str(sub) + '_psd_model.jpeg'
        # plt.savefig(file_path + file_name, format='jpeg')

        # Check if subject has enough trials per setsize condition
        if dp.reject_subject(sub_eeg):
            print("Subject rejected " + str(sub))
            sub_rejected.append(sub)
            continue

        # FOOOF the data per performance
        df_1, psd_bas_1, psd_ret_1, freq_x_axis = dp.get_fooof(sub, 1,
                                                sub_eeg, timepoints, dp.THETA, sfreq, analysis_type=analysis_type)
        df_3, psd_bas_3, psd_ret_3, _ = dp.get_fooof(sub, 3,
                                                sub_eeg, timepoints, dp.THETA, sfreq, analysis_type=analysis_type)
        df_6, psd_bas_6, psd_ret_6, _ = dp.get_fooof(sub, 6,
                                                sub_eeg, timepoints, dp.THETA, sfreq, analysis_type=analysis_type)
        

        # Also, check the model fits and add column to dataframes per condition/performance
        df_ext_1 = dp.get_bad_fits(df_1, dp.THETA)
        df_ext_3 = dp.get_bad_fits(df_3, dp.THETA)
        df_ext_6 = dp.get_bad_fits(df_6, dp.THETA)

        # Check if subject still has enough trials
        if dp.reject_subject(dfs=[df_ext_1, df_ext_3, df_ext_6], part='after', frequency=dp.THETA):
            print("Subject rejected " + str(sub))
            sub_rejected_after_r2.append(sub)
            continue

        sub_included.append(sub)

        # Concatenate conditions for this subject
        sub_df = pd.concat([df_ext_1, df_ext_3, df_ext_6])

        # Add aditional data to dataframes for each subject
        # 1. Average working memory performance over all trials
        sub_df['wm_capacity'] = theta_temp
        # 2. Whether this subject at theta oscillations or not
        sub_df['osc_presence'] = False # < -- needs to be if/else statement if I'm doing this in a single loop

        # Append to list of dataframes where all subjects are saved
        df_no_theta.append(sub_df)


        ### Now we also need to clean the power spectra arrays from bad fits
        psd_bas_clean_1, psd_ret_clean_1 = dp.del_bad_psd(df_ext_1, psd_bas_1, psd_ret_1, frequency=dp.THETA)
        psd_bas_clean_3, psd_ret_clean_3 = dp.del_bad_psd(df_ext_3, psd_bas_3, psd_ret_3, frequency=dp.THETA)
        psd_bas_clean_6, psd_ret_clean_6 = dp.del_bad_psd(df_ext_6, psd_bas_6, psd_ret_6, frequency=dp.THETA)

        # Insert into an array we can save; and take average over trials
                # psd_exp2_no_theta[ppn, freq_axis, setsize, timeperiod]
                # freq_axis: 138 stamps of frequencies
                # setsize: 0 = 1, 1 = 3, 2 = 6                
                # timeperiod: 0 = baseline, 1 = retention
        psd_exp1_no_theta[counter, :, 0, 0] = np.mean(psd_bas_clean_1, axis=0)
        psd_exp1_no_theta[counter, :, 0, 1] = np.mean(psd_ret_clean_1, axis=0)
        psd_exp1_no_theta[counter, :, 1, 0] = np.mean(psd_bas_clean_3, axis=0)
        psd_exp1_no_theta[counter, :, 1, 1] = np.mean(psd_ret_clean_3, axis=0)
        psd_exp1_no_theta[counter, :, 2, 0] = np.mean(psd_bas_clean_6, axis=0)
        psd_exp1_no_theta[counter, :, 2, 1] = np.mean(psd_ret_clean_6, axis=0)

        counter += 1

    print('We did it! Participants that have THETA oscillatory activity! :D')

    ############### Save data ################

    # Delete rows (ppns) with only nan
    to_delete = np.where(np.isnan(psd_exp1_no_theta[:,0,0,0]))
    psd_exp1_no_theta = np.delete(psd_exp1_no_theta, to_delete, axis=0)

    # Save subject lists as dictionary for later
    no_theta_exp1_inc = {'considered': dp.SUB_NO_THETA,
                    'included': sub_included,
                    'rejected': sub_rejected,
                    'rejected_r2': sub_rejected_after_r2}


    # Use pickle to save list of dataframes;save pkl files 
    file_name = 'no_theta_all_dfs.pkl'

    with open (save_path + file_name, 'wb') as f:
        pkl.dump(df_no_theta, f)

    np.save(save_path + 'no_theta_exp1_psd', psd_exp1_no_theta)
    np.save(save_path + 'no_theta_exp1_avg_perf', no_theta_avg_perform)
    np.save(save_path + 'freq_x_axis', freq_x_axis)
    np.save(save_path + 'no_theta_exp1_ppns.npy', no_theta_exp1_inc)

    print('NO THETA part is done and saved! :D')


