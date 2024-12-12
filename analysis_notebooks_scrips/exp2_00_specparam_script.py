'''Script to run over participants for both alpha power, theta power, and the group
that does not show theta power of experiment 2'''

# import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import random
import warnings
# imports module and variables from exp2_settings.py
from colourfulbirds import exp2_functionsforFOOOF as dp

random.seed(0)

'''Alpha: run specparam model and generate parameters'''

############## Initialize ################


# LOOP OVER MAIN ANALYSIS AND CONTROL ANALYSIS FOR WINDOW SIZE
for analysis_type in ['main_analysis']: # , 'control_analysis'

    # settings per main or control analysis
    if analysis_type == 'main_analysis':
        save_path = 'saved_files/fooof_results/exp2_output/main_analysis/'
    else:
        save_path = 'saved_files/fooof_results/exp2_output/control_analysis/'


    num_subjects = 48
    subs = 31 # 31 subjects after all the rejections and wether they have alpha or not
    # Initialize empty array structure to save psd's
    psd_exp2_alpha = np.full((subs, 138, 2, 2, 2), np.nan)
    psd_exp2_no_alpha = np.full((subs, 138, 2, 2, 2), np.nan)
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

        # create format with 4 lists --> left good, left poor, right good, right poor
        correct_eeg = dp.extract_correct_eeg(sub_accs, sub_eeg)

        # Calculate ppt average performance to average list
        alpha_temp = np.nanmean(beh_dat['data']['trialAcc'])

        # check if participant has alpha power
        peak, model = dp.check_sub_peak(sub_eeg, dp.ALPHA, sfreq, dp.SETS)
        print("Subject number " + str(sub) + ": " + peak)
        if peak == "Yay peak!":
            sub_with_alpha_peak.append(sub)

            # Check if subject has enough trials per performance condition: if not --> skip/continue
            if dp.reject_subject(correct_eeg):
                print("Subject rejected " + str(sub))
                sub_rejected.append(sub)
                continue


            # FOOOF the data per lateralization and performance
            df_gc, psd_bas_gc, psd_ret_gc, freq_x_axis = dp.get_fooof(sub, dp.GOOD,
                sub_eeg, sub_accs,timepoints,  dp.ALPHA, sfreq, dp.CONTRA, analysis_type)
            df_gi, psd_bas_gi, psd_ret_gi, _ = dp.get_fooof(sub, dp.GOOD,
                sub_eeg, sub_accs, timepoints, dp.ALPHA, sfreq, dp.IPSI, analysis_type)
            df_pc, psd_bas_pc, psd_ret_pc, _ = dp.get_fooof(sub, dp.POOR,
                sub_eeg, sub_accs, timepoints, dp.ALPHA, sfreq, dp.CONTRA, analysis_type)
            df_pi, psd_bas_pi, psd_ret_pi, _ = dp.get_fooof(sub, dp.POOR,
                sub_eeg, sub_accs, timepoints, dp.ALPHA, sfreq, dp.IPSI, analysis_type)

            # concatenate dataframes over lateralization --> so per performance
            df_g = pd.concat([df_gc, df_gi])
            df_p = pd.concat([df_pc, df_pi])

            # Also, check the model fits and add column to dataframes per condition/performance
            df_ext_g = dp.get_bad_fits(df_g, dp.ALPHA)
            df_ext_p = dp.get_bad_fits(df_p, dp.ALPHA)

            # Check if subject still has enough trials
            if dp.reject_subject(dfs=[df_ext_g, df_ext_p], part='after', frequency=dp.ALPHA):
                print("Subject rejected " + str(sub))
                sub_rejected_after_r2.append(sub)
                continue
            
            sub_included.append(sub) # subject has a peak and enough trials

            # Concatenate conditions for this subject
            sub_df = pd.concat([df_p, df_g])

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
            psd_bas_clean_p, psd_ret_clean_p = dp.del_bad_psd(df_ext_p, [psd_bas_pc, psd_bas_pi], 
                                                            [psd_ret_pc, psd_ret_pi], frequency=dp.ALPHA)
            psd_bas_clean_g, psd_ret_clean_g = dp.del_bad_psd(df_ext_g, [psd_bas_gc, psd_bas_gi], 
                                                            [psd_ret_gc, psd_ret_gi], frequency=dp.ALPHA)

            # psd_exp2_alpha[ppn, freq_axis, performance, lateralization, timeperiod]
                    # freq_axis: 138 stamps of frequencies
                    # performance: 0 = poor, 1 = good
                    # lateralization: 0 = contra, 1 = ipsi
                    # timeperiod: 0 = baseline, 1 = retention
            # for poor performance
            psd_exp2_alpha[power_detected, :, 0, 0, 0] = np.mean(psd_bas_clean_p[0], axis=0)
            psd_exp2_alpha[power_detected, :, 0, 0, 1] = np.mean(psd_ret_clean_p[0], axis=0)
            psd_exp2_alpha[power_detected, :, 0, 1, 0] = np.mean(psd_bas_clean_p[1], axis=0)
            psd_exp2_alpha[power_detected, :, 0, 1, 1] = np.mean(psd_ret_clean_p[1], axis=0)
            # for good performance
            psd_exp2_alpha[power_detected, :, 1, 0, 0] = np.mean(psd_bas_clean_g[0], axis=0)
            psd_exp2_alpha[power_detected, :, 1, 0, 1] = np.mean(psd_ret_clean_g[0], axis=0)
            psd_exp2_alpha[power_detected, :, 1, 1, 0] = np.mean(psd_bas_clean_g[1], axis=0)
            psd_exp2_alpha[power_detected, :, 1, 1, 1] = np.mean(psd_ret_clean_g[1], axis=0)

            power_detected += 1

        # else --> no alpha oscillation detected in this participant, and save psd only
        else:
            sub_no_alpha_peak.append(sub)
            
            df_no_alpha_gc, no_alpha_psd_bas_gc, no_alpha_psd_ret_gc, _ = dp.get_fooof(sub, dp.GOOD,
                sub_eeg, sub_accs, timepoints, dp.ALPHA, sfreq, dp.CONTRA, analysis_type)
            df_no_alpha_gi, no_alpha_psd_bas_gi, no_alpha_psd_ret_gi, _ = dp.get_fooof(sub, dp.GOOD,
                sub_eeg, sub_accs, timepoints, dp.ALPHA, sfreq, dp.IPSI, analysis_type)
            df_no_alpha_pc, no_alpha_psd_bas_pc, no_alpha_psd_ret_pc, _ = dp.get_fooof(sub, dp.POOR,
                sub_eeg, sub_accs, timepoints, dp.ALPHA, sfreq, dp.CONTRA, analysis_type)
            df_no_alpha_pi, no_alpha_psd_bas_pi, no_alpha_psd_ret_pi, _ = dp.get_fooof(sub, dp.POOR,
                sub_eeg, sub_accs, timepoints, dp.ALPHA, sfreq, dp.IPSI, analysis_type)
        
            ## Save no alpha details
            # concatenate dataframes over lateralization --> so per performance
            df_g = pd.concat([df_no_alpha_gc, df_no_alpha_gi])
            df_p = pd.concat([df_no_alpha_pc, df_no_alpha_pi])

            # Also, check the model fits and add column to dataframes per condition/performance
            df_ext_g = dp.get_bad_fits(df_g, dp.ALPHA)
            df_ext_p = dp.get_bad_fits(df_p, dp.ALPHA)

            sub_df = pd.concat([df_ext_g, df_ext_p])

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
            psd_bas_clean_p, psd_ret_clean_p = dp.del_bad_psd(df_ext_p, [no_alpha_psd_bas_pc, no_alpha_psd_bas_pi], 
                                                            [no_alpha_psd_ret_pc, no_alpha_psd_ret_pi], frequency=dp.ALPHA)
            psd_bas_clean_g, psd_ret_clean_g = dp.del_bad_psd(df_ext_g, [no_alpha_psd_bas_gc, no_alpha_psd_bas_gi], 
                                                            [no_alpha_psd_ret_gc, no_alpha_psd_ret_gi], frequency=dp.ALPHA)

            # psd_exp2_alpha[ppn, freq_axis, performance, lateralization, timeperiod]
                    # freq_axis: 138 stamps of frequencies
                    # performance: 0 = poor, 1 = good
                    # lateralization: 0 = contra, 1 = ipsi
                    # timeperiod: 0 = baseline, 1 = retention
            # for poor performance
            psd_exp2_no_alpha[no_power_detected, :, 0, 0, 0] = np.mean(psd_bas_clean_p[0], axis=0)
            psd_exp2_no_alpha[no_power_detected, :, 0, 0, 1] = np.mean(psd_ret_clean_p[0], axis=0)
            psd_exp2_no_alpha[no_power_detected, :, 0, 1, 0] = np.mean(psd_bas_clean_p[1], axis=0)
            psd_exp2_no_alpha[no_power_detected, :, 0, 1, 1] = np.mean(psd_ret_clean_p[1], axis=0)
            # for good performance
            psd_exp2_no_alpha[no_power_detected, :, 1, 0, 0] = np.mean(psd_bas_clean_g[0], axis=0)
            psd_exp2_no_alpha[no_power_detected, :, 1, 0, 1] = np.mean(psd_ret_clean_g[0], axis=0)
            psd_exp2_no_alpha[no_power_detected, :, 1, 1, 0] = np.mean(psd_bas_clean_g[1], axis=0)
            psd_exp2_no_alpha[no_power_detected, :, 1, 1, 1] = np.mean(psd_ret_clean_g[1], axis=0)

            no_power_detected += 1


    ############### Clean and Save ################

    # Delete rows (ppns) with only nan: np.array with psd's for plotting
    to_delete_psd_alpha = np.where(np.isnan(psd_exp2_alpha[:,0,0,0,0]))
    psd_exp2_alpha = np.delete(psd_exp2_alpha, to_delete_psd_alpha, axis=0)

    # Delete rows (ppns) with only nan: np.array with psd's for plotting
    to_delete_psd_no_alpha = np.where(np.isnan(psd_exp2_no_alpha[:,0,0,0,0]))
    psd_exp2_no_alpha = np.delete(psd_exp2_no_alpha, to_delete_psd_no_alpha, axis=0)


    # Create dictionary out of included/rejected subjects
    alpha_exp2_inc = {'included': sub_included,
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

    np.save(save_path + 'alpha_exp2_psd', psd_exp2_alpha)
    np.save(save_path + 'no_alpha_exp2_psd', psd_exp2_no_alpha)
    np.save(save_path + 'freq_x_axis', freq_x_axis)
    np.save(save_path + 'alpha_exp2_ppns', alpha_exp2_inc)


    print('ALPHA part is done and saved! :D')


    '''Theta: run specparam model and generate parameters'''

    ############### Initialize ################

    # Premade variable with subjects who exhibit oscillatory theta power (manually inspected)
    num_subjects = 48
    subs = len(dp.SUB_THETA) # either sub_theta or sub_no_theta

    # Initialize empty array structure to save psd's
    psd_exp2_theta = np.full((subs, 138, 2, 2), np.nan)
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
    # for sub in range(1, num_subjects + 1):  # to create average PSD for all participants'
        print(sub)
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

        # create format with 4 lists --> left good, left poor, right good, right poor
        correct_eeg = dp.extract_correct_eeg(sub_accs, sub_eeg)
        
        # check if participant has theta power <-- manually done by looping over specific ppn list
        # Keep code here so we can create the figures
    #     peak, model_fit = dp.check_sub_peak(sub_eeg, dp.THETA, sfreq, dp.SETS)
    #     print("Subject number " + str(sub) + ": " + peak)

    #     # make a figure and save it if running over all ppn's
    #     model_fit.plot()
    #     plt.title('ppn: ' + str(sub) + " " + peak)
    #     plt.axvspan(dp.THETA[0], dp.THETA[1], alpha = 0.2, color = 'red')
    #     file_path = 'saved_files/exp2_check_peak/temp/'
    #     file_name = 'ppn_' + str(sub) + '_psd_model.jpeg'
    #     plt.savefig(file_path + file_name, format='jpeg')

        # Check if subject has enough trials per performance condition: if not --> skip/continue
        if dp.reject_subject(correct_eeg):
            print("Subject rejected " + str(sub))
            sub_rejected.append(sub)
            continue

        # FOOOF the data per performance
        df_g, psd_bas_g, psd_ret_g, freq_x_axis = dp.get_fooof(sub, dp.GOOD,
            sub_eeg, sub_accs, timepoints, dp.THETA, sfreq, analysis_type=analysis_type)
        df_p, psd_bas_p, psd_ret_p, _ = dp.get_fooof(sub, dp.POOR,
            sub_eeg, sub_accs, timepoints, dp.THETA, sfreq, analysis_type=analysis_type)
        
        # Also, check the model fits and add column to dataframes per condition/performance
        df_ext_g = dp.get_bad_fits(df_g, dp.THETA)
        df_ext_p = dp.get_bad_fits(df_p, dp.THETA)

        # Check if subject still has enough trials
        if dp.reject_subject(dfs=[df_ext_g, df_ext_p], part='after', frequency=dp.THETA):
            print("Subject rejected " + str(sub))
            sub_rejected_after_r2.append(sub)
            continue

        sub_included.append(sub)

        # Concatenate conditions for this subject
        sub_df = pd.concat([df_ext_p, df_ext_g])

        # Add aditional data to dataframes for each subject
        # 1. Average working memory performance over all trials
        sub_df['wm_capacity'] = theta_temp
        # 2. Whether this subject at theta oscillations or not
        sub_df['osc_presence'] = True # < -- needs to be if/else statement if I'm doing this in a single loop

        # Append to list of dataframes where all subjects are saved
        df_all_theta.append(sub_df)


        ### Now we also need to clean the power spectra arrays from bad fits
        psd_bas_clean_p, psd_ret_clean_p = dp.del_bad_psd(df_ext_p, psd_bas_p, psd_ret_p, frequency=dp.THETA)
        psd_bas_clean_g, psd_ret_clean_g = dp.del_bad_psd(df_ext_g, psd_bas_g, psd_ret_g, frequency=dp.THETA)

        # Insert into an array we can save; and take average over trials
                # psd_exp2_no_theta[ppn, freq_axis, performance, timeperiod]
                # freq_axis: 138 stamps of frequencies
                # performance: 0 = poor, 1 = good
                # timeperiod: 0 = baseline, 1 = retention
        psd_exp2_theta[power_detected, :, 0, 0] = np.mean(psd_bas_clean_p, axis=0)
        psd_exp2_theta[power_detected, :, 0, 1] = np.mean(psd_ret_clean_p, axis=0)
        psd_exp2_theta[power_detected, :, 1, 0] = np.mean(psd_bas_clean_g, axis=0)
        psd_exp2_theta[power_detected, :, 1, 1] = np.mean(psd_ret_clean_g, axis=0)

        power_detected += 1 # add to counter

    print('We did it! Participants that have THETA oscillatory activity! :D')

    ############### Save data ################

    # Delete rows (ppns) with only nan
    to_delete = np.where(np.isnan(psd_exp2_theta[:,0,0,0]))
    psd_exp2_theta = np.delete(psd_exp2_theta, to_delete, axis=0)

    # Save subject lists as dictionary for later
    theta_exp2_inc = {'considered': dp.SUB_THETA,
                    'included': sub_included,
                    'rejected': sub_rejected,
                    'rejected_r2': sub_rejected_after_r2}


    # Use pickle to save list of dataframes;save pkl files 
    file_name = 'theta_all_dfs.pkl'

    with open (save_path + file_name, 'wb') as f:
        pkl.dump(df_all_theta, f)

    np.save(save_path + 'theta_exp2_psd', psd_exp2_theta)
    np.save(save_path + 'theta_exp2_avg_perf', theta_avg_perform)
    np.save(save_path + 'freq_x_axis', freq_x_axis)
    np.save(save_path + 'theta_exp2_ppns.npy', theta_exp2_inc)

    print('THETA part is done and saved! :D')


    '''NO Theta: run specparam model and generate parameters'''

    ############### Initialize ################

    # Premade variable with subjects who exhibit oscillatory theta power (manually inspected)
    subs = len(dp.SUB_NO_THETA) # either sub_theta or sub_no_theta

    # Initialize empty array structure to save psd's
    psd_exp2_no_theta = np.full((subs, 138, 2, 2), np.nan)
    power_detected = 0  # counter to fill in above array

    # Initialize empty lists for data storage
    no_theta_avg_perform = []
    df_all_no_theta = []

    # Other variables that have to be set at 0 before the loop
    sub_included = []
    sub_rejected = []
    sub_rejected_without = []
    sub_rejected_after_r2 = []

    ############### Start for-loop ################

    for sub in dp.SUB_NO_THETA:  # either dp.SUB_THETA or dp.SUB_NO_THETA to iterate over ppns
        print(sub)
        eeg_dat = None
        beh_dat = None

        # Load data
        try:
            eeg_dat, beh_dat = dp.load_data(sub)
        except FileNotFoundError:
            warnings.warn('File not found')
            continue

        # Add each ppt average performance to average list
        no_theta_temp = np.nanmean(beh_dat['data']['trialAcc'])
        no_theta_avg_perform.append(no_theta_temp)
        
        # Extract helpful data from the EEG file
        channels = eeg_dat['chanLabels']
        artifacts = eeg_dat['arf']
        timepoints = eeg_dat[dp.USE_UN_BASELINED[1]] # --> 'time_long' = unbaselined
        settings = eeg_dat['settings']
        sfreq = settings['srate']

        # Get subject accuracies and eeg with artifacts rejected
        sub_accs, sub_eeg = dp.reject_bad_trials(eeg_dat[dp.USE_UN_BASELINED[0]], beh_dat, artifacts)

        # create format with 4 lists --> left good, left poor, right good, right poor
        correct_eeg = dp.extract_correct_eeg(sub_accs, sub_eeg)

        # Check if subject has enough trials per performance condition
        if dp.reject_subject(correct_eeg):
            print("Subject rejected " + str(sub))
            sub_rejected.append(sub)
            continue
        
        # FOOOF the data per performance
        df_g, psd_bas_g, psd_ret_g, freq_x_axis = dp.get_fooof(sub, dp.GOOD,
            sub_eeg, sub_accs, timepoints, dp.THETA, sfreq, analysis_type=analysis_type)
        df_p, psd_bas_p, psd_ret_p, _ = dp.get_fooof(sub, dp.POOR,
            sub_eeg, sub_accs, timepoints, dp.THETA, sfreq, analysis_type=analysis_type)
        
        # Also, check the model fits and add column to dataframes per condition/performance
        df_ext_g = dp.get_bad_fits(df_g, dp.THETA)
        df_ext_p = dp.get_bad_fits(df_p, dp.THETA)

        # Check if subject still has enough trials
        if dp.reject_subject(dfs=[df_ext_g, df_ext_p], part='after', frequency=dp.THETA):
            print("Subject rejected " + str(sub))
            sub_rejected_after_r2.append(sub)
            continue

        sub_included.append(sub)

        # concatenate conditions for this subject
        sub_df = pd.concat([df_ext_p, df_ext_g])

        # Add aditional data to dataframes for each subject
        # 1. Average working memory performance over all trials
        sub_df['wm_capacity'] = no_theta_temp
        # 2. Whether this subject at theta oscillations or not
        sub_df['osc_presence'] = False # < -- needs to be if/else statement if I'm doing this in a single loop

        # Append to list of dataframes where all subjects are saved
        df_all_no_theta.append(sub_df)

        
        ### Now we also need to clean the power spectra arrays from bad fits
        psd_bas_clean_p, psd_ret_clean_p = dp.del_bad_psd(df_ext_p, psd_bas_p, psd_ret_p, frequency=dp.THETA)
        psd_bas_clean_g, psd_ret_clean_g = dp.del_bad_psd(df_ext_g, psd_bas_g, psd_ret_g, frequency=dp.THETA)

        # Insert into an array we can save; and take average over trials
                # psd_exp2_no_theta[ppn, freq_axis, performance, timeperiod]
                # freq_axis: 138 stamps of frequencies
                # performance: 0 = poor, 1 = good
                # timeperiod: 0 = baseline, 1 = retention
        psd_exp2_no_theta[power_detected, :, 0, 0] = np.mean(psd_bas_clean_p, axis=0)
        psd_exp2_no_theta[power_detected, :, 0, 1] = np.mean(psd_ret_clean_p, axis=0)
        psd_exp2_no_theta[power_detected, :, 1, 0] = np.mean(psd_bas_clean_g, axis=0)
        psd_exp2_no_theta[power_detected, :, 1, 1] = np.mean(psd_ret_clean_g, axis=0)

        power_detected += 1 # add to counter


    print('We did it! At least the THETA part for ppns without theta power! :D')

    ################ Clean and Save ################

    # Delete rows (ppns) with only nan
    to_delete = np.where(np.isnan(psd_exp2_no_theta[:,0,0,0]))
    psd_exp2_no_theta = np.delete(psd_exp2_no_theta, to_delete, axis=0)

    # Save subject lists as dictionary for later
    no_theta_exp2_inc = {'considered': dp.SUB_NO_THETA,
                    'included': sub_included,
                    'rejected': sub_rejected,
                    'rejected_r2': sub_rejected_after_r2}

    # Use pickle to save list of dataframes;save pkl files 
    file_name = 'no_theta_all_dfs.pkl'

    with open (save_path + file_name, 'wb') as f:
        pkl.dump(df_all_no_theta, f)

    np.save(save_path + 'no_theta_exp2_psd', psd_exp2_no_theta)
    np.save(save_path + 'no_theta_exp2_avg_perf', no_theta_avg_perform)
    np.save(save_path + 'freq_x_axis', freq_x_axis)


    print('NO THETA part is done and saved! :D')




