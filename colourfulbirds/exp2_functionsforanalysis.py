'''This module contains functions for exp2 to create long and wide format pandas dataframes from the .npy datafiles. '''

import numpy as np
import pandas as pd


def create_df_alpha_long(dataframe, parameter):
    '''Function that creates a pandas dataframe from the different conditions. Specifically, this function separates 
    de bigger dataframe into the 4 different parameters.

    Input:
        dataframe: pd.dataframe to convert from wide format
        parameter: str
    Output: 
        pd.dataframe for power, CF, exponent and offset and percentage peaks
        separately in long format'''
    
    # Create numpy data to convert into pandas dataframe
    conditions = 4   # 2 sides and 2 perfromances --> dataframe.shape[1] - 1
    subjects = dataframe.shape[0]
    lateralization = np.tile(np.repeat(['contra', 'ipsi'], subjects), int(conditions / 2))
    data = np.zeros((subjects * conditions, 3))
    data[:, 0] = np.tile(np.arange(subjects), conditions)
    performance = np.tile(np.repeat(['poor', 'good'], 2 * subjects), int(conditions / 4))
    
    
    # Put the amplitudes in the second col
    data[:subjects, 1] = dataframe['poor_contra']
    data[subjects:subjects*2, 1] = dataframe['poor_ipsi']
    data[subjects*2:subjects*3, 1] = dataframe['good_contra']
    data[subjects*3:subjects*4, 1] = dataframe['good_ipsi']
    
    df = pd.DataFrame({'sub_id': data[:,0], parameter: data[:,1], 'performance': performance,
                       'lateralization': lateralization})

    return df


def create_df_alpha_wide(dataframe):
    '''Function that creates a pandas dataframe from the different conditions. Specifically, this function separates 
    de bigger dataframe into the 4 different parameters.
    
    Input: numpy.array
        The giant numpy array
    Output: 
        pd.dataframe for power, CF, exponent and offset and percent peak'''

    
    # create array of subject ID
    subjects = np.arange(len(dataframe[:,0,0,0]))
    
    # dataframe infor: [ppn, parameter, performance, lateralization ]
    #  parameter: 0 = power, 1 = cf, 2 = exp, 3 = offset, 4 = percent_peak
    #  performance: 0 = poor, 1 = good
    #  lateralization: 0 = contra, 1 = ipsi
    
    # create wide-format of dataframes for the 4 different parameters 
    df_power = pd.DataFrame({'sub_id': subjects, "poor_contra": dataframe[:,0,0,0], "poor_ipsi": dataframe[:,0,0,1],
                        "good_contra": dataframe[:,0,1,0], "good_ipsi": dataframe[:,0,1,1]})
    
    df_CF = pd.DataFrame({'sub_id': subjects, "poor_contra": dataframe[:,1,0,0], "poor_ipsi": dataframe[:,1,0,1],
                        "good_contra": dataframe[:,1,1,0], "good_ipsi": dataframe[:,1,1,1]})
    
    df_exp = pd.DataFrame({'sub_id': subjects, "poor_contra": dataframe[:,2,0,0], "poor_ipsi": dataframe[:,2,0,1],
                        "good_contra": dataframe[:,2,1,0], "good_ipsi": dataframe[:,2,1,1]})
    
    df_offset = pd.DataFrame({'sub_id': subjects, "poor_contra": dataframe[:,3,0,0], "poor_ipsi": dataframe[:,3,0,1],
                        "good_contra": dataframe[:,3,1,0], "good_ipsi": dataframe[:,3,1,1]})

    df_cf_percent_peak = pd.DataFrame({'sub_id': subjects, "poor_contra": dataframe[:,4,0,0], "poor_ipsi": dataframe[:,4,0,1],
                        "good_contra": dataframe[:,4,1,0], "good_ipsi": dataframe[:,4,1,1]})
    
    return df_power, df_CF, df_exp, df_offset, df_cf_percent_peak


def create_df_theta_long(dataframe, parameter):
    '''Function that creates a pandas dataframe from the different conditions. Specifically, this function separates 
    de bigger dataframe into the 4 different parameters.
    
    Also deletes subs with only 0's 
    Input:
        pd.dataframe as long format
    Output: 
        pd.dataframe for power, CF, exponent and offset'''
    
    # Create numpy data to convert into pandas dataframe
    conditions = 2   # 2 perfromances --> dataframe.shape[1] - 1
    subjects = dataframe.shape[0]
    data = np.zeros((subjects * conditions, 2))
    data[:, 0] = np.tile(np.arange(subjects), conditions)
    performance = np.repeat(['poor', 'good'], subjects)
    
    
    # Put the parameters in the second col
    data[:subjects, 1] = dataframe['poor']
    data[subjects:subjects*2, 1] = dataframe['good']

    df = pd.DataFrame({'sub_id': data[:,0], parameter: data[:,1], 'performance': performance})

    return df


def create_df_theta_wide(dataframe):
    '''Function that creates a pandas dataframe from the different conditions. Specifically, this function separates 
    de bigger dataframe into the 4 different parameters.
    
    Also deletes subs with only 0's 
    Input:
        dataframe [subs, 4, 2] --> subjects, 4 parameters, 2 performances/conditions
    Output: 
        pd.dataframe for power, CF, exponent and offset'''

    
    # create array of subject ID
    subjects = np.arange(len(dataframe[:,0,0]))
    
    # dataframe infor: [ppn, parameter, performance, lateralization ]
    #  parameter: 0 = power, 1 = cf, 2 = exp, 3 = offset
    #  performance: 0 = poor, 1 = good
    
    # create wide-format of dataframes for the 4 different parameters 
    df_power = pd.DataFrame({'sub_id': subjects, "poor": dataframe[:,0,0], "good": dataframe[:,0,1]})
    
    df_CF = pd.DataFrame({'sub_id': subjects, "poor": dataframe[:,1,0], "good": dataframe[:,1,1]})
    
    df_exp = pd.DataFrame({'sub_id': subjects, "poor": dataframe[:,2,0], "good": dataframe[:,2,1]})
    
    df_offset = pd.DataFrame({'sub_id': subjects, "poor": dataframe[:,3,0], "good": dataframe[:,3,1]})

    df_cf_percent_peak = pd.DataFrame({'sub_id': subjects, "poor": dataframe[:,4,0], "good": dataframe[:,4,1]})
    
    return df_power, df_CF, df_exp, df_offset, df_cf_percent_peak