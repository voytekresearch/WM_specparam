'''This module contains functions for exp1 to create long and wide format pandas dataframes from the .npy datafiles. '''

import numpy as np
import pandas as pd


def create_df_alpha_long(dataframe, parameter):
    '''Function that transforms a wide format pd.df into a long format.
    Input:
        dataframe: dataframe to convert (wide format)
    Output: 
        pd.dataframe (long format)'''
    
    # Create numpy data to convert into pandas dataframe
    conditions = 6   # 2 sides and 3 set-sizes --> dataframe.shape[1] - 1
    subjects = dataframe.shape[0]
    side = np.tile(np.repeat(['contra', 'ipsi'], subjects), int(conditions / 2))
    data = np.zeros((subjects * conditions, 3))
    data[:, 0] = np.tile(np.arange(subjects), conditions)
    setsize = np.repeat(['size_1', 'size_3', 'size_6'], 2 * subjects)

    # Put the parameter in the second col
    data[:subjects, 1] = dataframe['size1_contra']
    data[subjects:subjects*2, 1] = dataframe['size1_ipsi']
    data[subjects*2:subjects*3, 1] = dataframe['size3_contra']
    data[subjects*3:subjects*4, 1] = dataframe['size3_ipsi']
    data[subjects*4:subjects*5, 1] = dataframe['size6_contra']
    data[subjects*5:subjects*6, 1] = dataframe['size6_ipsi']

    return pd.DataFrame({'sub_id': data[:,0], parameter: data[:,1], 'set_size': setsize, 'lateralization': side})


def create_df_alpha_wide(dataframe):
    # look at function in cell below and transform that for to fit exp1
    '''Function that creates a pandas dataframe from the different conditions. Specifically, this function separates 
    de bigger dataframe into the 4 different parameters.

    Input: numpy.array
        The giant numpy array
    Output: 
        pd.dataframe for power, CF, exponent and offset and percent peak'''
    
    # create array of subject ID
    subjects = np.arange(len(dataframe[:,0,0,0]))
    
    # dataframe infor: [ppn, parameter, setsize, lateralization ]
    #  parameter: 0 = power, 1 = cf, 2 = exp, 3 = offset, 4 = percent_peak
    #  setsize: 0 = 1, 1 = 3, 2 = 6
    #  lateralization: 0 = contra, 1 = ipsi
    
    # create wide-format of dataframes for the 4 different parameters 
    alpha_power = pd.DataFrame({'sub_id': subjects, "size1_contra": dataframe[:,0,0,0], "size1_ipsi": dataframe[:,0,0,1],
                        "size3_contra": dataframe[:,0,1,0], "size3_ipsi": dataframe[:,0,1,1],
                        "size6_contra": dataframe[:,0,2,0], "size6_ipsi": dataframe[:,0,2,1]})
    
    alpha_cf = pd.DataFrame({'sub_id': subjects, "size1_contra": dataframe[:,1,0,0], "size1_ipsi": dataframe[:,1,0,1],
                        "size3_contra": dataframe[:,1,1,0], "size3_ipsi": dataframe[:,1,1,1],
                        "size6_contra": dataframe[:,1,2,0], "size6_ipsi": dataframe[:,1,2,1]})
    
    alpha_exp = pd.DataFrame({'sub_id': subjects, "size1_contra": dataframe[:,2,0,0], "size1_ipsi": dataframe[:,2,0,1],
                        "size3_contra": dataframe[:,2,1,0], "size3_ipsi": dataframe[:,2,1,1],
                        "size6_contra": dataframe[:,2,2,0], "size6_ipsi": dataframe[:,2,2,1]})
    
    alpha_offset = pd.DataFrame({'sub_id': subjects, "size1_contra": dataframe[:,3,0,0], "size1_ipsi": dataframe[:,3,0,1],
                        "size3_contra": dataframe[:,3,1,0], "size3_ipsi": dataframe[:,3,1,1],
                        "size6_contra": dataframe[:,3,2,0], "size6_ipsi": dataframe[:,3,2,1]})

    alpha_perc_peak = pd.DataFrame({'sub_id': subjects, "size1_contra": dataframe[:,4,0,0], "size1_ipsi": dataframe[:,4,0,1],
                        "size3_contra": dataframe[:,4,1,0], "size3_ipsi": dataframe[:,4,1,1],
                        "size6_contra": dataframe[:,4,2,0], "size6_ipsi": dataframe[:,4,2,1]})  
    
    return alpha_power, alpha_cf, alpha_exp, alpha_offset, alpha_perc_peak


def create_df_theta_long(dataframe, parameter):
    '''Function that creates a pandas dataframe from the different conditions. Specifically, this function separates 
    de bigger dataframe into the 4 different parameters.
    
    Also deletes subs with only 0's 
    Input:
        pd.dataframe as long format
    Output: 
        pd.dataframe for power, CF, exponent and offset'''
    
    # Create numpy data to convert into pandas dataframe
    conditions = 3   # 3 conditions --> dataframe.shape[1] - 1
    subjects = dataframe.shape[0]
    data = np.zeros((subjects * conditions, 2))
    data[:, 0] = np.tile(np.arange(subjects), conditions)
    setsize = np.repeat(['size_1', 'size_3', 'size_6'], subjects)
    
    
    # Put the parameters in the second col
    data[:subjects, 1] = dataframe['size_1']
    data[subjects:subjects*2, 1] = dataframe['size_3']
    data[subjects*2:, 1] = dataframe['size_6']

    df = pd.DataFrame({'sub_id': data[:,0], parameter: data[:,1], 'set_size': setsize})

    return df


def create_df_theta_wide(dataframe):
    '''Function that creates a pandas dataframe from the different conditions. Specifically, this function separates 
    de bigger dataframe into the 5 different parameters.
    
    Also deletes subs with only 0's 
    Input:
        dataframe [subs, 5, 3] --> subjects, 5 parameters, 3 conditions
    Output: 
        pd.dataframe for power, CF, exponent,  offset, and cf percentages'''

    
    # create array of subject ID
    subjects = np.arange(len(dataframe[:,0,0]))
    
    # dataframe infor: [ppn, parameter, performance, lateralization ]
    #  parameter: 0 = power, 1 = cf, 2 = exp, 3 = offset, 4 = cf_perc
    #  setsize: 0 = 1, 1 = 3, 2 = 6
    
    # create wide-format of dataframes for the 4 different parameters 
    alpha_power = pd.DataFrame({'sub_id': subjects, "size_1": dataframe[:,0,0], 
        "size_3": dataframe[:,0,1], "size_6": dataframe[:,0,2]})
    
    alpha_CF = pd.DataFrame({'sub_id': subjects, "size_1": dataframe[:,1,0], 
        "size_3": dataframe[:,1,1], "size_6": dataframe[:,1,2]})
    
    alpha_exp = pd.DataFrame({'sub_id': subjects, "size_1": dataframe[:,2,0], 
        "size_3": dataframe[:,2,1], "size_6": dataframe[:,2,2]})
    
    alpha_offset = pd.DataFrame({'sub_id': subjects, "size_1": dataframe[:,3,0], 
        "size_3": dataframe[:,3,1], "size_6": dataframe[:,3,2]})

    alpha_perc_peak = pd.DataFrame({'sub_id': subjects, "size_1": dataframe[:,4,0], 
        "size_3": dataframe[:,4,1], "size_6": dataframe[:,4,2]})
    
    return alpha_power, alpha_CF, alpha_exp, alpha_offset, alpha_perc_peak
