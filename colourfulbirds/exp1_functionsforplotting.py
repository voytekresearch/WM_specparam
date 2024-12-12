'''This module contains functions for exp1 to create long and wide format pandas dataframes from the .npy datafiles. '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
import seaborn as sns

def plot_violin_swarms(data=None, parameter=None, frequency_band=None, zero=True, save_fig=False):
    '''Plotting tool for swarms overlapping a violin plot. It also adds a median line connecting the means of the data.
    Input
        data: pd.DataFrame
            long format pd dataframe
        parameter: str
            parameter you want to plot
        frequency_band: str
            Either alpha or theta
        save_fig: boolean
            False or True whether to save the figure as .pdf file
    Output
        figure
    '''
    # Color scheme and transparency reference:
    # https://stackoverflow.com/questions/70442958/seaborn-how-to-apply-custom-color-to-each-seaborn-violinplot
    sns.set_context('paper', font_scale=2)
    plt.figure(figsize=(6,4))
    
#     alpha_col = {"contra": '#beaed4', "ipsi": '#fdc086'}  # softer purple and orange
#     alpha_col = {"contra": '#7570b3', "ipsi": '#d95f02'}  # colorblind safe purple and orange
    # alpha_col = {"contra": '#984ea3', "ipsi": '#ff7f00'}  # brighter purple and orange

    pal_hue = ['1', '0.3']
    colors = ['g', 'r', 'b']
    theta_col = {1: 'g',
                 3: 'r',
                 6: 'b'}
    theta_no_theta = {'yes': 'darkviolet',
                    'no': 'teal'}
    
    # Let the plotting begin!
    if frequency_band == 'alpha':
        # Calculate mean and draw so you can draw a connecting line in the plots
        means = pd.DataFrame(data.groupby(['set_size', 'lateralization'])[parameter].mean())
        c1 = means.loc[(1, 'contra')]
        c3 = means.loc[(3, 'contra')]
        c6 = means.loc[(6, 'contra')]
        i1 = means.loc[(1, 'ipsi')]
        i3 = means.loc[(3, 'ipsi')]
        i6 = means.loc[(6, 'ipsi')]
        
        # plot violinplot and swarmplot on top of each other
        ax = sns.violinplot(x='set_size', y=parameter, hue='lateralization', data=data, dodge=True, inner=None, palette=pal_hue) # palette=alpha_col
        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            rgb = to_rgb(colors[ind // 2])
            if ind % 2 != 0:
                rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
            violin.set_facecolor(rgb)

        ax = sns.swarmplot(x='set_size', y=parameter, hue='lateralization', data=data, dodge=True, color='black', size=4)

        # Add mean lines
        sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x='set_size',
            y=parameter,
            hue='lateralization',
            data=data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax)
        
        # add lines to connect means
#         ax = sns.pointplot(x='set_size', 
#                            y=parameter, 
#                            hue='lateralization', 
#                            data=data, 
#                            ci=None, 
#                            color='black',
#                            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
#                            dodge=True)
        
        # add dashed line at y = 0 so it's easier to see the effect after baselining
        if zero:
            ax.axhline(y = 0, color = 'k', ls = '--', lw=2)
        
        # Fix legend outside of figure itself
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.legend('',frameon=False)

        # Add a labels
        plt.xlabel('set size') 
        ax.set_xticklabels(['1', '3', '6'])
        if parameter == 'power':
            plt.title('α peak height')
            plt.ylabel('peak height ($\mu$V$^2$)')
        elif parameter == 'bandpower':
            plt.title('α bandpower')
            plt.ylabel('power ($\mu$V$^2$/Hz)')
        elif parameter == 'peak_perc':
            plt.title('α abundance')
            plt.ylabel('%')
        elif parameter == 'exponent':
            plt.title('occipitoparietal exponent')
            plt.ylabel('exponent ($\mu$V$^2$/Hz)')
        elif parameter == 'offset':
            plt.title('occipitoparietal offset')
            plt.ylabel('offset ($\mu$V$^2$)')
        elif parameter == 'auc':
            plt.ylabel('bandpower - auc ($\mu$V$^2$/Hz)')
        elif parameter == 'auc_log_osc':
            plt.ylabel('log osc power ($\mu$V$^2$/Hz)')
        elif parameter == 'auc_lin_osc':
            plt.ylabel('linear osc power ($\mu$V$^2$/Hz)')

        sns.despine()    
        # Save figure if saving is set to True
        if save_fig == True:
            # create correct filename and path according to parameter
            path = '../figures/exp1_figures/'
            filename = 'fig4_exp1_alpha_' + parameter + '.pdf'
            plt.savefig(path + filename, format='pdf', bbox_inches='tight' )

    elif frequency_band == 'theta_presence':
        # Calculate mean and draw so you can draw a connecting line in the plots
        # means = pd.DataFrame(data.groupby('set_size')[parameter].mean())

        ax = sns.violinplot(x='osc_presence', y=parameter, data=data, inner=None, palette=theta_no_theta)
        ax = sns.swarmplot(x='osc_presence', y=parameter, data=data, color='black')
        # Add mean lines
        ax = sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x='osc_presence',
            y=parameter,
            data=data,
            showfliers=False,
            showbox=False,
            showcaps=False) # ax=ax

        
        # add dashed line at y = 0 so it's easier to see the effect after baselining
        # if zero:
        #     ax.axhline(y = 0, color = 'k', ls = '--', lw=2)

        # Add a y-label 
        plt.xlabel('θ oscillation presence')
        if parameter == 'power':
            plt.title('θ peak height ')
            plt.ylabel('peak height ($\mu$V$^2$)')
        elif parameter == 'bandpower':
            plt.title('θ bandpower ')
            plt.ylabel('power ($\mu$V$^2$/Hz)')
        elif parameter == 'percent_peak':
            plt.title('θ abundance ')
            plt.ylabel('%')
        elif parameter == 'exponent':
            plt.title('frontal midline exponent')
            plt.ylabel('exponent ($\mu$V$^2$/Hz)')
        elif parameter == 'offset':
            plt.title('frontal midline offset')
            plt.ylabel('offset ($\mu$V$^2$)')
        elif parameter == 'auc':
            plt.ylabel('bandpower - auc ($\mu$V$^2$/Hz)')
        elif parameter == 'auc_log_osc':
            plt.ylabel('log osc power ($\mu$V$^2$/Hz)')
        elif parameter == 'auc_lin_osc':
            plt.ylabel('linear osc power ($\mu$V$^2$/Hz)')

        sns.despine()
        plt.tight_layout()

            # Save figure if saving is set to True
        if save_fig == True:
            # create correct filename and path according to parameter
            path = '../figures/exp1_figures/'
            filename = 'fig4_exp1_theta_no_theta_' + parameter + '.pdf'
            plt.savefig(path + filename, bbox_inches='tight', format='pdf')

    
    elif frequency_band == 'alpha_diff':
            # Calculate mean and draw so you can draw a connecting line in the plots
        # means = pd.DataFrame(data.groupby('set_size')[parameter].mean())

        ax = sns.violinplot(x='set_size', y=parameter, data=data, inner=None, palette=theta_col)
        ax = sns.swarmplot(x='set_size', y=parameter, data=data, color='black')
        # Add mean lines
        sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x='set_size',
            y=parameter,
            data=data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax,)
        
        # add dashed line at y = 0 so it's easier to see the effect after baselining
        if zero:
            ax.axhline(y = 0, color = 'k', ls = '--', lw=2)


        # Add a labels
        plt.xlabel('set size') 
        ax.set_xticklabels(['1', '3', '6'])
        if parameter == 'power':
            plt.title('α peak height')
            plt.ylabel('peak height ($\mu$V$^2$)')
        elif parameter == 'cf':
            plt.ylabel('α center frequency (Hz)')
        elif parameter == 'peak_perc':
            plt.title('α abundance')
            plt.ylabel('%')
        elif parameter == 'exponent':
            plt.title('occipitoparietal exponent')
            plt.ylabel('exponent ($\mu$V$^2$/Hz)')
        elif parameter == 'offset':
            plt.title('occipitoparietal offset')
            plt.ylabel('offset ($\mu$V$^2$)')
        elif parameter == 'auc':
            plt.ylabel('bandpower - auc ($\mu$V$^2$/Hz)')
        elif parameter == 'auc_log_osc':
            plt.ylabel('log osc power ($\mu$V$^2$/Hz)')
        elif parameter == 'auc_lin_osc':
            plt.ylabel('linear osc power ($\mu$V$^2$/Hz)')
            
        plt.tight_layout()
        sns.despine()
        # Save figure if saving is set to True
        if save_fig == True:
            print(frequency_band)
            # create correct filename and path according to parameter
            path = '../figures/exp1_figures/'
            filename = 'fig4_exp1_alpha_diff_' + parameter + '.pdf'
            plt.savefig(path + filename, format='pdf', bbox_inches='tight' )
    
    
    else: # theta
        # Calculate mean and draw so you can draw a connecting line in the plots
        means = pd.DataFrame(data.groupby('set_size')[parameter].mean())
        # size1 = means.loc['size_1']
        # size3 = means.loc['size_3']
        # size6 = means.loc['size_6']

        ax = sns.violinplot(x='set_size', y=parameter, data=data, inner=None, palette=theta_col)
        ax = sns.swarmplot(x='set_size', y=parameter, data=data, color='black')
        # Add mean lines
        sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x='set_size',
            y=parameter,
            data=data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax)
        
        # add dashed line at y = 0 so it's easier to see the effect after baselining
        if zero:
            ax.axhline(y = 0, color = 'k', ls = '--', lw=2)

        # Add labels
        plt.xlabel('set size') 
        ax.set_xticklabels(['1', '3', '6'])
        if parameter == 'power':
            plt.title('θ peak height')
            plt.ylabel('peak height ($\mu$V$^2$)')
        elif parameter == 'bandpower':
            plt.title('θ bandpower')
            plt.ylabel('power ($\mu$V$^2$/Hz)')
        elif parameter == 'peak_perc':
            plt.title('θ abundance')
            plt.ylabel('%')
        elif parameter == 'exponent':
            plt.title('frontal midline exponent')
            plt.ylabel('exponent ($\mu$V$^2$/Hz)')
        elif parameter == 'offset':
            plt.title('frontal midline offset')
            plt.ylabel('offset ($\mu$V$^2$)')
        elif parameter == 'auc':
            plt.ylabel('bandpower - auc ($\mu$V$^2$/Hz)')
        elif parameter == 'auc_log_osc':
            plt.ylabel('log osc power ($\mu$V$^2$/Hz)')
        elif parameter == 'auc_lin_osc':
            plt.ylabel('linear osc power ($\mu$V$^2$/Hz)')

        sns.despine()
    
        # Save figure if saving is set to True
        if save_fig == True:
            # create correct filename and path according to parameter
            path = '../figures/exp1_figures/'
            filename = 'fig4_exp1_theta_' + parameter + '.pdf'
            plt.savefig(path + filename, format='pdf', bbox_inches='tight' )
    
    plt.show()

    return