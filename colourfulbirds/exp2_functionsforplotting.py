'''This module contains functions for exp2 to create long and wide format pandas dataframes from the .npy datafiles. '''

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

    sns.set_context('paper', font_scale=2)
    plt.figure(figsize=(6,4))
    
#     alpha_col = {"contra": '#beaed4', "ipsi": '#fdc086'}  # softer purple and orange
#     alpha_col = {"contra": '#7570b3', "ipsi": '#d95f02'}  # colorblind safe purple and orange
    alpha_col = {"contra": '#984ea3', "ipsi": '#ff7f00'}  # brighter purple and orange
#984ea3
    pal_hue = ['1', '0.3']
    colors = ['r', 'g']
    theta_col = {'poor': 'r',
                 'good': 'g'}
    
    theta_no_theta = {'yes': 'darkviolet',
                    'no': 'teal'}

    
    # Let the plotting begin!
    if frequency_band == 'alpha':
        # Calculate mean and draw so you can draw a connecting line in the plots
        means = pd.DataFrame(data.groupby(['performance', 'lateralization'])[parameter].mean())
        cpoor = means.loc[('poor', 'contra')]
        cgood = means.loc[('good', 'contra')]
        ipoor = means.loc[('poor', 'ipsi')]
        igood = means.loc[('good', 'ipsi')]
        
        # plot violinplot and swarmplot on top of each other
        ax = sns.violinplot(x='performance', y=parameter, hue='lateralization', data=data, dodge=True, inner=None, palette=pal_hue, order=['poor', 'good'])
        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            rgb = to_rgb(colors[ind // 2])
            if ind % 2 != 0:
                rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
            violin.set_facecolor(rgb)
        
        ax = sns.swarmplot(x='performance', y=parameter, hue='lateralization', data=data, dodge=True, color='black', order=['poor', 'good'], size=4)

        # Add mean lines
        sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x='performance',
            y=parameter,
            hue='lateralization',
            data=data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax,
            order=['poor', 'good'])
        
        # add lines to connect means
#         ax = sns.pointplot(x='performance', 
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


        # Add a title 
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
            
        plt.tight_layout()
        sns.despine()
        # Save figure if saving is set to True
        if save_fig == True:
            # create correct filename and path according to parameter
            path = '../figures/exp2_figures/'
            filename = 'fig4_exp2_alpha_' + parameter + '.pdf'
            plt.savefig(path + filename, format='pdf', bbox_inches='tight' )

    elif frequency_band == 'theta_presence':
        # Calculate mean and draw so you can draw a connecting line in the plots
        # means = pd.DataFrame(data.groupby('performance')[parameter].mean())

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
        if zero:
            ax.axhline(y = 0, color = 'k', ls = '--', lw=2)

        # Add a y-label 
        plt.xlabel('θ oscillation presence')
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

        sns.despine()
        plt.tight_layout()
            # Save figure if saving is set to True
        if save_fig == True:
            # create correct filename and path according to parameter
            path = '../figures/exp2_figures/'
            filename = 'fig4_exp2_theta_no_theta_' + parameter + '.pdf'
            plt.savefig(path + filename, bbox_inches='tight', format='pdf')

    
    elif frequency_band == 'alpha_diff':
        # Calculate mean and draw so you can draw a connecting line in the plots
        means = pd.DataFrame(data.groupby('performance')[parameter].mean())

        ax = sns.violinplot(x='performance', y=parameter, data=data, inner=None, palette=theta_col, order=['poor', 'good'])
        ax = sns.swarmplot(x='performance', y=parameter, data=data, color='black', order=['poor', 'good'])
        # Add mean lines
        sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x='performance',
            y=parameter,
            data=data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax,
            order=['poor', 'good'])
        
        # add dashed line at y = 0 so it's easier to see the effect after baselining
        if zero:
            ax.axhline(y = 0, color = 'k', ls = '--', lw=2)

        # Add a y-label 
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
            
        plt.tight_layout()
        sns.despine()
        # Save figure if saving is set to True
        if save_fig == True:
            print(frequency_band)
            # create correct filename and path according to parameter
            path = '../figures/exp2_figures/'
            filename = 'fig4_exp2_alpha_diff_' + parameter + '.pdf'
            plt.savefig(path + filename, format='pdf', bbox_inches='tight' )

    
    else: # theta
        # Calculate mean and draw so you can draw a connecting line in the plots
        means = pd.DataFrame(data.groupby('performance')[parameter].mean())

        ax = sns.violinplot(x='performance', y=parameter, data=data, inner=None, palette=theta_col, order=['poor', 'good'])
        ax = sns.swarmplot(x='performance', y=parameter, data=data, color='black', order=['poor', 'good'])
        # Add mean lines
        sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 3},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x='performance',
            y=parameter,
            data=data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax,
            order=['poor', 'good'])
        
        # add dashed line at y = 0 so it's easier to see the effect after baselining
        if zero:
            ax.axhline(y = 0, color = 'k', ls = '--', lw=2)

        # Add a y-label 
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
            
        plt.tight_layout()
        sns.despine()
    
        # Save figure if saving is set to True
        if save_fig == True:
            print(frequency_band)
            if frequency_band == 'theta':
                # create correct filename and path according to parameter
                path = '../figures/exp2_figures/'
                filename = 'fig4_exp2_theta_' + parameter + '.pdf'
                plt.savefig(path + filename, format='pdf', bbox_inches='tight' )
            else : # no_theta
                # create correct filename and path according to parameter
                path = '../figures/exp2_figures/'
                filename = 'fig4_exp2_no_theta_' + parameter + '.pdf'
                plt.savefig(path + filename, bbox_inches='tight', format='pdf')

    plt.show()

    return