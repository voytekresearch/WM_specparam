# ColourfulBirds
Predicting working memory contents from FOOOF.

fooof >= 1.0.0
neurodsp >= 2.1.0
pandas >= 1.3.0
seaborn = 0.9.0 (only for plotting)

## Data 
Open Data 2018: Contralateral delay activity tracks fluctuations in working memory performance
https://osf.io/8xuk3/

## Data Preparation
1. Rename files with 2 digits for the number
2. Resave file unpacked with older version of mat using the conv mat file in utils.


## What is this project about?
Colourfulbirds is a FOOOF project that uses data from Adam et al. (2018). The data is collected from a visual working memory task. 
<li>Experiment 1: different set-sizes and occipital alpha power. </li>
<li>Experiment 2: set-size kept at 6 items. Look at the performance within subjects (good vs. poor). Interested in occipital alpha power, and midline frontal theta power. </li>

### Replicating the original results
Our first goal was to replicate the results fom Adam et al. (2018). This was to verify that we understand the data structure and were able to combine the behavioral data correctly with the EEG data. <br>
The replicated results can be found in the "02-Replication.." notebooks <br>
Steps: <ol>
    <li>Delete all the to be rejected trials. This is based on the pre-processing from the original paper</li>
    <li>Select EEG data based on set-size/performance/side</li>
    <li>Per electrode and per trial, calculate the amplitude (using amp_by_time from neurodsp)</li>
    <li>Baseline per trial and electrode --> Baselined hilbert transfer</li>
    <li>Average the hilbert transfers per condition and per subject</li>
    <li>Lateralize the data: contralateral - ipsilateral</li>
    
### Statistical analysis -- original results    
Then, we performed the same statistical analysis in the "03-StatisticalAnalysis.." notebooks. Where we create the dataframes and save them to analyize it in JASP

### Our approach -- FOOOF
This part is of course the most interesting! <br>
Steps: <ol>
    <li>Delete all the to be rejected trials. This is based on the pre-processing from the original paper</li>
    <li>Select EEG data based on set-size/performance</li>
    <li>Per electrode group and per trial, calculate the PSD (using neurodsp) for the retention period and the baseline period</li>
    <li>Fooof the PSDs and save the peak values and aperiodic offset and exponent</li>
    <li>Subtract the baseline period output from the retention period output</li>
    <li>Lateralize the data: contralateral - ipsilateral</li>
    
### Statistical analysis -- FOOOF
Statistical tests are performed on all 4 fooof components: <br>
<ul>
    <li>Power (relative to 1/f fit)</li>
    <li>Central Frequency</li>
    <li>1/f exponent</li>
    <li>1/f offset</li>
    </ul>
    
A 2-way RM ANOVA is used to analyze the data for the occipital electrodes (alpha). <br>
A Pairwise t-test is used for midline frontal theta.

Furthermore, we're looking into a regression instead of a binary division of performance. <br>
The overall performance is measured as: total correct responses / (#trials * 6) <br>
And the power, 1/f exponent and offset are calculated over all trials as well (contra vs ipsi).

--> Nothing is significant, so we're dropping this idea
