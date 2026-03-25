#==========================================================
#--- coding: utf-8
#--- author: neele elbersgerd
#--- date of creation: 2026-03-25
#==========================================================
# purpose: replay analysis
# ==========================================================
# %% ===== IMPORT MODULES
import pandas as pd
import numpy as np
import glob
import re
import os
import ast
import random
from pathlib import Path
from os.path import join as opj
from itertools import permutations
from scipy.stats import pearsonr, linregress, ttest_1samp
from scipy.stats import false_discovery_control
import matplotlib.pyplot as plt
import seaborn as sns
root = Path(__file__).parent.parent
data_dir = opj(root, "data")

# %% ============ [DATA IMPORT] ============
def import_data(data_dir):
    """imports and merges data of all subjects 
    from psychopy behavioral data. 
    Returns: 
        df: pd.DataFrame of the concatenated psychopy experimental files
        id_dict: matching keys = participant_ids and values = design_ids
    """
    files = opj(
        data_dir, f"*decoding_with_order.csv",
    )
    files = glob.glob(files)
 
    print(f"Found {len(files)} files in path {data_dir}.")

    df = pd.concat(
            [pd.read_csv(f) for f in files],
            ignore_index=True
        )

    return df

# %% ============ [RESTRUCTURING INTO REPLAY TABLE] ============
def restructure(df, tITI=float):

    # filter the dataframe by only largest tITI and only mask == cv
    # filter to only get probabilities for category from category-vs-rest classifiers 
    # this kicks out multinomial log_reg classifier results as well
    df_replay = df[
        (df["test_set"]=="test-seq_long") &
        (df["mask"]=="cv") & 
        (df["tITI"]==tITI) &
        (df["class"]==df["classifier"])
        ].reset_index(drop=True)

    # look at one trial:
    # test = df_replay[(df_replay["trial"]==1)&(df_replay["classifier"]=="cat")]
    
    # 2. PIVOT THE DATA
    # We pivot so that:
    # - Each ROW is a unique trial (defined by id, test_set, classifier, stim, run, trial)
    # - Each COLUMN is a time point (seq_tr 1 to 13)
    # - VALUES are the probabilities
    df_pivot = df_replay.pivot_table(
        index=['id', 'classifier', 'stim_order', 'trial'], # Grouping columns
        columns='seq_tr',              # Time points become columns
        values='probability',          # Values to fill the table
        aggfunc='first'                # Safety: if duplicates exist, take the first
    ) 
    # Reset index to turn the grouping columns back into regular columns
    df_pivot = df_pivot.reset_index()
    
    # Rename the time columns from integers (1, 2...) to clear names (TR01, TR02...)
    # The last 13 columns are the time points
    time_cols = df_pivot.columns[-13:]
    new_names = [f'TR{int(t):02d}' for t in time_cols]
    
    # Create a mapping to rename only these columns
    rename_dict = dict(zip(time_cols, new_names))
    df_pivot = df_pivot.rename(columns=rename_dict)

    # Get index of classifier probability as separate column
    df_pivot['stim_order_list'] = df_pivot['stim_order'].apply(ast.literal_eval)
    df_pivot = df_pivot.drop('stim_order', axis=1)
    df_pivot["stim_position"] = df_pivot.apply(
        lambda row: row['stim_order_list'].index(row['classifier']),
        axis=1
    )
    print(df_pivot.head(3))
    
    # 3. SPLIT BY CLASSIFIER
    # Create a dictionary where each key is a classifier name
    data_by_classifier = {}
    classifiers_list = sorted(df_pivot['classifier'].unique())
    
    for clf in classifiers_list:
        # Extract subset for this classifier
        df_clf = df_pivot[df_pivot['classifier'] == clf].reset_index(drop=True)

        # Optional: Drop the 'classifier' column as it's now redundant in this subset
        df_clf = df_clf.drop(columns=['classifier'])
        data_by_classifier[clf] = df_clf

        df_clf.to_csv(opj(root, "output", f"replay_data_classifier{clf}.csv"))

    return data_by_classifier

# %% ============ [PLOT AVERAGE PROBABILITIES] ============
def plot_avg_probabilities(replay_dict):
    """
    Plot average probabilities across trials for each classifier.
    
    Parameters:
    -----------
    replay_dict : dict
        Dictionary where keys are classifier names and values are DataFrames
        with columns ['id', 'trial', 'TR1', 'TR2', ..., 'TR13']
    """
    # Get the time columns (TR1, TR2, ..., TR13)
    first_df = next(iter(replay_dict.values()))
    time_cols = [col for col in first_df.columns if col.startswith('TR')]
    
    # Restructure data into long format for seaborn
    data_list = []
    for clf_name, df in replay_dict.items():
        for _, row in df.iterrows():
            for tr_col in time_cols:
                tr_num = int(tr_col.replace('TR', ''))
                data_list.append({
                    'TR': tr_num,
                    'probability': row[tr_col],
                    'classifier': clf_name,
                    'trial': row['trial'],
                    'id': row['id'],
                    'stim_position': row['stim_position']
                })
    df_long = pd.DataFrame(data_list)
    
    # Create the plot with seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("white")
    sns.lineplot(data=df_long, x='TR', y='probability', hue='stim_position', palette="crest",
                 ax=ax, errorbar='ci', marker='o', linewidth=2)
    sns.despine()
    ax.set_xlabel('Time Point (TR)', fontsize=12)
    ax.set_ylabel('Average Probability', fontsize=12)
    ax.set_title('Average Probabilities Across Trials by Stimulus Position', fontsize=14)
    
    plt.tight_layout()
    #plt.savefig(opj(root, "output", "average_probabilities.png"), dpi=300)
    plt.show()
    
    return fig, ax

# %% ============ [REGRESSION ANALYSIS] ============
def replay_analysis(replay_dict):
    """
    For each subject and trial, correlate stim_position with probabilities at each TR.
    
    Parameters:
    -----------
    replay_dict : dict
        Dictionary where keys are classifier names and values are DataFrames
    
    Returns:
    --------
    results : list of dicts
        Each dict contains: subject_id, trial, TR, correlation, p_value
    """
    
    # First, reconstruct the full data with all classifiers
    # (since replay_dict has classifiers as keys)
    data_list = []
    for clf_name, df in replay_dict.items():
        df_copy = df.copy()
        df_copy['classifier'] = clf_name
        data_list.append(df_copy)
    
    df_full = pd.concat(data_list, ignore_index=True)
    
    # Get time columns
    time_cols = sorted([col for col in df_full.columns if col.startswith('TR')])
    
    results = []
    
    # Iterate through each subject and trial
    for subject_id in df_full['id'].unique():
        for trial in df_full[df_full['id'] == subject_id]['trial'].unique():
            # Get data for this subject-trial combination
            subset = df_full[(df_full['id'] == subject_id) & (df_full['trial'] == trial)]
            
            # Get stim_position and TR values
            stim_positions = subset['stim_position'].values
            
            # For each TR, correlate with stim_position
            for tr_col in time_cols:
                tr_values = subset[tr_col].values
                
                # Calculate regression slope
                if len(stim_positions) > 2:  # Need at least 3 points for meaningful regression
                    result = linregress(stim_positions, tr_values)
                    slope = result.slope
                    p_value = result.pvalue
                    results.append({
                        'id': subject_id,
                        'trial': trial,
                        'TR': tr_col,
                        'beta': slope,
                        'p': p_value
                    })
    
    correlation_df = pd.DataFrame(results)
    # Sort by subject_id, trial, and TR
    correlation_df = correlation_df.sort_values(['id', 'trial', 'TR']).reset_index(drop=True)
    
    correlation_df.to_csv(opj(root, "output", "correlation_results.csv"), index=False)

    return correlation_df

# %% ============ [PLOT REGRESSION SLOPE] ============
def plot_regression_slope(correlation_df):
    """
    Plot regression slopes across TRs with individual subject lines and overall average.
    
    Parameters:
    -----------
    correlation_df : pd.DataFrame
        DataFrame with columns: id, trial, TR, beta, p
    """
    # Extract TR number for plotting
    correlation_df['TR_num'] = correlation_df['TR'].str.replace('TR', '').astype(int)
    
    # Average slopes for each subject across trials
    subject_avg = correlation_df.groupby(['id', 'TR_num'])['beta'].mean().reset_index()
    subject_avg['group'] = 'subject'
    
    # Overall average slopes across all subjects and trials
    overall_avg = correlation_df.groupby('TR_num')['beta'].mean().reset_index()
    overall_avg['id'] = 'Overall'
    overall_avg['group'] = 'overall'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_style("ticks")
    
    # Plot individual subject lines with low opacity
    sns.lineplot(data=subject_avg, x='TR_num', y='beta', hue='id', palette="crest",
                 ax=ax, linewidth=1.5, alpha=0.4, legend=False)
    
    # Plot overall average with higher emphasis
    sns.lineplot(data=overall_avg, x='TR_num', y='beta', 
                 ax=ax, linewidth=3, color='black', marker='o', markersize=8)
    #label='Overall Average'
    sns.despine()
    ax.set_xlabel('Time Point (TR)', fontsize=12)
    ax.set_ylabel('Regression Slope (β)', fontsize=12)
    ax.set_title('Stimulus Position Encoding Across Time Points', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    #ax.legend()
    
    plt.tight_layout()
    plt.savefig(opj(root, "output", "regression_slopes.png"), dpi=300)
    plt.show()
    
    return fig, ax

# %% ============ [CREATE NULL DISTRIBUTION FUNCTION] ============

def permutation_options(l):
    """l: list of sequence positions (e.g. [1,2,3,4,5])"""
    def has_sequential_pair(perm):
        return any(abs(perm[i+1] - perm[i]) == 1 for i in range(len(perm)-1))

    valid = [p for p in permutations(l) if not has_sequential_pair(p)]

    print(f"Number of valid permutations: {len(valid)}")
    print(valid)
    return valid


def replay_nulldist(replay_dict):
    # First, reconstruct the full data with all classifiers
    # (since replay_dict has classifiers as keys)
    data_list = []
    for clf_name, df in replay_dict.items():
        df_copy = df.copy()
        df_copy['classifier'] = clf_name
        data_list.append(df_copy)
    
    df_full = pd.concat(data_list, ignore_index=True)
    valid = permutation_options([0,1,2,3,4])

    # Get time columns
    time_cols = sorted([col for col in df_full.columns if col.startswith('TR')])
    
    results = []
    
    # Iterate through each subject and trial
    for subject_id in df_full['id'].unique():
        for trial in df_full[df_full['id'] == subject_id]['trial'].unique():
            # Get data for this subject-trial combination
            subset = df_full[(df_full['id'] == subject_id) & (df_full['trial'] == trial)]
            subset = subset.sort_values(['stim_position']).reset_index(drop=True)
            subset["fake_position"] = random.choice(valid)
            
            # Get stim_position and TR values
            stim_positions = subset['fake_position'].values
            
            # For each TR, correlate with stim_position
            for tr_col in time_cols:
                tr_values = subset[tr_col].values
                
                # Calculate regression slope
                if len(stim_positions) > 2:  # Need at least 3 points for meaningful regression
                    result = linregress(stim_positions, tr_values)
                    slope = result.slope
                    p_value = result.pvalue
                    results.append({
                        'id': subject_id,
                        'trial': trial,
                        'TR': tr_col,
                        'beta': slope,
                        'p': p_value
                    })
    
    correlation_df = pd.DataFrame(results)
    # Sort by subject_id, trial, and TR
    correlation_df = correlation_df.sort_values(['id', 'trial', 'TR']).reset_index(drop=True)
    
    correlation_df.to_csv(opj(root, "output", "correlation_results_nulldistribution.csv"), index=False)

    return correlation_df

# %% ============ [QUANTIFY SIGNIFICANCE] ============
def quantify_beta_difference_significance(df_real, df_null):
    """
    Test whether the difference between real and null betas is significantly different from zero.
    Uses one-sample t-tests at each TR with FDR correction for multiple comparisons.
    
    Parameters:
    -----------
    df_real : pd.DataFrame
        Real regression results with columns: id, trial, TR, beta, p
    df_null : pd.DataFrame
        Null regression results with columns: id, trial, TR, beta, p
    
    Returns:
    --------
    sig_df : pd.DataFrame
        DataFrame with columns: TR, TR_num, t_stat, p_value, p_fdr_corrected, significant
    """
    # Make copies and extract TR info
    df_real = df_real.copy()
    df_null = df_null.copy()
    df_real['TR_num'] = df_real['TR'].str.replace('TR', '').astype(int)
    df_null['TR_num'] = df_null['TR'].str.replace('TR', '').astype(int)
    
    # Average for each subject at each TR
    real_subj = df_real.groupby(['id', 'TR_num'])['beta'].mean().reset_index()
    null_subj = df_null.groupby(['id', 'TR_num'])['beta'].mean().reset_index()
    
    # Merge and compute difference
    merged = real_subj.merge(null_subj, on=['id', 'TR_num'], suffixes=('_real', '_null'))
    merged['diff'] = merged['beta_real'] - merged['beta_null']
    
    # Test at each TR
    results = []
    all_p_values = []
    
    for tr_num in sorted(merged['TR_num'].unique()):
        diff_values = merged[merged['TR_num'] == tr_num]['diff'].values
        
        # One-sample t-test: H0 is that mean difference = 0
        t_stat, p_value = ttest_1samp(diff_values, popmean=0)
        all_p_values.append(p_value)
        
        results.append({
            'TR_num': tr_num,
            'TR': f'TR{tr_num:02d}',
            't_stat': t_stat,
            'p_value': p_value,
            'n_subjects': len(diff_values)
        })
    
    sig_df = pd.DataFrame(results)
    
    # Apply FDR correction
    sig_df['p_fdr_corrected'] = false_discovery_control(all_p_values, method='bh')
    sig_df['significant'] = sig_df['p_fdr_corrected'] < 0.05
    
    print("\n=== Significance Test Results (FDR corrected, alpha=0.05) ===")
    print(sig_df[['TR', 't_stat', 'p_value', 'p_fdr_corrected', 'significant']])
    print(f"\nSignificant TRs: {sig_df[sig_df['significant']]['TR'].tolist()}")
    
    sig_df.to_csv(opj(root, "output", "beta_difference_significance.csv"), index=False)
    
    return sig_df

def plot_beta_difference(df_real, df_null, sig_df=None):
    """
    Plot the difference between real and null regression slopes with SEM band.
    
    Parameters:
    -----------
    df_real : pd.DataFrame
        Real regression results with columns: id, trial, TR, beta, p
    df_null : pd.DataFrame
        Null regression results with columns: id, trial, TR, beta, p
    sig_df : pd.DataFrame, optional
        Significance test results to highlight significant TRs
    """
    # Make copies to avoid modifying originals
    df_real = df_real.copy()
    df_null = df_null.copy()
    
    # Extract TR number for plotting
    df_real['TR_num'] = df_real['TR'].str.replace('TR', '').astype(int)
    df_null['TR_num'] = df_null['TR'].str.replace('TR', '').astype(int)
    
    # Average slopes for each subject across trials
    real_subject_avg = df_real.groupby(['id', 'TR_num'])['beta'].mean().reset_index()
    null_subject_avg = df_null.groupby(['id', 'TR_num'])['beta'].mean().reset_index()
    
    # Merge to compute difference for each subject
    subject_diff = real_subject_avg.merge(null_subject_avg, on=['id', 'TR_num'], suffixes=('_real', '_null'))
    subject_diff['beta_diff'] = subject_diff['beta_real'] - subject_diff['beta_null']
    
    # Calculate mean and SEM for each TR
    diff_stats = subject_diff.groupby('TR_num')['beta_diff'].agg(['mean', 'sem']).reset_index()
    diff_stats.columns = ['TR_num', 'beta_diff', 'sem']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style("ticks")
    
    # Plot overall average with SEM band
    ax.plot(diff_stats['TR_num'], diff_stats['beta_diff'], 
            linewidth=3, color='black', marker='o', markersize=8, label='Mean Difference')
    ax.fill_between(diff_stats['TR_num'], 
                     diff_stats['beta_diff'] - diff_stats['sem'],
                     diff_stats['beta_diff'] + diff_stats['sem'],
                     alpha=0.2, color='black', label='±SEM')
    
    sns.despine()
    ax.set_xlabel('Time Point (TR)', fontsize=12)
    ax.set_ylabel('β Difference (Real - Null)', fontsize=12)
    ax.set_title('Real vs Null Stimulus Position Encoding', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Highlight significant TRs with shaded background
    if sig_df is not None:
        sig_trs = sig_df[sig_df['significant']]['TR_num'].values
        
        # Create shaded regions for significant TRs
        for tr_num in sig_trs:
            ax.axvspan(tr_num - 0.4, tr_num + 0.4, alpha=0.15, color='red')
        
        # Add legend with proper handling
        from matplotlib.lines import Line2D
        custom_lines = [ax.get_lines()[0], 
                       Line2D([0], [0], color='black', alpha=0.2, linewidth=6),
                       Line2D([0], [0], color='red', linewidth=0, marker='s', 
                             markersize=12, alpha=0.3)]
        ax.legend(custom_lines, ['Mean Difference', '±SEM', 'Significant TRs (FDR p<0.05)'], loc='best')
    else:
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(opj(root, "output", "beta_difference.png"), dpi=300)
    plt.show()
    
    return fig, ax

# %% ============ [ABSOLUTE & QUANTIFY] ============
def absolute_quantify(df_real, df_null):
    # take absolute and quantify
    df_real_abs = df_real.copy()
    df_null_abs = df_null.copy()
    df_real_abs["beta_abs"] = np.abs(df_real_abs["beta"])
    df_null_abs["beta_abs"] = np.abs(df_null_abs["beta"])

    print("Absolute Regression Slopes for REAL sequences")
    plot_regression_slope(df_real_abs)
    print("Absolute Regression Slopes for RANDOM sequences (Null distribution)")
    plot_regression_slope(df_null_abs)
    
    # Test significance of difference
    sig_df = quantify_beta_difference_significance(df_real_abs, df_null_abs)
    
    # Plot with significance highlighting
    plot_beta_difference(df_real_abs, df_null_abs, sig_df=sig_df)



# %% ============ [MAIN FUNCTION] ============
def main():
    # import
    df_all = import_data(data_dir)
    print(df_all.head())

    # restructure and select data
    replay_dict = restructure(df=df_all, tITI=0.512) #0.032, 0.064, 0.128, 0.512, 2.048
    plot_avg_probabilities(replay_dict)

    # perform replay analysis and plot slopes
    correlation_df = replay_analysis(replay_dict)
    print(correlation_df.head(10))
    print("Regression Slopes for REAL sequences")
    plot_regression_slope(correlation_df)

    # create null distribution
    correlation_null_df = replay_nulldist(replay_dict=replay_dict)
    print("Regression Slopes for RANDOM sequences (Null distribution)")
    plot_regression_slope(correlation_null_df)

    # 
    absolute_quantify(df_null=correlation_null_df, df_real=correlation_df)
    


main()