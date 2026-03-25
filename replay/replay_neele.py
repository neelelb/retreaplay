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
from pathlib import Path
from os.path import join as opj
from scipy.stats import pearsonr, linregress
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
                    'Probability': row[tr_col],
                    'Classifier': clf_name,
                    'Trial': row['trial'],
                    'ID': row['id']
                })
    
    df_long = pd.DataFrame(data_list)
    
    # Create the plot with seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("white")
    sns.lineplot(data=df_long, x='TR', y='Probability', hue='Classifier', 
                 ax=ax, errorbar='ci', marker='o', linewidth=2)
    sns.despine()
    ax.set_xlabel('Time Point (TR)', fontsize=12)
    ax.set_ylabel('Average Probability', fontsize=12)
    ax.set_title('Average Probabilities Across Trials by Classifier', fontsize=14)
    
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

# %% ============ [PLOT FUNCTION] ============
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

# %% ============ [MAIN FUNCTION] ============
def main():
    df_all = import_data(data_dir)
    print(df_all.head())

    replay_dict = restructure(df=df_all, tITI=2.048)
    plot_avg_probabilities(replay_dict)

    correlation_df = replay_analysis(replay_dict)
    print(correlation_df.head(10))

    plot_regression_slope(correlation_df)

main()