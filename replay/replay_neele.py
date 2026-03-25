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
from pathlib import Path
from os.path import join as opj
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
        data_dir, f"*decoding*.csv",
    )
    files = glob.glob(files)
 
    print(f"Found {len(files)} files in path {data_dir}.")

    df = pd.concat(
            [pd.read_csv(f) for f in files],
            ignore_index=True
        )

    return df

# %% ============ [RESTRUCTURING INTO REPLAY TABLE] ============
def restructure(df):

    # filter the dataframe by only largest tITI and only mask == cv
    # filter to only get probabilities for category from category-vs-rest classifiers 
    # this kicks out multinomial log_reg classifier results as well
    df_replay = df[
        (df["test_set"]=="test-seq_long") &
        (df["mask"]=="cv") & 
        (df["tITI"]==2.048) &
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
        index=['id', 'classifier', 'trial'], # Grouping columns
        columns='seq_tr',              # Time points become columns
        values='probability',          # Values to fill the table
        aggfunc='first'                # Safety: if duplicates exist, take the first
    ) 
    # Reset index to turn the grouping columns back into regular columns
    df_pivot = df_pivot.reset_index()
    
    # Rename the time columns from integers (1, 2...) to clear names (TR1, TR2...)
    # The last 13 columns are the time points
    time_cols = df_pivot.columns[-13:]
    new_names = [f'TR{int(t)}' for t in time_cols]
    
    # Create a mapping to rename only these columns
    rename_dict = dict(zip(time_cols, new_names))
    df_pivot = df_pivot.rename(columns=rename_dict)
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
    
    # 5. SAVE TO CSV (Optional)
    # Save each classifier's time courses to a separate file

# %% ============ [MAIN FUNCTION] ============
def main():
    df_all = import_data(data_dir)
    print(df_all.head())

    replay_dict = restructure(df=df_all)

main()