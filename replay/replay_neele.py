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
from pathlib import Path
from os.path import join as opj
root = Path(__file__).parent.parent
data_dir = opj(root, "data")

# %% ============ [DATA IMPORT] ============
def find_files(data_dir):
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
    return files

def import_sub(files, sub):
    files = [f for f in files if sub in f]
    df = pd.read_csv(files[0])
    return df

def preprocess_sub(df):
    df = df[df["test_set"]=="test-seq_long"]

    return df

# %% ============ [MAIN FUNCTION] ============
def main():
    files = find_files(data_dir)

    df = import_sub(files, "sub-01")

    df_sub = preprocess_sub(df)

    print(df_sub.head())


main()