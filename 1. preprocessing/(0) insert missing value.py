import pandas as pd
import numpy as np

df = pd.read_csv('dataset/original_dataset.csv')
target_columns = df.columns[3:]

df_missing_original = df.copy()

np.random.seed(42)
for col in target_columns:
    n = len(df_missing_original)
    missing_ratio = np.random.uniform(0.05, 0.10)  # random 5~10% missing
    num_missing = int(n * missing_ratio)
    missing_indices = np.random.choice(df_missing_original.index, num_missing, replace=False)
    df_missing_original.loc[missing_indices, col] = np.nan  # insert NaNs

df_missing_original.to_csv('dataset/(0) dirty_dataset.csv', index=False)  # save with missing values
