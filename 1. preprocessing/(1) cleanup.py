import pandas as pd
import numpy as np

df = pd.read_csv('dataset/(0) dirty_dataset.csv')
cleaned_df = df.iloc[:, 1:]
cleaned_df.to_csv('dataset/(1) cleaned_dataset.csv',index=False)