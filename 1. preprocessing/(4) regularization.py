import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scored_df = pd.read_csv("dataset/(3) scored_dataset.csv")

scaler = MinMaxScaler()

cols_to_scale = scored_df.columns[2:13]  # 인덱스 2 ~ 12

scored_df[cols_to_scale] = scaler.fit_transform(scored_df[cols_to_scale])

scored_df.to_csv("dataset/(4) regularized_dataset.csv", index=False)