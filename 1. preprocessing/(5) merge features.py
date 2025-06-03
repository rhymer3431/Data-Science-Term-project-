import pandas as pd
import numpy as np

# Load the normalized dataset
scaled_df = pd.read_csv("dataset/(4) regularized_dataset.csv")

# Create merged features by averaging related questions
merged_sleep_quality = scaled_df.iloc[:, [2, 4, 5]].mean(axis=1)
merged_sleep_quantity = scaled_df.iloc[:, [3]].mean(axis=1)
merged_sleep_impact = scaled_df.iloc[:, 6:10].mean(axis=1)
merged_lifestyle = scaled_df.iloc[:, 10:13].mean(axis=1)

# Select target columns for reference (optional)
sleep_quality = scaled_df.iloc[:, [2, 4]]
sleep_quantity = scaled_df.iloc[:, [3, 5]]
sleep_influence = scaled_df.iloc[:, 6:10]
lifestyle = scaled_df.iloc[:, 10:13]
stress = scaled_df.iloc[:, 13]
gpa = scaled_df.iloc[:, 14]

# Make final merged dataset
merged_df = pd.DataFrame({
    'Year': scaled_df.iloc[:, 0],
    'Gender': scaled_df.iloc[:, 1],
    'Sleep Quality': merged_sleep_quality,
    'Sleep Quantity': merged_sleep_quantity,
    'Sleep Impact': merged_sleep_impact,
    'Lifestyle': merged_lifestyle,
    'Stress': stress,
    'GPA': gpa,
})

# Save it
merged_df.to_csv('dataset/(5) merged_dataset.csv', index=False)
