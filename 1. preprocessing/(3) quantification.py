import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("dataset/(2) cleaned_filled_dataset.csv")
scored_df = df.copy()

# Set up score mappings for each question
q3_score = {
    'Never': 7,
    'Rarely (1-2 times a week)': 5.5,
    'Sometimes (3-4 times a week)': 3.5,
    'Often (5-6 times a week)': 1.5,
    'Every night': 0
}

q4_score = {
    'Less than 4 hours': 2,
    '4-5 hours': 5.5,
    '6-7 hours': 6.5,
    '7-8 hours': 7.5,
    'More than 8 hours': 10.0,
}

q5_score = q3_score.copy()
q6_score = {'Very poor': 0, 'Poor': 1, 'Average': 2, 'Good': 3, 'Very good': 4}
q7_score = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4}
q8_score = q7_score.copy()
q9_score = {
    'Never': 0,
    'Rarely (1-2 times a month)': 1,
    'Sometimes (1-2 times a week)': 2,
    'Often (3-4 times a week)': 3,
    'Always': 4,
}
q10_score = {
    'No impact': 0,
    'Minor impact': 1,
    'Moderate impact': 2,
    'Major impact': 3,
    'Severe impact': 4
}
q11_score = q3_score.copy()
q12_score = {
    'Never': 7,
    'Rarely (1-2 times a week)': 5.5,
    'Sometimes (3-4 times a week)': 3.5,
    'Often (5-6 times a week)': 1.5,
    'Every day': 0
}
q13_score = {
    'Never': 0,
    'Rarely (1-2 times a week)': 1.5,
    'Sometimes (3-4 times a week)': 3.5,
    'Often (5-6 times a week)': 5.5,
    'Every day': 7
}

# Replace original answers with scores
scored_df.iloc[:, 2].replace(q3_score, inplace=True)   # Q3
scored_df.iloc[:, 3].replace(q4_score, inplace=True)   # Q4
scored_df.iloc[:, 4].replace(q5_score, inplace=True)   # Q5
scored_df.iloc[:, 5].replace(q6_score, inplace=True)   # Q6
scored_df.iloc[:, 6].replace(q7_score, inplace=True)   # Q7
scored_df.iloc[:, 7].replace(q8_score, inplace=True)   # Q8
scored_df.iloc[:, 8].replace(q9_score, inplace=True)   # Q9
scored_df.iloc[:, 9].replace(q10_score, inplace=True)  # Q10
scored_df.iloc[:, 10].replace(q11_score, inplace=True) # Q11
scored_df.iloc[:, 11].replace(q12_score, inplace=True) # Q12
scored_df.iloc[:, 12].replace(q13_score, inplace=True) # Q13

# Save scored version
scored_df.to_csv("dataset/(3) scored_dataset.csv", index=False)
