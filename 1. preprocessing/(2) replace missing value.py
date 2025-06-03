import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv("dataset/(1) cleaned_dataset.csv")
df_copy = df.copy()

# Step 1: Temporarily fill missing values with the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
df_filled = pd.DataFrame(imputer.fit_transform(df_copy), columns=df_copy.columns)

# Step 2: Use decision trees to refill missing values group by group
def fill_missing_with_tree(df_filled, df_original, col_indices):
    for target_idx in col_indices:
        target_col = df.columns[target_idx]
        other_indices = [i for i in col_indices if i != target_idx]
        other_cols = df.columns[other_indices]

        not_null_mask = df_original[target_col].notnull()
        null_mask = df_original[target_col].isnull()
        if null_mask.sum() == 0:
            continue

        # Use other columns to predict missing values
        X_train = pd.get_dummies(df_filled.loc[not_null_mask, other_cols])
        y_train = df_filled.loc[not_null_mask, target_col]
        X_pred = pd.get_dummies(df_filled.loc[null_mask, other_cols])
        X_pred = X_pred.reindex(columns=X_train.columns, fill_value=0)

        # Train decision tree and predict missing values
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        df_filled.loc[null_mask, target_col] = clf.predict(X_pred)

    return df_filled

# Step 3: Define column groups (Q3~Q15)
group1_cols = list(range(2, 6))    # Q3~Q6 (sleep)
group2_cols = list(range(6, 10))   # Q7~Q10 (sensitivity)
group3_cols = list(range(10, 13))  # Q11~Q13 (lifestyle)
target_cols = list(range(13, 15))  # Q14~Q15 (stress, GPA)

# Step 4: Fill each group one by one
df_filled = fill_missing_with_tree(df_filled, df_copy, group1_cols)
df_filled = fill_missing_with_tree(df_filled, df_copy, group2_cols)
df_filled = fill_missing_with_tree(df_filled, df_copy, group3_cols)

# Fill Q14~Q15 using all Q3~Q13 features
all_feature_cols = df.columns[2:13]
for target_idx in target_cols:
    target_col = df.columns[target_idx]
    null_mask = df[target_col].isnull()
    not_null_mask = ~null_mask

    if null_mask.sum() == 0:
        continue

    X_train = pd.get_dummies(df_filled.loc[not_null_mask, all_feature_cols])
    y_train = df_filled.loc[not_null_mask, target_col]
    X_pred = pd.get_dummies(df_filled.loc[null_mask, all_feature_cols])
    X_pred = X_pred.reindex(columns=X_train.columns, fill_value=0)

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    df_filled.loc[null_mask, target_col] = clf.predict(X_pred)

# Save the final cleaned dataset
df_filled.to_csv("dataset/(2) cleaned_filled_dataset.csv", index=False)
