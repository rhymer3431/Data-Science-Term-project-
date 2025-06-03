import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load clustered dataset
df = pd.read_csv('dataset/clustered_dataset.csv')

# Compute correlation matrix (Pearson)
correlation_matrix = df[['Sleep Quality', 'Sleep Quantity', 'Sleep Impact', 'Lifestyle']].corr()

# Custom vintage-style colormap (blue → beige → red)
vintage_cmap = LinearSegmentedColormap.from_list(
    "vintage_redblue", 
    ["#5A7D7C", "#FAF3E0", "#7B2D26"]
)

# Draw heatmap
plt.figure(figsize=(10, 8))
sns.set(style="white", font_scale=1.1)
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap=vintage_cmap,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"fontsize": 10}
)
plt.title("Correlation Matrix (Vintage Style)", fontsize=14, weight='bold', pad=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Feature correlation summary ---

# Get mean of absolute correlation for each feature
abs_corr_mean = correlation_matrix.abs().mean().sort_values(ascending=False)
abs_corr_mean_df = abs_corr_mean.reset_index()
abs_corr_mean_df.columns = ['Feature', 'Mean Absolute Correlation']

# Bar plot of average correlation strength
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
sns.barplot(
    data=abs_corr_mean_df,
    x='Mean Absolute Correlation',
    y='Feature',
    palette='muted'
)
plt.title("Mean Absolute Correlation per Feature", fontsize=14, weight='bold')
plt.xlabel("Mean |Correlation|", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()
