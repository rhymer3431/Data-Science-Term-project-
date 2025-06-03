from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Load scored dataset
df = pd.read_csv('dataset/(3) scored_dataset.csv')
X = df[df.columns[2:13]]  # Q3 ~ Q13

# 1. Check how much variance is explained as we increase components
explained_variances = []
components_range = range(1, X.shape[1] + 1)

for n in components_range:
    pca = PCA(n_components=n)
    pca.fit(X)
    explained_variances.append(pca.explained_variance_ratio_.sum())

# 2. Plot the result
plt.figure(figsize=(8, 5))
plt.plot(components_range, explained_variances, marker='o', linewidth=2.5)
plt.xticks(components_range)
plt.xlabel('Number of Principal Components (n)', fontsize=12)
plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
plt.title('PCA - Explained Variance vs Number of Components', fontsize=14, weight='bold')
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

# Note:
# With 4 components, we explain ~75% of the variance.
# So using our merged 4 features might be a smart shortcut 👍
