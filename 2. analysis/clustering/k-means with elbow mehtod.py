from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('dataset/(5) merged_dataset.csv')
target_col = ['Sleep Quality', 'Sleep Quantity','Sleep Impact', 'Lifestyle']

X = df[target_col]

sse = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)  # data = scaled numeric features
    sse.append(kmeans.inertia_)

plt.plot(K_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method')
plt.grid(True)
plt.show()


# we can see that k value is 3~4 is good enough
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

df['Cluster'] = clusters
sns.pairplot(df, vars=['Sleep Quality', 'Sleep Impact', 'Lifestyle'], hue='Cluster')
plt.show()
print(df.groupby('Cluster')['GPA'].value_counts())
print(df.groupby('Cluster')['Stress'].value_counts())
print(df.groupby('Cluster')['Gender'].value_counts())
print(df.groupby('Cluster')['Year'].value_counts())


df.to_csv('dataset/clustered_dataset.csv', index=False)


