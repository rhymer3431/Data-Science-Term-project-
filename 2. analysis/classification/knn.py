from sklearn.model_selection import cross_val_score, StratifiedKFold,train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
df = pd.read_csv("dataset/(5) merged_dataset.csv")
X = df[['Sleep Quality', 'Sleep Quantity', 'Sleep Impact', 'Lifestyle']]
y = df['GPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# K-fold cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')


plt.figure(figsize=(8, 5))
plt.bar(range(1, 6), scores, color='skyblue')
plt.xticks(range(1, 6))
plt.ylim(0, 1)
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("K-Fold Cross Validation Accuracy")
plt.show()

print("교차검증 정확도 평균:", scores.mean())