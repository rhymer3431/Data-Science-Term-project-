import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load merged dataset
df = pd.read_csv("dataset/(5) merged_dataset.csv")

# Select features and target
X = df[['Sleep Quality', 'Sleep Quantity', 'Sleep Impact', 'Lifestyle']]
y = df['GPA']

# Encode GPA (categorical)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, random_state=42)

# Try different tree depths and store accuracy
depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=0)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))  # training accuracy
    test_scores.append(clf.score(X_test, y_test))     # test accuracy

# Plot accuracy vs depth
plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, label='Train Accuracy', marker='o')
plt.plot(depths, test_scores, label='Test Accuracy', marker='s')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs Max Depth')
plt.xticks(depths)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
