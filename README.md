# Data Science Term project
Group 10
202135746 김인규
202135751 김진하
202235126 조민주
202235036 노유정

# Sleep, Lifestyle, and GPA Analysis

This project analyzes the relationship between sleep patterns, lifestyle habits, stress levels, and academic performance (GPA) using Python.

## Dataset Description

The dataset (`dataset.csv`) includes the following main variables:

- Sleep Quality: categorical responses related to sleep satisfaction
- Sleep Quantity: hours of sleep on average
- Lifestyle: routines and habits (e.g., exercise, screen time)
- Stress: self-reported stress levels
- GPA: grade point average

## Analysis Workflow

### 1. Data Preprocessing

- Removed the first column (index or identifier)
- Mapped categorical survey responses to numerical scores (e.g., 'Never' → 7, 'Every night' → 0)
- Grouped related features into categories: Sleep Quality, Sleep Quantity, Sleep Influence, Lifestyle

### 2. Normalization

- Applied MinMaxScaler to normalize feature values between 0 and 1

### 3. Supervised Learning Models

- Linear Regression: predicted GPA based on sleep and lifestyle variables
- K-Nearest Neighbors (KNN): classified GPA into performance bands
- Decision Tree Classifier: identified key variables that influence GPA

### 4. Unsupervised Learning

- Performed K-Means clustering to group students by sleep and lifestyle characteristics
- Analyzed and visualized cluster-specific trends

### 5. Visualization

- Used matplotlib and seaborn to plot:
  - Heatmaps of Pearson correlations
  - Bar plots of average feature scores
  - Clustered data distributions

### 6. Correlation Analysis

- Computed pairwise correlations to investigate relationships between GPA and behavioral metrics

## Tools and Libraries

- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Key Insights

- Higher sleep quality and quantity are positively associated with GPA
- Consistent lifestyle patterns and lower stress levels correlate with better academic performance
- Clustering reveals distinct behavior groups with potential for targeted intervention
