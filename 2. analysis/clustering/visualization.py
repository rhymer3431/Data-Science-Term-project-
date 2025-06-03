import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('dataset/clustered_dataset.csv')
# 색상과 마커 정의 (GPA 색상 / Cluster 마커)
plotly_color_map = {
    'Below Average': 'lightgray',
    'Average': 'dodgerblue',
    'Good': 'mediumseagreen',
    'Excellent': 'darkorange'
}
plotly_marker_map = {
    0: 'circle',
    1: 'square',
    2: 'diamond'
}

# 산점도 데이터 생성
data = []
for gpa in df['GPA'].unique():
    for cluster in df['Cluster'].unique():
        sub_df = df[(df['GPA'] == gpa) & (df['Cluster'] == cluster)]
        if not sub_df.empty:
            data.append(go.Scatter3d(
                x=sub_df['Sleep Impact'],
                y=sub_df['Lifestyle'],
                z=sub_df['Sleep Quality'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=plotly_color_map.get(gpa, 'gray'),
                    symbol=plotly_marker_map.get(cluster, 'circle'),
                    opacity=0.7
                ),
                name=f'{gpa}, Cluster {cluster}'
            ))

# 회귀 모델 학습 (X = Sleep Impact + Lifestyle → y = Sleep Quality)
X = df[['Sleep Impact', 'Lifestyle']].values
y = df['Sleep Quality'].values
model = LinearRegression()
model.fit(X, y)

# 회귀면 생성
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
x_grid, y_grid = np.meshgrid(x_range, y_range)
z_pred = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

regression_surface = go.Surface(
    x=x_grid,
    y=y_grid,
    z=z_pred,
    colorscale='Greys',
    opacity=0.4,
    showscale=False,
    name='Regression Plane'
)

data.append(regression_surface)

# 레이아웃
layout = go.Layout(
    title='Sleep Impact - Lifestyle - Sleep Quality (3D) with Regression Plane',
    scene=dict(
        xaxis_title='Sleep Impact',
        yaxis_title='Lifestyle',
        zaxis_title='Sleep Quality'
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

fig = go.Figure(data=data, layout=layout)
fig.show()
