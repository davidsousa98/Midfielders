# Import libraries
import pandas as pd
from datetime import date, datetime
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.offline as pyo

# Import dataset
df = pd.read_excel('/Users/davidsousa/Documents/SportsDS/datasets/midfielders.xlsx')

# Retrieve age from date of birth
def calculate_age(born):
    born = datetime.strptime(born, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

df['Age'] = df['BirthDate'].apply(calculate_age)
df.drop(columns='BirthDate', inplace=True)

# Get an overall prespective of the data
info = df.describe()

# Outliers detection
f, axes = plt.subplots(3, 4, figsize=(10, 5))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
sb.boxplot(df["clearances_per_game"], color="red", ax=axes[0, 0])
sb.boxplot(df["interceptions_per_game"], color="red", ax=axes[0, 1])
sb.boxplot(df["blocks_per_game"], color="red", ax=axes[0, 2])
sb.boxplot(df["duels_per_game"], color="red", ax=axes[0, 3])
sb.boxplot(df["duelswon_per_game"], color="red", ax=axes[1, 0])
sb.boxplot(df["fouls_per_game"], color="red", ax=axes[1, 1])
sb.boxplot(df["passes_per_game"], color="red", ax=axes[1, 2])
sb.boxplot(df["completepasses_per_game"], color="red", ax=axes[1, 3])
sb.boxplot(df["smartpasses_per_game"], color="red", ax=axes[2, 0])
sb.boxplot(df["shots_per_game"], color="red", ax=axes[2, 1])
sb.boxplot(df["crosses_per_game"], color="red", ax=axes[2, 2])
sb.boxplot(df["sprints_per_game"], color="red", ax=axes[2, 3])

f, axes = plt.subplots(3, 4, figsize=(10, 5))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
sb.distplot(df["clearances_per_game"], color="red", ax=axes[0, 0], kde=True)
sb.distplot(df["interceptions_per_game"], color="red", ax=axes[0, 1], kde=True)
sb.distplot(df["blocks_per_game"], color="red", ax=axes[0, 2], kde=True)
sb.distplot(df["duels_per_game"], color="red", ax=axes[0, 3], kde=True)
sb.distplot(df["duelswon_per_game"], color="red", ax=axes[1, 0], kde=True)
sb.distplot(df["fouls_per_game"], color="red", ax=axes[1, 1], kde=True)
sb.distplot(df["passes_per_game"], color="red", ax=axes[1, 2], kde=True)
sb.distplot(df["completepasses_per_game"], color="red", ax=axes[1, 3], kde=True)
sb.distplot(df["smartpasses_per_game"], color="red", ax=axes[2, 0], kde=True)
sb.distplot(df["shots_per_game"], color="red", ax=axes[2, 1], kde=True)
sb.distplot(df["crosses_per_game"], color="red", ax=axes[2, 2], kde=True)
sb.distplot(df["sprints_per_game"], color="red", ax=axes[2, 3], kde=True)

# Excluding outlier
df['Outlier'] = 0
df.loc[df['fouls_per_game'] > 5.9, 'Outlier'] = 1 # this nonstandard behavior is due to the player possessing only 1 match
df = df.loc[df['Outlier'] == 0]
df.drop(columns='Outlier', inplace=True)

# Correlation analysis
numerical = df[['clearances_per_game', 'interceptions_per_game', 'blocks_per_game', 'duels_per_game', 'duelswon_per_game',
                'fouls_per_game', 'passes_per_game', 'completepasses_per_game', 'smartpasses_per_game',
                'shots_per_game', 'crosses_per_game', 'sprints_per_game']]
corr_matrix = numerical.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(data=corr_matrix, mask=mask, center=0, annot=True, linewidths=2, cmap='coolwarm')
plt.tight_layout()

# Select variables
X = df[['clearances_per_game', 'interceptions_per_game', 'duelswon_per_game',  'blocks_per_game',
        'fouls_per_game', 'completepasses_per_game', 'smartpasses_per_game', 'shots_per_game', 'crosses_per_game']]

# Data standardization
scaler = StandardScaler()
scaler_X = pd.DataFrame(scaler.fit_transform(X))
scaler_X.columns = ['clearances_per_game', 'interceptions_per_game', 'duelswon_per_game',  'blocks_per_game',
                    'fouls_per_game', 'completepasses_per_game', 'smartpasses_per_game', 'shots_per_game', 'crosses_per_game']

# KMeans algorithm
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=15).fit(scaler_X)

# Obtain cluster size
scaler_X['Cluster'] = kmeans.labels_
cluster_size = pd.DataFrame(scaler_X.Cluster.value_counts()).reset_index()
cluster_size.columns = ['Cluster', 'Size']

# Get clusters characterization based on the position of the centroids
centroids = pd.DataFrame(kmeans.cluster_centers_).reset_index()
centroids.columns = ['Cluster', 'clearances_per_game', 'interceptions_per_game', 'duelswon_per_game',  'blocks_per_game',
                     'fouls_per_game', 'completepasses_per_game', 'smartpasses_per_game', 'shots_per_game', 'crosses_per_game']
centroids = centroids.merge(cluster_size, on='Cluster', how='inner')

# Cluster labeling
df['Cluster'] = kmeans.labels_
df['Role'] = 'Offensive'
df.loc[df['Cluster'] == 0, 'Role'] = 'Defensive'
df.loc[df['Cluster'] == 1, 'Role'] = 'Inactive'

# Cluster visualization
cluster_1 = df.loc[df['Role'] == 'Defensive']
cluster_2 = df.loc[df['Role'] == 'Offensive']
cluster_3 = df.loc[df['Role'] == 'Inactive']

scatter1 = dict(
    mode="markers",
    name="Defensive",
    type="scatter3d",
    text=cluster_1['PlayerName'],
    x=cluster_1.values[:, 10], y=cluster_1.values[:, 16], z=cluster_1.values[:, 21],
    marker=dict(size=2, color="green")
)
scatter2 = dict(
    mode="markers",
    name="Offensive",
    type="scatter3d",
    text=cluster_2['PlayerName'],
    x=cluster_2.values[:, 10], y=cluster_2.values[:, 16], z=cluster_2.values[:, 21],
    marker=dict(size=2, color="blue")
)
scatter3 = dict(
    mode="markers",
    name="Inactive",
    type="scatter3d",
    text=cluster_3['PlayerName'],
    x=cluster_3.values[:, 10], y=cluster_3.values[:, 16], z=cluster_3.values[:, 21],
    marker=dict(size=2, color="red")
)

cluster1 = dict(
    alphahull=5,
    name="Defensive",
    opacity=.1,
    type="mesh3d",
    text=cluster_1['PlayerName'],
    x=cluster_1.values[:, 10], y=cluster_1.values[:, 16], z=cluster_1.values[:, 21],
    color='green', showscale=True
)
cluster2 = dict(
    alphahull=5,
    name="Offensive",
    opacity=.1,
    type="mesh3d",
    text=cluster_2['PlayerName'],
    x=cluster_2.values[:, 10], y=cluster_2.values[:, 16], z=cluster_2.values[:, 21],
    color='blue', showscale=True
)
cluster3 = dict(
    alphahull=5,
    name="Inactive",
    opacity=.1,
    type="mesh3d",
    text=cluster_3['PlayerName'],
    x=cluster_3.values[:, 10], y=cluster_3.values[:, 16], z=cluster_3.values[:, 21],
    color='red', showscale=True
)


layout = dict(
    title='3d point clustering',
    scene=dict(
        xaxis=dict(zeroline=True, title_text='Clearances per game'),
        yaxis=dict(zeroline=True, title_text='Fouls per game'),
        zaxis=dict(zeroline=True, title_text='Shots per game'),
    ),
)
fig = dict(data=[scatter1, scatter2, scatter3, cluster1, cluster2, cluster3], layout=layout)
pyo.plot(fig)

# Set Player ID to index
df.set_index('PlayerID', inplace=True)

# Export dataset to excel file
df.to_excel('/Users/davidsousa/Documents/SportsDS/datasets/players_clt.xlsx')

