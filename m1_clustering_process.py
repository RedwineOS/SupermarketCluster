import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import supermarket_cluster as sc

colors = ['navy', 'turquoise']


def specify_region(row):
    if row['Region'] == 1:
        row['is_region_1'] = 1
        row['is_region_2'] = 0
        row['is_region_3'] = 0

    elif row['Region'] == 2:
        row['is_region_1'] = 0
        row['is_region_2'] = 1
        row['is_region_3'] = 0

    elif row['Region'] == 3:
        row['is_region_1'] = 0
        row['is_region_2'] = 0
        row['is_region_3'] = 1

    return row


def specify_channel(row):
    if row['Channel'] == 1:
        row['is_channel_1'] = 1
        row['is_channel_2'] = 0

    elif row['Channel'] == 2:
        row['is_channel_1'] = 0
        row['is_channel_2'] = 1

    return row


# loading dataset into Pandas DataFrame
df = pd.read_csv('Wholesale customers data.csv')
# Duplication of the dataset to hold statistics
df_stats = df
original_features = list(df.columns)
print(original_features)
df.head()

# Checking none values
df.isna().sum()

# Changing data structure
df = df.apply(lambda row: specify_region(row), axis=1)
df = df.apply(lambda row: specify_channel(row), axis=1)
df = df.drop(columns=['Channel', 'Region'])

# Check correlation of the features
plt.figure(figsize=(12, 10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Drop unuseful columns
df = df.drop(columns=['is_region_1', 'is_region_2', 'is_region_3'])

df['Grocery'].mean()
df['Detergents_Paper'].mean()

# Label to color dict (automatic)
label_color_dict = {label: idx for idx, label in enumerate(np.unique(df['is_channel_1']))}

# Color vector creation
cvec = [label_color_dict[label] for label in df['is_channel_2']]

principalDf = sc.pca_df_creation(df)

sc.pca_plot(principalDf, 'PCA')

# Channel differentiation
sc.pca_plot(principalDf, 'PCA with channels code', cvec)

# First cycle

sc.plot_silhouette_score(df)

probabilities_cluster, centroids, labels = sc.fit_and_predict_gmm(df)

# Clusters visualization
sc.pca_plot(principalDf, 'PCA with defined clusters', labels=labels)

sc.obtain_number_of_coincidences(df, probabilities_cluster, labels, cvec)

# Second cycle
df = df.drop(columns=['is_channel_1', 'is_channel_2'])
sc.plot_silhouette_score(df)

probabilities_cluster, centroids, labels = sc.fit_and_predict_gmm(df)

# Clusters visualization
sc.pca_plot(principalDf, 'PCA with defined clusters', labels=labels)

sc.obtain_number_of_coincidences(df, probabilities_cluster, labels, cvec)

# Graph with probability
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.grid()

for i in range(len(df.index)):
    ax.scatter(principalDf['principal component 1'][i], principalDf['principal component 2'][i],
               c=([probabilities_cluster[i][0], probabilities_cluster[i][1], 1]))

plt.plot()

for column in df.columns:
    print(column)
    print("Mean: " + str(df[column].mean()))
    print("Median: " + str(df[column].median()))
    print("P90: " + str(df[column].quantile(0.9)))
    print("Standard deviation: " + str(df[column].std()))

df_stats['Total'] = df.sum(axis=1)
print("Total amount of money")
print("Mean: " + str(df_stats['Total'].mean()))
print("Median: " + str(df_stats['Total'].median()))
print("P90: " + str(df_stats['Total'].quantile(0.9)))
print("Standard deviation: " + str(df_stats['Total'].std()))

# Possible outliers 333, 86, 61, 47, 85 and 181
for outlier in [333, 86, 61, 47, 85, 181]:
    print(outlier)
    print(str(df.iloc[outlier]))
    print("Total: " + str(df.iloc[outlier].sum()))

# Third cycle
df = df.drop([333, 86, 61, 47, 85, 181])
df_stats_third_cycle = df_stats.drop([333, 86, 61, 47, 85, 181])
principalDf = principalDf.drop([333, 86, 61, 47, 85, 181])

sc.plot_silhouette_score(df)

probabilities_cluster, centroids, labels = sc.fit_and_predict_gmm(df)

# Clusters visualization
sc.pca_plot(principalDf, 'PCA with defined clusters', labels=labels)

sc.obtain_number_of_coincidences(df, probabilities_cluster, labels, cvec)

# Graph with probability
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)

ax.grid()

for i in range(len(df.index)):
    if i in principalDf.index:
        ax.scatter(principalDf['principal component 1'][i], principalDf['principal component 2'][i],
                   c=([probabilities_cluster[i][0], probabilities_cluster[i][1], 1]))

plt.plot()

# Possible outliers 65, 221, 183, 284 and 125
for outlier in [65, 221, 183, 284, 125]:
    print(outlier)
    print(str(df.loc[outlier]))
    print("Total: " + str(df.loc[outlier].sum()))

# Fourth cycle
df = df.drop([65, 221, 183, 284, 125])
df_stats_fourth_cycle = df_stats_third_cycle.drop([65, 221, 183, 284, 125])
principalDf = principalDf.drop([65, 221, 183, 284, 125])

sc.plot_silhouette_score(df)

top_20_percent = df_stats[df_stats['Total'] > df_stats['Total'].quantile(.8)]['Total']
bottom_80_percent = df_stats[df_stats['Total'] < df_stats['Total'].quantile(.8)]['Total']
total_sales = df_stats['Total'].sum()
print("The top 20% represent the " + str((top_20_percent.sum() / total_sales) * 100) + "% of the total amount of sales")

# Calculate at which percentile the 80% of the business is found
for i in range(1, 99):
    top_percent = df_stats[df_stats['Total'] > df_stats['Total'].quantile(1 - (i / 100))]['Total']
    if top_percent.sum() / total_sales >= 0.8:
        print("The top " + str(i) + "% represent the " + str((top_percent.sum() / total_sales) * 100) +
              "% of the total amount of sales")
        break
