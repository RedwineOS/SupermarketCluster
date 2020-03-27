from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import supermarket_cluster as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('Wholesale customers data.csv')
principal_df = sc.pca_df_creation(df)

# Divide main dataset into 6 the 6 possible permutations of channel and region
# G1 is C1 R1
# G2 is C1 R2
# G3 is C1 R3
# G4 is C2 R1
# G5 is C2 R2
# G6 is C2 R3

dataset_groups = []
for i in [1, 2]:

    for j in [1, 2, 3]:
        df_group = df[(df['Channel'] == i) & (df['Region'] == j)].drop(columns=['Region', 'Channel'])
        dataset_groups.append(df_group)

# Plot each group
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('PCA with final groups', fontsize=20)

ax.scatter(principal_df['principal component 1'],
           principal_df['principal component 2'],
           s=50,
           alpha=0)
ax.grid()

colours = [['magenta', 'blue', 'yellow'], ['red', 'purple', 'black']]
for i in range(len(principal_df.index)):

    if i in principal_df.index:
        c = colours[df['Channel'].loc[i] - 1][df['Region'].loc[i] - 1]
        plt.text(principal_df['principal component 1'][i], principal_df['principal component 2'][i], str(i),
                 ha="center",
                 va="center", c=c)

plt.plot()

# Calculate representative points
# The representative is the one that has the least SSE between itself and the mean of the current dataset

scaler = MinMaxScaler()
i = 1
representatives = []

for df_group in dataset_groups:
    # Deep copy to avoid dirtying the main dataset groups
    df_group_scaled = df_group.copy(deep=True)
    # Scaling is important to avoid that features that move bigger quantities of money have a bigger impact
    df_group_scaled[df_group_scaled.columns] = scaler.fit_transform(df_group_scaled[df_group_scaled.columns])
    representatives.append(
        df_group_scaled.loc[((df_group_scaled.mean() - df_group_scaled) ** 2).sum(axis=1).abs().idxmin()])
    print("Group " + str(i))
    print(df_group_scaled.mean())
    print("Representative of this group: ")
    print(str(representatives[i - 1]))
    i += 1

# Amount of money of each group
i = 1

for df_group in dataset_groups:
    df_group['Total'] = df_group.sum(axis=1)
    print("Group " + str(i))
    print("Total amount of money: " + str(df_group['Total'].sum()))
    print("Total amount of clients: " + str(len(df_group)))
    print("Money that each client spends on average: " + str(df_group['Total'].sum() / len(df_group)))

    i += 1

# Kruskal test for the null hypothesis
# Create a dataset for each of the values that need to be tested
df_fresh = pd.DataFrame()
df_milk = pd.DataFrame()
df_grocery = pd.DataFrame()
df_frozen = pd.DataFrame()
df_detergents_paper = pd.DataFrame()
df_delicassen = pd.DataFrame()
df_total = pd.DataFrame()

for df_group in dataset_groups:
    df_fresh = pd.concat([df_fresh, df_group['Fresh'].reset_index(drop=True)], axis=1)
    df_milk = pd.concat([df_milk, df_group['Milk'].reset_index(drop=True)], axis=1)
    df_grocery = pd.concat([df_grocery, df_group['Grocery'].reset_index(drop=True)], axis=1)
    df_frozen = pd.concat([df_frozen, df_group['Frozen'].reset_index(drop=True)], axis=1)
    df_detergents_paper = pd.concat([df_detergents_paper, df_group['Detergents_Paper'].reset_index(drop=True)], axis=1)
    df_delicassen = pd.concat([df_delicassen, df_group['Delicassen'].reset_index(drop=True)], axis=1)
    df_total = pd.concat([df_total, df_group['Total'].reset_index(drop=True)], axis=1)

df_fresh.columns = ['C1R1', 'C1R2', 'C1R3', 'C2R1', 'C2R2', 'C2R3']
df_milk.columns = ['C1R1', 'C1R2', 'C1R3', 'C2R1', 'C2R2', 'C2R3']
df_grocery.columns = ['C1R1', 'C1R2', 'C1R3', 'C2R1', 'C2R2', 'C2R3']
df_frozen.columns = ['C1R1', 'C1R2', 'C1R3', 'C2R1', 'C2R2', 'C2R3']
df_detergents_paper.columns = ['C1R1', 'C1R2', 'C1R3', 'C2R1', 'C2R2', 'C2R3']
df_delicassen.columns = ['C1R1', 'C1R2', 'C1R3', 'C2R1', 'C2R2', 'C2R3']
df_total.columns = ['C1R1', 'C1R2', 'C1R3', 'C2R1', 'C2R2', 'C2R3']

for df_group in [df_fresh, df_milk, df_grocery, df_frozen, df_detergents_paper, df_delicassen, df_total]:
    df_group_values = df_group.values
    print(stats.kruskal(*[df_group_values[x, :] for x in np.arange(df_group_values.shape[0])],
                        nan_policy='omit'))
