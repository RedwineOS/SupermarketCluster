import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %matplotlib inline - only used for iPython

# loading dataset into Pandas DataFrame
df = pd.read_csv('Wholesale customers data.csv')
features = list(df.columns)

# No standarization since they are on the same scale. Except for the first two features.
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2'])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

ax.scatter(principalDf['principal component 1']
           , principalDf['principal component 2']
           , s=50)
ax.grid()

plt.plot()
print(pca.explained_variance_ratio_)
