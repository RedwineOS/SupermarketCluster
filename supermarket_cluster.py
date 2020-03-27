import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.decomposition import PCA


# Utils library

def calc_metrics(data, labels, labels_pred, labels_true):
    # Homogeneity, completeness and V-measure¶
    # Given the knowledge of the ground truth class assignments of the samples,
    # it is possible to define some intuitive metric using conditional entropy analysis.
    #   homogeneity: each cluster contains only members of a single class.
    #   completeness: all members of a given class are assigned to the same cluster.
    #   V-measure: armonic means of both
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels_pred))

    # ADJUSTED RAND INDEX
    # Given the knowledge of the ground truth class assignments labels_true
    # and our clustering algorithm assignments of the same samples labels_pred,
    # the adjusted Rand index is a function that measures the similarity of the two assignments,
    # ignoring permutations and with chance normalization:
    # 1.0 is the perfect match score.
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))

    # the Mutual Information is a function that measures the agreement of the two assignments, ignoring permutations.
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))

    # The Silhouette Coefficient is defined for each sample and is composed of two scores:
    #    a: The mean distance between a sample and all other points in the same class.
    #    b: The mean distance between a sample and all other points in the next nearest cluster.
    # The Silhouette Coefficient s for a single sample is then given as: s = \frac{b - a}{max(a, b)}
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(data, labels))


def pca_df_creation(df):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['principal component 1', 'principal component 2'])

    print(pca.explained_variance_ratio_)  # 0.85
    return principal_df


def pca_plot(df, title, cvec=None, labels=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=20)

    ax.scatter(df['principal component 1'],
               df['principal component 2'],
               s=50,
               alpha=0)
    ax.grid()

    for i in range(len(df.index)):
        if i in df.index:
            if cvec:
                c = 'magenta' if cvec[i] == 0 else 'blue'
                plt.text(df['principal component 1'][i], df['principal component 2'][i], str(i),
                         ha="center",
                         va="center", c=c)
            elif labels is not None:
                c = 'magenta' if labels[i] == 0 else 'blue'
                plt.text(df['principal component 1'][i], df['principal component 2'][i], str(i),
                         ha="center",
                         va="center", c=c)
            else:
                plt.text(df['principal component 1'][i], df['principal component 2'][i], str(i), ha="center",
                         va="center", c='red')

    plt.plot()


def plot_silhouette_score(df):
    silhouettes = []

    for i in range(2, 30):
        em = GaussianMixture(n_components=i, covariance_type='full', init_params='kmeans')
        em.fit(df)
        labels = em.predict(df)
        silhouettes.append(metrics.silhouette_score(df, labels))

    # Plot Silhouette
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(range(2, 30), silhouettes, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette')
    plt.plot()


def fit_and_predict_gmm(df):
    # Model fit
    # Three hundred times to obtain the maximum likehood. Not done in the parametrization since
    # it's not as useful and requires
    # time to finish.
    em = GaussianMixture(n_components=2, n_init=300, covariance_type='full', init_params='kmeans')

    clusters = em.fit(df, 300)
    centroids = clusters.means_

    labels = clusters.predict(df)
    probabilities_cluster = em.predict_proba(df)

    return probabilities_cluster, centroids, labels


def obtain_number_of_coincidences(df, probabilities_cluster, labels, cvec):
    # Used to determine the coincidences between the clusters obtained and the channel that the customer belongs
    unique_values_cluster_1, frequency_values_cluster_1 = np.unique(probabilities_cluster[:, 0], return_counts=True)
    unique_values_cluster_2, frequency_values_cluster_2 = np.unique(probabilities_cluster[:, 1], return_counts=True)

    print("Unique values of cluster 1: " + str(unique_values_cluster_1))
    print("Frequency of values of cluster 1: " + str(frequency_values_cluster_1))

    print("Unique values of cluster 2: " + str(unique_values_cluster_2))
    print("Frequency of values of cluster 2: " + str(frequency_values_cluster_2))

    n_coincidences = 0
    for i in range(len(df.index)):
        if labels[i] == cvec[i]:
            n_coincidences += 1

    print(n_coincidences / len(df.index))
