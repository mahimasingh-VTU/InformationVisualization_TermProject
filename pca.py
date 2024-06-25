"""
This file contains the function to perform PCA on the data.
Strategy: Variance Explained with 95% retention
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def perform_pca(data, n_components=0.95):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(data)

    # Plot the explained variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')
    plt.title('Dataset Explained Variance')
    plt.show()

    return principalComponents