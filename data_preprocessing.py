import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, anderson

class DataProcessor:
    def __init__(self, df):
        self.df = df

    def perform_pca(self):
        # Select numeric features for PCA
        numeric_features = self.df.select_dtypes(include=[np.number])

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_features)

        # Initialize PCA
        pca = PCA()
        principal_components = pca.fit_transform(X_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Determine the number of components to explain 95% of variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

        # Plot cumulative explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_variance * 100, marker='o')
        plt.axhline(95, color='r', linestyle='--', label='95% Explained Variance')
        plt.axvline(n_components_95 - 1, color='b', linestyle='--', label=f'{n_components_95} components')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Variance Explained (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Singular values and condition number
        singular_values = pca.singular_values_
        condition_number = np.max(singular_values) / np.min(singular_values)

        # Optionally, visualize the correlation matrix of the principal components
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.corrcoef(principal_components.T), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of PCA Components")
        plt.show()

        return principal_components, n_components_95, cumulative_variance[n_components_95-1]

    def normality_test(self, data):
        results = {}
        for i, component in enumerate(data.T):  # Test each principal component
            stat, p = shapiro(component)
            results[f'Component {i + 1}'] = {'Statistic': stat, 'p-value': p, 'Normal': p > 0.05}
        return results

    def normality_test_Ksquare(self, data):
        results = {}
        for i, component in enumerate(data.T):  # Test each principal component
            stat, p = normaltest(component)
            results[f'Component {i + 1}'] = {'Statistic': stat, 'p-value': p, 'Normal': p > 0.05}
        return results