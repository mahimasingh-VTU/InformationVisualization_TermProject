
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, anderson, normaltest

from prettytable import PrettyTable


class DataProcessor:
    def __init__(self, df):
        self.df = df

    def perform_pca(self):
        # Select numeric features for PCA
        numeric_features = self.df.select_dtypes(include=[np.number])
        feature_names = numeric_features.columns

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_features)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

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
        plt.figure(figsize=(12, 8))
        corr_matrix = np.corrcoef(principal_components.T)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                    xticklabels=feature_names,
                    yticklabels=feature_names)
        plt.title("Correlation Matrix of PCA Components")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.show()

        return principal_components, n_components_95, cumulative_variance[n_components_95-1]



    def normality_test_Ksquare(self, data):
        print("K-square Normality Test on Original Features:")
        results_table = PrettyTable()
        results_table.field_names = ["Feature", "Statistic", "p-value", "Normal"]
        results = {}
        for i, feature in enumerate(data.columns):
            stat, p = normaltest(data[feature])
            normal = p > 0.05
            results_table.add_row([feature, stat, p, normal])
            results[feature] = {'Statistic': stat, 'p-value': p, 'Normal': normal}
        print(results_table)