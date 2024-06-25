
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, anderson, normaltest

from prettytable import PrettyTable
from scipy.stats import normaltest

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
        corr_matrix = np.corrcoef(principal_components.T)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                    xticklabels=[f'PC{i + 1}' for i in range(corr_matrix.shape[0])],
                    yticklabels=[f'PC{i + 1}' for i in range(corr_matrix.shape[0])])
        plt.title("Correlation Matrix of PCA Components")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.show()

        return principal_components, n_components_95, cumulative_variance[n_components_95-1]

    def normality_test(self, data):
        results_table = PrettyTable()
        results_table.field_names = ["Component", "Statistic", "p-value", "Normal"]
        results = {}
        for i, component in enumerate(data.T):  # Test each principal component
            stat, p = shapiro(component)
            normal = p > 0.05
            results_table.add_row([f'Component {i + 1}', stat, p, normal])
            results[f'Component {i + 1}'] = {'Statistic': stat, 'p-value': p, 'Normal': normal}
        print(results_table)


    def normality_test_Ksquare(self, data):
        results_table = PrettyTable()
        results_table.field_names = ["Component", "Statistic", "p-value", "Normal"]
        results = {}
        for i, component in enumerate(data.columns):
            stat, p = normaltest(data[component])
            normal = p > 0.05
            results_table.add_row([component, stat, p, normal])
            results[component] = {'Statistic': stat, 'p-value': p, 'Normal': normal}
        print(results_table)
