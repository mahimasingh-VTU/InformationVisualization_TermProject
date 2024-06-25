
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class DataExplorer:
    def __init__(self, df):
        self.df = df

    def explore_data(self):
        print("Data Shape:", self.df.shape)
        print("Data Info:")
        print(self.df.info())
        print("Data Description:")
        print(self.df.describe())
        print("Unique Values per Column:")
        for column in self.df.columns:
            print(f"{column} : {self.df[column].nunique()} unique values")
        print("Missing Values per Column:")
        print(self.df.isnull().sum())

    def plot_histograms(self):
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        self.df[numeric_cols].hist(bins=15, figsize=(15, 6), layout=(2, 4), color='skyblue')
        plt.tight_layout()
        plt.show()

    def plot_correration(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
