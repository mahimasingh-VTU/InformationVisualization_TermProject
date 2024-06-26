import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
pd.set_option('display.float_format', '{:.2f}'.format)
class DataExplorer:
    def __init__(self, df):
        self.df = df

    def explore_data(self):
        print("\nData Shape:", self.df.shape)

        # Data Info
        info_table = PrettyTable()
        info_table.field_names = ["Index", "Column", "Non-Null Count", "Dtype"]
        for idx, (col, non_null_count, dtype) in enumerate(zip(self.df.columns, self.df.count(), self.df.dtypes)):
            info_table.add_row([idx, col, non_null_count, dtype])
        print("\nData Info:\n",info_table)

        # Data Description
        desc = self.df.describe().T
        desc_table = PrettyTable()
        desc_table.field_names = desc.columns.tolist()
        for row in desc.iterrows():
            desc_table.add_row(row[1].tolist())
        print("\nData decription:\n", desc_table)

        # Feature type
        numerical_features = self.df.select_dtypes(include=[np.number]).columns
        categorical_features = self.df.select_dtypes(include=[object]).columns
        print(f'\nNumerical features: {numerical_features.to_list()}')
        print(f'Categorical features: {categorical_features.to_list()}\n')

        # Unique Values per Column
        unique_table = PrettyTable()
        unique_table.field_names = ["Column", "Unique Values"]
        for column in self.df.columns:
            unique_table.add_row([column, self.df[column].nunique()])
        print("\nUnique values",unique_table)

        # Missing Values per Column
        missing_table = PrettyTable()
        missing_table.field_names = ["Column", "Missing Values"]
        for column in self.df.columns:
            missing_table.add_row([column, self.df[column].isnull().sum()])
        print("\nMissing values",missing_table)
    def plot_histograms(self):
        relevant_cols = ['sellingprice', 'condition', 'mmr', 'year','odometer']
        numeric_cols = [col for col in relevant_cols if
                        col in self.df.columns and np.issubdtype(self.df[col].dtype, np.number)]
        for col in numeric_cols:
            plt.figure(figsize=(12, 8))
            self.df[col].hist(bins=15, color='skyblue', alpha=0.75)
            plt.title(f'Distribution of {col.capitalize()}', fontsize=16, fontfamily='serif', color='blue')
            plt.xlabel(col.capitalize(), fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(True)
            plt.tick_params(axis='x', labelrotation=45, labelsize=12)
            plt.tight_layout()
            plt.show()