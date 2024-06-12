import pandas as pd

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
