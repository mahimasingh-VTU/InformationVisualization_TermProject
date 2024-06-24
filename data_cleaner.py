import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        # Handle missing values
        self.df = self.df.dropna(subset=['make','trim','body','interior','transmission','color', 'model', 'sellingprice', 'mmr', 'condition', 'odometer'])

        # Remove duplicates
        self.df = self.df.drop_duplicates()

        # Fix incorrect data (e.g., unrealistic values for condition)
        self.df = self.df[(self.df['condition'] > 0) & (self.df['condition'] <= 5)]
        self.df = self.df[self.df['odometer'] > 0]

        # Standardize data (e.g., make all text data lowercase)
        self.df['make'] = self.df['make'].str.title()
        self.df['model'] = self.df['model'].str.title()
        self.df['trim'] = self.df['trim'].str.lower()
        self.df['body'] = self.df['body'].str.lower()
        self.df['transmission'] = self.df['transmission'].str.lower()
        self.df['color'] = self.df['color'].str.lower()
        self.df['interior'] = self.df['interior'].str.lower()
        self.df['state'] = self.df['state'].str.lower()
        self.df['seller'] = self.df['seller'].str.lower()

        return self.df
