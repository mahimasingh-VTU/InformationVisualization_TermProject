import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        # drop vin column
        self.df = self.df.drop(columns=['vin'])

        # Handle missing values
        self.df = self.df.dropna(subset=['make','trim','body','interior','transmission','color', 'model', 'sellingprice', 'mmr', 'condition', 'odometer'])

        # Remove duplicates
        self.df = self.df.drop_duplicates()

        # Fix incorrect data (e.g., unrealistic values for condition)
        self.df = self.df[(self.df['condition'] > 0) & (self.df['condition'] <= 5)]
        self.df = self.df[self.df['odometer'] > 0]

        # Standardize data (e.g., make all text data lowercase)
        self.df['make'] = self.df['make'].str.lower()
        self.df['model'] = self.df['model'].str.lower()
        self.df['trim'] = self.df['trim'].str.lower()
        self.df['body'] = self.df['body'].str.lower()
        self.df['transmission'] = self.df['transmission'].str.lower()
        self.df['color'] = self.df['color'].str.lower()
        self.df['interior'] = self.df['interior'].str.lower()
        self.df['state'] = self.df['state'].str.upper()
        self.df['seller'] = self.df['seller'].str.lower()

        # Remove rows with unrealistic values
        self.df = self.df[self.df['color'] != '—']
        self.df = self.df[self.df['interior'] != '—']

        # Convert date columns to datetime
        self.df['saledate'] = pd.to_datetime(self.df['saledate'], utc=True, format='mixed').dt.date

        # Clean the body column
        self.df['body'] = self.df['body'].str.replace(r'.*sedan.*', 'sedan', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*convertible.*', 'convertible', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*coupe.*', 'coupe', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*van.*', 'van', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*cab.*', 'pickup', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*wagon.*', 'wagon', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*supercrew.*', 'van', regex=True)
        self.df['body'] = self.df['body'].str.replace('koup', 'coupe')

        print(self.df.info())
        return self.df
