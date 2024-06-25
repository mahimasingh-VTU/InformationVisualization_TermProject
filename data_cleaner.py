import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)
class DataCleaner:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        # Filling missing values
        self.df['transmission'].fillna('automatic', inplace=True)
        self.df.dropna(axis=0, inplace=True)

        # Convert 'year' to datetime and extract the year
        self.df['year'] = pd.to_datetime(self.df['year'], format='%Y').dt.year

        # Replace conditions with mapped values
        condition_mapping = {range(10, 21): 1, range(20, 31): 2, range(30, 41): 3, range(40, 51): 4}
        for k, v in condition_mapping.items():
            self.df['condition'].replace(k, v, inplace=True)

        # Handling categorical data
        self.df['color'].replace('—', 'multicolor', inplace=True)
        self.df['interior'].replace('—', 'multicolor', inplace=True)
        self.df['body'] = self.df['body'].str.lower()

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
