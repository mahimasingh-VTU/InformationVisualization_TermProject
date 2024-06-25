import pandas as pd
from prettytable import PrettyTable
pd.set_option('display.float_format', '{:.2f}'.format)
class DataCleaner:
    def __init__(self, df):
        self.df = df

    def print_info(self):
            info_table = PrettyTable()
            info_table.field_names = ["Index", "Column", "Non-Null Count", "Dtype"]
            for idx, (col, non_null_count, dtype) in enumerate(zip(self.df.columns, self.df.count(), self.df.dtypes)):
                    info_table.add_row([idx, col, non_null_count, dtype])
            print(info_table)


       

    def clean_data(self):
        self.df['transmission'].fillna('automatic', inplace=True)
        self.df.dropna(axis=0, inplace=True)
        self.df['year'] = pd.to_datetime(self.df['year'], format='%Y').dt.year
        condition_mapping = {range(10, 21): 1, range(20, 31): 2, range(30, 41): 3, range(40, 51): 4}
        for k, v in condition_mapping.items():
            self.df['condition'].replace(k, v, inplace=True)


        self.df['color'].replace('—', 'multicolor', inplace=True)
        self.df['interior'].replace('—', 'multicolor', inplace=True)
        self.df['body'] = self.df['body'].str.lower()

        self.df['make'] = self.df['make'].str.title()
        self.df['model'] = self.df['model'].str.title()
        self.df['trim'] = self.df['trim'].str.lower()
        self.df['body'] = self.df['body'].str.lower()
        self.df['transmission'] = self.df['transmission'].str.lower()
        self.df['color'] = self.df['color'].str.lower()
        self.df['interior'] = self.df['interior'].str.lower()
        self.df['state'] = self.df['state'].str.upper()
        self.df['seller'] = self.df['seller'].str.lower()

        # Remove odometer outliers
        self.df = self.df[self.df['odometer'] < 500000]

        # Remove rows with unrealistic values
        self.df = self.df[self.df['color'] != '—']
        self.df = self.df[self.df['interior'] != '—']

        # Convert date columns to datetime
        self.df['saledate'] = pd.to_datetime(self.df['saledate'], utc=True, format='mixed').dt.date
        
         # Remove unnecessary columns
        self.df = self.df.drop(columns=['vin'])

        # Remove duplicates
        self.df = self.df.drop_duplicates()

        # Clean the body column
        self.df['body'] = self.df['body'].str.replace(r'.*sedan.*', 'sedan', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*convertible.*', 'convertible', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*coupe.*', 'coupe', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*van.*', 'van', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*cab.*', 'pickup', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*wagon.*', 'wagon', regex=True)
        self.df['body'] = self.df['body'].str.replace(r'.*supercrew.*', 'van', regex=True)
        self.df['body'] = self.df['body'].str.replace('koup', 'coupe')


        return self.df
