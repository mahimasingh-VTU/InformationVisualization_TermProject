import pandas as pd
from prettytable import PrettyTable
import numpy as np
import matplotlib as plt
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
        self.df['saledate'] = pd.to_datetime(self.df['saledate'], utc=True, format='mixed').dt.date # new format - yyyy-mm-dd
        self.df['saleyear'] = pd.to_datetime(self.df['saledate'], format='%Y').dt.year
        self.df['salemonth'] = pd.to_datetime(self.df['saledate'], format='%m').dt.month
        self.df['saleday'] = pd.to_datetime(self.df['saledate'], format='%d').dt.day
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
        self.df['state'] = self.df['state'].str.lower()
        self.df['seller'] = self.df['seller'].str.lower()

        self.df = self.df[self.df['odometer'] < 500000]
        return self.df

    def detect_and_remove_outliers(self):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for column in numeric_columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self.df
