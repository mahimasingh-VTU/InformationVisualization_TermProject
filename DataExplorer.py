import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        # Handle missing values
        self.df = self.df.dropna(subset=['make','trim','body','interior','transmission','color', 'model', 'sellingprice', 'mmr', 'condition', 'odometer'])
        # self.df['trim'].fillna('Unknown', inplace=True)
        # self.df['body'].fillna('Unknown', inplace=True)
        # self.df['transmission'].fillna('Unknown', inplace=True)
        # self.df['color'].fillna('Unknown', inplace=True)
        # self.df['interior'].fillna('Unknown', inplace=True)

        # Convert data types
        # self.df['saledate'] = pd.to_datetime(self.df['saledate'], errors='coerce')

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
        self.df['state'] = self.df['state'].str.lower()
        self.df['seller'] = self.df['seller'].str.lower()

        return self.df

class Plotter:
    def __init__(self, df):
        self.df = df

    def plot_bar(self, x, y, title, xlabel, ylabel):
        plt.figure(figsize=(10, 6))
        self.df.groupby(x)[y].count().plot(kind='bar', color='skyblue')
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(xlabel, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(ylabel, fontsize=12, fontfamily='serif', color='darkred')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def plot_line(self, x, y, title, xlabel, ylabel):
        plt.figure(figsize=(10, 6))
        self.df.groupby(x)[y].mean().plot(kind='line', color='skyblue')
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(xlabel, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(ylabel, fontsize=12, fontfamily='serif', color='darkred')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def plot_histogram(self, column, title):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], kde=True, color='skyblue')
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(column, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel('Frequency', fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_scatter(self, x, y, title, xlabel, ylabel):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df[x], self.df[y], color='skyblue')
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(xlabel, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(ylabel, fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_pie(self, column, title):
        plt.figure(figsize=(8, 8))
        self.df[column].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.ylabel('')
        plt.show()

    def plot_count(self, column, title, xlabel, ylabel):
        plt.figure(figsize=(10, 6))
        df.groupby('year')['make'].count().plot(kind='bar')
        plt.title('Vehicle Sales by Year')
        plt.xlabel('Year')
        plt.ylabel('Sales Volume')
        plt.xticks(rotation=45)
        plt.show()

        """
        2. Top 10 vehicle makes
        plot type - bar chart
        interactivity - none
        """
        plt.figure(figsize=(10, 6))
        df['make'].value_counts().head(10).plot(kind='bar')
        plt.title('Top 10 Vehicle Makes')
        plt.xlabel('Make')
        plt.ylabel('Sales Volume')
        plt.xticks(rotation=45)
        plt.show()

    def subplot_story(self):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Bar Plot: Vehicle Sales by Year
        self.df.groupby('year')['make'].count().plot(kind='bar', ax=axs[0, 0], color='skyblue')
        axs[0, 0].set_title('Vehicle Sales by Year', fontsize=12, fontfamily='serif', color='blue')
        axs[0, 0].set_xlabel('Year', fontsize=10, fontfamily='serif', color='darkred')
        axs[0, 0].set_ylabel('Sales Volume', fontsize=10, fontfamily='serif', color='darkred')

        # Bar Plot: Top 10 Vehicle Makes
        self.df['make'].value_counts().head(10).plot(kind='bar', ax=axs[0, 1], color='skyblue')
        axs[0, 1].set_title('Top 10 Vehicle Makes', fontsize=12, fontfamily='serif', color='blue')
        axs[0, 1].set_xlabel('Make', fontsize=10, fontfamily='serif', color='darkred')
        axs[0, 1].set_ylabel('Sales Volume', fontsize=10, fontfamily='serif', color='darkred')

        # Line Plot: Average Selling Price by Year
        self.df.groupby('year')['sellingprice'].mean().plot(kind='line', ax=axs[1, 0], color='skyblue')
        axs[1, 0].set_title('Average Selling Price by Year', fontsize=12, fontfamily='serif', color='blue')
        axs[1, 0].set_xlabel('Year', fontsize=10, fontfamily='serif', color='darkred')
        axs[1, 0].set_ylabel('Average Price', fontsize=10, fontfamily='serif', color='darkred')

        # Scatter Plot: Selling Price vs MMR
        axs[1, 1].scatter(self.df['sellingprice'], self.df['mmr'], color='skyblue')
        axs[1, 1].set_title('Selling Price vs MMR', fontsize=12, fontfamily='serif', color='blue')
        axs[1, 1].set_xlabel('Selling Price', fontsize=10, fontfamily='serif', color='darkred')
        axs[1, 1].set_ylabel('MMR', fontsize=10, fontfamily='serif', color='darkred')

        plt.tight_layout()
        plt.show()

    def plot_dist(self, column, title):
        plt.figure(figsize=(10, 6))
        sns.displot(self.df[column], color='skyblue')
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(column, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel('Density', fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_pair(self, columns, title):
        plt.figure(figsize=(10, 6))
        sns.pairplot(self.df[columns])
        plt.suptitle(title, fontsize=15, fontfamily='serif', color='blue')
        plt.show()

    def plot_heatmap(self, title):
        plt.figure(figsize=(10, 6))
        numeric_df = self.df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.show()

    def plot_hist_kde(self, column, title):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], kde=True, color='skyblue')
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(column, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel('Frequency', fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_qq(self, column, title):
        plt.figure(figsize=(10, 6))
        stats.probplot(self.df[column], dist="norm", plot=plt)
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.show()

    def plot_kde(self, column, title):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.df[column], fill=True, alpha=0.6, linewidth=2)
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(column, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel('Density', fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_im_reg(self, x, y, title):
        plt.figure(figsize=(10, 6))
        sns.regplot(x=self.df[x], y=self.df[y], scatter_kws={'s': 10}, line_kws={'color': 'red'})
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(x, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(y, fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_box(self, x, y, title):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[x], y=self.df[y], palette="pastel")
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(x, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(y, fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":

    # Load the data
    df = pd.read_csv('car_prices.csv')

    # Create instance of the DataExplorer
    explorer = DataExplorer(df)

    # Explore the data
    explorer.explore_data()

    # Apply the data cleaning process
    cleaner = DataCleaner(df)
    cleaned_df = cleaner.clean_data()

    # Verify the cleaning process
    cleaned_df.info()
    cleaned_df.head()

    # Create instance of the Plotter with cleaned data
    plotter = Plotter(cleaned_df)

    # Generate plots
    plotter.plot_bar('year', 'make', 'Vehicle Sales by Year', 'Year', 'Sales Volume')
    plotter.plot_histogram('condition', 'Vehicle Condition Distribution')
    plotter.plot_scatter('sellingprice', 'mmr', 'Selling Price vs MMR', 'Selling Price', 'MMR')
    plotter.plot_pie('make', 'Distribution of Vehicle Makes')
    plotter.plot_count('model', 'Count of Vehicle Models', 'Count', 'Model')
    plotter.subplot_story()
    # Additional plots
    plotter.plot_line('year', 'sellingprice', 'Average Selling Price by Year', 'Year', 'Average Selling Price')
    plotter.plot_dist('sellingprice', 'Selling Price Distribution')
    plotter.plot_pair(['sellingprice', 'mmr', 'odometer'], 'Pair Plot of Selling Price, MMR, and Odometer')
    plotter.plot_heatmap('Correlation Heatmap')
    plotter.plot_hist_kde('odometer', 'Odometer Distribution with KDE')
    plotter.plot_qq('sellingprice', 'QQ Plot of Selling Price')
    plotter.plot_kde('mmr', 'KDE Plot of MMR')
    plotter.plot_im_reg('sellingprice', 'odometer', 'Selling Price vs Odometer with Regression Line')
    plotter.plot_box('condition', 'sellingprice', 'Box Plot of Selling Price by Condition')
