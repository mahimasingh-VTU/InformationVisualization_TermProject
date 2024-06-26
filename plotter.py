import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)
class Plotter:
    def __init__(self, df):
        self.df = df

    def plot_bar(self, x, y, title, xlabel, ylabel):
        # Group by 'x' column, calculate count, and reset index to keep 'x' as a column
        data = self.df.groupby(x)[y].count().reset_index()

        plt.figure(figsize=(10, 6))
        # Use seaborn or matplotlib to plot, ensuring 'x' is used explicitly for the x-axis
        sns.barplot(data=data, x=x, y=y, color='skyblue')  # Seaborn automatically handles categorical axes better
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

    def plot_bar_vis(self, x, y, title, xlabel, ylabel):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.df[x], y=self.df[y], palette='viridis')
        plt.title(title, fontsize=20, fontfamily='serif', color='blue')
        plt.xlabel(xlabel, fontsize=16, fontfamily='serif', color='darkred')
        plt.ylabel(ylabel, fontsize=16, fontfamily='serif', color='darkred')
        plt.xticks(rotation=45)
        plt.show()

    def plot_boxen(self, x, y, title):
        plt.figure(figsize=(12, 8))
        sns.boxenplot(x=self.df[x], y=self.df[y])
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(x, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(y, fontsize=12, fontfamily='serif', color='darkred')
        plt.show()
    def plot_count(self, column, title, xlabel, ylabel):
        plt.figure(figsize=(12, 8))
        sns.countplot(x=self.df[column], palette='viridis')
        plt.title(title, fontsize=20, fontfamily='serif', color='blue')
        plt.xlabel(xlabel, fontsize=16, fontfamily='serif', color='darkred')
        plt.ylabel(ylabel, fontsize=16, fontfamily='serif', color='darkred')
        plt.xticks(rotation=90)
        plt.show()


    def plot_reg(self, x, y, title, xlabel, ylabel):
        plt.figure(figsize=(12, 6))
        sns.regplot(x=self.df[x], y=self.df[y], marker='o', color=".3", line_kws=dict(color="r"))
        plt.title(title, fontsize=20, fontfamily='serif', color='blue')
        plt.xlabel(xlabel, fontsize=16, fontfamily='serif', color='darkred')
        plt.ylabel(ylabel, fontsize=16, fontfamily='serif', color='darkred')
        plt.show()
    def plot_pie(self, column, title, threshold=0.007):
        # Calculate counts and create a new Series for plotting
        counts = self.df[column].value_counts(normalize=True)
        # Combine small categories into "Other"
        small_categories = counts[counts < threshold].sum()
        counts = counts[counts >= threshold]
        counts['Other'] = small_categories

        plt.figure(figsize=(12, 12))
        counts.plot.pie(autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.ylabel('')
        plt.show()

    def plot_grouped_bar(self, x, y, hue, title):
        plt.figure(figsize=(12, 8))
        sns.barplot(data=self.df, x=x, y=y, hue=hue)
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(x, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(y, fontsize=12, fontfamily='serif', color='darkred')
        plt.legend(title=hue)
        plt.show()


        """
        2. Top 10 vehicle makes
        plot type - bar chart
        interactivity - none
        """
        plt.figure(figsize=(10, 6))
        self.df['make'].value_counts().head(10).plot(kind='bar')
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
        sns.histplot(self.df[column], kde=True, color='skyblue')
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(column, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel('Density', fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_pair(self, columns=None, title='Pair Plot'):
        if columns is not None:
            g = sns.pairplot(self.df[columns], plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
        else:
            g = sns.pairplot(self.df, plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})

        g.fig.suptitle(title, fontsize=15, fontfamily='serif', color='blue')
        g.fig.subplots_adjust(top=0.95, right=0.95)

        # Set xlabel and ylabel for each plot
        for ax in g.axes.flatten():
            ax.set_xlabel(ax.get_xlabel(), fontsize=12, fontfamily='serif', color='darkred')
            ax.set_ylabel(ax.get_ylabel(), fontsize=12, fontfamily='serif', color='darkred')

        plt.show()

    def plot_heatmap(self, title='Correlation Heatmap'):
        plt.figure(figsize=(14, 12))  # Increase figure size
        features = ['sellingprice', 'mmr', 'odometer', 'condition', 'year']
        numeric_df = self.df[features].dropna()
        if numeric_df.empty:
            print("No numeric data available for heatmap.")
            return

        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=True,
                    xticklabels=corr.columns, yticklabels=corr.columns)

        plt.title(title, fontsize=20, fontfamily='serif', color='blue')
        plt.xlabel('Features', fontsize=15, fontfamily='serif', color='darkred')
        plt.ylabel('Features', fontsize=15, fontfamily='serif', color='darkred')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_hist_kde(self, column, title=None):
        if title is None:
            title = f'Histogram with KDE of {column}'
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column], kde=True, color='skyblue')
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(column, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel('Frequency', fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_qq(self, column, title=None):
        if title is None:
            title = f'QQ Plot of {column}'
        plt.figure(figsize=(10, 6))
        stats.probplot(self.df[column], dist="norm", plot=plt)
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel('Theoretical Quantiles', fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel('Sample Quantiles', fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_kde(self, column, title=None):
        if title is None:
            title = f'KDE Plot of {column}'
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

    def plot_box(self, x, y, hue=None, title='Default Title'):
        plt.figure(figsize=(10, 6))
        if hue:
            sns.boxplot(x=self.df[x], y=self.df[y], hue=self.df[hue], palette="pastel")
        else:
            sns.boxplot(x=self.df[x], y=self.df[y])
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(x, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(y, fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.legend(title=hue) if hue else None
        plt.show()

        # New plot methods for the missing types

    def plot_stacked_bar(self, x, y, title, xlabel, ylabel):
        plt.figure(figsize=(10, 6))
        self.df.groupby([x, y]).size().unstack().plot(kind='bar', stacked=True)
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(xlabel, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(ylabel, fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_area(self, x, y, title):
        plt.figure(figsize=(10, 6))
        self.df.plot.area(x=x, y=y)
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(x, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(y, fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_violin(self, x, y, title):
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=self.df, x=x, y=y)
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(x, fontsize=12, fontfamily='serif', color='darkred')
        plt.ylabel(y, fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_joint_kde_scatter(self, x, y, title):
        g = sns.jointplot(data=self.df, x=x, y=y, kind="kde")
        g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
        g.ax_joint.collections[0].set_alpha(0)
        plt.suptitle(title)
        plt.grid(True)
        plt.show()

    def plot_rug(self, column, title):
        plt.figure(figsize=(10, 6))
        sns.rugplot(data=self.df, x=column)
        plt.title(title, fontsize=15, fontfamily='serif', color='blue')
        plt.xlabel(column, fontsize=12, fontfamily='serif', color='darkred')
        plt.grid(True)
        plt.show()

    def plot_3d_scatter(self, x, y, z, title):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.df[x], self.df[y], self.df[z])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_title(title)
        plt.show()

    def plot_cluster_map(self):
        # Select only numeric columns from DataFrame
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("No numeric data available for cluster map.")
            return

        # Compute the correlation matrix
        corr_matrix = numeric_df.corr()

        # Generate a cluster map
        sns.clustermap(corr_matrix, cmap='coolwarm', annot=True)
        plt.show()

    def plot_hexbin(self, x, y, title, gridsize=40, cmap='inferno'):
        plt.hexbin(self.df[x], self.df[y], gridsize=gridsize, cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.show()

    def plot_strip(self, x, y, title):
        plt.figure(figsize=(12, 8))
        # sns.stripplot(data=self.df, x=x, y=y)
        sns.stripplot(data=self.df, x=x, y=y, size=4)
        plt.title(title)
        plt.show()

    def plot_swarm(self, x, y, title):
        plt.figure(figsize=(10, 6))
        sns.swarmplot(data=self.df, x=x, y=y,size=3)
        plt.title(title)
        plt.show()