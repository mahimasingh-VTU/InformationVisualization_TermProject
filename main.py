import pandas as pd
from data_explorer import DataExplorer
from data_cleaner import DataCleaner
from plotter import Plotter

if __name__ == "__main__":

    # Load the data
    file_path = 'car_prices.csv'
    df = pd.read_csv(file_path)

    # Data cleaning
    cleaner = DataCleaner(df)
    df_clean = cleaner.clean_data()
    # Verify the cleaning process
    print(df_clean.info())
    print(df_clean.head())

    # Create instance of the Plotter with cleaned data
    plotter = Plotter(df_clean)
    # Data exploration
    explorer = DataExplorer(df_clean)
    explorer.explore_data()
    explorer.plot_histograms()

    # Generate plots
    plotter.plot_bar('year', 'make', 'Vehicle Sales by Year', 'Year', 'Sales Volume')
    plotter.plot_histogram('condition', 'Vehicle Condition Distribution')
    plotter.plot_scatter('sellingprice', 'mmr', 'Selling Price vs MMR', 'Selling Price', 'MMR')
    plotter.plot_pie('make', 'Distribution of Vehicle Makes')
    plotter.plot_count('model', 'Count of Vehicle Models', 'Model', 'Count')
    plotter.subplot_story()

    # Additional plots
    plotter.plot_line('year', 'sellingprice', 'Average Selling Price by Year', 'Year', 'Average Selling Price')
    plotter.plot_dist('sellingprice', 'Selling Price Distribution')
    plotter.plot_pair(columns=['sellingprice', 'mmr', 'odometer'], title='Pair Plot of Selling Price, MMR, and Odometer')
    plotter.plot_heatmap('Correlation Heatmap')
    plotter.plot_hist_kde('odometer', 'Odometer Distribution with KDE')
    plotter.plot_qq('sellingprice', 'QQ Plot of Selling Price')
    plotter.plot_kde('mmr', 'KDE Plot of MMR')
    plotter.plot_im_reg('sellingprice', 'odometer', 'Selling Price vs Odometer with Regression Line')
    plotter.plot_box('condition', 'sellingprice', None, 'Box Plot of Selling Price by Condition')


    # New plot calls for the additional plots
    plotter.plot_stacked_bar('year', 'condition', 'Vehicle Sales by Condition and Year', 'Year', 'Count')
    plotter.plot_area('year', 'sellingprice', 'Area Plot of Selling Prices Over Years')
    plotter.plot_violin('year', 'sellingprice', 'Violin Plot of Selling Prices by Year')
    plotter.plot_joint_kde_scatter('sellingprice', 'mmr', 'Joint KDE and Scatter Plot of Selling Price vs MMR')
    plotter.plot_rug('sellingprice', 'Rug Plot of Selling Prices')
    plotter.plot_3d_scatter('year', 'sellingprice', 'mmr', '3D Scatter Plot of Year, Selling Price and MMR')
    plotter.plot_cluster_map()
    plotter.plot_hexbin('sellingprice', 'mmr', 'Hexbin Plot of Selling Price vs MMR')
    plotter.plot_strip('condition', 'sellingprice', 'Strip Plot of Selling Price by Condition')
    plotter.plot_swarm('year', 'sellingprice', 'Swarm Plot of Selling Prices by Year')