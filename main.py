import pandas as pd
from data_explorer import DataExplorer
from data_cleaner import DataCleaner
from plotter import Plotter
from dash_app import create_dash_app

if __name__ == "__main__":

    # Load the data
    df = pd.read_csv('car_prices.csv')
    # # Create instance of the DataExplorer
    # explorer = DataExplorer(df)
    #
    # # Explore the data
    # explorer.explore_data()
    #
    # # Apply the data cleaning process
    # cleaner = DataCleaner(df)
    # cleaned_df = cleaner.clean_data()
    #
    # # Verify the cleaning process
    # print(cleaned_df.info())
    # print(cleaned_df.head())
    #
    # # Create instance of the Plotter with cleaned data
    # plotter = Plotter(cleaned_df)
    #
    # # Generate plots
    # plotter.plot_bar('year', 'make', 'Vehicle Sales by Year', 'Year', 'Sales Volume')
    # plotter.plot_histogram('condition', 'Vehicle Condition Distribution')
    # plotter.plot_scatter('sellingprice', 'mmr', 'Selling Price vs MMR', 'Selling Price', 'MMR')
    # plotter.plot_pie('make', 'Distribution of Vehicle Makes')
    # plotter.plot_count('model', 'Count of Vehicle Models', 'Model', 'Count')
    # plotter.subplot_story()
    #
    # # Additional plots
    # plotter.plot_line('year', 'sellingprice', 'Average Selling Price by Year', 'Year', 'Average Selling Price')
    # plotter.plot_dist('sellingprice', 'Selling Price Distribution')
    # plotter.plot_pair(['sellingprice', 'mmr', 'odometer'], 'Pair Plot of Selling Price, MMR, and Odometer')
    # plotter.plot_heatmap('Correlation Heatmap')
    # plotter.plot_hist_kde('odometer', 'Odometer Distribution with KDE')
    # plotter.plot_qq('sellingprice', 'QQ Plot of Selling Price')
    # plotter.plot_kde('mmr', 'KDE Plot of MMR')
    # plotter.plot_im_reg('sellingprice', 'odometer', 'Selling Price vs Odometer with Regression Line')
    # plotter.plot_box('condition', 'sellingprice', 'Box Plot of Selling Price by Condition')

    # Create the Dash app
    my_app = create_dash_app()
    my_app.run_server(port=8050, host='0.0.0.0')