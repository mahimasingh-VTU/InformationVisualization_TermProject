import pandas as pd
from data_explorer import DataExplorer
from data_cleaner import DataCleaner
from data_preprocessing import DataProcessor
from plotter import Plotter
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from dash_app import create_dash_app

pd.set_option('display.float_format', '{:.2f}'.format)


def print_pretty_table(title, data):
    table = PrettyTable()
    table.field_names = data.columns.tolist()
    for index, row in data.iterrows():
        table.add_row(row)
    print(table.get_string(title=title))


def plot_initial_histograms(dataframe):
    columns_to_plot = ['sellingprice', 'odometer', 'condition', 'year']  # Add or remove columns as needed
    for column in columns_to_plot:
        if column in dataframe.columns:
            plt.figure(figsize=(10, 6))
            dataframe[column].hist(bins=20, color='gray', alpha=0.7)
            plt.title(f'Initial Distribution of {column.capitalize()}')
            plt.xlabel(column.capitalize())
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()


if __name__ == "__main__":
    # Load the data
    file_path = 'car_prices.csv'
    df = pd.read_csv(file_path)
    # Print basic data info
    # print("Basic Data Info")
    # table_info = PrettyTable()
    # table_info.field_names = ["Description", "Value"]
    # table_info.add_row(["Shape", df.shape])
    # table_info.add_row(["Columns", list(df.columns)])
    # print(table_info)
    #
    # print("Head of the DataFrame")
    # print_pretty_table("Head of the DataFrame", df.head())
    #
    # print("Descriptive Statistics (Numeric Columns)")
    # print_pretty_table("Descriptive Statistics (Numeric Columns)", df.describe().T.reset_index())
    #
    # print("Descriptive Statistics (Non-Numeric Columns)")
    # print_pretty_table("Descriptive Statistics (Non-Numeric Columns)", df.describe(exclude='number').T.reset_index())
    #
    # print("Missing Values")
    # missing_values = (df.isnull().sum() / df.count() * 100).reset_index()
    # missing_values.columns = ["Column", "Missing Values (%)"]
    # print_pretty_table("Missing Values", missing_values)
    # # Call the function to plot histograms
    # plot_initial_histograms(df)
    # cleaner = DataCleaner(df)
    # df_clean = cleaner.clean_data()
    # print(df_clean.info())
    # print_pretty_table("Head of Cleaned DataFrame", df_clean.head())
    # # PCA
    # processor = DataProcessor(df_clean)
    # principal_components, n_components_95, variance_explained = processor.perform_pca()
    #
    # print(f"Number of components to explain 95% of the variance: {n_components_95}")
    # print(f"Cumulative explained variance by {n_components_95} components: {variance_explained * 100:.2f}%")
    #
    # # Perform normality test on PCA results
    # processor.normality_test(principal_components)
    #
    # # Perform Ksquare normality test on PCA results
    # processor.normality_test(principal_components)
    # plotter = Plotter(df_clean)
    # # Data exploration
    # explorer = DataExplorer(df_clean)
    # explorer.explore_data()
    # explorer.plot_histograms()
    #
    # # Generate plots
    #
    # plotter.plot_bar_vis('make', 'sellingprice', 'Brands vs Selling Price by Transmission', 'Brands', 'Selling Price')
    # plotter.plot_scatter('sellingprice', 'mmr', 'Selling Price vs MMR', 'Selling Price', 'MMR')
    # plotter.plot_heatmap()
    # plotter.plot_line('year', 'sellingprice', 'Manufacturing Year vs Selling Price', 'Manufacturing Year',
    #                   'Selling Price')
    # # plotter.plot_box('year', 'Distribution of Manufacturing Years')
    # plotter.plot_histogram('mmr', 'Manheim Market Report (MMR) Distribution')
    # plotter.plot_count('state', 'Distribution by State', 'State', 'Count')
    # plotter.plot_reg('sellingprice', 'mmr', 'Regression: Selling Price vs MMR', 'Selling Price', 'MMR')
    #
    # plotter.plot_histogram('condition', 'Vehicle Condition Distribution')
    #
    # plotter.plot_pie('make', 'Distribution of Vehicle Makes')
    # plotter.plot_count('model', 'Count of Vehicle Models', 'Model', 'Count')
    # plotter.subplot_story()
    #
    # # Additional plots
    # plotter.plot_line('year', 'sellingprice', 'Average Selling Price by Year', 'Year', 'Average Selling Price')
    # plotter.plot_dist('sellingprice', 'Selling Price Distribution')
    # plotter.plot_pair(columns=['sellingprice', 'mmr', 'odometer'],
    #                   title='Pair Plot of Selling Price, MMR, and Odometer')
    # plotter.plot_heatmap('Correlation Heatmap')
    # plotter.plot_hist_kde('odometer', 'Odometer Distribution with KDE')
    # plotter.plot_qq('sellingprice', 'QQ Plot of Selling Price')
    # plotter.plot_kde('mmr', 'KDE Plot of MMR')
    # plotter.plot_im_reg('sellingprice', 'odometer', 'Selling Price vs Odometer with Regression Line')
    # plotter.plot_box('condition', 'sellingprice', None, 'Box Plot of Selling Price by Condition')
    #
    # # New plot calls for the additional plots
    # plotter.plot_stacked_bar('year', 'condition', 'Vehicle Sales by Condition and Year', 'Year', 'Count')
    # plotter.plot_area('year', 'sellingprice', 'Area Plot of Selling Prices Over Years')
    # plotter.plot_violin('year', 'sellingprice', 'Violin Plot of Selling Prices by Year')
    # plotter.plot_joint_kde_scatter('sellingprice', 'mmr', 'Joint KDE and Scatter Plot of Selling Price vs MMR')
    # plotter.plot_rug('sellingprice', 'Rug Plot of Selling Prices')
    # plotter.plot_3d_scatter('year', 'sellingprice', 'mmr', '3D Scatter Plot of Year, Selling Price and MMR')
    # plotter.plot_cluster_map()
    # plotter.plot_hexbin('sellingprice', 'mmr', 'Hexbin Plot of Selling Price vs MMR')
    # plotter.plot_strip('condition', 'sellingprice', 'Strip Plot of Selling Price by Condition')
    # plotter.plot_swarm('year', 'sellingprice', 'Swarm Plot of Selling Prices by Year')

    #Create the Dash app
    my_app = create_dash_app()
    my_app.run_server(port=8050, host='0.0.0.0')
