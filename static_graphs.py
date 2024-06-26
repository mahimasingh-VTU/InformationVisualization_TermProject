import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_cleaner import DataCleaner
from sklearn.preprocessing import StandardScaler

df_raw = pd.read_csv('car_prices.csv')
df = DataCleaner(df_raw).clean_data()

"""
Instructions:
1. include legend (hue) in your static plot for comparison.
2. All figures must include title, legend, x-label, y-label, and grid and they must be customized with assorted color, font size, line width.
3. Make sure the information on the graph is not blocked.
4. display all numbers with 2-digit decimal precision
5. Title = [font: ‘serif’, color:’blue, size: large enough]
6. X, Y label = [font: ‘serif’, ‘color’, darkred] size large enough

Numerical features: ['year', 'condition', 'odometer', 'mmr', 'sellingprice']
Categorical features: ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior', 'seller', 'saledate', 'salemonth', 'saleday', 'saleyear']
"""

# 1.========= Line Plot: Total Sales Volume Over Time =========
sales_volume = df.groupby('year')['sellingprice'].sum()
plt.figure(figsize=(10, 6))
plt.plot(sales_volume.index, sales_volume.values, color='skyblue', linewidth=2)
plt.title('Total Sales Volume Over Time', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Year', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Total Sales Volume', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.show()

# 2.========= Histogram Plot with KDE: Distribution of Selling Prices =========
plt.figure(figsize=(10, 6))
sns.histplot(df['sellingprice'], kde=True, color='skyblue', bins=30)
plt.title('Distribution of Selling Prices', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Frequency', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()


# 3.========= KDE Plot: Distribution of Odometer Reading =========
plt.figure(figsize=(10, 6))
sns.kdeplot(df['odometer'], color='skyblue', shade=True)
plt.title('Distribution of Odometer Readings', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Odometer Reading', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Density', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()


# 4.========= Bar Plot (Grouped): Average Selling Prices by Make and Condition =========
avg_prices = df.groupby(['make', 'condition'])['sellingprice'].mean().reset_index()
plt.figure(figsize=(15, 8))
sns.barplot(x='make', y='sellingprice', hue='condition', data=avg_prices)
plt.title('Average Selling Prices by Make and Condition', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Make', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Average Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()


# 5.========= Count Plot: Frequency of Different Vehicle Models =========
plt.figure(figsize=(15, 8))
sns.countplot(x='model', data=df, order=df['model'].value_counts().index)
plt.title('Frequency of Different Vehicle Models', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Vehicle Model', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Frequency', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()

# 6.========= Pie Chart: Market Share of Top 10 Vehicle Makes =========
top_10_makes = df['make'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_10_makes.plot(kind='pie', autopct='%1.2f%%')
plt.title('Market Share of Top 10 Vehicle Makes', fontsize=20, fontfamily='serif', color='blue')
plt.tight_layout()
plt.show()


# 7.========= Scatter Plot with Regression Line: Selling Price vs. Odometer Readings =========
plt.figure(figsize=(10, 6))
sns.regplot(x='odometer', y='sellingprice', data=df, color='skyblue', line_kws={'color': 'red'})
plt.title('Selling Price vs. Odometer Readings', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Odometer Reading', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()


# 8.========= Pair Plot: Relationships Between Selling Price, MMR, Odometer, and Year =========
selected_columns = df[['sellingprice', 'mmr', 'odometer', 'year']]
sns.pairplot(selected_columns)
plt.show()


# 9.========= Heatmap with Color Bar: Correlation Between Numerical Variables =========
corr_matrix = df[['year', 'condition', 'odometer', 'mmr', 'sellingprice']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Numerical Variables', fontsize=20, fontfamily='serif', color='blue')
plt.tight_layout()
plt.show()


# 10.========= Box Plot: Selling Price Distribution by Vehicle Condition =========
plt.figure(figsize=(10, 6))
sns.boxplot(x='condition', y='sellingprice', data=df, palette='coolwarm')
plt.title('Selling Price Distribution by Vehicle Condition', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Vehicle Condition', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()


# 11.========= Violin Plot: Selling Price Distribution by Year of Manufacture =========
plt.figure(figsize=(15, 8))
sns.violinplot(x='year', y='sellingprice', data=df, palette='coolwarm')
plt.title('Selling Price Distribution by Year of Manufacture', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Year of Manufacture', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()

# 12.========= Strip Plot: Selling Prices Across Different Body Types =========
plt.figure(figsize=(15, 8))
sns.stripplot(x='body', y='sellingprice', data=df, palette='coolwarm')
plt.title('Selling Prices Across Different Body Types', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Body Type', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
# plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()


# 13.========= Area Plot: Cumulative Sales by Make Over Years =========
cumulative_sales = df.groupby(['make', 'year'])['sellingprice'].sum().groupby('make').cumsum()
cumulative_sales_unstacked = cumulative_sales.unstack(level=0)
plt.figure(figsize=(15, 8))
cumulative_sales_unstacked.plot.area()
plt.title('Cumulative Sales by Make Over Years', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Year', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Cumulative Sales', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()


# 14.========= Stacked Bar Plot: Proportion of Sales by Condition for Each Year =========
pivot_table = df.pivot_table(index='year', columns='condition', values='sellingprice', aggfunc='count', fill_value=0)
proportions = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
proportions.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('Proportion of Sales by Condition for Each Year', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Year', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Proportion', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()


# 15.========= Joint Plot: KDE and Scatter Representation of Selling Price vs. MMR =========
sns.jointplot(x='mmr', y='sellingprice', data=df, kind='scatter', color='skyblue',
              marginal_kws=dict(bins=30, fill=True))
plt.suptitle('Selling Price vs. MMR', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('MMR', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()

# 16.========= Hexbin Plot: Density of Data Points for Selling Price vs. MMR =========
plt.figure(figsize=(10, 6))
plt.hexbin(df['mmr'], df['sellingprice'], gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')
plt.title('Density of Data Points for Selling Price vs. MMR', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('MMR', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()


# 17.========= 3D Scatter Plot: Year, Selling Price, and Odometer Readings =========
fig = px.scatter_3d(df, x='year', y='sellingprice', z='odometer', color='year')
fig.update_layout(
    title={
        'text': 'Year, Selling Price, and Odometer Readings',
        'font': dict(
            family="Serif",
            size=20,
            color='blue'
        )
    },
    scene=dict(
        xaxis_title='Year',
        yaxis_title='Selling Price',
        zaxis_title='Odometer Reading',
        xaxis=dict(
            titlefont=dict(
                family="Serif",
                size=15,
                color="darkred"
            )
        ),
        yaxis=dict(
            titlefont=dict(
                family="Serif",
                size=15,
                color="darkred"
            )
        ),
        zaxis=dict(
            titlefont=dict(
                family="Serif",
                size=15,
                color="darkred"
            )
        )
    )
)
fig.show()


# 18.========= Swarm Plot: Selling Prices by Make for Top 5 Selling Makes =========
top_5_makes = df['make'].value_counts().nlargest(5).index
df_top_5_makes = df[df['make'].isin(top_5_makes)]
plt.figure(figsize=(15, 8))
sns.swarmplot(x='make', y='sellingprice', data=df_top_5_makes)
plt.title('Selling Prices by Make for Top 5 Selling Makes', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Make', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\18_selling_prices_top_5_makes.png')

# 19.========= QQ-Plot: Compare Selling Price Distribution to Normal Distribution =========
import matplotlib.pyplot as plt
from scipy.stats import probplot
plt.figure(figsize=(10, 6))
probplot(df['sellingprice'], plot=plt)
plt.title('QQ-Plot: Compare Selling Price Distribution to Normal Distribution', fontsize=20, fontfamily='serif',
          color='blue')
plt.xlabel('Theoretical Quantiles', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Ordered Values', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\19_qq_plot_selling_price.png')


# 20.========= Dist Plot: Compare Selling Price Distributions for Different Transmission Types =========
transmission_types = df['transmission'].unique()
plt.figure(figsize=(10, 6))
for transmission in transmission_types:
    subset = df[df['transmission'] == transmission]
    sns.distplot(subset['sellingprice'], hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label=transmission)
plt.title('Selling Price Distributions for Different Transmission Types', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Density', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.legend(prop={'size': 10}, title='Transmission Type')
plt.show()
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\20_selling_price_distribution_transmission.png')

# 21.========= Cluster Map: Hierarchical Clustering of Vehicles Based on Numerical Features =========
import seaborn as sns
numerical_features = df[['year', 'condition', 'odometer', 'mmr', 'sellingprice']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)
df_scaled = pd.DataFrame(scaled_features, columns=numerical_features.columns)
sns.clustermap(df_scaled, cmap='coolwarm', standard_scale=1)
plt.show()
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\21_cluster_map.png')


# 22.========= Contour Plot: Density of Sales in the Space of Year and Selling Price =========
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='year', y='sellingprice', fill=True)
plt.title('Density of Sales in the Space of Year and Selling Price', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Year', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\22_density_sales_year_sellingprice.png')


# 23.========= Rug Plot: Distribution of Individual Data Points for MMR Values =========
plt.figure(figsize=(10, 6))
sns.rugplot(data=df, x='mmr', height=0.5)
plt.title('Distribution of Individual Data Points for MMR Values', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('MMR', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\23_distribution_mmr_values.png')


# 24.========= Boxen Plot: Selling Prices Across Different States (Top 10 States by Sales Volume) =========
sales_volume = df.groupby('state')['sellingprice'].sum()
top_10_states = sales_volume.nlargest(10).index
df_top_10_states = df[df['state'].isin(top_10_states)]
plt.figure(figsize=(15, 8))
sns.boxenplot(x='state', y='sellingprice', data=df_top_10_states, palette='coolwarm')
plt.title('Selling Prices Across Different States (Top 10 States by Sales Volume)', fontsize=20, fontfamily='serif',
          color='blue')
plt.xlabel('State', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\24_selling_prices_states.png')

# SUBPLOTS

# 25.========= Price Trends and Factors  =========
"""
    Line plot: Average selling price trend over years 

    Scatter plot: Selling price vs. Odometer reading 

    Box plot: Selling price by vehicle condition 

    Bar plot: Average selling price by top 10 makes 
"""

avg_price_trend = df.groupby('year')['sellingprice'].mean()
avg_price_makes = df.groupby('make')['sellingprice'].mean().nlargest(10)

# Create a new figure
fig, axs = plt.subplots(2, 2, figsize=(20, 15))

# Line plot: Average selling price trend over years
axs[0, 0].plot(avg_price_trend.index, avg_price_trend.values, color='skyblue', linewidth=2)
axs[0, 0].set_title('Average Selling Price Trend Over Years', fontsize=15, fontfamily='serif', color='blue')
axs[0, 0].set_xlabel('Year', fontsize=12, fontfamily='serif', color='darkred')
axs[0, 0].set_ylabel('Average Selling Price', fontsize=12, fontfamily='serif', color='darkred')
axs[0, 0].grid(True)

# Scatter plot: Selling price vs. Odometer reading
axs[0, 1].scatter(df['odometer'], df['sellingprice'], color='skyblue')
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\25_price_trends_factors.png')


# 26.========= Sales Volume Analysis =========
"""
    Line plot: Total sales volume over years 

    Stacked bar plot: Sales volume by vehicle condition for each year 

    Pie chart: Market share of top 5 makes 

    Bar plot: Sales volume by month (to show seasonality) 
"""
# Calculate total sales volume over years
sales_volume_years = df.groupby('year')['sellingprice'].sum()

# Plot line plot of total sales volume over years
plt.figure(figsize=(10, 6))
plt.plot(sales_volume_years.index, sales_volume_years.values, color='skyblue', linewidth=2)
plt.title('Total Sales Volume Over Years', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Year', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Total Sales Volume', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.show()

# Calculate sales volume by vehicle condition for each year
sales_volume_condition_year = df.groupby(['year', 'condition'])['sellingprice'].sum().unstack()

# Plot stacked bar plot of sales volume by vehicle condition for each year
sales_volume_condition_year.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sales Volume by Vehicle Condition for Each Year', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Year', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Sales Volume', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.show()

# Calculate market share of top 5 makes
top_5_makes = df['make'].value_counts().nlargest(5)

# Plot pie chart of market share of top 5 makes
top_5_makes.plot(kind='pie', autopct='%1.2f%%', figsize=(10, 6))
plt.title('Market Share of Top 5 Makes', fontsize=20, fontfamily='serif', color='blue')
plt.show()

# Calculate sales volume by month
sales_volume_month = df.groupby(df['saledate'].dt.month)['sellingprice'].sum()

# Plot bar plot of sales volume by month
sales_volume_month.plot(kind='bar', figsize=(10, 6))
plt.title('Sales Volume by Month', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Month', fontsize=15, fontfamily='serif', color='darkred')
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\26_sales_volume_analysis.png')

# 27.========= Vehicle Characteristics and Their Impact =========
"""
    Scatter plot: Year vs. Odometer reading, colored by selling price 

    Histogram: Distribution of vehicle ages in the dataset 

    Bar plot: Average MMR by vehicle condition 

    Heatmap: Correlation between numerical features (year, odometer, selling price, MMR) 
"""
# Calculate vehicle age
df['age'] = 2024 - df['year']  # Assuming the current year is 2024

# Create a new figure for subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 15))

# Scatter plot: Year vs. Odometer reading, colored by selling price
sns.scatterplot(x='year', y='odometer', hue='sellingprice', data=df, ax=axs[0, 0])
axs[0, 0].set_title('Year vs. Odometer Reading', fontsize=15, fontfamily='serif', color='blue')
axs[0, 0].set_xlabel('Year', fontsize=12, fontfamily='serif', color='darkred')
axs[0, 0].set_ylabel('Odometer Reading', fontsize=12, fontfamily='serif', color='darkred')
axs[0, 0].grid(True)

# Histogram: Distribution of vehicle ages in the dataset
sns.histplot(df['age'], kde=False, ax=axs[0, 1])
axs[0, 1].set_title('Distribution of Vehicle Ages', fontsize=15, fontfamily='serif', color='blue')
axs[0, 1].set_xlabel('Age', fontsize=12, fontfamily='serif', color='darkred')
axs[0, 1].set_ylabel('Frequency', fontsize=12, fontfamily='serif', color='darkred')
axs[0, 1].grid(True)

# Bar plot: Average MMR by vehicle condition
avg_mmr_condition = df.groupby('condition')['mmr'].mean()
sns.barplot(x=avg_mmr_condition.index, y=avg_mmr_condition.values, ax=axs[1, 0])
axs[1, 0].set_title('Average MMR by Vehicle Condition', fontsize=15, fontfamily='serif', color='blue')
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\27_vehicle_characteristics_impact.png')


# 28.========= Regional and Categorical Analysis =========
"""
    Choropleth map: Sales volume by state 

    Bar plot: Top 10 states by average selling price 

    Stacked bar plot: Proportion of transmission types for top 5 makes 

    Violin plot: Distribution of selling prices for different body types 
"""
# Choropleth map: Sales volume by state
sales_volume_state = df.groupby('state')['sellingprice'].sum().reset_index()
fig = go.Figure(data=go.Choropleth(
    locations=sales_volume_state['state'],
    z=sales_volume_state['sellingprice'].astype(float),
    locationmode='USA-states',
    colorscale='Reds',
))
fig.update_layout(
    title_text='Sales Volume by State',
    geo_scope='usa',
)
fig.show()

# Bar plot: Top 10 states by average selling price
avg_price_state = df.groupby('state')['sellingprice'].mean().nlargest(10).reset_index()
fig = px.bar(avg_price_state, x='state', y='sellingprice', title='Top 10 States by Average Selling Price')
fig.show()

# Stacked bar plot: Proportion of transmission types for top 5 makes
top_5_makes = df['make'].value_counts().nlargest(5).index
df_top_5_makes = df[df['make'].isin(top_5_makes)]
transmission_proportions = df_top_5_makes.groupby(['make', 'transmission']).size().unstack().apply(lambda x: x/x.sum(), axis=1)
transmission_proportions.plot(kind='bar', stacked=True, title='Proportion of Transmission Types for Top 5 Makes')

# Violin plot: Distribution of selling prices for different body types
fig = px.violin(df, x='body', y='sellingprice', box=True, title='Distribution of Selling Prices for Different Body Types')
fig.show()
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\28_regional_categorical_analysis.png')