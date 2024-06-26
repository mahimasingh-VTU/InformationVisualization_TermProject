import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
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
plt.show()


# 3.========= KDE Plot: Distribution of Odometer Reading =========
plt.figure(figsize=(10, 6))
sns.kdeplot(df['odometer'], color='skyblue', shade=True)
plt.title('Distribution of Odometer Readings', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Odometer Reading', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Density', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.show()

# TODO - change make to body
# 4.========= Bar Plot (Grouped): Average Selling Prices by Body and Condition =========
avg_prices = df.groupby(['make', 'condition'])['sellingprice'].mean().reset_index()
plt.figure(figsize=(15, 8))
sns.barplot(x='make', y='sellingprice', hue='condition', data=avg_prices)
plt.title('Average Selling Prices by Make and Condition', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Make', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Average Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.show()

# TODO - change models to body clean body first
# 5.========= Count Plot: Frequency of Different Vehicle Models =========
plt.figure(figsize=(15, 8))
sns.countplot(x='model', data=df, order=df['model'].value_counts().index)
plt.title('Frequency of Different Vehicle Models', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Vehicle Model', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Frequency', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
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
plt.show()


# 8.========= Pair Plot: Relationships Between Selling Price, MMR, Odometer, and Year =========
selected_columns = df[['sellingprice', 'mmr', 'odometer', 'year']]
sns.pairplot(selected_columns)
plt.subplots_adjust(top=0.9)
plt.suptitle('Relationships Between Selling Price, MMR, Odometer, and Year', fontsize=20, fontfamily='serif', color='blue')
plt.show()


# 9.========= Heatmap with Color Bar: Correlation Between Numerical Variables =========
corr_matrix = df[['year', 'condition', 'odometer', 'mmr', 'sellingprice']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Numerical Variables', fontsize=20, fontfamily='serif', color='blue')
plt.tight_layout()
plt.show()


# # 10.========= Box Plot: Selling Price Distribution by Vehicle Condition =========
plt.figure(figsize=(10, 6))
sns.boxplot(x='condition', y='sellingprice', data=df, palette='coolwarm')
plt.title('Selling Price Distribution by Vehicle Condition', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Vehicle Condition', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.show()


# 11.========= Violin Plot: Selling Price Distribution by Year of Manufacture =========
plt.figure(figsize=(15, 8))
sns.violinplot(x='year', y='sellingprice', data=df, palette='coolwarm')
plt.title('Selling Price Distribution by Year of Manufacture', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Year of Manufacture', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.show()

# 12.========= Strip Plot: Selling Prices Across Different Body Types =========
plt.figure(figsize=(15, 8))
sns.stripplot(x='body', y='sellingprice', data=df, palette='coolwarm')
plt.title('Selling Prices Across Different Body Types', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Body Type', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.show()


# 13.========= Area Plot: Stacked Sales Volume by Top 5 Makes Over Time =========
sales_volume = df.groupby(['make', 'year'])['sellingprice'].sum()
sales_volume_df = sales_volume.unstack(level=0)
total_sales_volume = sales_volume_df.sum()
top_5_makes = total_sales_volume.nlargest(5).index
sales_volume_top_5 = sales_volume_df[top_5_makes]
plt.figure(figsize=(10, 6))
sales_volume_top_5.plot.area(stacked=True)
plt.title('Stacked Sales Volume by Top 5 Makes Over Time', fontsize=15, fontfamily='serif', color='blue')
plt.xlabel('Year', fontsize=12, fontfamily='serif', color='darkred')
plt.ylabel('Sales Volume', fontsize=12, fontfamily='serif', color='darkred')
plt.grid(True)
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
plt.show()


# 15.========= Joint Plot: KDE and Scatter Representation of Selling Price vs. MMR =========
sns.jointplot(x='mmr', y='sellingprice', data=df, kind='scatter', color='skyblue',
              marginal_kws=dict(bins=30, fill=True))
plt.suptitle('Selling Price vs. MMR', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('MMR', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.show()

# 16.========= Hexbin Plot: Density of Data Points for Selling Price vs. MMR =========
plt.figure(figsize=(10, 6))
plt.hexbin(df['mmr'], df['sellingprice'], gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')
plt.title('Density of Data Points for Selling Price vs. MMR', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('MMR', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.xlim(0, df['mmr'].max()/2)
plt.ylim(0, df['sellingprice'].max()/2)
plt.tight_layout()
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

# TODO - change
# 18.========= Swarm Plot: Selling Prices by Make for Top 5 Selling Makes =========
import seaborn as sns
import matplotlib.pyplot as plt

# Create the swarm plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='condition', y='sellingprice', data=df, size=3)

# Set the title and labels
plt.title('Selling Prices by Vehicle Condition', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('Vehicle Condition', fontsize=15, fontfamily='serif', color='darkred')
plt.ylabel('Selling Price', fontsize=15, fontfamily='serif', color='darkred')

# Display the grid and tighten the layout
plt.grid(True)
plt.tight_layout()
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
plt.legend(prop={'size': 10}, title='Transmission Type')
plt.show()
plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\20_selling_price_distribution_transmission.png')

# TODO - change
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

# TODO - change
# 23.========= Rug Plot: Distribution of Individual Data Points for MMR Values =========
plt.figure(figsize=(10, 6))
sns.rugplot(data=df, x='mmr', height=0.5)
plt.title('Distribution of Individual Data Points for MMR Values', fontsize=20, fontfamily='serif', color='blue')
plt.xlabel('MMR', fontsize=15, fontfamily='serif', color='darkred')
plt.grid(True)
plt.tight_layout()
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))
plt.show()
#plt.savefig('C:\\Github\\InformationVisualization_TermProject\\staticgraphs\\23_distribution_mmr_values.png')

# TODO - change
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

# ---------------------------SUBPLOTS-----------------------------------

# 25.========= Pricing Dynamics and Market Value  =========
"""

    Line plot: Average selling price vs. average MMR over time
    Scatter plot: Selling price vs. MMR, colored by vehicle condition
    Box plot: Selling price distribution by make (top 10 makes)
    Bar plot: Average price difference (Selling price - MMR) by vehicle age group
"""

avg_price_mmr_year = df.groupby('year')[['sellingprice', 'mmr']].mean()
top_10_makes = df.groupby('make')['sellingprice'].mean().nlargest(10)
df['age'] = 2024 - df['year']  # Assuming the current year is 2024
avg_price_diff_age = df.groupby('age').apply(lambda x: (x['sellingprice'] - x['mmr']).mean())
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

# Plot line plot of average selling price vs. average MMR over time
axs[0, 0].plot(avg_price_mmr_year.index, avg_price_mmr_year['sellingprice'], label='Selling Price')
axs[0, 0].plot(avg_price_mmr_year.index, avg_price_mmr_year['mmr'], label='MMR')
axs[0, 0].set_title('Average Selling Price vs. Average MMR Over Time')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Price')
axs[0, 0].legend()

# Scatter plot: Selling price vs. MMR, colored by vehicle condition
sns.scatterplot(x='mmr', y='sellingprice', hue='condition', data=df, ax=axs[0, 1])
axs[0, 1].set_title('Selling Price vs. MMR')
axs[0, 1].set_xlabel('MMR')
axs[0, 1].set_ylabel('Selling Price')

# Box plot: Selling price distribution by make (top 10 makes)
sns.boxplot(x='make', y='sellingprice', data=df[df['make'].isin(top_10_makes.index)], ax=axs[1, 0])
axs[1, 0].set_title('Selling Price Distribution by Make (Top 10 Makes)')
axs[1, 0].set_xlabel('Make')
axs[1, 0].set_ylabel('Selling Price')

# Bar plot: Average price difference (Selling price - MMR) by vehicle age group
sns.barplot(x=avg_price_diff_age.index, y=avg_price_diff_age.values, ax=axs[1, 1])
axs[1, 1].set_title('Average Price Difference (Selling Price - MMR) by Vehicle Age Group')
axs[1, 1].set_xlabel('Age Group')
axs[1, 1].set_ylabel('Average Price Difference')

plt.tight_layout()
plt.show()



# 26.========= Sales Volume Analysis =========
"""
    Line plot: Total sales volume over years

    Stacked bar plot: Sales volume by vehicle condition for each year

    Pie chart: Market share of top 5 makes

    Bar plot: Sales volume by month (to show seasonality)
"""

sales_volume_years = df.groupby('year')['sellingprice'].sum()

sales_volume_condition_year = df.groupby(['year', 'condition'])['sellingprice'].sum().unstack()

top_5_makes = df['make'].value_counts().nlargest(5)

sales_volume_month = df.groupby('salemonth')['sellingprice'].sum()

fig, axs = plt.subplots(2, 2, figsize=(20, 20))

# Plot line plot of total sales volume over years
axs[0, 0].plot(sales_volume_years.index, sales_volume_years.values, color='skyblue', linewidth=2)
axs[0, 0].set_title('Total Sales Volume Over Years')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Total Sales Volume')

# Plot stacked bar plot of sales volume by vehicle condition for each year
sales_volume_condition_year.plot(kind='bar', stacked=True, ax=axs[0, 1])
axs[0, 1].set_title('Sales Volume by Vehicle Condition for Each Year')
axs[0, 1].set_xlabel('Year')
axs[0, 1].set_ylabel('Sales Volume')

# Plot pie chart of market share of top 5 makes
axs[1, 0].pie(top_5_makes, labels=top_5_makes.index, autopct='%1.1f%%')
axs[1, 0].set_title('Market Share of Top 5 Makes')

# Plot bar plot of sales volume by month
axs[1, 1].bar(sales_volume_month.index, sales_volume_month.values, color='skyblue')
axs[1, 1].set_title('Sales Volume by Month')
axs[1, 1].set_xlabel('Month')
axs[1, 1].set_ylabel('Sales Volume')

plt.tight_layout()
plt.show()



# 27.========= Vehicle Characteristics and Their Impact =========
"""
    Scatter plot: Year vs. Odometer reading, colored by selling price

    Histogram: Distribution of vehicle ages in the dataset

    Bar plot: Average MMR by vehicle condition

    Heatmap: Correlation between numerical features (year, odometer, selling price, MMR)
"""

df['age'] = 2024 - df['year']  # Assuming the current year is 2024

fig, axs = plt.subplots(2, 2, figsize=(20, 15))

# Scatter plot: Year vs. Odometer reading, colored by selling price
sns.scatterplot(x='year', y='odometer', hue='sellingprice', data=df, ax=axs[0, 0])
axs[0, 0].set_title('Year vs. Odometer Reading')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Odometer Reading')

# Histogram: Distribution of vehicle ages in the dataset
sns.histplot(df['age'], kde=False, ax=axs[0, 1])
axs[0, 1].set_title('Distribution of Vehicle Ages')
axs[0, 1].set_xlabel('Age')
axs[0, 1].set_ylabel('Frequency')

# Bar plot: Average MMR by vehicle condition
avg_mmr_condition = df.groupby('condition')['mmr'].mean()
sns.barplot(x=avg_mmr_condition.index, y=avg_mmr_condition.values, ax=axs[1, 0])
axs[1, 0].set_title('Average MMR by Vehicle Condition')
axs[1, 0].set_xlabel('Condition')
axs[1, 0].set_ylabel('Average MMR')

# Heatmap: Correlation between numerical features (year, odometer, selling price, MMR)
numerical_features = df[['year', 'odometer', 'sellingprice', 'mmr']]
correlation = numerical_features.corr()
sns.heatmap(correlation, annot=True, ax=axs[1, 1])
axs[1, 1].set_title('Correlation Between Numerical Features')

plt.tight_layout()
plt.show()


# 28.========= Regional Analysis and Seasonal Trends =========
"""
1. Bar plot: Top 10 states by average selling price

2. Bar plot: Top 10 states by sales volume

3. Heatmap: Monthly sales volume by year

4. Line plot: Average selling price trend for top 5 states over time
"""
avg_price_states = df.groupby('state')['sellingprice'].mean().nlargest(10)
sales_volume_states = df.groupby('state')['sellingprice'].sum().nlargest(10)
sales_volume_month_year = df.groupby(['saleyear', 'salemonth'])['sellingprice'].sum().unstack()
top_5_states = df['state'].value_counts().nlargest(5).index
avg_price_trend_top_5_states = df[df['state'].isin(top_5_states)].groupby(['saleyear', 'state'])['sellingprice'].mean().unstack()
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
sns.barplot(x=avg_price_states.index, y=avg_price_states.values, ax=axs[0, 0])
axs[0, 0].set_title('Top 10 States by Average Selling Price')
axs[0, 0].set_xlabel('State')
axs[0, 0].set_ylabel('Average Selling Price')
sns.barplot(x=sales_volume_states.index, y=sales_volume_states.values, ax=axs[0, 1])
axs[0, 1].set_title('Top 10 States by Sales Volume')
axs[0, 1].set_xlabel('State')
axs[0, 1].set_ylabel('Sales Volume')
sns.heatmap(sales_volume_month_year, annot=True, ax=axs[1, 0])
axs[1, 0].set_title('Monthly Sales Volume by Year')
axs[1, 0].set_xlabel('Month')
axs[1, 0].set_ylabel('Year')
for state in top_5_states:
    axs[1, 1].plot(avg_price_trend_top_5_states.index, avg_price_trend_top_5_states[state], label=state)
axs[1, 1].set_title('Average Selling Price Trend for Top 5 States Over Time')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Average Selling Price')
axs[1, 1].legend()
plt.tight_layout()
plt.show()