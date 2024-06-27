import pandas as pd
from data_explorer import DataExplorer
from data_cleaner import DataCleaner
from data_preprocessing import DataProcessor
from plotter import Plotter
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np
from dash_app import create_dash_app
import urllib
import dash as dash
import pandas as pd
import plotly.express as px
from dash import html, dcc
from dash.dependencies import Input, Output, State
from plotly import graph_objs as go
from data_cleaner import DataCleaner
import base64
import os

pd.set_option('display.float_format', '{:.2f}'.format)

def plot_box_plots(df, title):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = len(numeric_columns)
    num_rows = (num_cols + 2) // 3  # Create a grid with 3 columns

    fig, axes = plt.subplots(num_rows, 3, figsize=(18, 5 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        df.boxplot(column=column, ax=ax)
        ax.set_title(f'Box Plot of {column.capitalize()} ({title})')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
def print_pretty_table(title, data):
    table = PrettyTable()
    table.field_names = data.columns.tolist()
    for index, row in data.iterrows():
        table.add_row(row)
    print(table.get_string(title=title))
def print_dataframe_in_one_line(data):
    rows = data.apply(lambda row: ', '.join(row.astype(str)), axis=1).tolist()
    for row in rows:
        print(row)
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
# Function to encode image file to base64
def encode_image(image_file):
    with open(image_file, 'rb') as file:
        return base64.b64encode(file.read()).decode('ascii')

# Assuming 'logo.png' is in the same directory as your script
logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
encoded_logo = encode_image(logo_path)

df_raw = pd.read_csv('car_prices.csv')
df = DataCleaner(df_raw).clean_data()
print(df.info())
print(df.head())
"""
Numerical features: ['year', 'condition', 'odometer', 'mmr', 'sellingprice']
Categorical features: ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior', 'seller', 'saledate', 'salemonth', 'saleday', 'saleyear']
"""


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash("Car Sales Dashboard", external_stylesheets=external_stylesheets,
                   suppress_callback_exceptions=True)
server = my_app.server
# =============================creating tabs================================
my_app.layout = html.Div([
    html.Div([
        html.Img(src=f'data:image/png;base64,{encoded_logo}', style={'height': '60px', 'width': '60px'}),
        html.Pre("  "),
        html.H2('Car Sales Dashboard', style={'textAlign': 'center'}),
    ], style={'display': 'flex', 'align-items': 'center'}),
    html.Br(),
    dcc.Tabs(
        id='tabs',
        value='overview',
        children=[
            dcc.Tab(label='Overview', value='overview'),
            dcc.Tab(label='Market Trends', value='market_trends'),
            dcc.Tab(label='Make and Model Insights', value='insights'),
            dcc.Tab(label='Static graphs',value='static-graphs'),
            dcc.Tab(label='User Feedback', value='feedback')
        ]
    ),
    html.Div(id='layout'),
    html.Button("Download Data", id="download-button"),
    dcc.Download(id="download-data")
])

# =============================Overview tab layout================================
overview_layout = html.Div([
    html.Header('Overview', style={'textAlign': 'center'}),
    html.Br(),
    html.Div([
        # 1. _______________Revenue by state choropleth map_______________
        dcc.Loading(
            id="ov-loading-1",
            type="circle",
            children=dcc.Graph(id='ov-revenue-by-state-graph')
        ),
    ]),
    html.Br(),
    html.Div([
        # 2. _______________Top 10 selling vehicles by year - drop_______________
        dcc.Loading(
            id="ov-loading-2",
            type="circle",
            children=dcc.Graph(id='ov-top-10-selling-vehicles')
        ),
        html.Br(),
        html.Label('Select the year:', htmlFor='ov-year-dropdown'),
        dcc.Dropdown(
            id='ov-year-dropdown',
            options=[{'label': year, 'value': year} for year in df['year'].unique()],
            value=df['year'].min()
        )
    ]),
    html.Br(),
    html.Div([
        # 3. _______________Treemap of Sales Volume by State_______________
        dcc.Loading(
            id="ov-loading-3",
            type="circle",
            children=[dcc.Graph(id='ov-sales-volume-by-state-graph')]

        )
    ]),
    html.Br(),
    html.Div([
        # 4. _______________Pie chart of Top 10 makes by Sales volume_______________
        dcc.Loading(
            id="ov-loading-4",
            type="circle",
            children=dcc.Graph(id='ov-top-10-makes-pie-graph')
        ),
        html.Br(),
        html.P('Select the states:'),
        dcc.Checklist(
            id='ov-state-checklist',
            options=[{'label': state, 'value': state} for state in df['state'].unique()],
            value=df['state'].unique()[:5]
        ),
    ]),
    html.Br(),
    html.Div([
        # 5. _______________Total Sales by body type_______________
        dcc.Loading(
            id="ov-loading-5",
            type="circle",
            children=dcc.Graph(id='ov-total-sales-by-bodytype-graph')
        ),
    ]),
])

# 1. _______________Revenue by state choropleth map_______________
@my_app.callback(
    Output('ov-revenue-by-state-graph', 'figure'),
    [Input('tabs', 'value')]
)
def update_revenue_by_state_graph(tab):
    if tab == 'overview':
        revenue_by_state = df.groupby('state')['sellingprice'].sum().reset_index()
        fig = px.choropleth(
            revenue_by_state,
            locations='state',
            color='sellingprice',
            locationmode='USA-states',
            scope="usa",
            title='Revenue by State',
            hover_data=['state', 'sellingprice'],
            hover_name='state',
        )

        fig.update_layout(
            title={
                'x': 0.5,
                'font': dict(
                    family="Serif",
                    size=20,
                    color='blue'
                )
            },
        )
        return fig

# 2. _______________Top 10 selling vehicles by year - drop_______________
@my_app.callback(
    Output('ov-top-10-selling-vehicles', 'figure'),
    [Input('ov-year-dropdown', 'value')]
)
def update_top_10_selling_vehicles(selected_year):
    df_filtered = df[df['year'] == selected_year]
    top_10_vehicles = df_filtered.groupby('make')['sellingprice'].sum().nlargest(10).reset_index()
    fig = px.bar(
        top_10_vehicles,
        x='make',
        y='sellingprice',
        title=f'Top 10 Selling Vehicles in {selected_year}'
    )
    fig.update_layout(
        title={
            'x': 0.5,
            'font': dict(
                family="Serif",
                size=20,
                color='blue'
            )
        },
        xaxis=dict(
            title='Make',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        ),
        yaxis=dict(
            title='Revenue',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        )
    )
    return fig

# 3. _______________Treemap of Sales Volume by State_______________
@my_app.callback(
    Output('ov-sales-volume-by-state-graph', 'figure'),
    [Input('tabs', 'value')]
)
def update_sales_volume_by_state_graph(tab):
    if tab == 'overview':
        sales_by_state = df.groupby('state')['sellingprice'].sum().reset_index()
        fig = px.treemap(
            sales_by_state,
            path=['state'],
            values='sellingprice',
            title='Sales Volume by State'
        )
        fig.update_layout(
            title={
                'x': 0.5,
                'font': dict(
                    family="Serif",
                    size=20,
                    color='blue'
                )
            }
        )
        return fig

# 4. _______________Pie chart of Top 10 makes by Sales volume_______________
@my_app.callback(
    Output('ov-top-10-makes-pie-graph', 'figure'),
    [Input('ov-state-checklist', 'value')]
)
def update_top_10_makes_pie_graph(selected_states):
    df_filtered = df[df['state'].isin(selected_states)]
    sales_by_make = df_filtered.groupby('make')['sellingprice'].sum().nlargest(10).reset_index()
    fig = px.pie(
        sales_by_make,
        values='sellingprice',
        names='make',
        title='Top 10 Makes by Sales Volume'
    )
    fig.update_layout(
        title={
            'x': 0.5,
            'font': dict(
                family="Serif",
                size=20,
                color='blue'
            )
        }
    )
    return fig

# 5. _______________Total Sales by Body Type_______________
@my_app.callback(
    Output('ov-total-sales-by-bodytype-graph', 'figure'),
    [Input('tabs', 'value')]
)
def update_total_sales_by_bodytype_graph(tab):
    if tab == 'overview':
        sales_by_bodytype = df.groupby('body')['sellingprice'].sum().reset_index()
        fig = px.pie(
            sales_by_bodytype,
            values='sellingprice',
            names='body',
            title='Total Sales by Body Type',
            hole=.5,
        )
        fig.update_layout(
            title={
                'x': 0.5,
                'font': dict(
                    family="Serif",
                    size=20,
                    color='blue'
                )
            }
        )
        return fig

# ======================Market Trends tab layout===============================
market_trends_layout = html.Div([
    html.Header('Market Trends', style={'textAlign': 'center'}),
    html.Br(),
    html.Div([
        # 1. __________________Top 10 highest prices by make
        dcc.Loading(
            id="mt-loading-1",
            type="circle",
            children=dcc.Graph(id='mt-top-10-highest-prices')
        )
    ]),
    html.Br(),
    html.Div([
        # 2. Manufacturer distribution by state
        dcc.Loading(
            id="mt-loading-2",
            type="circle",
            children=dcc.Graph(id='mt-manufacturer-distribution-graph')
        ),
    ]),
    html.Div([
        # 3. Make Market Share over the years
        dcc.Loading(
            id="mt-loading-3",
            type="circle",
            children=dcc.Graph(id='mt-market-share-graph'),
        ),
        html.Br(),
        html.P('Select the year range:'),
        dcc.RangeSlider(
            id='mt-year-slider-1',
            min=df['year'].min(),
            max=df['year'].max(),
            value=[df['year'].min(), df['year'].max()],
            marks={str(year): str(year) for year in df['year'].unique()},
            step=None
        )
    ]),
    html.Br(),
    html.Div([
        # 4. Sales distribution by category
        dcc.Graph(id='mt-sales-distribution'),
        html.Br(),
        html.P('Select the category:'),
        dcc.RadioItems(
            id='mt-distribution-category',
            options=[
                {'label': 'Make', 'value': 'make'},
                {'label': 'Body Type', 'value': 'body'},
                {'label': 'State', 'value': 'state'}
            ],
            value='make'
        )
    ]),
    html.Br(),
    html.Div([
        # 5. Average selling price of [body type] over the years - provide a dropdown to select the body type
        dcc.Loading(
            id="mt-loading-5",
            type="circle",
            children=dcc.Graph(id='mt-selling-price-body-type-graph')
        ),
        html.Br(),
        html.P('Select the body type:'),
        dcc.Dropdown(
            id='mt-body-type-dropdown',
            options=[{'label': body, 'value': body} for body in df['body'].unique()],
            value='sedan'
        ),
    ])
])

# 1. Top 10 highest prices by make
@my_app.callback(
    Output('mt-top-10-highest-prices', 'figure'),
    [Input('tabs', 'value')]
)
def update_top_10_highest_prices_graph(tab):
    if tab == 'market_trends':
        highest_prices = df.groupby('make')['sellingprice'].max().nlargest(10).reset_index()
        fig = px.bar(
            highest_prices,
            x='make',
            y='sellingprice',
            title='Top 10 Highest Prices by Make',
            labels={'sellingprice': 'Highest Price ($)', 'make': 'Make'},
            height=400
        )
        fig.update_layout(
            title={
                'x': 0.5,
                'font': dict(
                    family="Serif",
                    size=20,
                    color='blue'
                )
            },
            xaxis=dict(

                titlefont=dict(
                    family="Serif",
                    size=18,
                    color="darkred"
                )
            ),
            yaxis=dict(

                titlefont=dict(
                    family="Serif",
                    size=18,
                    color="darkred"
                )
            )
        )
        return fig

# 2. Manufacturer distribution by state
@my_app.callback(
    Output('mt-manufacturer-distribution-graph', 'figure'),
    [Input('tabs', 'value')]
)
def update_manufacturer_distribution_graph(tab):
    if tab == 'market_trends':
        manufacturer_distribution = df.groupby(['state', 'make']).size().reset_index(name='counts')
        fig = px.bar(
            manufacturer_distribution,
            x='state',
            y='counts',
            color='make',
            title='Manufacturer Distribution by State',
            labels={'counts': 'Number of Cars', 'state': 'State', 'make': 'Manufacturer'},
            height=400
        )
        fig.update_layout(
            title={
                'x': 0.5,
                'font': dict(
                    family="Serif",
                    size=20,
                    color='blue'
                )
            },
            xaxis=dict(

                titlefont=dict(
                    family="Serif",
                    size=18,
                    color="darkred"
                )
            ),
            yaxis=dict(

                titlefont=dict(
                    family="Serif",
                    size=18,
                    color="darkred"
                )
            )
        )
        return fig

# 3. Make Market Share over the years
@my_app.callback(
    Output('mt-market-share-graph', 'figure'),
    [Input('mt-year-slider-1', 'value')]
)
def update_graph(year_range):
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

    market_share = df_filtered['make'].value_counts(normalize=True) * 100

    fig = go.Figure(data=[go.Scatter(
        x=market_share.index,
        y=market_share.values,
        mode='lines',
        fill='tozeroy'
    )])
    fig.update_layout(
        title={
            'text': 'Market Share by Car Make',
            'x': 0.5,
            'font': dict(
                family="Serif",
                size=20,
                color='blue'
            )
        },
        xaxis=dict(
            title='Car Make',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        ),
        yaxis=dict(
            title='Market Share (%)',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        )
    )
    return fig

# 4. Sales distribution by category
@my_app.callback(
Output('mt-sales-distribution', 'figure'),
Input('mt-distribution-category', 'value')
)
def update_sales_distribution(category):
    distribution = df[category].value_counts().nlargest(10)
    fig = px.pie(values=distribution.values, names=distribution.index,
                 title=f'Sales Distribution by {category.capitalize()}')
    fig.update_layout(
        title={

            'x': 0.5,
            'font': dict(
                family="Serif",
                size=20,
                color='blue'
            )
        },
        xaxis=dict(
            title='Category',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        ),
        yaxis=dict(
            title='Sales Volume',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        )
    )
    return fig

# 5. Average selling price of [bodytype] over the years - provide a dropdown to select the body type
@my_app.callback(
    Output('mt-selling-price-body-type-graph', 'figure'),
    [Input('mt-body-type-dropdown', 'value')]
)
def update_selling_price_body_type_graph(selected_body_type):
    df_filtered = df[df['body'] == selected_body_type]
    avg_sellingprice_body_type = df_filtered.groupby('year')['sellingprice'].mean().reset_index()
    fig = px.line(
        avg_sellingprice_body_type,
        x='year',
        y='sellingprice',
        title=f'Average Selling Price of {selected_body_type} Over the Years',
        labels={'sellingprice': 'Average Selling Price ($)', 'year': 'Year'}
    )
    fig.update_layout(
        title={
            'x': 0.5,
            'font': dict(
                family="Serif",
                size=20,
                color='blue'
            )
        },
        xaxis=dict(

            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        ),
        yaxis=dict(

            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        )
    )
    return fig

# =======================Make and Model Insights tab layout=======================
insights_layout = html.Div([
    html.Header('Car Insights', style={'textAlign': 'center'}),
    html.Br(),
    html.Div([
        # 1. _________________________________
        dcc.Graph(id='mm-price-range-distribution'),
        html.Br(),
        html.P('Choose a make:'),
        dcc.Dropdown(
            id='mm-make-model-price-dropdown',
            options=[{'label': make, 'value': make} for make in df['make'].unique()],
            value=df['make'].unique()[0]
        ),
        html.Br(),
        html.P('Choose the price range on the slider:'),
        dcc.RangeSlider(
            id='mm-price-range-slider',
            min=df['sellingprice'].min(),
            max=df['sellingprice'].max(),
            step=1000,
            value=[df['sellingprice'].min(), df['sellingprice'].max()],
            marks={i: f'{i}' for i in range(0, int(df['sellingprice'].max()), 10000)}
        )
    ]),
    html.Br(),
    html.Div([
        # 2. ________________Top selling makes models_____________________
        dcc.Graph(id='mm-top-selling-makes-models'),
        html.Br(),
        html.P('Select the type of data to display:'),
        dcc.RadioItems(
            id='mm-make-model-selector',
            options=[
                {'label': 'Top Makes', 'value': 'make'},
                {'label': 'Top Models', 'value': 'model'}
            ],
            value='make'
        ),
        html.Br(),
        html.P('Select the number of top makes/models to display:'),
        dcc.Slider(
            id='mm-top-n-slider',
            min=5,
            max=20,
            step=1,
            value=10,
            marks={i: str(i) for i in range(5, 21, 5)}
        )
    ]),
    html.Br(),
    html.Div([
        # 3. _____________________________________
        dcc.Graph(id='mm-avg-price-make-model'),
        html.Br(),
        html.P('Choose the make(s) to display:'),
        dcc.Dropdown(
            id='mm-make-dropdown',
            options=[{'label': make, 'value': make} for make in df['make'].unique()],
            multi=True,
            value=df['make'].unique()[:5].tolist()
        )
    ]),
    html.Br(),
    html.Div([
        # 4. _____________________________________
        dcc.Graph(id='mm-body-type-distribution'),
        html.Br(),
        html.P('Choose a make:'),
        dcc.Dropdown(
            id='mm-make-model-dropdown',
            options=[{'label': make, 'value': make} for make in df['make'].unique()],
            value=df['make'].unique()[0]
        )
    ]),
    html.Br(),
    html.Div([
        # 5. _____________________________________
        dcc.Graph(id='mm-price-vs-odometer'),
        html.Br(),
        html.P('Choose a make:'),
        dcc.Dropdown(
            id='mm-make-model-scatter-dropdown',
            options=[{'label': make, 'value': make} for make in df['make'].unique()],
            value=df['make'].unique()[0]
        )
    ]),
])

@my_app.callback(
    Output('mm-price-range-distribution', 'figure'),
    [Input('mm-make-model-price-dropdown', 'value'),
     Input('mm-price-range-slider', 'value')]
)
def update_price_range_distribution(selected_make, price_range):
    filtered_df = df[(df['make'] == selected_make) &
                     (df['sellingprice'] >= price_range[0]) &
                     (df['sellingprice'] <= price_range[1])]
    fig = px.histogram(
        filtered_df,
        x='sellingprice',
        nbins=30,
        title=f'Price Distribution for {selected_make}'
    )
    fig.update_layout(
        title={
            'x': 0.5,
            'font': dict(
                family="Serif",
                size=20,
                color='blue'
            )
        },
        xaxis=dict(
            title='Price ($)',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        ),
        yaxis=dict(
            title='Count',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        )
    )
    return fig

@my_app.callback(
    Output('mm-top-selling-makes-models', 'figure'),
    [Input('mm-make-model-selector', 'value'),
     Input('mm-top-n-slider', 'value')]
)
def update_top_selling(selection, top_n):
    filtered_df = df[df[selection].isin(df[selection].value_counts().nlargest(top_n).index)]
    counts = filtered_df.groupby([selection, 'transmission']).size().reset_index(name='counts')
    fig = px.bar(counts, x=selection, y='counts', color='transmission',
                 title=f'Top {top_n} Selling {selection.capitalize()}s by Transmission Mode')
    fig.update_layout(
        title={
            'x':0.5,
            'font' : dict(
                family="Serif",
                size=20,
                color='blue'
            )
        },
        xaxis=dict(
            title=selection.capitalize(),
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        ),
        yaxis=dict(
            title='Count',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        )
    )
    return fig

@my_app.callback(
    Output('mm-avg-price-make-model', 'figure'),
    Input('mm-make-dropdown', 'value')
)
def update_avg_price(selected_makes):
    filtered_df = df[df['make'].isin(selected_makes)]
    avg_prices = filtered_df.groupby('make')['sellingprice'].mean().sort_values(ascending=False)
    fig = px.bar(x=avg_prices.index, y=avg_prices.values,
                 title='Average Selling Price by Make')
    fig.update_layout(
        title={
            'x': 0.5,
            'font': dict(
                family="Serif",
                size=20,
                color='blue'
            )
        },
        xaxis=dict(
            title='Make',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        ),
        yaxis=dict(
            title='Average Selling Price ($)',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        )
    )
    return fig

@my_app.callback(
    Output('mm-body-type-distribution', 'figure'),
    Input('mm-make-model-dropdown', 'value')
)
def update_body_type_distribution(selected_make):
    filtered_df = df[df['make'] == selected_make]
    body_type_counts = filtered_df['body'].value_counts()
    fig = px.pie(values=body_type_counts.values, names=body_type_counts.index,
                 title=f'Body Type Distribution for {selected_make}')
    fig.update_layout(
        title={
            'x': 0.5,
            'font': dict(
                family="Serif",
                size=20,
                color='blue'
            )
        }
    )
    return fig

@my_app.callback(
    Output('mm-price-vs-odometer', 'figure'),
    Input('mm-make-model-scatter-dropdown', 'value')
)
def update_price_vs_odometer(selected_make):
    filtered_df = df[df['make'] == selected_make]
    fig = px.scatter(filtered_df, x='odometer', y='sellingprice',
                     title=f'Price vs. Odometer Reading for {selected_make}')
    fig.update_layout(
        title={
            'x': 0.5,
            'font': dict(
                family="Serif",
                size=20,
                color='blue'
            )
        },
        xaxis=dict(
            title='Odometer Reading',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        ),
        yaxis=dict(
            title='Price ($)',
            titlefont=dict(
                family="Serif",
                size=18,
                color="darkred"
            )
        )
    )
    return fig



# ======================Static graphs tab layout===============================
static_graphs_layout = html.Div([
    html.Title('Static Graphs', style={'textAlign': 'center'}),
    html.Br(),
    html.H3('Total Sales Volume by Year'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/1_total_sales_volume.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Distribution of Selling Prices'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/2_distribution_selling_prices.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Distribution of Odometer Readings'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/3_distribution_odometer_readings.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    # html.H3('Average Selling Prices by Make and Condition'),
    # html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/4_avg_selling_prices_make_condition.png', style={'height': '500px', 'width': '800px'}),
    # html.Br(),
    # html.H3('Frequency of Vehicle Models'),
    # html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/5_frequency_vehicle_models.png', style={'height': '500px', 'width': '800px'}),
    # html.Br(),
    html.H3('Market Share of Top 10 Makes'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/6_market_share_top_10_makes.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Selling Price vs Odometer Reading'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/7_selling_price_vs_odometer.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Pair Plot of Numeric Features'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/8_pair_plot.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Correlation Heatmap of Numeric Features'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/9_correlation_heatmap.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Selling Price Distribution by Condition'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/10_selling_price_distribution_condition.png',
             style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Selling Price Distribution by Year'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/11_selling_price_distribution_year.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Selling Prices by Body Types'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/12_selling_prices_body_types.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Stacked Sales Volume by Top 5 Makes'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/13_stacked_sales_volume_top5_makes.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Proportion of Sales by Condition and Year'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/14_proportion_sales_condition_year.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Joint Plot of Selling Price and MMR'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/15_jointplot_sellingprice_mmr.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Hexbin Plot of Selling Price and MMR'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/16_hexbin_sellingprice_mmr.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('3D Scatter Plot'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/17_3d_scatter_year_sellingprice_odometer_1.png',
             style={'height': '500px', 'width': '800px'}),
    html.Br(),
    # html.H3('Top 5 Makes by Selling Prices'),
    # html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/18_selling_prices_top_5_makes.png', style={'height': '500px', 'width': '800px'}),
    # html.Br(),
    html.H3('QQ Plot of Selling Price'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/19_selling_price_qq_plot.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Distribution Plot of Transmission'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/20_distplot_transmission.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Contour Plot'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/22_density_sales_year_sellingprice.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Rug Plot of Selling Price'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/23_rug_plot.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Price Distribution by Make'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/25_pricing_dynamics_market_value.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Sales Volume Analysis'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/26_sales_volume_analysis.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Vehicle Characteristics Impact on Selling Price'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/27_vehicle_characteristics_impact.png', style={'height': '500px', 'width': '800px'}),
    html.Br(),
    html.H3('Count Distribution by State'),
    html.Img(src='https://raw.githubusercontent.com/mahimasingh-VTU/InformationVisualization_TermProject/main/staticgraphs/28_Count_Distribution_State.png', style={'height': '500px', 'width': '800px'}),

])

# =======================Feedback tab layout=======================
feedback_layout = html.Div([
    html.Br(),
    html.Div([
        html.Label('Your feedback is valuable to us:', htmlFor='comments-textarea'),
        dcc.Textarea(
            id='comments-textarea',
            placeholder='Enter your comments here...',
            style={'width': '100%', 'height': 100},
        ),
        html.Button('Submit', id='submit-button'),
    ]),
])
@my_app.callback(
    Output("submit-button", "n_clicks"),
    Input("submit-button", "n_clicks"),
    State("comments-textarea", "value"),
    prevent_initial_call=True,
)
def print_comments(n_clicks, comments):
    commentslist = []
    if n_clicks > 0:
        commentslist.append(comments)

# =======================Update layout based on selected tab=======================
@my_app.callback(
    Output('layout', 'children'),
    [Input('tabs', 'value')]
)
def update_layout(tab):
    if tab == 'overview':
        return overview_layout
    elif tab == 'market_trends':
        return market_trends_layout
    elif tab == 'insights':
        return insights_layout
    elif tab == 'static-graphs':
        return static_graphs_layout
    elif tab == 'feedback':
        return feedback_layout

@my_app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True,
)
def generate_download(n_clicks):
    csv_string = df.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string




if __name__ == "__main__":

    # # Load the data
    # file_path = 'car_prices.csv'
    # df = pd.read_csv(file_path)
    #
    # #Print basic data info
    # print("Basic Data Info")
    # table_info = PrettyTable()
    # table_info.field_names = ["Description", "Value"]
    # table_info.add_row(["Shape", df.shape])
    # table_info.add_row(["Columns", list(df.columns)])
    # print(table_info)
    #
    # print("Head of the DataFrame")
    # print_dataframe_in_one_line(df.head())
    #
    # print("Descriptive Statistics (Numeric Columns)")
    # print_pretty_table("Descriptive Statistics (Numeric Columns)", df.describe().T.reset_index().round(2))
    #
    # print("Descriptive Statistics (Non-Numeric Columns)")
    # print_pretty_table("Descriptive Statistics (Non-Numeric Columns)", df.describe(exclude='number').T.reset_index())
    #
    # print("Missing Values")
    # missing_values = (df.isnull().sum() / df.count() * 100).reset_index().round(2)
    # missing_values.columns = ["Column", "Missing Values (%)"]
    # print_pretty_table("Missing Values", missing_values)
    # # Call the function to plot histograms
    # plot_initial_histograms(df)
    # cleaner = DataCleaner(df)
    # df_clean = cleaner.clean_data()
    # print(df_clean.info())
    # print("Head of Cleaned DataFrame")
    # print_dataframe_in_one_line(df_clean.head())
    # # Plot box plots before removing outliers
    # print("Box Plots Before Removing Outliers")
    # plot_box_plots(df_clean, "Before Removing Outliers")
    #
    # # Outlier detection and removal
    # df_no_outliers = cleaner.detect_and_remove_outliers()
    # print("DataFrame after removing outliers")
    # print(df_no_outliers.describe().T.reset_index().round(2).to_string(index=False))
    #
    # # Plot box plots after removing outliers
    # print("Box Plots After Removing Outliers")
    # plot_box_plots(df_no_outliers, "After Removing Outliers")
    #
    # # PCA
    # processor = DataProcessor(df_no_outliers)
    # principal_components, n_components_95, variance_explained = processor.perform_pca()
    #
    # print(f"Number of components to explain 95% of the variance: {n_components_95}")
    # print(f"Cumulative explained variance by {n_components_95} components: {variance_explained * 100:.2f}%")
    #
    # # Perform normality test on original features
    # numeric_features = df_no_outliers.select_dtypes(include=[np.number])
    #
    #
    # # Perform K-square normality test on original features
    # processor = DataProcessor(df_no_outliers)
    # processor.normality_test_Ksquare(numeric_features)
    #
    # plotter = Plotter(df_no_outliers)
    # # Data exploration
    # explorer = DataExplorer(df_no_outliers)
    # explorer.explore_data()
    # explorer.plot_histograms()

    # # Generate plots
    #
    # plotter.plot_bar_vis('make', 'sellingprice', 'Brands vs Selling Price by Transmission', 'Brands', 'Selling Price')
    # plotter.plot_scatter('sellingprice', 'mmr', 'Selling Price vs MMR', 'Selling Price', 'MMR')
    # plotter.plot_heatmap()
    # plotter.plot_line('year', 'sellingprice', 'Manufacturing Year vs Selling Price', 'Manufacturing Year',
    #                   'Selling Price')
    # # plotter.plot_box('year', 'Distribution of Manufacturing Years')
    # plotter.plot_histogram('mmr', 'Manheim Market Report (MMR) Distribution')
    # plotter.plot_count('state', 'Count of Distribution by State', 'State', 'Count')
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
    # plotter.plot_violin('year', 'sellingprice', 'Violin Plot of Selling Prices by Year')
    # plotter.plot_joint_kde_scatter('sellingprice', 'mmr', 'Joint KDE and Scatter Plot of Selling Price vs MMR')
    # plotter.plot_rug('sellingprice', 'Rug Plot of Selling Prices')
    # plotter.plot_3d_scatter('year', 'sellingprice', 'mmr', '3D Scatter Plot of Year, Selling Price and MMR')
    #
    # plotter.plot_hexbin('sellingprice', 'mmr', 'Hexbin Plot of Selling Price vs MMR')
    # plotter.plot_strip('condition', 'sellingprice', 'Strip Plot of Selling Price by Condition')
    # plotter.plot_swarm('year', 'sellingprice', 'Swarm Plot of Selling Prices by Year')
    # print("Grouped Bar Plot")
    # plotter.plot_grouped_bar('make', 'sellingprice', 'year', 'Sales Price by Make and Year')
    #
    # print("Boxen Plot")
    # plotter.plot_boxen('year', 'sellingprice', 'Boxen Plot of Selling Prices by Year')

    #Create the Dash app
    # my_app = create_dash_app()
    my_app.run_server(debug=True, port=8050, host='0.0.0.0')
