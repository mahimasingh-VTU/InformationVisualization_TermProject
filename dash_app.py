import dash as dash
import pandas as pd
import plotly.express as px
from dash import html, dcc
from dash.dependencies import Input, Output
from plotly import graph_objs as go
from data_cleaner import DataCleaner

df_raw = pd.read_csv('car_prices.csv')
df = DataCleaner(df_raw).clean_data()

"""
Numerical features: ['year', 'condition', 'odometer', 'mmr', 'sellingprice']
Categorical features: ['make', 'model', 'trim', 'body', 'transmission', 'vin', 'state', 'color', 'interior', 'seller', 'saledate']
"""
def create_dash_app():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    my_app = dash.Dash("Car Sales Dashboard", external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

    # =============================creating tabs================================
    my_app.layout = html.Div([
        html.H2('Car Sales Dashboard', style={'textAlign': 'center'}),
        html.Br(),
        dcc.Tabs(
            id='tabs',
            children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Market Trends', value='market_trends'),
                dcc.Tab(label='Make and Model Insights', value='insights')
            ]
        ),
        html.Div(id='layout')
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
            # dcc.Graph(id='revenue-by-state-graph'),
        ]),
        html.Br(),
        html.Div([
            # 2. _______________Top 10 selling vehicles by year - drop_______________
            dcc.Loading(
                id="ov-loading-2",
                type="circle",
                children=dcc.Graph(id='ov-top-10-selling-vehicles')
            ),
            dcc.Dropdown(
                id = 'ov-year-dropdown',
                options = [{'label': year, 'value': year} for year in df['year'].unique()],
                value = df['year'].min()
            )
        ]),
        html.Br(),
        html.Div([
            # 3. _______________Treemap of Sales Volume by State_______________
            dcc.Loading(
                id="ov-loading-3",
                type="circle",
                children=dcc.Graph(id='ov-sales-volume-by-state-graph')
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
                title='Revenue by State'
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
            return fig


    # ======================Market Trends tab layout===============================
    market_trends_layout = html.Div([
        html.Header('Market Trends', style={'textAlign': 'center'}),
        html.Br(),
        html.Div([
            # 1. __________________
            dcc.Loading(
                id="loading-average-price",
                type="circle",
                children=dcc.Graph(id='average-price-graph')
            ),
            dcc.Dropdown(
                id='category_dropdown',
                options=[
                    {'label': 'Condition', 'value': 'condition'},
                    {'label': 'State', 'value': 'state'},
                    {'label': 'Color', 'value': 'color'},
                    {'label': 'Make', 'value': 'make'},
                    {'label': 'Year', 'value': 'year'}
                ],
                value='condition'
            ),
        ]),
        html.Br(),
        html.Div([
            # 2. Manufacturer distribution by state
            dcc.Loading(
                id="loading-average-price",
                type="circle",
                children=dcc.Graph(id='manufacturer-distribution-graph')
            ),
        ]),
        html.Div([
            # 3. Make Market Share over the years
            dcc.Loading(
                id="loading-average-price",
                type="circle",
                children=dcc.Graph(id='market-share-graph'),
            ),
            dcc.RangeSlider(
                id='year-slider',
                min=df['year'].min(),
                max=df['year'].max(),
                value=[df['year'].min(), df['year'].max()],
                marks={str(year): str(year) for year in df['year'].unique()},
                step=None
            )
        ]),
        html.Br(),
        html.Div([
            # 4. Monthly sales trend by year - input year via slider
            dcc.Slider(
                id='year-slider',
                min=df['year'].min(),
                max=df['year'].max(),
                value=df['year'].min(),
                marks={str(year): str(year) for year in df['year'].unique()},
                step=None
            ),
            dcc.Graph(id='monthly-sales-trend-graph'),
        ]),
        html.Br(),
        html.Div([
            # 5. Average selling price of [body type] over the years - provide a dropdown to select the body type
            dcc.Loading(
                id="loading-average-price",
                type="circle",
                children=dcc.Graph(id='selling-price-body-type-graph')
            ),
            dcc.Dropdown(
                id='body-type-dropdown',
                options=[{'label': body, 'value': body} for body in df['body'].unique()],
                value='sedan'
            ),
        ])
    ])

    # 1. ___________Average price by [condition,state,color,make,year] - radiobuttons to select the category_______
    @my_app.callback(
        Output('average-price-graph', 'figure'),
        [Input('category_dropdown', 'value')]
    )
    def update_tab2(category):
        df['sellingprice'] = pd.to_numeric(df['sellingprice'], errors='coerce')
        avg_price_fig = px.bar(df.groupby(category).mean()['sellingprice'],
                               x=category, y='sellingprice', title=f'Average price by {category}')
        return avg_price_fig

    # 2. Manufacturer distribution by state
    @my_app.callback(
        Output('manufacturer-distribution-graph', 'figure'),
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
            return fig
    # 3. Make Market Share over the years
    @my_app.callback(
        Output('market-share-graph', 'figure'),
        [Input('year-slider', 'value')]
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
        fig.update_layout(title='Market Share by Car Make', xaxis_title='Car Make', yaxis_title='Market Share (%)')
        return fig

    # 4. Monthly sales trend by year - input year via slider
    @my_app.callback(
        Output('monthly-sales-trend-graph', 'figure'),
        [Input('year-slider', 'value')]
    )
    def update_monthly_sales_trend_graph(selected_year):
        df_filtered = df[df['year'] == selected_year]
        df_filtered['saledate'] = pd.to_datetime(df_filtered['saledate'])
        df_filtered = df_filtered.set_index('saledate')
        monthly_sales = df_filtered['sellingprice'].resample('M').sum()
        fig = go.Figure(data=go.Scatter(x=monthly_sales.index, y=monthly_sales.values))
        fig.update_layout(title='Monthly Sales Trend', xaxis_title='Month', yaxis_title='Sales')

        return fig

    # 5. Average selling price of [bodytype] over the years - provide a dropdown to select the body type
    @my_app.callback(
        Output('selling-price-body-type-graph', 'figure'),
        [Input('body-type-dropdown', 'value')]
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
        return fig

    # =======================Make and Model Insights tab layout=======================
    insights_layout = html.Div([
        html.Header('Car Insights', style={'textAlign': 'center'}),
        html.Br(),
        # 1.
        html.Div([
            html.H5('Top 10 makes by Sales volume'),
            dcc.Graph(id='top-10-makes-graph'),
        ]),
        dcc.RadioItems(
            id='category_radio',
            options=[
                {'label': 'State', 'value': 'state'},
                {'label': 'Color', 'value': 'color'},
                {'label': 'Year', 'value': 'year'}
            ],
            value='state'
        ),
    ])

    @my_app.callback(
        Output('top-10-makes-graph', 'figure'),
        [Input('category_radio', 'value')]
    )
    def update_tab3(category):
        top_10_makes_fig = px.bar(
            df.groupby('make').sum().nlargest(10, 'sellingprice')[category],
            x=category,
            y='color',
            title=f'Top 10 makes across {category} by sales volume'
        )
        return top_10_makes_fig


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

    return my_app
