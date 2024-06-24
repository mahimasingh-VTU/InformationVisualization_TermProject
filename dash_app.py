import urllib
import dash as dash
import pandas as pd
import plotly.express as px
from dash import html, dcc
from dash.dependencies import Input, Output, State
from plotly import graph_objs as go

from data_cleaner import DataCleaner

df_raw = pd.read_csv('car_prices.csv')
df = DataCleaner(df_raw).clean_data()

def create_dash_app():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    my_app = dash.Dash("Car Sales Dashboard", external_stylesheets=external_stylesheets,
                       suppress_callback_exceptions=True)

    # =============================creating tabs================================
    my_app.layout = html.Div([
        html.Div([
            html.Img(src='logo.png', style={'height': '60px', 'width': '60px'}),
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
                dcc.Tab(label='Make and Model Insights', value='insights')
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
        html.Br(),
        html.Div([
            # 6. _______________User comments_______________
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

    @my_app.callback(
        Output("submit-button", "n_clicks"),
        Input("submit-button", "n_clicks"),
        State("comments-textarea", "value"),
        prevent_initial_call=True,
    )
    def print_comments(n_clicks, comments):
        if n_clicks > 0:
            print(comments)

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

    @my_app.callback(
        Output("download-data", "data"),
        Input("download-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def generate_download(n_clicks):
        csv_string = df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string

    return my_app
