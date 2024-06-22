import dash as dash
from dash import html, dcc
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash("Car Sales Dashboard", external_stylesheets=external_stylesheets)

# creating tabs
my_app.layout = html.Div([
    html.H1('Car Sales Dashboard', style={'textAlign': 'center'}),
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

# Overview layout
overview_layout = html.Div([
    html.H1('Overview', style={'textAlign': 'center'}),
])
@my_app.callback(
)
def update_tab1(text):
    pass

# Market Trends layout
market_trends_layout = html.Div([
    html.H1('Market Trends', style={'textAlign': 'center'}),
])
@my_app.callback(
   )

def update_tab2(text):
    pass


# Make and Model Insights layout
insights_layout = html.Div([
    html.H1('Car Insights', style={'textAlign': 'center'}),
])
@my_app.callback(
)
def update_tab3(text):
    pass

@my_app.callback(
    Output('layout', 'children'),
    [Input('tabs', 'value')]
)
def update_layout(tab):
    if tab == 'overview':
        return overview_layout
    elif tab == 'market_trends':
        return market_trends_layout

my_app.run_server( port=8050, host='0.0.0.0')