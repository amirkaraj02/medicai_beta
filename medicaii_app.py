import dash
from dash.dependencies import Input, Output, State
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
import dash_auth

# pandas and plotly library
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from flask_sqlalchemy import SQLAlchemy
from flask import Flask

# import ml library
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors, svm, naive_bayes, neural_network
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn import metrics, datasets
from sklearn.preprocessing import StandardScaler
from sklearn import *
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier

# app requires "pip install psycopg2" as well

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
new_style_sheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
PLOTLY_LOGO = "/assets/mylogo.jpg"

server = Flask(__name__)
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css, new_style_sheet])
server = app.server

app.server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

app.title = "MEDIC-AI"
app._favicon = '/assets/favicon.ico'

auth = dash_auth.BasicAuth(
    app,
    {'marmara': 'demo'}
)

# connection string for local PostgresSQL test table
# app.server.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:amir02@localhost/medicAI_app_test"

# connection string for live database on heroku
# app.server.config["SQLALCHEMY_DATABASE_URI"] = "postgres://tctazcptsemgbx:f5571aba7ec0ee16ba3daa04037580d825844d2a0bdade4224fe3295dfd4d7c5@ec2-54-228-32-29.eu-west-1.compute.amazonaws.com:5432/db8f6qmenvsgf2"
# app.server.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://fgwbratbblzjzt:d74375080115398d280f39aa59178a7a28725864cfae35a9f0dfd52019e49e50@ec2-44-205-63-142.compute-1.amazonaws.com:5432/ddcvvddf5lh5v2"
app.server.config[
    "SQLALCHEMY_DATABASE_URI"] = "postgresql://evfjbosb:RJzpZoiQ3EmojPvg-NI4Q8IrwLMTQNSl@ella.db.elephantsql.com/evfjbosb"
db = SQLAlchemy(app.server)

# ---------------------------------------------------------------------------------
# ML Models
# ---------------------------------------------------------------------------------
predict_models = {'Logistic Regression': linear_model.LogisticRegression,
                  'Decision Tree': tree.DecisionTreeClassifier,
                  'k-NN': neighbors.KNeighborsClassifier,
                  'Linear Regression': linear_model.LinearRegression,
                  'Naive Bayes': naive_bayes.GaussianNB,
                  'SVM': svm.SVC,
                  'ANN': neural_network.MLPClassifier
                  }

# roc_models = {'Regression': linear_model.LinearRegression,
#               'Decision Tree': tree.DecisionTreeRegressor,
#               'k-NN': neighbors.KNeighborsRegressor}

# ---------------------------------------------------------------------------------
# app Navbar
# ---------------------------------------------------------------------------------
navbar = dbc.Navbar(
    dbc.Container(
        [
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="60px", width="60px")),
                    dbc.Col(dbc.NavbarBrand("MEDIC-AI Karar Destek Sistemi", className="ms-2")),
                ],
                align="center",
                className="g-0",
            ),
            dbc.Button('Veriler', id='veri_collapse-button',
                       style={'padding': 10, 'top': '10px', 'left': '10px', 'width': '25%'}),
        ]
    ),
    color="dark",
    dark=True,
)

# ---------------------------------------------------------------------------------
# app layout
# ---------------------------------------------------------------------------------

app.layout = html.Div([
    navbar,
    html.Div(children=[
        dbc.Collapse(
            dbc.Card(dbc.CardBody(
                html.Div([
                    dcc.Input(
                        id='adding-rows-name',
                        placeholder='Yeni ozellik adi...',
                        value='',
                        style={'padding': 10, "display": "inline-block"}
                    ),
                    dcc.Dropdown(
                        id='column_data_type',
                        placeholder='Select datatype',
                        options=['numeric', 'text'],
                        value='',
                        style={"display": "inline-block", "left": "10px", "width": "150px"}
                    ),
                    dbc.Button('Ozellik Ekle', id='adding-columns-button', n_clicks=0,
                               style={"display": "inline-block", "margin-left": "20px"}),
                    dcc.Interval(id='interval_pg', interval=86400000 * 7, n_intervals=0),
                    # activated once/week or when page refreshed
                    html.Br(),
                    html.Br(),
                    html.Div(id='postgres_datatable'),
                    #
                    html.Br(),
                    dbc.Button('Orneklem Ekle', id='editing-rows-button', n_clicks=0),
                    dbc.Button('Kaydet', id='save_to_postgres', color="success", n_clicks=0,
                               style={"margin-left": "20px"}),

                    dbc.Button('Veri Icerigi', id='data_details', color="info", style={"margin-left": "20px"}),

                    # Create notification when saving to database
                    html.Div(id='placeholder', children=[]),
                    dcc.Store(id="store", data=0),
                    dcc.Interval(id='interval', interval=1000),
                    dbc.Collapse([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(id='table_columns_details')
                            ])
                        ], className='text-center'),
                    ], id='data_component-detail')
                ]),

            )),
            id="collapse-data",
        ),
    ]),
    html.Br(),
    html.Br(),
    html.Div(children=[
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.P(children=["Ozellikler Sayisi: ", html.P(id='ozellik_sayisi')],
                                       style={"font-weight": "bold"}),
                                html.P(children=["Orneklem Sayisi: ", html.P(id='orneklem_sayisi')],
                                       style={"font-weight": "bold"}),
                            ])
                        ], className='text-center'),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.P("Model Tahmini:"),
                                dcc.Dropdown(
                                    id='ml_model_selection',
                                    options=["ANN", "SVM", "k-NN", "Decision Tree", "Naive Bayes", "CatBoost"],
                                    placeholder='Tahmin icin bir Model secin',
                                    value='',
                                    clearable=False,
                                ),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col([
                                        html.P("Sonuc:"),
                                    ]),
                                    dbc.Col([
                                        dcc.Input(id='ml_model_pred', type='text', disabled=True),
                                        html.Br(),
                                        html.Br(),
                                        dbc.Button('Tahmin Yap', color="warning", id='ml_pred_start',
                                                   n_clicks=0)
                                    ]),

                                ])
                            ])
                        ], className='text-center')
                    ])
                ], className='pt-2'),
                dbc.Row([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Button('Model Tahmin Sonuclari', id='model_pred_roc', style={'padding': 10}),
                        ])
                    ], className='text-center'),
                ], className='pt-2'),
            ], className='pt-2', xs=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Label('Scatter matrix kullanarak data analisi yapmak:'),
                        dcc.Dropdown(
                            id='scatter_matrix_dropdown',
                            value='',
                            multi=True,
                            persistence=True,
                            persistence_type='session',
                        ),
                        dcc.Graph(id="Graff_scatter_matrix"),
                    ])
                ], className='h-100 text-center')
            ], xs=5),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([html.Label("Grafikler:")]),
                            dbc.Col([
                                dcc.Dropdown(
                                    id='selection',
                                    options=["Scatter", "Bar", "Histogram", "Box", "Violin"],
                                    value='Scatter',
                                    persistence=True,
                                    persistence_type='session',
                                )]),
                        ]),
                        html.Br(),
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    id="x-variable",
                                    value='',
                                    persistence=True,
                                    persistence_type='session',
                                )
                            ]),
                            dbc.Col([
                                dcc.Dropdown(
                                    id="y-variable",
                                    value='',
                                    persistence=True,
                                    persistence_type='session',
                                ),
                            ]),
                        ]),
                        dcc.Loading(dcc.Graph(id="multi-Graph"), type="default")
                    ], className='h-100 text-center')
                ])
            ], xs=5)
        ], className='p-2 align-items-stretch'),
        # ML Grafikleri
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([html.P("ML Grafik Secimi:")]),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id='ml_selection',
                                            options=["ANN", "SVM", "k-NN", "Decision Tree", "Naive Bayes",
                                                     "Linear Regression",
                                                     "Logistic Regression"],
                                            value='Decision Tree',
                                            clearable=False,
                                            persistence=True,
                                            persistence_type='session',
                                        )]),
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id="ml_x_vale",
                                            value='',
                                            persistence=True,
                                            persistence_type='session',
                                        ),
                                    ]),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id="ml_y_vale",
                                            value='',
                                            persistence=True,
                                            persistence_type='session',
                                        ),
                                    ]),
                                ]),
                                dcc.Graph(id="ml_model"),
                            ])
                        ], className='h-100 text-center')
                    ], xs=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Label("k-means clustering"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("X variable"),
                                        dcc.Dropdown(
                                            id="xk-variable",
                                            value="",
                                            persistence=True,
                                            persistence_type='session',
                                        ),
                                    ]),
                                    dbc.Col([
                                        dbc.Label("Y variable"),
                                        dcc.Dropdown(
                                            id="yk-variable",
                                            value="",
                                            persistence=True,
                                            persistence_type='session',
                                        ),
                                    ]),
                                    dbc.Col([
                                        dbc.Label("Cluster count"),
                                        dbc.Input(id="cluster-count", type="number", value=3),
                                    ]),
                                ]),
                                dbc.Col(dcc.Graph(id="cluster-graph")),

                            ], className='h-100 text-center')
                        ])
                    ], xs=6)
                ])
            ])
        ], className='p-2 align-items-stretch'),
        # ML model prediction
        dbc.Collapse([
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.P('Classification Report Sonuclari'),
                                    html.Div(id="ml_model_pred_result"),
                                    html.P('Confusion Matrix Sonuclari'),
                                    dcc.Graph(id="ml_model_confusion_matrix"),
                                ])
                            ], className='h-100 text-center')
                        ], xs=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.P('ROC and PR Curves Graphs'),
                                    dcc.Graph(id="roc-curve-model", figure={})

                                ], className='h-100 text-center')
                            ])
                        ], xs=6)
                    ])
                ])
            ], className='p-2 align-items-stretch'),
        ], id='model_pred')

    ])

],
)


# ---------------------------------------------------------------------------------
# app callbacks
# ---------------------------------------------------------------------------------
# callback for collapse button
@app.callback(Output('collapse-data', 'is_open'),
              [Input('veri_collapse-button', 'n_clicks')],
              [State('collapse-data', 'is_open')])
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# callback for table detail collapse button
@app.callback(Output('data_component-detail', 'is_open'),
              [Input('data_details', 'n_clicks')],
              [State('data_component-detail', 'is_open')])
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# callback for collapse modal for model prediction
@app.callback(Output('model_pred', 'is_open'),
              [Input('model_pred_roc', 'n_clicks')],
              [State('model_pred', 'is_open')])
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# callback for populating table from database
@app.callback(Output('postgres_datatable', 'children'),
              [Input('interval_pg', 'n_intervals')])
def populate_datatable(n_intervals):
    df = pd.read_sql_table('hastaveri', con=db.engine)
    return [
        dash_table.DataTable(
            id='our-table',
            columns=[{
                         'name': str(x),
                         'id': str(x),
                         'deletable': False,
                     } if x == 'SONUC' or x == 'OUTPUT'
                     else {
                'name': str(x),
                'id': str(x),
                'deletable': True,
            } for x in df.columns],
            data=df.to_dict('records'),
            editable=True,
            row_deletable=True,
            fixed_rows={'headers': True},
            row_selectable='single',
            filter_action="native",
            sort_action="native",  # give user capability to sort columns
            sort_mode="single",  # sort across 'multi' or 'single' columns
            # page_action='none',  # render all of the data at once. No paging.
            style_table={'height': '300px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'minWidth': '90px', 'width': '100px', 'maxWidth': '150px'},
            page_size=15,
            export_format='xlsx',
        ),
    ]


# callback for adding new column
@app.callback(
    Output('our-table', 'columns'),
    [Input('adding-columns-button', 'n_clicks')],
    [State('adding-rows-name', 'value'),
     State('column_data_type', 'value'),
     State('our-table', 'columns')],
    prevent_initial_call=True)
def add_columns(n_clicks, value, dtype, existing_columns):
    if n_clicks > 0:
        existing_columns.append({
            'name': value,
            'id': value,
            'type': dtype,
            'renamable': True,
            'deletable': True
        })
    return existing_columns


# callback fo adding new row
@app.callback(
    Output('our-table', 'data'),
    [Input('editing-rows-button', 'n_clicks')],
    [State('our-table', 'data'),
     State('our-table', 'columns')],
    prevent_initial_call=True)
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})

    return rows


# callback for saving into database
@app.callback(
    [Output('placeholder', 'children'),
     Output("store", "data")],
    [Input('save_to_postgres', 'n_clicks'),
     Input("interval", "n_intervals")],
    [State('our-table', 'data'),
     State('store', 'data')],
    prevent_initial_call=True)
def df_to_db(n_clicks, n_intervals, dataset, s):
    output = html.Plaintext("The data has been saved to your PostgreSQL database.",
                            style={'color': 'green', 'font-weight': 'bold', 'font-size': 'large'})
    no_output = html.Plaintext("", style={'pading': ''})

    input_triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if input_triggered == "save_to_postgres":
        s = 6
        pg = pd.DataFrame(dataset)
        pg.to_sql('hastaveri', con=db.engine, if_exists='replace', index=False)
        return output, s
    elif input_triggered == 'interval' and s > 0:
        s = s - 1
        if s > 0:
            return output, s
        else:
            return no_output, s
    elif s == 0:
        return no_output, s


# callback for scatter-matrix with dd
@app.callback(
    Output("Graff_scatter_matrix", "figure"),
    Input("scatter_matrix_dropdown", "value"),
    State('our-table', 'data'),
    prevent_initial_call=True
)
def update_bar_chart(dims, data):
    df_colName = pd.DataFrame(data)
    fig = px.scatter_matrix(df_colName,
                            dimensions=dims)  # if we want to open first with a color value --> color=df.columns[2])
    return fig


# callback for multi-graph
@app.callback(
    Output("multi-Graph", "figure"),
    Input("selection", "value"),
    Input("x-variable", "value"),
    Input("y-variable", "value"),
    State('our-table', 'data'),
    prevent_initial_call=True)
def display_animated_graph(selection, x_var, y_var, data):
    df_multi = pd.DataFrame(data)
    # multi_fig = make_subplots(rows=1, cols=2)
    # multi_fig.add_trace(go.Scatter(x=df_multi["ALT"], y=df_multi["OUTPUT"], mode="lines"), row=1, col=1)
    # multi_fig.add_trace(go.Bar(x=df_multi["Gender"], y=df_multi["OUTPUT"]), row=1, col=2)
    animations = {
        'Scatter': px.scatter(df_multi, x=x_var, y=y_var),

        'Bar': px.bar(df_multi, x=x_var, y=y_var, barmode="group"),

        'Histogram': px.histogram(df_multi, x=x_var, y=y_var, color="SONUC", marginal="rug",
                                  hover_data=df_multi.columns),
        'Box': px.box(df_multi, x=x_var, y=y_var, color="SONUC", notched=True),
        'Violin': px.violin(df_multi, x=x_var, y=y_var, color="SONUC", box=True, points='all',
                            hover_data=df_multi.columns),
        # 'Go_Scatter': px.scatter(df_multi, x="RBC", y="HGB", color="DISEASE"),
        # 'Go_Scatter': multi_fig,
        # 'Go_Scatter': go.Scatter(x=df_multi.columns[x], y=df_multi.columns[y], mode="markers", marker={"size": 8})
        # 'Go_Scatter': px.scatter(df_multi, x=x_var, y=y_var)
        # 'Go_Scatter': px.histogram(
        #     df_multi,  x="RBC", y="HGB", color="DISEASE", range_x=[-5, 60],
        #     hover_data=df_multi.columns)
    }
    return animations[selection]


# callback for updating dropdown
@app.callback(
    Output("scatter_matrix_dropdown", "options"),
    Input('our-table', 'data'),
    prevent_initial_call=True)
def update_dd_value(data):
    df_multi = pd.DataFrame(data)
    return [{"label": i, "value": i} for i in df_multi.columns]


# callback for column number
@app.callback(
    Output("ozellik_sayisi", "children"),
    Input('our-table', 'data'),
    prevent_initial_call=True)
def update_dd_value(data):
    df_columns_no = pd.DataFrame(data)
    return len(df_columns_no.columns)


# callback for row number
@app.callback(
    Output("orneklem_sayisi", "children"),
    Input('our-table', 'data'),
    prevent_initial_call=True)
def update_dd_value(data):
    df_columns_no = pd.DataFrame(data)
    return len(df_columns_no.index)


# callback for columns detail
@app.callback(
    Output('table_columns_details', 'children'),
    Input('our-table', 'data'),
    prevent_initial_call=True)
def update_dd_value(data):
    df = pd.DataFrame(data)
    return [
        html.P(
            children=[
                html.Ul(children=[
                    html.Li(children=[i], style={
                        "display": "inline-block",
                        "padding": "10px",
                        "font-weight": "bold",
                        "text-align": "center"
                    }) for i in df.columns])
            ],
        ),
    ]


# callback for updating x-variable value dropdown
@app.callback(
    Output("x-variable", "options"),
    Input('our-table', 'data'),
    prevent_initial_call=True)
def update_dd_value(data):
    df_x_variable = pd.DataFrame(data)
    return [{"label": i, "value": i} for i in df_x_variable.columns]


# callback for updating y-variable value dropdown
@app.callback(
    Output("y-variable", "options"),
    Input('our-table', 'data'),
    prevent_initial_call=True)
def update_dd_value(data):
    df_y_variable = pd.DataFrame(data)
    return [{"label": i, "value": i} for i in df_y_variable.columns]


# ---------------------------------------------------------------------------------
# callbacks for ML Models
# ---------------------------------------------------------------------------------
@app.callback(
    Output('ml_model', 'figure'),
    [Input('our-table', 'data'),
     Input('ml_x_vale', 'value'),
     Input('ml_y_vale', 'value'),
     Input('ml_selection', 'value')],
    prevent_initial_call=True
)
def ml_train_display(data, x_ml, y_ml, model_name):
    dff = pd.DataFrame(data)

    # converting datas from object to numeric
    convert_df = dff._convert(numeric=True)

    num_data = convert_df._get_numeric_data()

    x_val = convert_df[x_ml].values[:, None]
    target_data = convert_df[y_ml]

    # convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(target_data)

    X_train, X_test, y_train, y_test = train_test_split(
        x_val, y_transformed, random_state=20)

    model = predict_models[model_name]()
    model.fit(X_train, y_train)
    x_range = np.linspace(x_val.min(), x_val.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train,
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test,
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range,
                   name='prediction')
    ])
    return fig


@app.callback(
    Output("ml_x_vale", "options"),
    Input('our-table', 'data'),
    prevent_initial_call=True)
def update_x_dd_value(data):
    df_ml_x_variable = pd.DataFrame(data)
    convert_df_x_ml = df_ml_x_variable._convert(numeric=True)
    df_x_ml_num = convert_df_x_ml._get_numeric_data()
    del df_x_ml_num['OUTPUT']
    return [{"label": i, "value": i} for i in df_x_ml_num.columns]


@app.callback(
    Output("ml_y_vale", "options"),
    Input('our-table', 'data'),
    prevent_initial_call=True)
def update_y_dd_value(data):
    df_ml_y_variable = pd.DataFrame(data)
    convert_df_y_ml = df_ml_y_variable._convert(numeric=True)
    df_y_ml_num = convert_df_y_ml._get_numeric_data()
    del df_y_ml_num['OUTPUT']
    return [{"label": i, "value": i} for i in df_y_ml_num.columns]


# test callbacks for K-means cluster Models
@app.callback(
    Output("xk-variable", "options"),
    Input('our-table', 'data'),
    prevent_initial_call=True)
def update_dd_value(data):
    df_multi = pd.DataFrame(data)
    convert_df_x_cluster = df_multi._convert(numeric=True)
    df_x_cluster_num = convert_df_x_cluster._get_numeric_data()
    return [{"label": i, "value": i} for i in df_x_cluster_num.columns]


@app.callback(
    Output("yk-variable", "options"),
    Input('our-table', 'data'),
    prevent_initial_call=True)
def update_dd_value(data):
    df_multi = pd.DataFrame(data)
    convert_df_y_cluster = df_multi._convert(numeric=True)
    df_y_cluster_num = convert_df_y_cluster._get_numeric_data()
    return [{"label": i, "value": i} for i in df_y_cluster_num.columns]


@app.callback(
    Output("cluster-graph", "figure"),
    [
        Input("xk-variable", "value"),
        Input("yk-variable", "value"),
        Input("cluster-count", "value"),
        Input('our-table', 'data'),
    ],
    prevent_initial_call=True
)
def make_graph(x, y, n_clusters, data):
    dff = pd.DataFrame(data)

    # converting datatype from object to numeric
    convert_df = dff._convert(numeric=True)
    num_data = convert_df._get_numeric_data()

    # minimal input validation, make sure there's at least one cluster
    km = KMeans(n_clusters=max(n_clusters, 1))
    df_dff = num_data.loc[:, [x, y]]
    # print(df_dff)
    km.fit(df_dff.values)
    df_dff["cluster"] = km.labels_

    centers = km.cluster_centers_

    data = [
        go.Scatter(
            x=df_dff.loc[df_dff.cluster == c, x],
            y=df_dff.loc[df_dff.cluster == c, y],
            mode="markers",
            marker={"size": 8},
            name="Cluster {}".format(c),
        )
        for c in range(n_clusters)
    ]

    data.append(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker={"color": "#000", "size": 12, "symbol": "diamond"},
            name="Cluster centers",
        )
    )

    layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}

    return go.Figure(data=data, layout=layout)


# callbacks for algorithm predictions
@app.callback(
    Output('ml_model_pred', 'value'),
    Input('ml_model_selection', 'value'),
    Input('ml_pred_start', 'n_clicks'),
    Input('our-table', 'data'),
    Input('our-table', 'selected_rows'),
    prevent_initial_call=True,
)
def model_prediction(ml_selection, n_clicks, data, selected_rows):
    dff = pd.DataFrame(data)
    convert_df = dff._convert(numeric=True)
    numeric_data = convert_df.select_dtypes(include=np.number)
    del numeric_data['OUTPUT']
    targetData = np.nan_to_num(convert_df['OUTPUT'])

    # targetData= np.nan_to_num(targetData)

    # reset button click state
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if ml_selection == 'SVM' and 'ml_pred_start' in changed_id:
        SVMnumeric_data_train, SVMnumeric_data_test, SVMtarget_data_train, SVMtarget_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)
        scaler = StandardScaler()
        scaler.fit(SVMnumeric_data_train)

        SVMnumeric_data_train = scaler.transform(SVMnumeric_data_train)
        SVMnumeric_data_test = scaler.transform(SVMnumeric_data_test)
        svcclassifier = SVC(kernel='poly', degree=3)
        svcclassifier.fit(SVMnumeric_data_train, SVMtarget_data_train)

        # Selected row prediction matrix
        sel_row = [data[i] for i in selected_rows]
        dff_sel_row = pd.DataFrame(sel_row)
        dff_sel_non_nan = dff_sel_row.fillna(0)
        convert_df = dff_sel_non_nan._convert(numeric=True)
        new_numeric_data = convert_df.select_dtypes(include=np.number)
        del new_numeric_data['OUTPUT']

        SVMnewTestData = scaler.transform(new_numeric_data)

        # Yeni datanin tahmini yapma
        SVMprediction = svcclassifier.predict(SVMnewTestData)

        # Yeni datanin tahmini sonuclandirmak
        if SVMprediction[:] == 1:
            return "DIPPER"
        elif SVMprediction[:] == 2:
            return "NON-DIPPER"
        else:
            return "NaN"

        # return algthPredict


    elif ml_selection == 'ANN' and 'ml_pred_start' in changed_id:
        ANN_numeric_data_train, ANN_numeric_data_test, ANN_target_data_train, ANN_target_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)

        scaler = StandardScaler()
        scaler.fit(ANN_numeric_data_train)

        ANN_numeric_data_train = scaler.transform(ANN_numeric_data_train)
        ANN_numeric_data_test = scaler.transform(ANN_numeric_data_test)

        mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50))
        model_ANN = mlp.fit(ANN_numeric_data_train, ANN_target_data_train)

        pred = model_ANN.predict(ANN_numeric_data_test)

        # Selected row prediction matrix
        sel_row = [data[i] for i in selected_rows]
        dff_sel_row = pd.DataFrame(sel_row)
        dff_sel_non_nan = dff_sel_row.fillna(0)
        convert_df = dff_sel_non_nan._convert(numeric=True)
        new_numeric_data = convert_df.select_dtypes(include=np.number)
        print("before:", new_numeric_data)
        del new_numeric_data['OUTPUT']

        ann_newTestData = scaler.transform(new_numeric_data)

        # Yeni datanin tahmini yapma
        model_ANN_prediction = model_ANN.predict(ann_newTestData)

        # Yeni datanin tahmini sonuclandirmak
        if model_ANN_prediction[:] == 1:
            return "DIPPER"
        elif model_ANN_prediction[:] == 2:
            return "NON-DIPPER"
        else:
            return "NaN"

    elif ml_selection == 'k-NN' and 'ml_pred_start' in changed_id:
        knn_numeric_data_train, knn_numeric_data_test, knn_target_data_train, knn_target_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)
        scaler = StandardScaler()
        scaler.fit(knn_numeric_data_train)

        knn_numeric_data_train = scaler.transform(knn_numeric_data_train)
        knn_numeric_data_test = scaler.transform(knn_numeric_data_test)

        knn_model = KNeighborsClassifier(n_neighbors=3)

        knn_model.fit(knn_numeric_data_train, knn_target_data_train)

        algthPredict = knn_model.predict(knn_numeric_data_test)

        # print(confusion_matrix(SVMtarget_data_test, algthPredict))
        #
        # print(classification_report(SVMtarget_data_test, algthPredict))

        # Selected row prediction matrix
        sel_row = [data[i] for i in selected_rows]
        dff_sel_row = pd.DataFrame(sel_row)
        dff_sel_non_nan = dff_sel_row.fillna(0)
        convert_df = dff_sel_non_nan._convert(numeric=True)
        new_numeric_data = convert_df.select_dtypes(include=np.number)
        del new_numeric_data['OUTPUT']

        knn_newTestData = scaler.transform(new_numeric_data)

        # Yeni datanin tahmini yapma
        knn_prediction = knn_model.predict(knn_newTestData)

        # Yeni datanin tahmini sonuclandirmak
        if knn_prediction[:] == 1:
            return "DIPPER"
        elif knn_prediction[:] == 2:
            return "NON-DIPPER"
        else:
            return "NaN"

    elif ml_selection == 'Decision Tree' and 'ml_pred_start' in changed_id:
        d_tree_numeric_data_train, d_tree_numeric_data_test, d_tree_target_data_train, d_tree_target_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)
        scaler = StandardScaler()
        scaler.fit(d_tree_numeric_data_train)

        knn_numeric_data_train = scaler.transform(d_tree_numeric_data_train)
        knn_numeric_data_test = scaler.transform(d_tree_numeric_data_test)

        d_tree_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        # d_tree_classifier = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
        #                                            max_features=None, max_leaf_nodes=None,
        #                                            min_impurity_decrease=0.0, min_impurity_split=None,
        #                                            min_samples_leaf=1, min_samples_split=2,
        #                                            min_weight_fraction_leaf=0.0, presort=False,
        #                                            random_state=0, splitter='best')

        d_tree_classifier.fit(d_tree_numeric_data_train, d_tree_target_data_train)

        # Selected row prediction matrix
        sel_row = [data[i] for i in selected_rows]
        dff_sel_row = pd.DataFrame(sel_row)
        dff_sel_non_nan = dff_sel_row.fillna(0)
        convert_df = dff_sel_non_nan._convert(numeric=True)
        new_numeric_data = convert_df.select_dtypes(include=np.number)
        del new_numeric_data['OUTPUT']

        d_tree_newTestData = scaler.transform(new_numeric_data)

        # Yeni datanin tahmini yapma
        d_tree_prediction = d_tree_classifier.predict(d_tree_newTestData)

        # Yeni datanin tahmini sonuclandirmak
        if d_tree_prediction[:] == 1:
            return "DIPPER"
        elif d_tree_prediction[:] == 2:
            return "NON-DIPPER"
        else:
            return "NaN"

    elif ml_selection == 'Naive Bayes' and 'ml_pred_start' in changed_id:
        naiveB_numeric_data_train, naiveB_numeric_data_test, naiveB_target_data_train, naiveB_target_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)
        scaler = StandardScaler()
        scaler.fit(naiveB_numeric_data_train)

        naiveB_numeric_data_train = scaler.transform(naiveB_numeric_data_train)
        naiveB_numeric_data_test = scaler.transform(naiveB_numeric_data_test)

        naiveB_classifier = GaussianNB()

        naiveB_classifier.fit(naiveB_numeric_data_train, naiveB_target_data_train)

        # y_pred = naiveB_classifier.predict(naiveB_numeric_data_test)

        # Selected row prediction matrix
        sel_row = [data[i] for i in selected_rows]
        dff_sel_row = pd.DataFrame(sel_row)
        dff_sel_non_nan = dff_sel_row.fillna(0)
        convert_df = dff_sel_non_nan._convert(numeric=True)
        new_numeric_data = convert_df.select_dtypes(include=np.number)
        del new_numeric_data['OUTPUT']

        naiveB_newTestData = scaler.transform(new_numeric_data)

        # Yeni datanin tahmini yapma
        naiveB_prediction = naiveB_classifier.predict(naiveB_newTestData)
        print(naiveB_prediction)

        # Yeni datanin tahmini sonuclandirmak
        if naiveB_prediction[:] == 1:
            return "DIPPER"
        elif naiveB_prediction[:] == 2:
            return "NON-DIPPER"
        else:
            return "NaN"

    elif ml_selection == 'CatBoost' and 'ml_pred_start' in changed_id:
        CatBoost_numeric_data_train, CatBoost_numeric_data_test, CatBoost_target_data_train, CatBoost_target_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)
        # scaler = StandardScaler()
        # scaler.fit(CatBoost_numeric_data_train)

        # CatBoost_numeric_data_train = scaler.transform(CatBoost_numeric_data_train)
        # CatBoost_numeric_data_test = scaler.transform(CatBoost_numeric_data_test)

        CatBoost_classifier = CatBoostClassifier(
            iterations=5,
            learning_rate=0.1,
            # loss_function='CrossEntropy'
        )

        CatBoost_classifier.fit(CatBoost_numeric_data_train, CatBoost_target_data_train)

        # y_pred = naiveB_classifier.predict(naiveB_numeric_data_test)

        # Selected row prediction matrix
        sel_row = [data[i] for i in selected_rows]
        dff_sel_row = pd.DataFrame(sel_row)
        dff_sel_non_nan = dff_sel_row.fillna(0)
        convert_df = dff_sel_non_nan._convert(numeric=True)
        new_numeric_data = convert_df.select_dtypes(include=np.number)
        del new_numeric_data['OUTPUT']

        # CatBoost_newTestData = scaler.transform(new_numeric_data)

        # Yeni datanin tahmini yapma
        CatBoost_prediction = CatBoost_classifier.predict(new_numeric_data)

        # Yeni datanin tahmini sonuclandirmak
        if CatBoost_prediction[:] == 1:
            return "DIPPER"
        elif CatBoost_prediction[:] == 2:
            return "NON-DIPPER"
        else:
            return "NaN"


# callback for model prediction result
@app.callback(
    [
        Output('ml_model_pred_result', 'children'),
        Output('ml_model_confusion_matrix', 'figure'),
        Output('roc-curve-model', 'figure'),
    ],
    [
        Input('ml_model_selection', 'value'),
        Input('ml_pred_start', 'n_clicks'),
        Input('our-table', 'data'),
        Input('our-table', 'selected_rows')
    ],
    prevent_initial_call=True,
)
def model_pred_result(ml_selection, n_clicks, data, selected_rows):
    dff = pd.DataFrame(data)
    convert_df = dff._convert(numeric=True)
    numeric_data = convert_df.select_dtypes(include=np.number)
    del numeric_data['OUTPUT']
    targetData = np.nan_to_num(convert_df['OUTPUT'])

    # targetData= np.nan_to_num(targetData)

    # reset button click state
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if ml_selection == 'SVM' and 'ml_pred_start' in changed_id:
        SVMnumeric_data_train, SVMnumeric_data_test, SVMtarget_data_train, SVMtarget_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)
        scaler = StandardScaler()
        scaler.fit(SVMnumeric_data_train)

        SVMnumeric_data_train = scaler.transform(SVMnumeric_data_train)
        SVMnumeric_data_test = scaler.transform(SVMnumeric_data_test)
        svcclassifier = SVC(kernel='poly', degree=3, probability=True)

        svcclassifier.fit(SVMnumeric_data_train, SVMtarget_data_train)
        algthPredict = svcclassifier.predict(SVMnumeric_data_test)

        dff_pred_res = pd.DataFrame.from_dict(
            classification_report(SVMtarget_data_test, algthPredict, output_dict=True)).transpose()
        df_table = dff_pred_res.reset_index()

        clas_report_table = [
            dash_table.DataTable(
                columns=[{
                    'name': str(x),
                    'id': str(x),
                }
                    for x in df_table.columns.tolist()],
                data=df_table.to_dict('records'),
                fixed_rows={'headers': True},
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '90px', 'width': '100px', 'maxWidth': '150px'},
                # export_format='xlsx'

            ),
        ]

        dff_conf_matrix = pd.DataFrame(confusion_matrix(SVMtarget_data_test, algthPredict))
        fig = px.imshow(dff_conf_matrix, text_auto=True)

        prod_roc = svcclassifier.predict_proba(SVMnumeric_data_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(SVMtarget_data_test, prod_roc, pos_label=1)
        score = metrics.auc(fpr, tpr)

        roc_fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={score:.4f})',
            labels=dict(
                x='False Positive Rate',
                y='True Positive Rate'))
        roc_fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)

        return [clas_report_table, fig, roc_fig]

    elif ml_selection == 'ANN' and 'ml_pred_start' in changed_id:
        ANN_numeric_data_train, ANN_numeric_data_test, ANN_target_data_train, ANN_target_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)

        scaler = StandardScaler()
        scaler.fit(ANN_numeric_data_train)

        ANN_numeric_data_train = scaler.transform(ANN_numeric_data_train)
        ANN_numeric_data_test = scaler.transform(ANN_numeric_data_test)

        mlp = MLPClassifier(hidden_layer_sizes=(50, 50))
        model_ANN = mlp.fit(ANN_numeric_data_train, ANN_target_data_train)

        ann_pred = model_ANN.predict(ANN_numeric_data_test)

        ann_dff_pred_res = pd.DataFrame.from_dict(
            classification_report(ANN_target_data_test, ann_pred, output_dict=True)).transpose()
        ann_df_table = ann_dff_pred_res.reset_index()

        ann_clas_report_table = [
            dash_table.DataTable(
                columns=[{
                    'name': str(x),
                    'id': str(x),
                }
                    for x in ann_df_table.columns.tolist()],
                data=ann_df_table.to_dict('records'),
                fixed_rows={'headers': True},
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '90px', 'width': '100px', 'maxWidth': '150px'},
                # export_format='xlsx'

            ),
        ]

        ann_dff_conf_matrix = pd.DataFrame(confusion_matrix(ANN_target_data_test, ann_pred))
        ann_fig = px.imshow(ann_dff_conf_matrix, text_auto=True)

        ann_prod_roc = model_ANN.predict_proba(ANN_numeric_data_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(ANN_target_data_test, ann_prod_roc, pos_label=1)
        score = metrics.auc(fpr, tpr)

        ann_roc_fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={score:.4f})',
            labels=dict(
                x='False Positive Rate',
                y='True Positive Rate'))
        ann_roc_fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)

        return [ann_clas_report_table, ann_fig, ann_roc_fig]



    elif ml_selection == 'k-NN' and 'ml_pred_start' in changed_id:
        knn_numeric_data_train, knn_numeric_data_test, knn_target_data_train, knn_target_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)
        scaler = StandardScaler()
        scaler.fit(knn_numeric_data_train)

        knn_numeric_data_train = scaler.transform(knn_numeric_data_train)
        knn_numeric_data_test = scaler.transform(knn_numeric_data_test)

        knn_model = KNeighborsClassifier(n_neighbors=3)

        knn_model.fit(knn_numeric_data_train, knn_target_data_train)

        knn_algthPredict = knn_model.predict(knn_numeric_data_test)

        knn_dff_pred_res = pd.DataFrame.from_dict(
            classification_report(knn_target_data_test, knn_algthPredict, output_dict=True)).transpose()
        knn_df_table = knn_dff_pred_res.reset_index()

        knn_clas_report_table = [
            dash_table.DataTable(
                columns=[{
                    'name': str(x),
                    'id': str(x),
                }
                    for x in knn_df_table.columns.tolist()],
                data=knn_df_table.to_dict('records'),
                fixed_rows={'headers': True},
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '90px', 'width': '100px', 'maxWidth': '150px'},
                # export_format='xlsx'

            ),
        ]

        knn_result = confusion_matrix(knn_target_data_test, knn_algthPredict)
        knn_dff_conf_matrix = pd.DataFrame(knn_result)
        knn_fig = px.imshow(knn_dff_conf_matrix, text_auto=True)

        knn_prod_roc = knn_model.predict_proba(knn_numeric_data_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(knn_target_data_test, knn_prod_roc, pos_label=1)
        score = metrics.auc(fpr, tpr)

        knn_roc_fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={score:.4f})',
            labels=dict(
                x='False Positive Rate',
                y='True Positive Rate'))
        knn_roc_fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)

        return [knn_clas_report_table, knn_fig, knn_roc_fig]


    elif ml_selection == 'Decision Tree' and 'ml_pred_start' in changed_id:
        d_tree_numeric_data_train, d_tree_numeric_data_test, d_tree_target_data_train, d_tree_target_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)
        scaler = StandardScaler()
        scaler.fit(d_tree_numeric_data_train)

        knn_numeric_data_train = scaler.transform(d_tree_numeric_data_train)
        knn_numeric_data_test = scaler.transform(d_tree_numeric_data_test)

        d_tree_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        # d_tree_classifier = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
        #                                            max_features=None, max_leaf_nodes=None,
        #                                            min_impurity_decrease=0.0, min_impurity_split=None,
        #                                            min_samples_leaf=1, min_samples_split=2,
        #                                            min_weight_fraction_leaf=0.0, presort=False,
        #                                            random_state=0, splitter='best')

        d_tree_classifier.fit(d_tree_numeric_data_train, d_tree_target_data_train)

        d_tree_algthPredict = d_tree_classifier.predict(knn_numeric_data_test)

        d_tree_dff_pred_res = pd.DataFrame.from_dict(
            classification_report(d_tree_target_data_test, d_tree_algthPredict, output_dict=True)).transpose()
        d_tree_df_table = d_tree_dff_pred_res.reset_index()

        d_tree_clas_report_table = [
            dash_table.DataTable(
                columns=[{
                    'name': str(x),
                    'id': str(x),
                }
                    for x in d_tree_df_table.columns.tolist()],
                data=d_tree_df_table.to_dict('records'),
                fixed_rows={'headers': True},
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '90px', 'width': '100px', 'maxWidth': '150px'},
                # export_format='xlsx'

            ),
        ]

        d_tree_dff_conf_matrix = pd.DataFrame(confusion_matrix(d_tree_target_data_test, d_tree_algthPredict))
        d_tree_fig = px.imshow(d_tree_dff_conf_matrix, text_auto=True)

        d_tree_prod_roc = d_tree_classifier.predict_proba(knn_numeric_data_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(d_tree_target_data_test, d_tree_prod_roc, pos_label=1)
        score = metrics.auc(fpr, tpr)

        d_tree_roc_fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={score:.4f})',
            labels=dict(
                x='False Positive Rate',
                y='True Positive Rate'))
        d_tree_roc_fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)

        return [d_tree_clas_report_table, d_tree_fig, d_tree_roc_fig]



    elif ml_selection == 'Naive Bayes' and 'ml_pred_start' in changed_id:
        naiveB_numeric_data_train, naiveB_numeric_data_test, naiveB_target_data_train, naiveB_target_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)
        scaler = StandardScaler()
        scaler.fit(naiveB_numeric_data_train)

        naiveB_numeric_data_train = scaler.transform(naiveB_numeric_data_train)
        naiveB_numeric_data_test = scaler.transform(naiveB_numeric_data_test)

        naiveB_classifier = GaussianNB()

        naiveB_classifier.fit(naiveB_numeric_data_train, naiveB_target_data_train)

        naiveB_algthPredict = naiveB_classifier.predict(naiveB_numeric_data_test)

        naiveB_dff_pred_res = pd.DataFrame.from_dict(
            classification_report(naiveB_target_data_test, naiveB_algthPredict, output_dict=True)).transpose()
        naiveB_df_table = naiveB_dff_pred_res.reset_index()

        naiveB_clas_report_table = [
            dash_table.DataTable(
                columns=[{
                    'name': str(x),
                    'id': str(x),
                }
                    for x in naiveB_df_table.columns.tolist()],
                data=naiveB_df_table.to_dict('records'),
                fixed_rows={'headers': True},
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '90px', 'width': '100px', 'maxWidth': '150px'},
                # export_format='xlsx'

            ),
        ]

        naiveB_dff_conf_matrix = pd.DataFrame(confusion_matrix(naiveB_target_data_test, naiveB_algthPredict))
        naiveB_fig = px.imshow(naiveB_dff_conf_matrix, text_auto=True)

        naiveB_prod_roc = naiveB_classifier.predict_proba(naiveB_numeric_data_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(naiveB_target_data_test, naiveB_prod_roc, pos_label=1)
        score = metrics.auc(fpr, tpr)

        naiveB_roc_fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={score:.4f})',
            labels=dict(
                x='False Positive Rate',
                y='True Positive Rate'))
        naiveB_roc_fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1)

        return [naiveB_clas_report_table, naiveB_fig, naiveB_roc_fig]


    elif ml_selection == 'CatBoost' and 'ml_pred_start' in changed_id:
        CatBoost_numeric_data_train, CatBoost_numeric_data_test, CatBoost_target_data_train, CatBoost_target_data_test = train_test_split(
            numeric_data, targetData, test_size=0.30)

        CatBoost_classifier = CatBoostClassifier(
            iterations=5,
            learning_rate=0.1,
            # loss_function='CrossEntropy'
        )

        CatBoost_classifier.fit(CatBoost_numeric_data_train, CatBoost_target_data_train)

        CatBoost_prediction = CatBoost_classifier.predict(CatBoost_numeric_data_test)

    CatBoost_dff_pred_res = pd.DataFrame.from_dict(
        classification_report(CatBoost_target_data_test, CatBoost_prediction, output_dict=True)).transpose()
    CatBoost_df_table = CatBoost_dff_pred_res.reset_index()

    CatBoost_clas_report_table = [
        dash_table.DataTable(
            columns=[{
                'name': str(x),
                'id': str(x),
            }
                for x in CatBoost_df_table.columns.tolist()],
            data=CatBoost_df_table.to_dict('records'),
            fixed_rows={'headers': True},
            style_table={'height': '300px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'minWidth': '90px', 'width': '100px', 'maxWidth': '150px'},
            # export_format='xlsx'

        ),
    ]

    CatBoost_dff_conf_matrix = pd.DataFrame(confusion_matrix(CatBoost_target_data_test, CatBoost_prediction))
    CatBoost_fig = px.imshow(CatBoost_dff_conf_matrix, text_auto=True)

    naiveB_prod_roc = CatBoost_classifier.predict_proba(CatBoost_numeric_data_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(CatBoost_target_data_test, naiveB_prod_roc, pos_label=1)
    score = metrics.auc(fpr, tpr)

    CatBoost_roc_fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate',
            y='True Positive Rate'))
    CatBoost_roc_fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    return [CatBoost_clas_report_table, CatBoost_fig, CatBoost_roc_fig]


if __name__ == '__main__':
    # app.run_server(debug=True, use_reloader=False, port=4000)
    app.run_server(debug=True, port=4000)

# Not: Linear Regresion Prediction has problems, will return back and see more in the documentation
