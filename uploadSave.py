import base64
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import plotly.express as px
import pandas as pd

from flask_sqlalchemy import SQLAlchemy
from flask import Flask

from sqlalchemy import Table, Column, String, MetaData, select

# app requires "pip install psycopg2" as well

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server, suppress_callback_exceptions=True)
app.server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# app.server.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:amir02@localhost/medicAI_app_test"
app.server.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://fgwbratbblzjzt:d74375080115398d280f39aa59178a7a28725864cfae35a9f0dfd52019e49e50@ec2-44-205-63-142.compute-1.amazonaws.com:5432/ddcvvddf5lh5v2"

db = SQLAlchemy(app.server)

query = db.session.execute("select tables.table_name from information_schema.tables where table_schema='public'")
names = [row[0] for row in query]
# print(names)


#
# are giving error on saving into db
# query_from_dd = db.session.execute("select * from hastatest")
# df_columns_name = db.session.execute("select * from hastatest").keys()
# df_table = pd.DataFrame(query_from_dd, columns=df_columns_name)


app.layout = html.Div([
    dcc.Upload(id='upload-data',
               children=html.Button('Upload File'),
               multiple=True),
    html.Br(),
    dcc.Dropdown(
        id='dd_input',
        options=[{'label': i, 'value': i} for i in names],
        value='',
        persistence=True
    ),
    html.Div(id="show-query"),
    html.Br(),
    dcc.Interval(id='interval_pg', interval=86400000 * 7, n_intervals=0),  # activated once/week or when page refreshed
    # html.Div(id='output-data-upload'),  # it is same as the div postgres
    html.Div(id='postgres_datatable'),
    html.Div(id='show_db_datatable'),
    html.Br(),
    html.Button('Save to Database', id='save_to_postgres', n_clicks=0),

    dcc.Dropdown(id='x_dd_input', value='', persistence=True),
    dcc.Dropdown(id='y_dd_input', value='', persistence=True),
    dcc.Graph(id="dd_graf"),
    #
    # Create notification when saving to excel
    html.Div(id='placeholder', children=[]),
    dcc.Store(id="store", data=0),
    dcc.Interval(id='interval', interval=1000),

])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    fileNameCSV = filename.split('.csv')[0]
    fileNameXLS = filename.split('.xlsx')[0]

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df.to_sql(name=fileNameCSV.lower(), con=db.engine, if_exists='replace', index=False)
        elif 'xlsx' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            df.to_sql(name=fileNameXLS.lower(), con=db.engine, if_exists='replace', index=False)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            id='our-table',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),
    ])


@app.callback(Output('postgres_datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children


@app.callback(Output('show_db_datatable', 'children'),
              Input('interval_pg', 'n_intervals'),
              Input('dd_input', 'value'))
def populate_datatable(n_intervals, dd_input):
    # df = pd.read_sql_table('hastaveri', con=db.engine)
    query_from_dd = db.session.execute("select * from "f"{dd_input}""")
    query_from_dd_clm = db.session.execute("select * from "f"{dd_input}""").keys()
    df = pd.DataFrame(query_from_dd, columns=query_from_dd_clm)
    # # df = pd.DataFrame(query_from_dd)
    # print(query_to_df)
    return [
        dash_table.DataTable(
            id='our-table1',
            columns=[{
                'name': str(x),
                'id': str(x),
                'deletable': True,
            }
                for x in df.columns],
            data=df.to_dict('records'),
            editable=True,
            row_deletable=True,
            filter_action="native",
            sort_action="native",  # give user capability to sort columns
            sort_mode="single",  # sort across 'multi' or 'single' columns
            page_action='none',  # render all of the data at once. No paging.
            style_table={'height': '300px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'}

        ),
    ]


@app.callback(
    [Output('placeholder', 'children'),
     Output("store", "data")],
    [Input('save_to_postgres', 'n_clicks'),
     Input("interval", "n_intervals")],
    [State('our-table1', 'data'),
     State('store', 'data')],
    prevent_initial_call=True)
def df_to_db(n_clicks, n_intervals, dataset, s):
    output = html.Plaintext("The data has been saved to your PostgreSQL database.",
                            style={'color': 'green', 'font-weight': 'bold', 'font-size': 'large'})
    no_output = html.Plaintext("", style={'margin': "0px"})

    input_triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if input_triggered == "save_to_postgres":
        s = 6
        pg = pd.DataFrame(dataset)
        pg.to_sql("data.xls", con=db.engine, if_exists='replace', dtype=False, index=False)
        return output, s
    elif input_triggered == 'interval' and s > 0:
        s = s - 1
        if s > 0:
            return output, s
        else:
            return no_output, s
    elif s == 0:
        return no_output, s


# @app.callback(
#     Output("show-query", "children"),
#     Input("dd_input", "value"))
# def select_dd_table(dd_input):
#     query_from_dd = db.session.execute("select * from "f"{dd_input}""")
#     query_from_dd_clm = db.session.execute("select * from "f"{dd_input}""").keys()
#     query_to_df = pd.read_sql(query_from_dd, con=db.engine, columns=query_from_dd_clm)
#     # df = pd.DataFrame(query_from_dd)
#     print(query_from_dd_clm)
#     # query_from_dd = db.session.execute("select * from "f"{dd_input}""")
#     # df_table = pd.DataFrame(query_from_dd)
#     # print(df_table)
#     # return html.Div([
#     #     dash_table.DataTable(
#     #         data=df_table.to_dict('records'),
#     #         columns=[{'name': str(i), 'id': str(i)} for i in df_table.columns]
#     #     )])

@app.callback(
    Output("x_dd_input", "options"),
    Input("dd_input", "value"))
def update_dd_value(dd_input):
    # query_from_dd = db.session.execute("select * from "f"{dd_input}""")
    query_from_dd_clm = db.session.execute("select * from "f"{dd_input}""").keys()
    return [{"label": i, "value": i} for i in query_from_dd_clm]


@app.callback(
    Output("y_dd_input", "options"),
    Input("dd_input", "value"))
def update_dd_value(dd_input):
    # query_from_dd = db.session.execute("select * from "f"{dd_input}""")
    query_from_dd_clm = db.session.execute("select * from "f"{dd_input}""").keys()
    return [{"label": i, "value": i} for i in query_from_dd_clm]


@app.callback(
    Output("dd_graf", "figure"),
    Input("x_dd_input", "value"),
    Input("y_dd_input", "value"),
    State('our-table1', 'data'),
    State('store', 'data'),
    prevent_initial_call=True
)
def update_graf_from(data, x_dd_input, y_dd_input):
    df_fig = pd.DataFrame(data)
    print(df_fig)

    fig = px.scatter(df_fig, x=x_dd_input, y=y_dd_input)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=5000)
