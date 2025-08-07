import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dash
from dash import Dash, html, dcc, Input, Output, State
from dash_ag_grid import AgGrid
import pandas as pd
from glass_dag.dag_model import GlassDag

# Initialize DAG model with a 3x3 grid (can be changed as needed)
dag = GlassDag(rows=3, cols=3)

# Initialize DataFrame from DAG templates
def dag_to_df(dag):
    data = {}
    for col in range(dag.grid.shape[1]):
        data[f"Col {col+1}"] = [dag.grid[row, col].template for row in range(dag.grid.shape[0])]
    return pd.DataFrame(data)

def update_dag_from_df(dag, df):
    for col in range(dag.grid.shape[1]):
        for row in range(dag.grid.shape[0]):
            dag.grid[row, col].template = df.iloc[row, col]
    dag.resolve_grid()

def get_resolved_df(dag):
    data = {}
    for col in range(dag.grid.shape[1]):
        data[f"Col {col+1}"] = [dag.grid[row, col].resolved_template for row in range(dag.grid.shape[0])]
    return pd.DataFrame(data)

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Glass DAG Spreadsheet (Dash AG Grid)"),
    html.P("Enter values or references like cell(0,1) in any cell. All references are resolved by the DAG model."),
    dcc.Store(id="spreadsheet-store", data=dag_to_df(dag).to_dict("records")),
    html.Button("Resolve DAG", id="resolve-button", n_clicks=0),
    AgGrid(
        id="spreadsheet",
        rowData=dag_to_df(dag).to_dict("records"),
        columnDefs=[{"field": col, "editable": True} for col in dag_to_df(dag).columns],
        defaultColDef={"editable": True, "resizable": True},
        style={"height": 400, "width": "100%"}
    ),
    html.H4("Resolved Output"),
    html.Div(id="resolved-output")
])

@app.callback(
    Output("spreadsheet", "rowData"),
    Output("resolved-output", "children"),
    Input("spreadsheet", "cellValueChanged"),
    State("spreadsheet", "rowData"),
    prevent_initial_call=True
)
def update_grid(new_data, old_data):
    print("Updating DAG with new data...")
    df = pd.DataFrame(new_data)
    update_dag_from_df(dag, df)
    resolved_df = get_resolved_df(dag)
    return new_data, html.Pre(resolved_df.to_string(index=False))

@app.callback(
    Input('resolve-button', 'n_clicks'),
)
def resolve_dag(n_clicks):
    if n_clicks > 0:
        print("Resolving DAG...")
    return dash.no_update

if __name__ == "__main__":
    app.run(debug=True)
