# app.py
import dash
from dash import dcc, html, dash_table
import pandas as pd
import requests
import plotly.express as px

# ETL-Service URL
ETL_URL = "http://localhost:8000/data"

# Daten vom ETL ziehen
response = requests.get(ETL_URL)
data = response.json()
df = pd.DataFrame(data)

# Dash App
app = dash.Dash(__name__)

fig = px.scatter(df, x="sepal length (cm)", y="sepal width (cm)", color="target")

app.layout = html.Div([
    html.H1("Iris Dataset Dashboard"),
    
    html.H2("Tabelle"),
    dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=10
    ),
    
    html.H2("Scatterplot"),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run(debug=True, port=8050)
