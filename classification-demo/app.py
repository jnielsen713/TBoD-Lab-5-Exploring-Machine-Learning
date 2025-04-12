# Imports --------------------------------------------------------------------------------------------------------------
import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from main import app
import layout
import callbacks

# App Layout -----------------------------------------------------------------------------------------------------------

app.layout = layout.create_layout()

# Run ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
