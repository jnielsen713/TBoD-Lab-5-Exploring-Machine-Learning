# Imports --------------------------------------------------------------------------------------------------------------
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from main import pokemon_dataset

# Components -----------------------------------------------------------------------------------------------------------

header = html.Div(
    [
        html.H1("Classifying Legendary Pokemon", className="mt-4", style={"color": "black"}),
        html.H4("Joshua Nielsen - CS-150 - Prof. Mike Ryu", style={"color": "black"}),
        html.Br()
    ],
    style={"text-align": "center", "background-color": "#a82a2a"}
)

left_body = html.Div(
    dbc.Card(
        [
            html.H2("Threshold", style={"color": "black"}),
            dcc.Slider(id="threshold", max=1, min=0, step=0.1, value=0.5),
            html.H2("Sample Size", style={"color": "black"}),
            dcc.Slider(id="sample-size", max=600, min=0, step=50, value=100)
        ],
        style={"background-color": "#e3baba"}
    )
)

confusion_matrix = html.Table(
    [
        html.Tr(
            [
                html.Td("", style={"border-style": "solid", "border-color": "#000000", "border-width": "medium"}),
                html.Td("Actual Legendary", style={"border-style": "solid", "border-color": "#000000", "border-width": "medium"}),
                html.Td("Not Actual Legendary", style={"border-style": "solid", "border-color": "#000000", "border-width": "medium"})
            ]
        ),
        html.Tr(
            [
                html.Td("Predicted Legendary", style={"border-style": "solid", "border-color": "#000000", "border-width": "medium"}),
                html.Td("", id="true-positive", style={"border-style": "solid", "border-color": "#000000", "border-width": "medium"}),
                html.Td("", id="false-positive", style={"border-style": "solid", "border-color": "#000000", "border-width": "medium"})
            ]
        ),
        html.Tr(
            [
                html.Td("Predicted Not Legendary", style={"border-style": "solid", "border-color": "#000000", "border-width": "medium"}),
                html.Td("", id="false-negative", style={"border-style": "solid", "border-color": "#000000", "border-width": "medium"}),
                html.Td("", id="true-negative", style={"border-style": "solid", "border-color": "#000000", "border-width": "medium"})
            ],
        )
    ],
    style={"margin": "10px"}
)

right_body = html.Div(
    dcc.Graph(id="results")
)

footer = html.Div(
    [],
    style={"text-align": "right", "background-color": "#a82a2a", "height": "75px", "color": "black"},
)


# Main Layout ----------------------------------------------------------------------------------------------------------

def create_layout():
    return html.Div(
        [
            dbc.Row(
                header
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [left_body, confusion_matrix],
                        width=6
                    ),
                    dbc.Col(
                        [right_body],
                        width=6
                    )
                ]
            ),
            dbc.Row(
                footer
            )
        ],
    )
