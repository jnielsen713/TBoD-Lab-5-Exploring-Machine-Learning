# Imports --------------------------------------------------------------------------------------------------------------
import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np


from main import app, make_visual, get_globals
import layout


# Callbacks ------------------------------------------------------------------------------------------------------------

@app.callback(Output("results", "figure"),
              Input("sample-size", "value"),
              Input("threshold", "value"))
def update_results(sample_size, threshold):
    return make_visual(sample_size, threshold)


@app.callback(Output("true-positive", "children"),
              Output("false-positive", "children"),
              Output("false-negative", "children"),
              Output("true-negative", "children"),
              Input("results", "clickData"))
def update_matrix(click_data):

    model, settings, test_features, test_labels = get_globals()

    if click_data is None:
        return ["Select a point!"] * 4
    if model is None or settings is None or test_features is None or test_labels is None:
        return ["Updating"] * 4

    # print("Callback triggered!")
    # print("model:", model)
    # print("test_features:", test_features)
    # print("test_labels:", test_labels)
    # print("settings:", settings)

    test_features = test_features[["base_total", "height_m", "weight_kg"]]

    print("test_features:", test_features)

    # epoch = click_data["points"][0]["x"]

    y_probs = model.predict(test_features)
    y_pred = (y_probs > settings.classification_threshold).astype(int)
    y_true = test_labels
    cm = confusion_matrix(y_true, y_pred)

    a, b, c, d = cm.ravel()
    return str(a), str(b), str(c), str(d)
