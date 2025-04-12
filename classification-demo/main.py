# Imports --------------------------------------------------------------------------------------------------------------
from dash import Dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.tools as tls

import keras
from sklearn.utils.class_weight import compute_class_weight
import ml_edu.experiment
import ml_edu.results
import numpy as np

model = None
settings = None
test_features = None
test_labels = None

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME],
)


# Initialization and Data Processing -----------------------------------------------------------------------------------

def create_model(
        settings: ml_edu.experiment.ExperimentSettings,
        metrics: list[keras.metrics.Metric],
) -> keras.Model:
    """Create and compile a simple classification model."""
    model_inputs = [
        keras.Input(name=feature, shape=(1,))
        for feature in settings.input_features
    ]
    # Use a Concatenate layer to assemble the different inputs into a single
    # tensor which will be given as input to the Dense layer.
    # For example: [input_1[0][0], input_2[0][0]]

    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    model_output = keras.layers.Dense(
        units=1, name='dense_layer', activation=keras.activations.sigmoid
    )(concatenated_inputs)
    model = keras.Model(inputs=model_inputs, outputs=model_output)
    # Call the compile method to transform the layers into a model that
    # Keras can execute.  Notice that we're using a different loss
    # function for classification than for regression.
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            settings.learning_rate
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return model


def train_model(
        experiment_name: str,
        model: keras.Model,
        dataset: pd.DataFrame,
        labels: np.ndarray,
        settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    """Feed a dataset into the model in order to train it."""

    # Compute class weights to handle imbalance
    # print(str(labels[:5]) + " (test)")
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weights = dict(zip(classes, weights))

    # The x parameter of keras.Model.fit can be a list of arrays, where
    # each array contains the data for one feature.
    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
        class_weight=class_weights,
    )

    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )


pokemon_dataset_raw = pd.read_csv("../Task 2 - Problem to solve/pokemon.csv")
pokemon_dataset = pokemon_dataset_raw[[
    'name',
    'hp',
    'attack',
    'defense',
    'speed',
    'sp_attack',
    'sp_defense',
    'base_total',
    'height_m',
    'weight_kg',
    'capture_rate',
    'is_legendary'
]]


def compare_train_test(experiment: ml_edu.experiment.Experiment, test_metrics: dict[str, float]):
    print('Comparing metrics between train and test:')
    for metric, test_value in test_metrics.items():
        print('------')
        print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
        print(f'Test {metric}:  {test_value:.4f}')


# Functions ------------------------------------------------------------------------------------------------------------

def make_visual(sample_size, threshold):
    global model, settings, test_features, test_labels

    keras.utils.set_random_seed(200)
    # Create indices at the 80th and 90th percentiles
    number_samples = len(pokemon_dataset)
    index_80th = round(number_samples * 0.8)
    index_90th = index_80th + round(number_samples * 0.1)

    # Randomize order and split into train, validation, and test with a .8, .1, .1 split
    shuffled_dataset = pokemon_dataset.sample(frac=1, random_state=100)
    train_data = shuffled_dataset.iloc[0:index_80th]
    validation_data = shuffled_dataset.iloc[index_80th:index_90th]
    test_data = shuffled_dataset.iloc[index_90th:]
    label_columns = ['is_legendary']

    train_features = train_data.drop(columns=label_columns)
    train_labels = train_data['is_legendary'].to_numpy()
    validation_features = validation_data.drop(columns=label_columns)
    validation_labels = validation_data['is_legendary'].to_numpy()
    test_features = test_data.drop(columns=label_columns)
    test_labels = test_data['is_legendary'].to_numpy()

    input_features = [
        'base_total',
        'height_m',
        'weight_kg'
    ]

    # Let's define our first experiment settings.
    settings = ml_edu.experiment.ExperimentSettings(
        learning_rate=0.005,  # 0.001
        number_epochs=60,
        batch_size=sample_size,
        classification_threshold=threshold,  # 0.35
        input_features=input_features,
    )

    metrics = [
        keras.metrics.BinaryAccuracy(name='accuracy', threshold=settings.classification_threshold),
        keras.metrics.Precision(name='precision', thresholds=settings.classification_threshold),
        keras.metrics.Recall(name='recall', thresholds=settings.classification_threshold),
        keras.metrics.AUC(num_thresholds=100, name='auc'),
    ]

    # Establish the model's topography.
    model = create_model(settings, metrics)

    # Train the model on the training set.
    experiment = train_model(
        'baseline', model, train_features, train_labels, settings
    )

    print("Metrics history columns:", experiment.metrics_history.columns)
    print(experiment.metrics_history.head())

    metrics_df = experiment.metrics_history

    # Had some help from ChatGPT in converting the ml plot into a plotly figure.
    fig = go.Figure()

    for metric in ['accuracy', 'precision', 'recall']:
        if metric in metrics_df.columns:
            fig.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df[metric], mode='lines+markers', name=metric))

    fig.update_layout(title="Pokemon Graph", xaxis_title="Epochs", yaxis_title="Percentage", legend_title="Metric")

    return fig


def get_globals():
    return model, settings, test_features, test_labels
