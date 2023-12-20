import json
import os
from pathlib import Path
from typing import Annotated, Any, Iterable
from urllib.parse import urlparse

import mlflow
import pandas as pd
import typer
from numpyencoder import NumpyEncoder

from attrition_pred.config import MLFLOW_TRACKING_URI, logger, mlflow
from attrition_pred.data import preprocess_test
from attrition_pred.models import CustomRandomForestClassifier

# TODO Save the label/onehot encoder and load with job lib for
# Inference to Test Set
"""
EXample:
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories="auto", handle_unknown="ignore")
X_train_encoded = encoder.fit_transform(X_train)

import joblib
joblib.dump(encoder, 'onehot_encoder.joblib')



# Load the saved encoder
loaded_encoder = joblib.load('onehot_encoder.joblib')

# Assuming X_test is your test data
X_test_encoded = loaded_encoder.transform(X_test)

"""

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# Initialize Typer CLI app
app = typer.Typer()


def decode(indices: Iterable[Any], index_to_class: dict) -> list:
    """Decode indices to labels.

    Args:
        indices (Iterable[Any]): Iterable (list, array, etc.) with indices.
        index_to_class (Dict): mapping between indices and labels.

    Returns:
        List: list of labels.
    """
    return [index_to_class[index] for index in indices]


def get_artifact_path(run_id: int) -> Path:
    artifact_dir = urlparse(mlflow.get_run(run_id).info.artifact_uri).path
    return artifact_dir


@app.command()
def get_best_run_id(experiment_name: str, metric: str, mode: str) -> str:  # pragma: no cover, mlflow logic
    """Get the best run_id from an MLflow experiment.

    Args:
        experiment_name (str): name of the experiment.
        metric (str): metric to filter by.
        mode (str): direction of metric (ASC/DESC).

    Returns:
        str: best run id from experiment.
    """
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} {mode}"],
    )
    run_id = sorted_runs.iloc[0]  # .run_id
    # run_id = run_id.split("\n")
    print(run_id["run_id"])
    # return run_id


@app.command()
def predict(
    unit_data: str,
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")],
) -> str:  # pragma: no cover, tested with inference workload
    """Predict Attrition given all attributes

    Args:
        unit_data (dict): test data
        run_id (str): id of the specific run to load from. Defaults to None.

    Returns:
        Dict: prediction results for the input data.
    """
    artifact_path = get_artifact_path(run_id)
    model_weight_path = os.path.join(artifact_path, "model_weight", "random_forest_model.joblib")
    rf_model = CustomRandomForestClassifier()
    rf_model = rf_model.load_model(model_weight_path, rf_model.rf_params)

    # y_true
    encoder_path = os.path.join(artifact_path, "encoder", "encoder.joblib")
    unit_data_json = json.loads(unit_data)
    unit_data_df = pd.DataFrame(unit_data_json)
    # unit_data_df = pd.DataFrame([unit_data])
    test_X = preprocess_test(unit_data_df, encoder_path)
    # pred_Y = rf_model.predict(test_X)
    results = rf_model.predict_proba(test_X)
    results = results.to_dict(orient="records")
    logger.info(json.dumps(results, cls=NumpyEncoder, indent=2))
    return results


if __name__ == "__main__":
    app()
