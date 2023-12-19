import datetime
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import typer
from pandas import DataFrame
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slice_dataframe, slicing_function
from typing_extensions import Annotated

from attrition_pred import predict, utils
from attrition_pred.config import logger
from attrition_pred.data import CLASS_TO_INDEX, preprocess, preprocess_test
from attrition_pred.models import CustomRandomForestClassifier

# Initialize Typer CLI app
app = typer.Typer()


def get_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:  # pragma: no cover, eval workload
    """Get overall performance metrics.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.

    Returns:
        Dict: overall metrics.
    """
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    overall_metrics = {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": np.float64(len(y_true)),
    }
    return overall_metrics


def get_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_to_index: dict) -> dict:  # pragma: no cover, eval workload
    """Get per class performance metrics.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        class_to_index (Dict): dictionary mapping class to index.

    Returns:
        Dict: per class metrics.
    """
    per_class_metrics = {}
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(class_to_index):
        if i < len(metrics[0]):  # Test class is less than all classes in org data
            per_class_metrics[_class] = {
                "precision": metrics[0][i],
                "recall": metrics[1][i],
                "f1": metrics[2][i],
                "num_samples": np.float64(metrics[3][i]),
            }
    sorted_per_class_metrics = OrderedDict(sorted(per_class_metrics.items(), key=lambda tag: tag[1]["f1"], reverse=True))
    return sorted_per_class_metrics


@slicing_function()
def research_pos(x):
    """Look At research Department with attrition"""
    reseach_dep_attrition = "Yes" in x.Attrition
    terms = ["research"]
    research_deps = any(s.lower() in x.Department.lower() for s in terms)
    return reseach_dep_attrition and research_deps


@slicing_function()
def old_population(x):
    yes_attrition = "Yes" in x.Attrition
    return (x.Age > 50) and yes_attrition


def get_slice_metrics(y_true: np.ndarray, y_pred: np.ndarray, df: DataFrame) -> dict:  # pragma: no cover, eval workload
    """Get performance metrics for slices.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        df (Dataset): Ray dataset with labels.
    Returns:
        Dict: performance metrics for slices.
    """
    slice_metrics = {}
    slices = PandasSFApplier([research_pos, old_population]).apply(df)
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            metrics = precision_recall_fscore_support(y_true[mask], y_pred[mask], average="micro")
            slice_metrics[slice_name] = {}
            slice_metrics[slice_name]["precision"] = metrics[0]
            slice_metrics[slice_name]["recall"] = metrics[1]
            slice_metrics[slice_name]["f1"] = metrics[2]
            slice_metrics[slice_name]["num_samples"] = len(y_true[mask])
    return slice_metrics


@app.command()
def evaluate(
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")] = None,
    dataset_loc: Annotated[str, typer.Option(help="dataset (with labels) to evaluate on")] = None,
    results_fp: Annotated[str, typer.Option(help="location to save evaluation results to")] = None,
) -> dict:
    """Evaluate on the holdout dataset.

    Args:
        run_id (str): id of the specific run to load from. Defaults to None.
        dataset_loc (str): dataset (with labels) to evaluate on.
        results_fp (str, optional): location to save evaluation results to. Defaults to None.

    Returns:
        Dict: model's performance metrics on the dataset.
    """
    # Load
    ds = pd.read_csv(dataset_loc)
    artifact_path = predict.get_artifact_path(run_id)
    model_weight_path = os.path.join(artifact_path, "model_weight", "random_forest_model.joblib")
    rf_model = CustomRandomForestClassifier()
    rf_model = rf_model.load_model(model_weight_path, rf_model.rf_params)

    # y_true
    encoder_path = os.path.join(artifact_path, "encoder", "encoder.joblib")
    test_X, test_Y = preprocess_test(ds, encoder_path, label=True)
    pred_Y = rf_model.predict(test_X)

    # Metrics
    metrics = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": run_id,
        "overall": get_overall_metrics(y_true=test_Y, y_pred=pred_Y),
        "per_class": get_per_class_metrics(y_true=test_Y, y_pred=pred_Y, class_to_index=CLASS_TO_INDEX),
        "slices": get_slice_metrics(y_true=test_Y, y_pred=pred_Y, df=ds),
    }
    logger.info(json.dumps(metrics, indent=2))
    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(d=metrics, path=results_fp)
    return metrics


if __name__ == "__main__":  # pragma: no cover, checked during evaluation workload
    app()
