# from sklearn.ensemble import RandomForestClassifier


seed = 0  # We set our random seed to zero for reproducibility


# def get_rf_model() -> RandomForestClassifier:
#     # Random Forest parameters
#     rf_params = {
#         'n_jobs': -1,
#         'n_estimators': 1000,
#     #     'warm_start': True,
#         'max_features': 0.3,
#         'max_depth': 4,
#         'min_samples_leaf': 2,
#         'max_features' : 'sqrt',
#         'random_state' : seed,
#         'verbose': 0
#     }
#     rf = RandomForestClassifier(**rf_params)
#     return rf

from typing import Any, Dict, Optional

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


# #TODO: Change This to multiple models and ability to choose one
class CustomRandomForestClassifier:
    def __init__(self, rf_params: Optional[Dict[str, Any]] = None, smote_params: Optional[Dict[str, Any]] = None) -> None:
        self.rf_params = rf_params or {
            "n_jobs": -1,
            "n_estimators": 1000,
            "max_features": 0.3,
            "max_depth": 4,
            "min_samples_leaf": 2,
            "random_state": seed,  # Specify your seed here
            "verbose": 0,
        }
        self.smote_params = smote_params
        self.rf: Optional[RandomForestClassifier] = None

    def fit(self, train_X: pd.DataFrame, train_Y: pd.Series) -> None:
        if self.smote_params:
            oversampler = SMOTE(**self.smote_params)
            train_X, train_Y = oversampler.fit_resample(train_X, train_Y)

        self.rf = RandomForestClassifier(**self.rf_params)
        self.rf.fit(train_X, train_Y)

    def predict(self, val_X: pd.DataFrame) -> pd.Series:
        if self.rf is None:
            raise ValueError("Model has not been fitted yet.")
        return pd.Series(self.rf.predict(val_X))

    def predict_proba(self, val_X: pd.DataFrame) -> pd.DataFrame:
        if self.rf is None:
            raise ValueError("Model has not been fitted yet.")
        proba_array = self.rf.predict_proba(val_X)
        # print(self.rf.classes_)
        return pd.DataFrame(proba_array, columns=self.rf.classes_)

    def save_model(self, model_path: str) -> None:
        if self.rf is None:
            raise ValueError("Model has not been fitted yet.")
        joblib.dump(self.rf, model_path)

    def load_model(self, model_path: str, rf_params: Dict[str, Any], smote_params: Optional[Dict[str, Any]] = None) -> "CustomRandomForestClassifier":
        rf_params = rf_params or self.rf_params
        instance = CustomRandomForestClassifier(rf_params=rf_params, smote_params=smote_params)
        instance.rf = joblib.load(model_path)
        return instance


# Example Usage:
# custom_rf = CustomRandomForestClassifier()
# custom_rf.fit(train_X, train_Y)
# custom_rf.save_model('../model/saved_model/random_forest_model.joblib')
