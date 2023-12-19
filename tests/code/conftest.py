import pytest


@pytest.fixture
def dataset_loc():
    return "data/processed/HR_Employee_data.csv"


@pytest.fixture
def encoder_path():
    return "model/saved_encoder/encoder.joblib"
