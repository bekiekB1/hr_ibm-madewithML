import pandas as pd
import pytest

from attrition_pred import data


@pytest.fixture(scope="module")
def df():
    data = pd.read_csv("data/processed/testing_data.csv")
    df = pd.DataFrame(data)
    return df


def test_load_data(dataset_loc):
    result = data.load_data(dataset_loc)
    assert isinstance(result, pd.DataFrame)

    original_dataset = pd.read_csv(dataset_loc)
    assert result.shape[0] == original_dataset.shape[0]
    assert not result.equals(original_dataset)
    assert result.index.is_monotonic_increasing


def test_stratify_split(dataset_loc):
    df = pd.read_csv(dataset_loc)
    train_df, val_df, test_df = data.stratify_split(df, val_size=0.2, test_size=0.5)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert train_df["Attrition"].value_counts().min() > 0
    assert val_df["Attrition"].value_counts().min() > 0
    assert test_df["Attrition"].value_counts().min() > 0


def test_preprocess(dataset_loc):
    attrition = pd.read_csv(dataset_loc)
    result_data, result_target = data.preprocess(attrition, save_encoder=False)

    assert isinstance(result_data, pd.DataFrame)
    assert isinstance(result_target, pd.Series)
    assert len(result_data) == len(result_target)
    assert set(result_target.unique()) == {0, 1}


def test_preprocess_test(dataset_loc, encoder_path):
    attrition = pd.read_csv(dataset_loc)

    # Test without label
    attrition_no_label = attrition.drop(columns="Attrition")
    result_no_label = data.preprocess_test(attrition_no_label, encoder_path)
    assert isinstance(result_no_label, pd.DataFrame)

    # Test with label
    result_with_label = data.preprocess_test(attrition, encoder_path, label=True)
    assert isinstance(result_with_label, tuple)
    assert isinstance(result_with_label[0], pd.DataFrame)
    assert isinstance(result_with_label[1], pd.Series)
    assert len(result_with_label[0]) == len(result_with_label[1])
