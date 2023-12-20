from pathlib import Path

import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

CLASS_TO_INDEX = {"Yes": 1, "No": 0}


def load_data(dataset_loc: str) -> DataFrame:
    """Load data from source into a Ray Dataset.

    Args:
        dataset_loc (str): Location of the dataset.

    Returns:
        Dataset: Our dataset represented as a pandas dataframe
    """
    ds = pd.read_csv(dataset_loc)
    ds = ds.sample(frac=1, random_state=1234).reset_index(drop=True)
    return ds


def stratify_split(df: DataFrame, val_size: int = 0.2, test_size: int = 0.5) -> list[DataFrame]:
    """Split a dataset into train, val and test splits with equal
    amounts of data points from each class in the column we
    want to stratify on.

    Args:
        df (DataFrame): full dataset represented as DataFrame
        val_size (int, optional): Percentatge of data allocated to val and test. Defaults to 0.2.
        test_size (int, optional): Split of val and test. Defaults to 0.5.

    Returns:
        DataFrame: tran, val, and test data frames
    """
    train_df, temp_df = train_test_split(df, stratify=df["Attrition"], test_size=val_size, random_state=2023)
    val_df, test_df = train_test_split(temp_df, stratify=temp_df["Attrition"], test_size=test_size, random_state=2023)
    return train_df, val_df, test_df


def preprocess(attrition: DataFrame, save_encoder=True) -> DataFrame:
    """Preprocess Data

    Args:
        attrition (DataFrame): dataset that needs to be preprocessed
        save_encoder (DataFrame): Save Encoder for later test use. default=True

    Returns:
        list[DataFrame]: datamatrix and target represented as DataFrame
    """
    # Define a dictionary for the target mapping
    target_map = {"Yes": 1, "No": 0}
    # Use the pandas apply method to numerically encode our attrition target variable
    target = attrition["Attrition"].apply(lambda x: target_map[x])

    attrition.drop(columns=["Attrition"], inplace=True)

    # Empty list to store columns with categorical data
    categorical = []
    for col, value in attrition.items():
        if value.dtype == "object":
            categorical.append(col)

    # Store the numerical columns in a list numerical
    numerical = attrition.columns.difference(categorical)

    # Store the categorical data in a dataframe called attrition_cat
    attrition_cat = attrition[categorical]

    # Store the numerical features to a dataframe attrition_num
    attrition_num = attrition[numerical]

    ohe = OneHotEncoder()

    # Step 2: Fit the encoder on the extracted categorical columns
    ohe.fit(attrition_cat)

    feature_names = ohe.get_feature_names_out()

    if save_encoder:  # pragma: no cover, basic save
        joblib.dump(ohe, "../model/saved_encoder/encoder.joblib")  # pragma: no cover, basic save

    # Step 4: Transform the original DataFrame
    df_encoded = ohe.transform(attrition_cat).toarray()

    # If you want to create a DataFrame with feature names
    df_encoded_cat = pd.DataFrame(df_encoded, columns=feature_names)

    # Reset indices if needed
    attrition_num = attrition_num.reset_index(drop=True)
    df_encoded_cat = df_encoded_cat.reset_index(drop=True)

    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    attrition_final = pd.concat([attrition_num, df_encoded_cat], axis=1)

    # attrition_final['target'] = target
    return attrition_final, target


def preprocess_test(attrition: DataFrame, encoder_path, label=False) -> DataFrame:
    """Preprocess for Testinng

    Args:
        attrition (DataFrame): _description_
        encoder_path (_type_): _description_

    Returns:
        DataFrame: _description_
    """
    target = None
    attrition = attrition.copy()
    if label:
        # Define a dictionary for the target mapping
        target_map = {"Yes": 1, "No": 0}
        # Use the pandas apply method to numerically encode our attrition target variable
        target = attrition["Attrition"].apply(lambda x: target_map[x])
        attrition.drop(columns=["Attrition"], inplace=True)

    # Empty list to store columns with categorical data
    categorical = []
    for col, value in attrition.items():
        if value.dtype == "object":
            categorical.append(col)

    # Store the numerical columns in a list numerical
    numerical = attrition.columns.difference(categorical)

    # Store the categorical data in a dataframe called attrition_cat
    attrition_cat = attrition[categorical]

    # Store the numerical features to a dataframe attrition_num
    attrition_num = attrition[numerical]
    loaded_encoder = joblib.load(encoder_path)
    feature_names = loaded_encoder.get_feature_names_out()

    # Step 4: Transform the original DataFrame
    df_encoded = loaded_encoder.transform(attrition_cat).toarray()

    # If you want to create a DataFrame with feature names
    df_encoded_cat = pd.DataFrame(df_encoded, columns=feature_names)

    # Reset indices if needed
    attrition_num = attrition_num.reset_index(drop=True)
    df_encoded_cat = df_encoded_cat.reset_index(drop=True)

    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    attrition_final = pd.concat([attrition_num, df_encoded_cat], axis=1)
    out = (attrition_final, target) if label else attrition_final
    return out
