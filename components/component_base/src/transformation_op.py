# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DISCLAIMER: This code is generated as part of the AutoMLOps output.

import argparse
import json
from kfp.dsl import executor

import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *

def transformation_op(
    project: str,
    location: str,
    X_train_dataset: Output[Dataset],
    X_test_dataset: Output[Dataset],
    y_train_dataset: Output[Dataset],
    y_test_dataset: Output[Dataset],
    current_year: int = 2025,  # Input parameter remains the same
    test_split_ratio: float = 0.1,  # Made split ratio a parameter
    random_state_seed: int = 42,  # Made random state a parameter
):
    """
    Loads Ames housing data, preprocesses features, splits into train/test sets,
    and saves them as separate Dataset artifacts (.npy format).
    """
    # === Imports inside the function ===
    # This is standard practice for KFP components

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # === Data Loading ===

    data_url = "https://raw.githubusercontent.com/melindaleung/Ames-Iowa-Housing-Dataset/master/data/ames%20iowa%20housing.csv"
    print(f"Loading data from: {data_url}")
    try:
        df = pd.read_csv(data_url)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise  # Re-raise the exception to fail the component
    # === Feature Selection ===
    # Keep only necessary features plus the target variable initially

    features_to_keep = ["LotFrontage", "LotArea", "YearBuilt", "SaleType", "SalePrice"]
    print(f"Selecting features: {features_to_keep}")
    # Select columns, creating a copy to avoid SettingWithCopyWarning later

    df = df[features_to_keep].copy()

    # === Preprocessing ===

    print("Starting preprocessing...")
    # Impute missing LotFrontage with the mean

    lot_frontage_mean = df["LotFrontage"].mean()
    df["LotFrontage"].fillna(lot_frontage_mean, inplace=True)
    print(f"Imputed 'LotFrontage' NaN with mean: {lot_frontage_mean:.2f}")

    # Log transform LotArea

    df["LotArea"] = np.log1p(df["LotArea"])
    print("Applied log1p transformation to 'LotArea'")

    # Create HouseAge feature

    df["HouseAge"] = current_year - df["YearBuilt"]
    df.drop("YearBuilt", axis=1, inplace=True)
    print(
        f"Created 'HouseAge' using current_year={current_year} and dropped 'YearBuilt'"
    )

    # Label Encode SaleType
    # Check if 'SaleType' column exists and handle potential errors

    if "SaleType" in df.columns:
        # Ensure 'SaleType' is treated as string/category before encoding if it contains NaNs or mixed types

        df["SaleType"] = df["SaleType"].astype(
            str
        )  # Convert to string to handle potential NaN/mixed types robustly
        label_encoder = LabelEncoder()
        df["SaleType"] = label_encoder.fit_transform(df["SaleType"])
        print("Label encoded 'SaleType'")
        # Optional: Save the encoder if needed downstream (e.g., using joblib to a separate Output[Artifact])
        # encoder_path = ... # define an Output[Artifact] for the encoder
        # joblib.dump(label_encoder, encoder_path)
    else:
        print("Warning: 'SaleType' column not found. Skipping encoding.")
    # === Define Features X and Target y ===

    target_column = "SalePrice"
    feature_columns = ["LotFrontage", "LotArea", "HouseAge", "SaleType"]

    # Ensure all expected feature columns exist after preprocessing

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing expected feature columns after preprocessing: {missing_features}"
        )
    print(f"Defining features (X) from columns: {feature_columns}")
    print(f"Defining target (y) from column: {target_column}")
    X = df[feature_columns].values  # Convert features to numpy array
    y = df[target_column].values  # Convert target to numpy array

    # === Split into Training and Testing Sets ===

    print(
        f"Splitting data with test_size={test_split_ratio} and random_state={random_state_seed}"
    )
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X, y, test_size=test_split_ratio, random_state=random_state_seed
    )

    # === Save Outputs to KFP Provided Paths ===
    # Instead of returning, save each array to the path provided by the Output[Dataset] artifact.
    # We'll save as .npy files, a common format for NumPy arrays. KFP will handle uploading these.

    print(f"Saving X_train to: {X_train_dataset.path}")
    np.save(X_train_dataset.path, X_train_np)
    # Add metadata (optional but good practice)

    X_train_dataset.metadata["npy_shape"] = X_train_np.shape
    X_train_dataset.metadata["description"] = "Training features"

    print(f"Saving X_test to: {X_test_dataset.path}")
    np.save(X_test_dataset.path, X_test_np)
    X_test_dataset.metadata["npy_shape"] = X_test_np.shape
    X_test_dataset.metadata["description"] = "Test features"

    print(f"Saving y_train to: {y_train_dataset.path}")
    np.save(y_train_dataset.path, y_train_np)
    y_train_dataset.metadata["npy_shape"] = y_train_np.shape
    y_train_dataset.metadata["description"] = "Training target"

    print(f"Saving y_test to: {y_test_dataset.path}")
    np.save(y_test_dataset.path, y_test_np)
    y_test_dataset.metadata["npy_shape"] = y_test_np.shape
    y_test_dataset.metadata["description"] = "Test target"

    print("Transformation component finished successfully.")

def main():
    """Main executor."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--executor_input', type=str)
    parser.add_argument('--function_to_execute', type=str)

    args, _ = parser.parse_known_args()
    executor_input = json.loads(args.executor_input)
    function_to_execute = globals()[args.function_to_execute]

    executor.Executor(
        executor_input=executor_input,
        function_to_execute=function_to_execute).execute()

if __name__ == '__main__':
    main()
