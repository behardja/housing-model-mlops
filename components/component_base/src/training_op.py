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

def training_op(
    project: str,
    location: str,
    BUCKET_URI: str,
    X_train_dataset: Input[Dataset],
    X_test_dataset: Input[Dataset],
    y_train_dataset: Input[Dataset],
    y_test_dataset: Input[Dataset],
    pickled_model: Output[Model],
    filename: str = 'house_price_model.pkl',
) -> NamedTuple("Outputs", [("model_gcs_path", str),]):

    import numpy as np
    import pickle
    from sklearn.linear_model import LinearRegression
    from google.cloud import storage
    from collections import namedtuple

    # --- MODIFICATION START ---
    print(f"Loading X_train from base path: {X_train_dataset.path}")
    X_train = np.load(X_train_dataset.path + ".npy") # Append .npy

    print(f"Loading X_test from base path: {X_test_dataset.path}")
    X_test = np.load(X_test_dataset.path + ".npy")   # Append .npy

    print(f"Loading y_train from base path: {y_train_dataset.path}")
    y_train = np.load(y_train_dataset.path + ".npy") # Append .npy

    print(f"Loading y_test from base path: {y_test_dataset.path}")
    y_test = np.load(y_test_dataset.path + ".npy")   # Append .npy
    # --- MODIFICATION END ---

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    if X_test.shape[0] > 0:
        predictions = model.predict(X_test[:1])
        print(f"Sample prediction on first test instance: {predictions}")
    else:
        print("Test set is empty, skipping sample prediction.")

    with open(pickled_model.path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model pickled and saved to KFP managed path: {pickled_model.path}")
    pickled_model.metadata["framework"] = "scikit-learn"
    pickled_model.metadata["model_type"] = "LinearRegression"
    pickled_model.metadata["description"] = "House price prediction model."

    local_upload_filename = filename
    with open(local_upload_filename, 'wb') as f:
        pickle.dump(model, f)

    gcs_model_path_in_bucket = f"housing_models/{filename}"
    bucket_name = BUCKET_URI.replace("gs://", "").split('/')[0]

    print(f"Uploading {local_upload_filename} to gs://{bucket_name}/{gcs_model_path_in_bucket}")
    storage_client = storage.Client(project=project if project else None)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_model_path_in_bucket)
    blob.upload_from_filename(local_upload_filename)

    uploaded_model_gcs_uri = f"gs://{bucket_name}/{gcs_model_path_in_bucket}"
    print(f"Model additionally uploaded to: {uploaded_model_gcs_uri}")

    Outputs = namedtuple("Outputs", ["model_gcs_path"])
    return Outputs(model_gcs_path=uploaded_model_gcs_uri)

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
