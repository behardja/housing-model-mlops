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

def model_registry_op(
    project: str,
    location: str,
    model_gcs_uri: str,
    serving_container_image: str, # Full URI of the serving container
    model_display_name: str = "housing-model-registered", # Default display name, can be overridden
) -> NamedTuple("RegistryOutputs", [("model_resource_name", str), ("model_version_id", str)]):
    """
    Uploads a trained model to Vertex AI Model Registry.
    """
    import os
    from google.cloud import aiplatform
    from collections import namedtuple

    aiplatform.init(project=project, location=location)

    # The artifact_uri for Model.upload should be the GCS *directory*
    # containing the model artifact. model_gcs_uri is the path to the file itself.
    upload_artifact_uri = os.path.dirname(model_gcs_uri)

    # aiplatform.Model.upload will create a new model or a new version
    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=upload_artifact_uri, # GCS directory containing the model
        serving_container_image_uri=serving_container_image,
        serving_container_predict_route="/predict", 
        serving_container_health_route="/health",   
        serving_container_ports=[8080],             
        project=project,
        location=location,
    )

    RegistryOutputs = namedtuple("RegistryOutputs", ["model_resource_name", "model_version_id"])
    return RegistryOutputs(
        model_resource_name=uploaded_model.resource_name,
        model_version_id=uploaded_model.version_id
    )

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
