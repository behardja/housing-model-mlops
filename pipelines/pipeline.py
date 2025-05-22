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

"""Kubeflow Pipeline Definition"""

import argparse
from typing import *
import os

from google.cloud import storage
import kfp
from kfp import compiler, dsl
from kfp.dsl import *
import yaml

def upload_pipeline_spec(gs_pipeline_job_spec_path: str,
                         pipeline_job_spec_path: str,
                         storage_bucket_name: str):
    '''Upload pipeline job spec from local to GCS'''
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(storage_bucket_name)
    filename = '/'.join(gs_pipeline_job_spec_path.split('/')[3:])
    blob = bucket.blob(filename)
    blob.upload_from_filename(pipeline_job_spec_path)

def load_custom_component(component_name: str):
    component_path = os.path.join('components',
                                component_name,
                              'component.yaml')
    return kfp.components.load_component_from_file(component_path)

def create_training_pipeline(pipeline_job_spec_path: str):
    import_data_op = load_custom_component(component_name='import_data_op')
    transformation_op = load_custom_component(component_name='transformation_op')
    training_op = load_custom_component(component_name='training_op')
    model_registry_op = load_custom_component(component_name='model_registry_op')

    @dsl.pipeline(
        name='automlops-pipeline',
        description='placeholder',
    )
    def pipeline(
        PROJECT_ID: str,      
        LOCATION: str,        
        VERSION: str,         
        REPO_NAME: str,       
        JOB_IMAGE_ID: str,    
        BUCKET_URI: str,            
    ):

        import_data = import_data_op().set_display_name("Import Data Source")

        transformation = transformation_op(
            project=PROJECT_ID,
            location=LOCATION,
        ).after(import_data).set_display_name("Feature Transformation")

        training = training_op(
            project=PROJECT_ID,
            location=LOCATION,
            BUCKET_URI=BUCKET_URI,
            X_train_dataset=transformation.outputs["X_train_dataset"],
            X_test_dataset=transformation.outputs["X_test_dataset"],
            y_train_dataset=transformation.outputs["y_train_dataset"],
            y_test_dataset=transformation.outputs["y_test_dataset"],
        ).after(transformation).set_display_name("Model Training")

        # ADD A NEW COMPONENT THAT DOES TRANSOFRMATIONS
        
        model_registry = model_registry_op(
            project=PROJECT_ID,
            location=LOCATION,
            model_gcs_uri=training.outputs["model_gcs_path"],
            serving_container_image=f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{JOB_IMAGE_ID}:{VERSION}",
            model_display_name="housing_model"
        ).after(training).set_display_name("Register Model to Vertex AI")

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_job_spec_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                       help='The config file for setting default values.')

    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    create_training_pipeline(
        pipeline_job_spec_path=config['pipelines']['pipeline_job_spec_path'])

    upload_pipeline_spec(
        gs_pipeline_job_spec_path=config['pipelines']['gs_pipeline_job_spec_path'],
        pipeline_job_spec_path=config['pipelines']['pipeline_job_spec_path'],
        storage_bucket_name=config['gcp']['storage_bucket_name'])
