#!/bin/bash 
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

# Publishes a message to a Pub/Sub topic to invoke the
# pipeline job submission service.
# This script should run from the AutoMLOps/ directory
# Change directory in case this is not the script root.

gcloud pubsub topics publish housingmodel-queueing-svc --message "$(cat pipelines/runtime_parameters/pipeline_parameter_values.json)"