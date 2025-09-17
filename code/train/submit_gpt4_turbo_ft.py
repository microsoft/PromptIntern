# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

SUBSCRIPTION_ID = "" # Subscription ID of the Azure subscription to use
RESOURCE_GROUP = "" # Resource group of the Azure ML workspace
WORKSPACE_NAME = "" # Name of the Azure ML workspace
COMPUTE_TARGET = ""

# We HAVE to set these environment variables before importing azureml-core in order to enable private features
# and be able to import components registered in the workspace
import os
os.environ["AZURE_ML_INTERNAL_COMPONENTS_ENABLED"] = "True"
os.environ["AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED"] = "true"

# Add CLI to identity scope
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline


def connect_to_aml(subscription_id, resource_group, workspace_name=None, registry_name=None):
    assert workspace_name is not None or registry_name is not None, "Either workspace_name or registry_name must be provided."
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential does not work
        credential = InteractiveBrowserCredential()

    ML_CLIENT = MLClient(
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
        credential=credential,
        registry_name=registry_name
    )

    return ML_CLIENT

# Connect to the workspace and OAI v2 registry
ml_client_registry_oai_v2 = connect_to_aml(subscription_id=SUBSCRIPTION_ID, resource_group=RESOURCE_GROUP, registry_name="azure-openai-v2-1p")
ml_client_workspace = connect_to_aml(subscription_id=SUBSCRIPTION_ID, resource_group=RESOURCE_GROUP, workspace_name=WORKSPACE_NAME)

# Get the components and datasets we need
dataset_train = ml_client_workspace.data.get(name="mbpp_train_",version="1") # TODO: Name your dataset name & version here 
# Requirement to provide validation set will be removed in the upcoming version
dataset_valid = ml_client_workspace.data.get(name="mbpp_valid_", version="1")
oai_data_import = ml_client_registry_oai_v2.components.get(name="openai_data_import", version="0.3.6")
ft_model = ml_client_registry_oai_v2.components.get(name="openai_completions_finetune", version="0.5.11")

#TODO: define the name you want to display in AML job page
@pipeline(
    name = "compression_ft_mbpp",
    display_name = "compression_ft_mbpp",
    description="",
    compute="serverless"
)
def ft_pipeline():

    # Feed train and validation set to data import component
    oai_data_import_step = oai_data_import(
        train_dataset=dataset_train,
        validation_dataset=dataset_valid
    )
    # Set the compute target to the Singularity A100 IPP cluster
    oai_data_import_step.compute = COMPUTE_TARGET
    oai_data_import_step.resources = {
        "instance_type": "Singularity.ND12am_A100_v4",
        "virtual_cluster_arm_id" : COMPUTE_TARGET,
        "instance_count": 1,
        "properties": {
            "singularity": {
                "slaTier": "Standard", # Basic, Standard, Premium
                "priority": "Medium" # Low, Medium, High
            }
        }
    }
    
    # Finetune the model
    ft_step = ft_model(
        input_dataset=oai_data_import_step.outputs.out_dataset,
        model="gpt-4", #TODO: name your model here
        task_type="chat",
        export_merged_weights=True,
        registered_model_name="gpt-4-mbpp-base", #TODO: name the model name after FTed e.g. gpt-4-<task name>-v#
        #TODO: define the parameters here 
        n_ctx=4096,
        lora_dim=32,
        n_epochs= 10,
        batch_size=-1,
        learning_rate_multiplier=1.0,
        weight_decay_multiplier=1e-05,
        prompt_loss_weight = 0.0,
        trim_mode = "right",
        shuffle_type = "none",
        checkpoint_interval=200,
        n_steps=1
    )
    # Set the compute target to the Singularity A100 IPP cluster
    ft_step.compute = COMPUTE_TARGET
    ft_step.resources = {
        "instance_type": "Singularity.ND96amrs_A100_v4",
        "virtual_cluster_arm_id" : COMPUTE_TARGET,
        "instance_count": 3,
        "properties": {
            "singularity": {
                "slaTier": "Standard", # Basic, Standard, Premium
                "priority": "Medium" # Low, Medium, High
            }
        }
    }
    return ft_step


def main():
    # Submit the pipeline job
    pipeline_job = ft_pipeline()
    pipeline_job = ml_client_workspace.jobs.create_or_update(
        pipeline_job, experiment_name="LLM_FT"
    )

    print("The url to see your live job running is returned by the sdk:")
    print(pipeline_job.services["Studio"].endpoint)

if __name__ == "__main__":
    main()
