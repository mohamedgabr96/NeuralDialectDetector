from azureml.core import Workspace
import azureml
from azureml.core.model import Model
import azureml.core
from azureml._restclient.artifacts_client import ArtifactsClient


model_name_registration = "neural_model_registration_2"

ws = Workspace.from_config(path=r"./config_aml.json")

service_name = 'dialect-detector-service'

path_to_folder_w_artifacts = r"./transformers_dummy_files"

model = Model.register(ws,  path_to_folder_w_artifacts, model_name_registration)

artifact = model._get_asset().artifacts[0]

print(artifact.prefix)
