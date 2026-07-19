import os
import sys
import botocore
import sagemaker
from importlib import util

try:
    from sagemaker.huggingface.model import HuggingFaceModel
    from sagemaker.serverless import ServerlessInferenceConfig
except ImportError as e:
    hf_spec = util.find_spec("sagemaker.huggingface")
    print("Required SageMaker submodules not found:", e)
    print(f"sagemaker package loaded from: {sagemaker.__file__}")
    print(f"sagemaker.huggingface available? {'yes' if hf_spec else 'no'}")
    print("This usually means you are using SageMaker SDK 3.x, which does not include the v2 HuggingFace integration.")
    print("Install a v2-compatible SageMaker SDK instead: pip install 'sagemaker>=2.185.0,<3'")
    sys.exit(1)

# Ensure role is provided
role = os.environ.get("SAGEMAKER_ROLE_ARN")
if not role:
    print("Environment variable SAGEMAKER_ROLE_ARN is not set. Set it before running deploy.py")
    sys.exit(1)

hub = {
    'HF_MODEL_ID': 'soorajshet5/finbert-nse-sentiment',
    'HF_TASK': 'text-classification',
}

huggingface_model = HuggingFaceModel(
    env=hub,
    role=role,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
)

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=3072,
    max_concurrency=5,
)

endpoint_name = "finbert-nse-sentiment-serverless"

sm_client = sagemaker.Session().sagemaker_client

try:
    sm_client.describe_endpoint(EndpointName=endpoint_name)
    exists = True
except botocore.exceptions.ClientError:
    exists = False

if exists:
    # v2 SDK's update_endpoint path assumes a real-time endpoint with an
    # instance_type string, which serverless configs don't have (instance_type
    # is None). That crashes with "Failed to parse instance type 'None'".
    # Safe fix for serverless: delete the old endpoint, then deploy fresh.
    print(f"Endpoint {endpoint_name} exists — deleting before redeploy...")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    waiter = sm_client.get_waiter("endpoint_deleted")
    waiter.wait(EndpointName=endpoint_name)
    print("Old endpoint deleted.")

print(f"Deploying endpoint {endpoint_name}...")
predictor = huggingface_model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name=endpoint_name,
)

print("Deployed:", predictor.endpoint_name)