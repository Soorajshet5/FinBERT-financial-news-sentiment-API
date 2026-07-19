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
except botocore.exceptions.ClientError as e:
    # If the endpoint does not exist, a ClientError will be raised
    exists = False

if exists:
    print(f"Endpoint {endpoint_name} exists — updating...")
    predictor = huggingface_model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=endpoint_name,
        update_endpoint=True,
    )
else:
    print(f"Creating new endpoint {endpoint_name}...")
    predictor = huggingface_model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=endpoint_name,
    )

print("Deployed:", predictor.endpoint_name)