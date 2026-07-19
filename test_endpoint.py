import boto3
import json

client = boto3.client("sagemaker-runtime", region_name="us-east-1")

response = client.invoke_endpoint(
    EndpointName="finbert-nse-sentiment-serverless",
    ContentType="application/json",
    Body=json.dumps({"inputs": "NSE Nifty hits record high on strong FII inflows"}),
)

result = json.loads(response["Body"].read())
print(result)