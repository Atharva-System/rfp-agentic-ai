# RFPAI

Proof of concept for AI Agents.

## Setup

Requires Python 3.12. PyEnv can be used to manage local python versions.

Setting up environment and installing dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

.env file:

```
MODEL=bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0

OPENAI_MODEL=model_name
OPENAI_KEY=<Api-Key>

AWS_ACCESS_KEY_ID=<Access key for dev environment user>
AWS_SECRET_ACCESS_KEY=<Secret access key for dev environment user>
AWS_REGION_NAME=us-east-1
AWS_DEFAULT_REGION=us-east-1

AWS_S3_BUCKET_NAME=<Bucket for knowledge base>
AWS_OPENSEARCH_COLLECTION_ARN=<OpenSearch collection for knowledge base>
AWS_KNOWLEDGE_BASE_ROLE_ARN=<Knowledge base role used by Bedrock>
AWS_EMBEDDING_MODEL_ARN=arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0
AWS_EMBEDDING_MODEL_DIMENSIONS=1024
AWS_ENVIRONMENT=Development
AWS_PROJECT=rfpai-agents
AWS_OPENSEARCH_SERVERLESS_COLLECTION_HOST=<OpenSearch Serverless host for collection>

pp_bucket=<Past Performace S3 Bucket Location>
RFP_S3_BUCKET=
RFP_S3_BASE_FOLDER=
```

Starting FastAPI server:

```sh
fastapi dev src/serve.py
```
