# %%
import boto3
import json
import base64
from PIL import Image

from marker.schema.blocks import Block
from marker.services import BaseService


# %%
class BedrockService(BaseService):
    bedrock_model_name: Annotated[
        str, "The name of the Bedrock model to use for the service."
    ] = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    bedrock_profile: Annotated[str, "Relevant AWS profile name"] = None
    max_tokens: Annotated[
        int, "The maximum number of tokens to use for a single request."
    ] = 8192
