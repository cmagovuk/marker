# %%
import json
from typing import Annotated, List

import boto3
import PIL
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService


# %%
class BedrockService(BaseService):
    bedrock_model_name: Annotated[
        str, "The name of the Bedrock model to use for the service."
    ] = "anthropic.claude-3-sonnet-20240229-v1:0"
    bedrock_profile: Annotated[str, "Relevant AWS profile name"] = ""
    region: Annotated[str, "AWS region name"] = "eu-west-2"
    max_tokens: Annotated[
        int, "The maximum number of tokens to use for a single request."
    ] = 8192

    def process_images(self, images: List[PIL.Image.Image]) -> List[dict]:
        """Process images so they can be included in input prompt."""
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/webp",
                    "data": self.img_to_base64(img),
                },
            }
            for img in images
        ]

    def get_client(self):
        """Initialise bedrock client."""

        bedrock_session = boto3.session.Session(profile_name=self.bedrock_profile)
        return bedrock_session.client(service_name="bedrock-runtime")

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        print("called")
        schema_example = response_schema.model_json_schema()
        system_prompt = f"""
Follow the instructions given by the user prompt.  You must provide your response in JSON format matching this schema:

{json.dumps(schema_example, indent=2)}

Respond only with the JSON schema, nothing else.  Do not include ```json, ```,  or any other formatting.
""".strip()

        client = self.get_client()
        image_data = self.format_image_for_llm(image)

        messages = [
            {
                "role": "user",
                "content": [
                    *image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            print(tries)
            try:
                response = client.messages.create(
                    system=system_prompt,
                    model=self.bedrock_model_name,
                    max_tokens=self.max_tokens,
                    messages=messages,
                    timeout=timeout,
                )
                # Extract and validate response
                response_text = response.content[0].text
                return response_text
            except Exception as e:
                print(f"Error during Bedrock API call: {e}")
                break
