import json
from typing import Annotated, List, T

import boto3
import PIL
from pydantic import BaseModel

from marker.logger import get_logger
from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


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

    def validate_response(self, response_text: str, schema: type[T]) -> T:
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        try:
            # Try to parse as JSON first
            out_schema = schema.model_validate_json(response_text)
            out_json = out_schema.model_dump()
            return out_json
        except Exception:
            try:
                # Re-parse with fixed escapes
                escaped_str = response_text.replace("\\", "\\\\")
                out_schema = schema.model_validate_json(escaped_str)
                return out_schema.model_dump()
            except Exception:
                return

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = 0,
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

        prompt_config = prompt_config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": 0,
        }

        body = json.dumps(prompt_config)
        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                response = client.invoke_model(
                    body=body,
                    modelId=self.bedrock_model_name,
                    accept="application/json",
                    contentType="application/json",
                )

                response_body = json.loads(response.get("body").read())

                total_tokens = response_body.get("usage").get("output_tokens")
                if block:
                    block.update_metadata(
                        llm_request_count=1, llm_tokens_used=total_tokens
                    )
                # Extract response
                response_text = response_body.get("content")[0].get("text")
                # logger.info(f"response: {response_text}")
                return self.validate_response(response_text, response_schema)
            except Exception as e:
                logger.error(f"Error during Bedrock API call: {e}")
                break
