# %%
import os
import PIL
from PIL import Image
import boto3
import json
from io import BytesIO
import base64

# %%
# run script to convert files
# it's easier to run in command line as script already done
output_dir = "sample/"
workers = 4
format_lines = ""  # "--format_lines"
use_llm = "--use_llm"
llm_service = "--llm_service=marker.services.bedrock.BedrockService"
llm_model = ""  # "--ollama_model=minicpm-v:8b"

# %%
file = "/notebooks/marker/pdf.pdf"

command = (
    f"marker_single {file} --output_dir {output_dir} --force_ocr"
    f"{format_lines} "
    f"{use_llm} {llm_service} {llm_model}"
)
print(command)

# %%
os.system(command)

# %%
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="eu-west-2",
)


# %%
def img_to_base64(img: PIL.Image.Image):
    image_bytes = BytesIO()
    img.save(image_bytes, format="WEBP")
    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")


# %%
base64_string = img_to_base64(Image.open("a.png"))

# %%
prompt_config = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 4096,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_string,
                    },
                },
                {"type": "text", "text": "what is this image?"},
            ],
        }
    ],
    "temperature": 0.2,
}

bedrock_session = boto3.session.Session(profile_name="personal")

bedrock_runtime = bedrock_session.client(service_name="bedrock-runtime")

body = json.dumps(prompt_config)

response = bedrock_runtime.invoke_model(
    body=body,
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())

results = response_body.get("content")[0].get("text")

# %%
results

# %%
