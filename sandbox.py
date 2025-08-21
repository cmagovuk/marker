# %% [markdown]
# ### Test Bedrock connector
#
# Connector is in `marker/services/bedrock.py`
#
# This notebook will extract text from a single pdf file. To run it:
#
# 1. You will need an insecure GPU instance (e.g. g5).
# 2. You'll need to first install marker using `poetry install`.
# 3. Replace the file with a pdf file you want to test.
# 4. Replace the parameters below. If you want to run without LLM for comparison/benchmarking, repace all llm parameters with an empty string. You can also change the LLM model, it defaults to Claude 3. The profile is required for the LLM to run, you will need to use an AWS profile with permission to access Bedrock.
#
# If you want to get some logs about what the LLM is doing, `bedrock.py` contains some commented logger lines that can return the prompt, the schema and the output of the LLM. You can uncomment them, restart the kernel and rerun the notebook. The log will include the pydantic schema in json, the prompt as it appears in the processor modules, and the output of the LLM, including a corrected version of the input and its rationale for the change.
#
# The `--force_ocr` parameter will convert all the pdf to image and perform OCR, ignoring any text already in the pdf. Uncomment the parameter if you want to try it.
#
# The `--debug` flag will generate images for each page of the pdf, indicating how the page was segmented in blocks and how those blocks have been labelled. This is really useful for debugging so I recommend having it activated. It will still generate the normal output in markdown.

# %%
import os

# %%
# run script to convert files
# it's easier to run in command line as script already done
output_dir = "sample/"
use_llm = "--use_llm"
llm_service = "--llm_service=marker.services.bedrock.BedrockService"
llm_model = ""  # "--bedrock_profile=anthropic.claude-3-sonnet-20240229-v1:0"
profile = ""

# %%
file = "/notebooks/marker/sample-tables-single.pdf"

command = (
    f"marker_single {file} --output_dir {output_dir} "
    # "--force_ocr "
    f"{use_llm} {llm_service} {llm_model} "
    f"--bedrock_profile {profile} "
    "--debug "
    "--output_format markdown"
)
print(command)

# %%
os.system(command)

# %%
