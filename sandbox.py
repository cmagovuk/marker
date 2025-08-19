# %%
import os

# %%
# run script to convert files
# it's easier to run in command line as script already done
output_dir = "sample/"
workers = 4
format_lines = ""  # "--format_lines"
use_llm = ""  # "--use_llm"
llm_service = ""  # "--llm_service=marker.services.bedrock.BedrockService"
llm_model = ""  # "--ollama_model=minicpm-v:8b"
profile = ""

# %%
file = "/notebooks/marker/Online_choice_architecture_discussion_paper.pdf"

command = (
    f"marker_single {file} --output_dir {output_dir} "
    # "--force_ocr "
    f"{format_lines} "
    f"{use_llm} {llm_service} {llm_model} "
    f"--bedrock_profile {profile} "
    "--debug "
    " --output_format markdown"
)
print(command)

# %%
os.system(command)

# %%
