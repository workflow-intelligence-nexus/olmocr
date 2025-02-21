import sys
import glob
import json
import os
import shutil
import asyncio
import olmocr.pipeline

# Set sys.argv as if you were running the script from the command line.

workspace_dir = "olmocr/bench/sample_data/olmocr/workspace"

sys.argv = [
    "pipeline.py",              # The script name (can be arbitrary)
    "olmocr/bench/sample_data/olmocr/workspace",            # Positional argument: workspace
    "--pdfs", *list(glob.glob("olmocr/bench/sample_data/pdfs/*.pdf")),  # PDF paths
]

# Call the async main() function.
# asyncio.run(olmocr.pipeline.main())

# Now, take a produced jsonl files and unpack them into mds
for jsonl_path in glob.glob(workspace_dir + "/results/*.jsonl"):
    with open(jsonl_path, "r") as jsonl_f:
        for line in jsonl_f:
            data = json.loads(line)

            name = os.path.basename(data["metadata"]["Source-File"])

            with open(f"olmocr/bench/sample_data/olmocr/{name.replace('.pdf', '.md')}", "w") as out_f:
                out_f.write(data["text"])

shutil.rmtree(workspace_dir)