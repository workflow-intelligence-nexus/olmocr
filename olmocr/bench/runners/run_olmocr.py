import sys
import glob
import asyncio
import olmocr.pipeline

# Set sys.argv as if you were running the script from the command line.
sys.argv = [
    "pipeline.py",              # The script name (can be arbitrary)
    "olmocr/bench/sample_data/olmocr/workspace",            # Positional argument: workspace
    "--pdfs", *list(glob.glob("olmocr/bench/sample_data/pdfs/*.pdf")),  # PDF paths
]

# Call the async main() function.
asyncio.run(olmocr.pipeline.main())

