# pdelfin

Toolkit for training language models to work with PDF documents in the wild.

<img src="https://github.com/user-attachments/assets/984a645c-096d-4b9a-9c5b-44063004cd8c" alt="image" width="300"/>


What is included:
 - A prompting strategy to get really good natural text parsing using ChatGPT 4o - [buildsilver.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/data/buildsilver.py)
 - An eval toolkit for comparing different pipeline versions - [runeval.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/eval/runeval.py)
 - Basic filtering by language and SEO spam removal - [filter.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/filter/filter.py)
 - Finetuning code for Qwen2-VL (and soon other VLMs) - [train.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/train/train.py)
 - Processing millions of PDFs through a finetuned model using Sglang - [beakerpipeline.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/beakerpipeline.py)
 - Viewing Dolma Docs created from PDFs - [dolmaviewer.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/viewer/dolmaviewer.py)

### Installation

You will need to install poppler-utils and then also some fonts on your computer so that any pdfs you render come out looking nice.

```bash
sudo apt-get install poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```

Then, clone and install the pdelfin package
```bash
git clone https://github.com/allenai/pdelfin.git
cd pdelfin
pip install -e .
```

You will also need to install the latest pypdf, which contains some fixes regarding processing PDF documents. Hopefully soon it will be included in the next full release.
```bash
pip install git+https://github.com/py-pdf/pypdf.git@9e0fce7b9810d3e09e2af66481ea3429c42e0d11
```

### Beaker Usage

If you want to linearize millions of PDFs efficiently using [beaker](https://www.beaker.org), follow these instructions.
This is the preferred method for best performance, and lets you get results quickly for iterating and debugging.

It also runs at 2,800+ tokens per second per H100 GPU.

For example:
```bash
python -m pdelfin.beakerpipeline s3://ai2-oe-data/[your username]/pdfworkspaces/[workspacename] --pdfs s3://ai2-oe-data/jakep/gnarly_pdfs/*.pdf --beaker
```

This will convert all the pdfs at `s3://ai2-oe-data/jakep/gnarly_pdfs/*.pdf` and output dolma formatted documents at `s3://ai2-oe-data/[your username]/pdfworkspaces/[workspacename]/results`

You can specify more GPUs with `--beaker_gpus [int]` to get through the work faster. You can also specify your workspace, and allowed beaker clusters to use.
With default settings, it should work fine on any available GPUs.


```bash
python -m pdelfin.beakerpipeline --help
usage: beakerpipeline.py [-h] [--pdfs PDFS] [--workspace_profile WORKSPACE_PROFILE] [--pdf_profile PDF_PROFILE] [--pages_per_group PAGES_PER_GROUP]
                         [--max_page_retries MAX_PAGE_RETRIES] [--max_page_error_rate MAX_PAGE_ERROR_RATE] [--workers WORKERS] [--stats]
                         [--model MODEL] [--model_max_context MODEL_MAX_CONTEXT] [--model_chat_template MODEL_CHAT_TEMPLATE]
                         [--target_longest_image_dim TARGET_LONGEST_IMAGE_DIM] [--target_anchor_text_len TARGET_ANCHOR_TEXT_LEN] [--beaker]
                         [--beaker_workspace BEAKER_WORKSPACE] [--beaker_cluster BEAKER_CLUSTER] [--beaker_gpus BEAKER_GPUS]
                         [--beaker_priority BEAKER_PRIORITY]
                         workspace

Manager for running millions of PDFs through a batch inference pipeline

positional arguments:
  workspace             The S3 path where work will be done e.g., s3://bucket/prefix/

options:
  -h, --help            show this help message and exit
  --pdfs PDFS           Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list
                        of pdf paths
  --workspace_profile WORKSPACE_PROFILE
                        S3 configuration profile for accessing the workspace
  --pdf_profile PDF_PROFILE
                        S3 configuration profile for accessing the raw pdf documents
  --pages_per_group PAGES_PER_GROUP
                        Aiming for this many pdf pages per work item group
  --max_page_retries MAX_PAGE_RETRIES
                        Max number of times we will retry rendering a page
  --max_page_error_rate MAX_PAGE_ERROR_RATE
                        Rate of allowable failed pages in a document, 1/250 by default
  --workers WORKERS     Number of workers to run at a time
  --stats               Instead of running any job, reports some statistics about the current workspace
  --model MODEL         List of paths where you can find the model to convert this pdf. You can specify several different paths here, and the script
                        will try to use the one which is fastest to access
  --model_max_context MODEL_MAX_CONTEXT
                        Maximum context length that the model was fine tuned under
  --model_chat_template MODEL_CHAT_TEMPLATE
                        Chat template to pass to sglang server
  --target_longest_image_dim TARGET_LONGEST_IMAGE_DIM
                        Dimension on longest side to use for rendering the pdf pages
  --target_anchor_text_len TARGET_ANCHOR_TEXT_LEN
                        Maximum amount of anchor text to use (characters)
  --beaker              Submit this job to beaker instead of running locally
  --beaker_workspace BEAKER_WORKSPACE
                        Beaker workspace to submit to
  --beaker_cluster BEAKER_CLUSTER
                        Beaker clusters you want to run on
  --beaker_gpus BEAKER_GPUS
                        Number of gpu replicas to run
  --beaker_priority BEAKER_PRIORITY
                        Beaker priority level for the job
```


### TODOs for future versions
 - Equations could be specified to be in a more specific format (they are "LaTeX" now)
 - Ask model to predict footnotes in a structured format separately
 - Add training data for complex tables
 - More training augmentations to improve performance
 - Fix pages which are all-references sometimes rendering as empty-text
 
