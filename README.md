# olmOCR

Toolkit for training language models to work with PDF documents in the wild.


<img src="https://github.com/user-attachments/assets/d70c8644-3e64-4230-98c3-c52fddaeccb6" alt="olmOCR Logo" width="300"/>
<br/>

Online demo: [https://olmocr.allen.ai/](https://olmocr.allen.ai/)

What is included:
 - A prompting strategy to get really good natural text parsing using ChatGPT 4o - [buildsilver.py](https://github.com/allenai/olmocr/blob/main/olmocr/data/buildsilver.py)
 - An side-by-side eval toolkit for comparing different pipeline versions - [runeval.py](https://github.com/allenai/olmocr/blob/main/olmocr/eval/runeval.py)
 - Basic filtering by language and SEO spam removal - [filter.py](https://github.com/allenai/olmocr/blob/main/olmocr/filter/filter.py)
 - Finetuning code for Qwen2-VL and Molmo-O - [train.py](https://github.com/allenai/olmocr/blob/main/olmocr/train/train.py)
 - Processing millions of PDFs through a finetuned model using Sglang - [pipeline.py](https://github.com/allenai/olmocr/blob/main/olmocr/pipeline.py)
 - Viewing [Dolma docs](https://github.com/allenai/dolma) created from PDFs - [dolmaviewer.py](https://github.com/allenai/olmocr/blob/main/olmocr/viewer/dolmaviewer.py)

### Installation

You will need to install poppler-utils and some additional fonts as a prerequisite. olmOCR uses poppler to render its PDF images.

Linux Ubuntu/Debian
```bash
sudo apt-get install poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```

Set up a conda environment, then clone and install the olmocr package
```bash
conda create -n olmocr python=3.11
conda activate olmocr

git clone https://github.com/allenai/olmocr.git
cd olmocr
pip install -e .
```

Finally, make sure you have sglang with [flashinfer](https://github.com/flashinfer-ai/flashinfer) installed if you want to run inference on your own GPU.
```bash
pip install sgl-kernel==0.0.3.post1 --force-reinstall --no-deps
pip install "sglang[all]==0.4.2" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```

**BETA TESTER NOTE:**

If you are a beta tester, you will need to login using the hugging-face CLI
to make sure you have access to https://huggingface.co/allenai/olmocr-preview
 
`huggingface-cli login`

### Local Usage Example

The easiest way to try out olmOCR on one or two PDFs is to check out the [web demo](https://olmocr.allen.ai/).

Once you are ready to run locally, a local GPU is required, as inference is powered by [sglang](https://github.com/sgl-project/sglang) 
under the hood.

This command will convert one PDF into a directory called `localworkspace`:
```bash
python -m olmocr.pipeline ./localworkspace --pdfs tests/gnarly_pdfs/horribleocr.pdf
```

You can also bulk convert many PDFS with a glob pattern:
```bash
python -m olmocr.pipeline ./localworkspace --pdfs tests/gnarly_pdfs/*.pdf
```

#### Viewing Results

Once that finishes, output is stored as [Dolma](https://github.com/allenai/dolma)-style JSONL inside of the `./localworkspace/results` directory.

```bash
cat localworkspace/results/output_*.jsonl  
```

You can view your documents side-by-side with the original PDF renders using the `dolmaviewer` command.

```bash
python -m olmocr.viewer.dolmaviewer localworkspace/results/output_*.jsonl
```

Now open `./dolma_previews/tests_gnarly_pdfs_horribleocr_pdf.html` in your favorite browser.

![image](https://github.com/user-attachments/assets/128922d1-63e6-4d34-84f2-d7901237da1f)


### Multi-node / Cluster Usage

If you want to convert millions of PDFs, using multiple nodes running in parallel, then olmOCR supports
reading your PDFs from AWS S3, and coordinating work using an AWS S3 output bucket.

For example, you can start this command on your first worker node, and it will set up
a simple work queue in your AWS bucket and start converting PDFs.

```bash
python -m olmocr.pipeline s3://my_s3_bucket/pdfworkspaces/exampleworkspace --pdfs s3://my_s3_bucket/jakep/gnarly_pdfs/*.pdf
```

Now on any subsequent nodes, just run this and they will start grabbing items from the same workspace queue. 
```bash
python -m olmocr.pipeline s3://my_s3_bucket/pdfworkspaces/exampleworkspace
```

If you are at AI2 and want to linearize millions of PDFs efficiently using [beaker](https://www.beaker.org), just add the `--beaker`
flag. This will prepare the workspace on your local machine, and then launch N GPU workers in the cluster to start
converting PDFs.

For example:
```bash
python -m olmocr.pipeline s3://my_s3_bucket/pdfworkspaces/exampleworkspace --pdfs s3://my_s3_bucket/jakep/gnarly_pdfs/*.pdf --beaker --beaker_gpus 4
```

### Full documentation for the pipeline

```bash
python -m olmocr.pipeline --help
usage: pipeline.py [-h] [--pdfs PDFS] [--workspace_profile WORKSPACE_PROFILE] [--pdf_profile PDF_PROFILE] [--pages_per_group PAGES_PER_GROUP]
                   [--max_page_retries MAX_PAGE_RETRIES] [--max_page_error_rate MAX_PAGE_ERROR_RATE] [--workers WORKERS] [--apply_filter] [--stats] [--model MODEL]
                   [--model_max_context MODEL_MAX_CONTEXT] [--model_chat_template MODEL_CHAT_TEMPLATE] [--target_longest_image_dim TARGET_LONGEST_IMAGE_DIM]
                   [--target_anchor_text_len TARGET_ANCHOR_TEXT_LEN] [--beaker] [--beaker_workspace BEAKER_WORKSPACE] [--beaker_cluster BEAKER_CLUSTER]
                   [--beaker_gpus BEAKER_GPUS] [--beaker_priority BEAKER_PRIORITY]
                   workspace

Manager for running millions of PDFs through a batch inference pipeline

positional arguments:
  workspace             The filesystem path where work will be stored, can be a local folder, or an s3 path if coordinating work with many workers, s3://bucket/prefix/

options:
  -h, --help            show this help message and exit
  --pdfs PDFS           Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths
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
  --apply_filter        Apply basic filtering to English pdfs which are not forms, and not likely seo spam
  --stats               Instead of running any job, reports some statistics about the current workspace
  --model MODEL         List of paths where you can find the model to convert this pdf. You can specify several different paths here, and the script will try to use the
                        one which is fastest to access
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


#### TODOs for future versions
 - Ask model to predict footnotes in a structured format separately
 - Add training data for complex tables
 - More training augmentations to improve performance
 - Fix pages which are all-references sometimes rendering as empty-text
 - Automated benchmarking
 - More efficient inference with 8-bit KV cache
 
