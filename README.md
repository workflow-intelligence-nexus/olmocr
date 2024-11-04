# pdelfin

Toolkit for training language models to work with PDF documents in the wild.

<img src="https://github.com/user-attachments/assets/984a645c-096d-4b9a-9c5b-44063004cd8c" alt="image" width="300"/>


What is included:
 - A prompting strategy to get really good natural text parsing using ChatGPT 4o - [buildsilver.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/data/buildsilver.py)
 - An eval toolkit for comparing different pipeline versions - [runeval.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/eval/runeval.py)
 - Basic filtering by language and SEO spam removal - [filter.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/filter/filter.py)
 - Finetuning code for Qwen2-VL (and soon other VLMs) - [train.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/train/train.py)
 - Processing millions of PDFs through a finetuned model using VLLM (requires [birr](https://github.com/allenai/mise/tree/main/birr)) - [birrpipeline.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/birrpipeline.py)
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

### Batch Inference Usage

If you want run a fine tuned model in order to linearize millions of PDFs, you need to use the [birrpipeline.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/birrpipeline.py) script.

birrpipeline.py will take as input all of your PDFs (stored in S3), and generate the inputs needed to run those through your fine-tuned model.
After that, you will use [birr](https://github.com/allenai/mise/tree/main/birr) (part of mise) in order to run those batch inference files efficiently via VLLM.

You should expect somewhere between 1,400 to 1,800 tokens per second per H100 GPU.

```
usage: birrpipeline.py [-h] [--add_pdfs ADD_PDFS] [--target_longest_image_dim TARGET_LONGEST_IMAGE_DIM] [--target_anchor_text_len TARGET_ANCHOR_TEXT_LEN] [--workspace_profile WORKSPACE_PROFILE]
                       [--pdf_profile PDF_PROFILE] [--max_size_mb MAX_SIZE_MB] [--workers WORKERS] [--reindex] [--skip_build_queries]
                       workspace

Manager for running millions of PDFs through a batch inference pipeline

positional arguments:
  workspace             The S3 path where work will be done e.g., s3://bucket/prefix/)

options:
  -h, --help            show this help message and exit
  --add_pdfs ADD_PDFS   Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths
  --target_longest_image_dim TARGET_LONGEST_IMAGE_DIM
                        Dimension on longest side to use for rendering the pdf pages
  --target_anchor_text_len TARGET_ANCHOR_TEXT_LEN
                        Maximum amount of anchor text to use (characters)
  --workspace_profile WORKSPACE_PROFILE
                        S3 configuration profile for accessing the workspace
  --pdf_profile PDF_PROFILE
                        S3 configuration profile for accessing the raw pdf documents
  --max_size_mb MAX_SIZE_MB
                        Max file size in MB
  --workers WORKERS     Number of workers to run in the processpool
  --reindex             Reindex all of the page_results
  --skip_build_queries  Skip generation of new pdf page queries for batch inferencing
```

```bash
python -m pdelfin.birrpipeline [s3_workspace_path] --add_pdfs [s3_glob_path or path to file with s3 paths (one per line)]
```

For example:
```bash
python -m pdelfin.birrpipeline s3://ai2-oe-data/[your username]/pdfworkspaces/[workspacename] --pdf_profile s2 --add_pdfs s3://ai2-oe-data/jakep/gnarly_pdfs/*.pdf
```

After this runs the first time, you should have a whole bunch of json files generated in 

`s3://ai2-oe-data/[your username]/pdfworkspaces/[workspacename]/round_0/`

Now you need to run them using birr. 
You can use the [qwen2-vl-7b-pdf-weka.yaml](https://github.com/allenai/pdelfin/blob/main/scripts/birr/config/qwen2-vl-7b-pdf-weka.yaml) file here as a template for your birr config.
You will need to edit your queue name, priority level, etc.

```bash
mise birr create-queue -n [your_queue] --owner [your username] --project ai2-oe-data 

mise birr populate-queue -n [your_queue] "s3://ai2-oe-data/[your username]/pdfworkspaces/[workspacename]/inference_inputs/round_0/*.jsonl"

mise birr submit-job -c pdelfin/scripts/birr/config/qwen2-vl-7b-pdf-weka-customized.yaml
```

Once the batch inference job completes, you will want to run the birrpipeline again (witthout the --add_pdfs argument). This will index all of the 
batch inference files, and assemble dolma docs, which you can preview with [dolmaviewer.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/viewer/dolmaviewer.py)

Because of the nature of vlms, you will need to run multiple rounds of inference in order to convert the majority of your files. This is because
sometimes generation will fail due to repetition errors, (or if the pdf page was rotated incorrectly, the system will attempt to classify that and rotate it properly on
the next round). Usually 2 to 3 complete rounds is enough to get most of your files.


### TODOs for future versions
 - Equations could be specified to be in a more specific format (they are "LaTeX" now)
 - Ask model to predict footnotes in a structured format separately
 - Add training data for complex tables
 - More training augmentations to improve performance
 - Fix pages which are all-references sometimes rendering as empty-text
 
