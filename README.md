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

```bash
python -m pdelfin.birrpipeline [s3_workspace_path] --add_pdfs [s3_glob_path or path to file with s3 paths (one per line)]
```



### TODOs for future versions
 - Equations could be specified to be in a more specific format (they are "LaTeX" now)
 - Ask model to predict footnotes in a structured format separately
 - Add training data for complex tables
 - More training augmentations to improve performance
 - Fix pages which are all-references sometimes rendering as empty-text
 
