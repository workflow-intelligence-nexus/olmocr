# pdelfin

Toolkit for training language models to work with PDF documents in the wild.

<img src="https://github.com/user-attachments/assets/984a645c-096d-4b9a-9c5b-44063004cd8c" alt="image" width="300"/>


What is included:
 - A prompting strategy to get really good natural text parsing using ChatGPT 4o [buildsilver.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/data/buildsilver.py)
 - An eval toolkit for comparing different pipeline versions [runeval.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/eval/runeval.py)
 - Basic filtering by language and SEO spam removal [filter.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/filter/filter.py)
 - Finetuning code for Qwen2-VL (and soon other VLMs) [train.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/train/train.py)
 - Processing millions of PDFs through a finetuned model using VLLM (requires [birr](https://github.com/allenai/mise/tree/main/birr)) [birrpipeline.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/birrpipeline.py)
 - Viewing Dolma Docs created from PDFs [dolmaviewer.py](https://github.com/allenai/pdelfin/blob/main/pdelfin/viewer/dolmaviewer.py)

### Note: Poppler and Font installation

You will probably need to install poppler-utils and some fonts on your computer so that any pdfs you render come out looking nice.

```
sudo apt-get install poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

```


### TODOs for future versions
 - Equations could be specified to be in a more specific format (they are "LaTeX" now)
 - Ask model to predict footnotes in a structured format separately
 - Add training data for complex tables
 - More training augmentations to improve performance
 - Fix pages which are all-references sometimes rendering as empty-text
 
