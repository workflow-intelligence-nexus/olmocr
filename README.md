# pdelfin

Toolkit for truly understanding PDF documents in the wild.

<img src="https://github.com/user-attachments/assets/984a645c-096d-4b9a-9c5b-44063004cd8c" alt="image" width="300"/>

Things supported:
 - A prompting strategy to get really good natural text parsing using ChatGPT 4o (silver_data)
 - An eval toolkit for comparing different pipeline versions
 - Basic filtering by language and SEO spam removal
 - Finetuning code for Qwen2-VL (and soon other VLMs)

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
 
