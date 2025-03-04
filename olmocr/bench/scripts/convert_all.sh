#!/bin/bash

set -e

# Assuming olmocr env already exists
source activate olmocr
python -m olmocr.bench.convert olmocr --repeats 5

pip install marker-pdf
python -m olmocr.bench.convert marker

pip install verovio
python -m olmocr.bench.convert gotocr

python -m olmocr.bench.convert chatgpt


#python -m olmocr.bench.convert mineru