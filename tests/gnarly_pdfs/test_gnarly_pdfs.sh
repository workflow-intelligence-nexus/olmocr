#/usr/bin/bash

set -ex

python -m olmocr.pipeline ./localworkspace --pdfs tests/gnarly_pdfs/*.pdf \
  && pytest tests/gnarly_pdfs/test_gnarly_pdfs.py
