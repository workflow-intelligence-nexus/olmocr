#/usr/bin/bash

set -ex

python -m olmocr.pipeline ./localworkspace --pdfs tests/gnarly_pdfs/ambiguous.pdf \
  && pytest tests/gnarly_pdfs/test_gnarly_pdfs.py
