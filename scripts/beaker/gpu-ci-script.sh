set -ex

git clone git@github.com:allenai/olmocr.git olmocr \
  && cd olmocr \
  && git checkout $GIT_REVISION \
  && /root/.local/bin/uv pip install --system --no-cache . \
  && /root/.local/bin/uv pip install --system --no-cache sgl-kernel==0.0.3.post1 --force-reinstall --no-deps \
  && /root/.local/bin/uv pip install --system --no-cache "sglang[all]==0.4.2" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
  && /root/.local/bin/uv pip install --system --no-cache pytest \
  && python -m olmocr.pipeline ./localworkspace --pdfs tests/gnarly_pdfs/*.pdf \
  && python tests/gnarly_pdfs/evaluate_gnarly_pdfs.py



