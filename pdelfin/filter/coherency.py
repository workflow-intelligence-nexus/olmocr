# Uses a premade kenLM filter trained on good DCLM filtered web data to help identify pdfs where the 
# content has been very poorly parsed
import kenlm

from functools import lru_cache
from cached_path import cached_path

KENLM_S3_PATH = "s3://ai2-oe-data/jakep/kenlm-dclm/5gramtok.bin"

@lru_cache()
def load_kenlm():
    local_path = cached_path(KENLM_S3_PATH)
    model = kenlm.Model(local_path)

    return model


def get_document_coherency(text: str) -> float:
    model = load_kenlm()

    return model.score(text)