# In this script, we assume that you have used the olmocr.bench.convert script to convert
# a number of PDFs, each perhaps number of repeat times.

# Now, we automatically search for some rules amongst these. Such that the chosen
# diff is correct amongst most of the "clean" model, and perhaps wrong most often


def merge_median_document(base: str, candidates: list[str]) -> str:
    # Split base into sentences using syntok
    # Find matching sentences using from fuzzysearch import find_near_matches from amongst candidates
    # For each sentence, we build a list of candidates, with their counts

    # Return a new document where for each sentence, we replace each sentence using the one which
    # appears most often inside of candidates