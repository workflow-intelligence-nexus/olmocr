# olmOCR-Bench

We develop olmOCR-Bench in order to automatically and effectively evaluate document-level
parsing and OCR of various tools.

olmOCR-Bench works by testing various "facts" or "properties" about document pages at the PDF-level.
Our intention is that each "fact" is very simple, unambiguous, and machine-checkable. 

We stay away from soft metrics like edit distance comparisons, because they may not correctly capture things like
two articles appearing on the same page. You want the text of each article to be grouped together, but the relative order of the two articles is less important. Also, some documents may have critical details, like switching x and y in an equation that can make all the difference in understanding the document, but would appear as just a small edit in an edit-distance metric.

We also choose PDFs directly, because PDFs do preserve some digital metadata and information which is helpful
and commonly available. Almost any other format can be converted to a PDF, but not the reverse.

## Test classes

- Text presence
  - This task makes sure that a given small piece of text (ex. 1-3 sentence level) is present within
    a parsed document. Soft/fuzzy matching is allowed, as well as specifying if the text must be in the first N or last N characters of the document. Case sensitive by default.
- Text absense
  - This task makes sure that a given piece of next does NOT appear in the OCR'ed version of a document. We generally want our OCR systems to filter out content like headers/footers/page numbers from documents. The same fuzzy matching as in Text Presence tests is allowed.
- Natural Reading Order
  - This task ensures that blocks of text which are present have a defined order relative to one another. For example,
  on a document that contains multiple news articles on one page, you'd want to see that the first sentence of the 
  first article appears after the heading of that article. But, you may be okay with swapping the order of those 
  two articles.
- Table Accuracy
  - Both Markdown and HTML based tables are supported. These tests check that a cell with a given text exists somewhere in the table, and that its neighbors have certain properties. Ex. A cell exists on this page with text "4.5%" and above that is a cell with the text "2.4%"
- Math Formula Accuracy
  - We render a given Latex style equation using Katex in a headless browser. And then see if it exists anywhere in the final OCRed document. Matching is performed on a relative symbol level, ex. in "\f\relax{x} = \int_{-\infty}^\infty
    x^2dx" we check that a ∫ appears to the left of a x, x appears to the left of dx, etc...
  
## Downloading and running the benchmark

Currently the full benchmark data is located here, but it's private until we are done reviewing and checking all of the tests:
https://huggingface.co/datasets/allenai/olmOCR-bench

To run a benchmark, first install the bench requirements
```bash
conda create -n olmocr python=3.11
conda activate olmocr

git clone https://github.com/allenai/olmocr.git
cd olmocr

pip install -e .[bench]
```

Convert your documents
```bash
# convert using a single OCR-engine, see the olmocr/bench/runners directory for options
python -m olmocr.bench.convert olmocr_pipeline --dir ./olmOCR-bench/bench_data

# or use convert_all.sh to run OCR with many common frameworks all at once, API keys will be required
./olmocr/bench/scripts/convert_all.sh
```

Now run the benchmark
```bash
python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data
```

## How were the tests made

Several categories of tests have been made so far:
1. arxiv_math -> We downloaded recent math papers from arxiv, filtered to those which had a single tex source file, and a rendered pdf, using https://github.com/allenai/olmocr/blob/main/olmocr/bench/miners/download_math.py. Then we matched up the text on a pdf page to the location in the tex source mostly likely to match to it using a dynamic programming matching algorithm in https://github.com/allenai/olmocr/blob/main/olmocr/bench/miners/mine_math.py. From there, Latex equations from the matching page were then parsed out, and we checked they rendered in Katex before adding them as test cases.
2. headers_footers -> We sampled documents from our internal crawled PDF repository. (The same from which olmOCR-mix was derived, though the likelyhood of duplicates is low, as there are 200M+ pdfs in this set). Then we used [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) to identify regions of the pages which were marked as headers/footers using the abandon category. We then got the text of those headers/footers regions by extracting them out and prompting Gemini, and we added them as test cases which should be absent. Manual review was then performed to remove mistakenly filtered text, and to set conditions such as limiting the search area to the first N or last N characters. Ex. if a page number "5" appears on the bottom a page, you want to test that your OCR system does not output a "5" in the last 20 characters of the page, but "5" could apepar earlier if in the actual body text.
3. table_tests -> We sampled documents from our internal crawled PDF repository, and found those which had tables using gemini-flash-2.0. https://github.com/allenai/olmocr/blob/main/olmocr/bench/miners/mine_tables_gemini.py On pages that had tables, we then further asked gemini-flash-2.0 to tell us the relationships between randomly chosen cells. Those tests were then manually checked.
4. multi_column -> We sampled documents from our internal crawled PDF repository manually, to find documents which had multi-column layouts and multiple articles on one page. Then, we used claude-sonnet-3.7 to render those pages to html, and from that html, we extracted text segments which were before/after one another. Then we manually reviewed each entry.
5. old_scans -> We sampled documents from the library of congress which contained handwriting or typewritten text. (TODO)
6. book_math -> We found old math textbooks in the public domain from the Internet Archive. We then extracted random pages from them, OCRed them, filtered down to pages which contained equations, and picked several random equations from each page to use as test cases. We then manually checked each test case to see that it was accurate capturing what was on the page. (TODO)
