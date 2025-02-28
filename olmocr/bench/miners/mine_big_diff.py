# The point of this script is to sample documents from the s2pdf set
# Then, you sample a page, and OCR it with several "ground truth" models.
# We then save the results and once the number of requested pages are processed with all models,
# we go in look for the ones with the largest textual difference between them.


# Then, prompt again to generate the set of absent/present rules, given the diffs presented

# Then, run those rules through a tinyhost verification/edit system to quickly build up a big set
