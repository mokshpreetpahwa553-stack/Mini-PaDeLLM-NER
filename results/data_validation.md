# CoNLL-2003 data validation

Source: Hugging Face "lhoestq/conll2003"
SPLIT - Train, Validation, Test
SENTENCES - 14041, 3250, 3453
MENTIONS - 23499, 5942, 5648

Train mention count matches the paper’s reported 23,499 mentions.

- BIO tags were converted to (label, mention_text) pairs.
- Unit tests for the converter pass.
- The first training example was manually checked.
- Total mention counts were calculated for all three splits.