# Word cloud improvements — 2 November 2026

The existing project has been updated in place. This package is a portable copy.
Keep `app.py` and the new `text_analysis.py` together. Existing requirements.txt,
packages.txt and runtime.txt are unchanged; no additional production dependency
is required by the new phrase-processing module.

## Run

Install requirements as for the original app, then run:

```sh
streamlit run app.py
```

For an existing deployment, deploy both Python files and restart the app. Generate
the analysis again; results from an already-running older session lack the new
structural token stream.

## Changes

- Preserve “not too X” separately from “not X” and “not X enough”.
- Handle “not only” constructions in the six Latin-script spaCy languages.
- Require negative evidence before changing a post-position “enough” into
  “not enough” in the token-list language rules.
- Preserve punctuation as hard boundaries for negation and phrase extraction;
  preserve Stanza sentence boundaries. Conjunctions stop spaCy target searches.
- Detect adjacent bigrams and trigrams before stopword filtering. Selected
  grammatical connectors can occur inside a phrase (e.g. “easy to use”).
- Require support in distinct response rows, with an adjustable PMI threshold.
- Resolve overlaps by phrase length, response support, PMI, then position.
- Review detected phrases with counts and original-response examples; exclusions
  update the cloud immediately.
- Words / Phrases / Mixed views. Words preserves individual-token counts. Mixed
  replaces covered words with selected phrases. Phrases displays selected phrases
  only. Negation and intensity expressions remain atomic analytical tokens even
  in Words mode; a displayed expression may contain more words than its token count.
- Size by percentage of responses (default), total occurrences, or TF-IDF.
  Percentages use all rows in the selected product/group as the denominator,
  including rows with no surviving terms. Repetition within a row counts once.
- When English translations merge several labels, percentage sizing counts the
  union of matching rows, rather than adding percentages.
- Excel exports include response counts and percentages. Batch ZIP downloads are
  invalidated when relevant settings change.
- Retain short meaningful words instead of dropping them solely for length.
- Describe cloud colors as response co-occurrence clusters.

Phrase-generation settings require Generate again. Display mode, phrase exclusions,
sizing and existing appearance controls update without rerunning the NLP models.
The existing minimum word-occurrence threshold still applies independently.

## Validation

```sh
python -m unittest discover -s tests -v
python -X utf8 tests/smoke_app.py
python -X utf8 tests/smoke_generate.py
```

All 28 regression tests passed. Regression tests exercise phrase boundaries, overlap selection, distinct-response
support, counting, translated-label unions, short words, and negation rules.
English/French model tests use the pinned 3.8.0 small models and skip explicitly
when those models are unavailable. Streamlit smoke checks cover Excel ingestion,
Generate in English and French, display modes, exclusions, filters and batch
PNG/XLSX exports. The Generate test supplies a synthetic workbook in memory.

Validation was performed on local Python 3.12; the supplied deployment runtime
remains Python 3.11. The code uses syntax supported by Python 3.11.

## Limits

Negation still uses language-specific heuristics, not a complete grammatical scope
resolver. Complex coordination, idioms and double negatives need further labeled
examples. Full model pipelines were exercised for English and French; token-rule
regressions cover Thai, Indonesian, Arabic and Chinese, but all other language
models and translation model downloads were not exercised end to end.

Phrase labels use lemmas; original response examples help review grammatical or
domain-specific wording. PMI is a candidate association measure, not a confidence
probability. Punctuation handling is deliberately conservative, so punctuation-
containing expressions may be split. Colors still describe co-occurrence within
responses, not sentiment or semantic equivalence.
