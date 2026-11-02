# Vietnamese and Tagalog support

Both languages are now available under **Verbatim language**.

- **Vietnamese:** Stanza word segmentation with its identity lemma processor;
  word forms are retained. Model-recognized compound words are preserved.
  A source-span check prevents splitting inside a written syllable and separates
  known negation/intensity markers from their targets. Learned word boundaries
  can vary with context. Unicode is normalized to NFC before Stanza processing.
- **Tagalog:** spaCy's built-in `tl` tokenizer, with retained word forms and no
  trained POS tagger or lemmatizer. Basic mixed Tagalog/English negation is covered.
- Conservative stopwords, domain exclusions and phrase connectors for both.
- Rules distinguish negation, not-too, not-enough, excess, and not-only constructions.
  Unresolved markers are retained; punctuation and conjunctions stop target searches.
- Existing phrase review, bigrams/trigrams, weighting and export features apply.
- Optional English translation is configured using the official Helsinki-NLP
  `opus-mt-vi-en` and `opus-mt-tl-en` models. Translation model downloads/inference
  were not tested in this update; native-language processing was tested.

## Install / deploy

Keep **app.py**, **text_analysis.py**, and **language_rules.py** together.
Install the supplied requirements.txt, then restart Streamlit and click Generate.
Stanza now requires version 1.11 or later (under 2), for the verified Vietnamese
default_fast package. No new production package was added. Vietnamese downloads
its small tokenizer on first use; Tagalog needs no trained model download.

```sh
pip install -r requirements.txt
streamlit run app.py
```

## Checks

44 regression tests passed, including real spaCy English, French and Tagalog
tokenizer tests. A Streamlit integration check exercised synthetic Excel input,
Generate, phrases and displayed labels in both new languages, using the actual
Vietnamese Stanza model. Synthetic span tests cover syllable-boundary repair and
compound preservation. These examples are engineering regressions, not a native-
speaker evaluation or a measured language-accuracy benchmark.

```sh
python -m unittest discover -s tests -v
python -X utf8 tests/smoke_languages.py
```

The integration check downloads the Vietnamese tokenizer if needed; set
STANZA_RESOURCES_DIR to choose its cache location. Tests ran on Python 3.12;
production files were also parsed with Python 3.11 syntax rules.

Complex negation, idioms, inflection variants and extensive Taglish still need
review on representative responses. In particular, Tagalog word variants are
not automatically grouped into lemmas.

References:
- https://github.com/stanfordnlp/stanza-resources
- https://spacy.io/usage/models#languages
- https://huggingface.co/Helsinki-NLP/opus-mt-vi-en
- https://huggingface.co/Helsinki-NLP/opus-mt-tl-en
