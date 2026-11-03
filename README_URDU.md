# Urdu support — 3 November 2026

Select **Urdu** in the Verbatim language menu and Generate again.
This release includes all previous languages, including Vietnamese and Tagalog.

## Processing

- Stanza Urdu (`ur`) default_fast pipeline: tokenize, POS, lemma, loaded only on
  first Generate. Its model is downloaded automatically and cached for later use.
- Conservative Urdu stopwords, default domain exclusions and phrase connectors.
- Urdu-script normalization handles common Arabic keyboard variants of kaf/yeh,
  optional vowel marks and presentation forms, while retaining Urdu-specific letters.
- Local negation rules support common postposed negation, negated excessive
  intensity, insufficient intensity, and the additive “نہ صرف” construction.
- “بہت” (very) is not treated as excessive “too”. Positive excess requires an
  explicit “حد سے زیادہ” or “ضرورت سے زیادہ” expression.
- Urdu labels display negation after the target word. Unresolved negation markers
  remain visible rather than silently disappearing.
- Words/Phrases/Mixed modes, phrase exclusions, response percentages, product/group
  filters and PNG/XLSX batch exports use the existing app features.
- Right-to-left shaping uses Pillow/libraqm when available, otherwise the existing
  arabic-reshaper and python-bidi dependencies. Noto Nastaliq Urdu is discoverable
  on Linux/local fonts; Windows Arial is also available when installed. You can
  still upload your own font. Exported Excel text remains in logical Unicode order.
- Optional English translation is configured with Helsinki-NLP/opus-mt-ur-en.
  Translation model inference was not exercised during this update.

## Deploy

Deploy the full folder, including **all four Python modules**:

- app.py
- text_analysis.py
- language_rules.py
- urdu_rules.py

Use the supplied requirements.txt, packages.txt and runtime.txt as before.
No new production dependency was added for Urdu. Restart the app after deployment.
The first Urdu analysis needs network access to download the Stanza model.

## Validation

All 58 regression tests passed. Tests include the actual Urdu Stanza model,
normalization, negation, phrase connectors, and both RTL rendering code paths.
Streamlit integration checked a synthetic Excel workbook through Generate, local
labels, font glyph coverage, percentage counts and PNG/XLSX batch exports.
An exported PNG was visually checked for connected letters and right-to-left order.

```sh
python -m unittest discover -s tests -v
python -X utf8 tests/smoke_urdu.py
```

Set STANZA_RESOURCES_DIR to your model cache to include the Urdu model regression
test. Without a downloaded cache, that model test skips explicitly; the integration
test can download the model on first run. Local validation used Python 3.12 and
Stanza 1.15; production modules also pass Python 3.11 syntax parsing.

## Limits

This is for Urdu-script responses, not Roman Urdu. Negation scope remains
heuristic, particularly for complex clauses, idioms and mixed languages.
Lemmatized labels may lose grammatical agreement. Native-speaker review on your
own survey responses is still needed; these tests are not an accuracy benchmark.

Official model references:
- https://huggingface.co/stanfordnlp/stanza-ur
- https://huggingface.co/Helsinki-NLP/opus-mt-ur-en
