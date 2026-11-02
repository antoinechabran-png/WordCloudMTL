"""Sentence-aware phrase selection, independent of Streamlit and NLP models."""
from collections import Counter
import math

BOUNDARY = "|"
CONNECTORS = {
    "en": {"to", "of", "for", "in", "on", "with"},
    "fr": {"de", "du", "des", "à", "au", "en", "pour"},
    "es": {"de", "del", "a", "para", "con"},
    "it": {"di", "del", "da", "a", "per"},
    "de": {"zu", "zum", "zur", "von", "für", "mit"},
    "pt": {"de", "do", "da", "para", "com"},
    "zh": {"的"}, "ja": {"の"}, "th": {"ของ"},
    "vi": {"của", "cho", "với", "để"}, "tl": {"ng", "sa", "na", "para"},
    "id": {"untuk", "dengan"}, "ar": {"من", "في", "ب"},
}


def keep_word(token, stops):
    return token != BOUNDARY and token not in stops and any(c.isalpha() for c in token)


def extract_phrases(documents, stops, lang, min_responses=4, max_n=3, min_pmi=0.0):
    """Adjacent lemma phrases only; never bridge punctuation or removed tokens.

    Support counts distinct input rows. Internal grammatical connectors are
    allowed, but conjunctions/other stopwords cannot create artificial phrases.
    """
    counts, support, examples, unigrams = Counter(), Counter(), {}, Counter()
    totals = Counter()
    for row, tokens in enumerate(documents):
        unigrams.update(t for t in tokens if t != BOUNDARY)
        seen = set()
        for size in range(2, max_n + 1):
            for i in range(len(tokens) - size + 1):
                phrase = tuple(tokens[i:i + size])
                if BOUNDARY in phrase:
                    continue
                totals[size] += 1
                if not keep_word(phrase[0], stops) or not keep_word(phrase[-1], stops):
                    continue
                if any(t in stops and t not in CONNECTORS.get(lang, set()) for t in phrase[1:-1]):
                    continue
                counts[phrase] += 1
                seen.add(phrase)
                if len(examples.setdefault(phrase, [])) < 3 and row not in examples[phrase]:
                    examples[phrase].append(row)
        support.update(seen)
    total = sum(unigrams.values())
    results = {}
    for phrase, count in counts.items():
        if support[phrase] < min_responses:
            continue
        expected = math.prod(unigrams[t] / max(1, total) for t in phrase)
        pmi = math.log2((count / max(1, totals[len(phrase)])) / expected)
        if pmi >= min_pmi:
            results[phrase] = {"occurrences": count, "responses": support[phrase],
                               "pmi": pmi, "examples": examples[phrase]}
    return results


def select_terms(tokens, phrases, stops, mode="Mixed"):
    """Words retain their original counts. Mixed uses non-overlapping phrases.

    Resolve overlaps globally: longer supported phrases first, then response
    support, association, and position. Phrases mode uses the same selection.
    """
    if mode == "Words":
        return [t for t in tokens if keep_word(t, stops)]
    candidates = []
    for i in range(len(tokens)):
        for size in (2, 3):
            phrase = tuple(tokens[i:i + size])
            if len(phrase) == size and phrase in phrases:
                meta = phrases[phrase]
                candidates.append((-size, -meta["responses"], -meta["pmi"], i, phrase))
    occupied, chosen = set(), {}
    for _, _, _, i, phrase in sorted(candidates):
        span = set(range(i, i + len(phrase)))
        if not span & occupied:
            chosen[i] = "_".join(phrase)
            occupied.update(span)
    out = []
    for i, token in enumerate(tokens):
        if i in chosen:
            out.append(chosen[i])
        elif mode == "Mixed" and i not in occupied and keep_word(token, stops):
            out.append(token)
    return out


def response_statistics(documents):
    documents = list(documents)
    counts = Counter(t for doc in documents for t in set(doc))
    percentages = {t: 100.0 * n / len(documents) for t, n in counts.items()} if documents else {}
    return counts, percentages


def translated_weights(documents, labels, weights, percentage=False):
    """Count the union of responses when several terms share a display label."""
    if percentage:
        mapped = [[labels[t] for t in doc if t in labels] for doc in documents]
        return response_statistics(mapped)[1]
    result = Counter()
    for term, label in labels.items():
        result[label] += weights.get(term, 0)
    return dict(result)
