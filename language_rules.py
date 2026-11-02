"""Conservative token rules for Vietnamese and basic Tagalog/Taglish support.

These rules do not claim trained Tagalog lemmatization or full negation scope.
Keep punctuation/clauses as barriers and preserve Vietnamese compound words.
"""
import re
import unicodedata

BOUNDARY = "|"
EXTRA_STOPWORDS = {
    "vi": set("tôi bạn chúng tôi chúng_tôi chúng_ta họ nó này đó những các một là và hoặc nhưng mà của với cho ở trong trên thì cũng đã đang sẽ rất thật khá được có như về khi đến từ cái việc còn chỉ".split()),
    "tl": set("ako ikaw siya kami tayo kayo sila ko mo niya namin natin ninyo nila ang mga ng sa na ay at o pero ngunit ito iyan iyon itong isang mga rin din naman po ho pa ba lang lamang dahil para mula nang kong mong niyang naming nating ninyong nilang the a an is are was were be it its and or but this that very really".split()),
}

CONFIG = {
    "vi": {
        "negative": {"không", "chẳng", "chả", "chưa"},
        "too": {"quá", "quá_mức"}, "enough": {"đủ"}, "insufficient": {"thiếu"},
        "skip": {"hề", "là", "có", "thật", "sự", "thật_sự", "rất", "phải"},
        "barriers": {"và", "hoặc", "nhưng", "mà", "vì", "nếu", "tuy"},
        "only": {"chỉ"},
        "expand": {"không_đủ", "chưa_đủ", "không_quá", "không_chỉ", "không_hề", "không_phải", "không_có"},
    },
    "tl": {
        "negative": {"hindi", "di", "'di", "wala", "walang", "huwag", "wag", "not", "no", "never", "n't"},
        "too": {"masyado", "masyadong", "sobra", "sobrang", "too", "overly"},
        "enough": {"sapat", "enough"}, "insufficient": {"kulang"},
        "skip": {"ang", "ng", "sa", "na", "talaga", "talagang", "gaanong", "gaano", "very", "really", "at", "all"},
        "barriers": {"at", "o", "pero", "ngunit", "dahil", "kung", "kahit", "and", "or", "but", "because"},
        "only": {"lang", "lamang", "only", "just"}, "expand": set(),
    },
}


def normalize_lexical_token(token, lang):
    token = unicodedata.normalize("NFC", token).lower().strip()
    # Stanza represents Vietnamese lexical words using internal spaces.
    if lang == "vi":
        token = re.sub(r"\s+", "_", token)
    return token


def lexical_or_boundary(token, lang):
    token = normalize_lexical_token(token, lang)
    if token in CONFIG[lang]["negative"]:
        return token
    chars = token.replace("_", "") if lang == "vi" else token
    return token if chars and all(unicodedata.category(c)[0] in "LM" for c in chars) else BOUNDARY


def vietnamese_sentence_tokens(sentence, text):
    """Respect model word boundaries only at whole written-syllable boundaries.

    Reconstruct from source spans, so model splits inside a syllable cannot
    introduce fragments or attach punctuation to a lexical item. Known modifier
    syllables stay separate even when the model groups them with their target.
    """
    words = list(sentence.words)
    if not words:
        return []
    start, end = words[0].start_char, words[-1].end_char
    if start is None or end is None:
        return [w.text for w in words]
    model_ends = {w.end_char for w in words if w.end_char is not None}
    cfg = CONFIG['vi']
    markers = cfg['negative'] | cfg['too'] | cfg['enough'] | cfg['insufficient'] | cfg['only']
    pieces = list(re.finditer(r'\w+|[^\w\s]', text[start:end], re.UNICODE))
    out, previous_end = [], None
    for match in pieces:
        piece = match.group()
        a, b = start + match.start(), start + match.end()
        gap = text[previous_end:a] if previous_end is not None else ''
        merge = (out and piece.isalpha() and out[-1].replace(' ', '').isalpha()
                 and gap.isspace() and '\n' not in gap
                 and not any(previous_end <= stop <= a for stop in model_ends)
                 and piece.lower() not in markers and out[-1].lower() not in markers)
        if merge:
            out[-1] += ' ' + piece
        else:
            out.append(piece)
        previous_end = b
    return out


def merge_language_modifiers(tokens, lang):
    cfg = CONFIG[lang]
    expanded = []
    for token in tokens:
        expanded.extend(token.split("_") if token in cfg["expand"] else [token])
    tokens = expanded
    markers = cfg["negative"] | cfg["too"] | cfg["enough"] | cfg["insufficient"]
    stops = EXTRA_STOPWORDS[lang]
    out, i = [], 0
    while i < len(tokens):
        token = tokens[i]
        if token in cfg["negative"] and i + 1 < len(tokens) and tokens[i + 1] in cfg["only"]:
            i += 2   # "hindi lamang", "không chỉ": additive construction, not negation
            continue
        if token not in cfg["negative"] | cfg["too"] | cfg["insufficient"]:
            out.append(token)
            i += 1
            continue
        negative = token in cfg["negative"]
        kind = "not" if negative else ("too" if token in cfg["too"] else "notenough")
        j = i + 1
        matched = False
        while j < len(tokens) and j <= i + 5:
            word = tokens[j]
            if word == BOUNDARY or word in cfg["barriers"] or word in cfg["negative"]:
                break
            if negative and word in cfg["too"]:
                kind = "nottoo"
            elif negative and word in cfg["enough"]:
                kind = "notenough"
            elif word in cfg["skip"]:
                pass
            elif word in markers or word in stops:
                break  # don't skip arbitrary subjects or grammatical structures
            else:
                end = j + 1
                if negative and end < len(tokens) and tokens[end] in cfg["enough"] and kind == "not":
                    kind, end = "notenough", end + 1
                # Preserve "not yet" / absence / prohibition wording when possible.
                prefix = token if kind == "not" and token in {"chưa", "wala", "walang", "huwag", "wag", "never"} else kind
                out.append(prefix + "_" + word)
                i = end
                matched = True
                break
            j += 1
        if not matched:
            if token in cfg["too"] and out and out[-1] != BOUNDARY and out[-1] not in stops | markers:
                if not out[-1].startswith(("not_", "nottoo_", "notenough_", "too_")):
                    out[-1] = "too_" + out[-1]
                    i += 1
                    continue
            # Successful matches set i beyond their target. Otherwise retain
            # the unresolved marker instead of silently dropping it.
            out.append(token)
            i += 1
    return out
