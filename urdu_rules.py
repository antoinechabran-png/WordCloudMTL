"""Urdu-script normalization and conservative local modifier scope rules."""
import re
import unicodedata

BOUNDARY = '|'
STOPWORDS = set('یہ وہ اس اُس اِس ان اُن میں ہم تم آپ مجھے ہمیں اسے اسے کا کی کے کو سے پر تک لیے لئے ہے ہیں ہوں ہو تھا تھی تھے ہونا اور یا لیکن مگر بلکہ بھی تو ہی ایک کچھ جو کہ جب اگر کیونکہ بہت واقعی بالکل صرف'.split())
NEGATIVE = {'نہیں', 'نہ', 'مت'}
BARRIERS = {'اور', 'یا', 'لیکن', 'مگر', 'بلکہ', 'کیونکہ', 'اگر', BOUNDARY}
AUXILIARIES = {'ہے', 'ہیں', 'ہوں', 'ہو', 'تھا', 'تھی', 'تھے', 'ہونا'}
INTENSITY = {'زیادہ', 'کافی'}


def normalize_urdu(token):
    # Keep Urdu-specific letters (especially ے and ھ) distinct.
    token = unicodedata.normalize('NFKC', token).strip().lower()
    token = token.translate(str.maketrans({'ك': 'ک', 'ي': 'ی', '\u200c': '', '\u200d': ''}))
    token = re.sub(r'[\u064b-\u0652\u0670\u0640]', '', token)
    return re.sub(r'\s+', '_', token)


def lexical_or_boundary(token):
    token = normalize_urdu(token)
    chars = token.replace('_', '')
    return token if chars and all(unicodedata.category(c)[0] in 'LM' for c in chars) else BOUNDARY


def merge_modifiers(tokens):
    """Prefer the preceding predicate; never cross punctuation/conjunctions.

    'بہت' means very, and is deliberately not converted into excessive 'too'.
    Only explicit 'حد/ضرورت سے زیادہ' creates a positive too-expression.
    """
    tokens = [lexical_or_boundary(t) for t in tokens]
    consumed, modified = set(), {}
    # Remove the additive 'not only' construction before negation handling.
    for i in range(len(tokens) - 1):
        if tokens[i:i + 2] == ['نہ', 'صرف']:
            consumed.update((i, i + 1))

    def target(start, step):
        skipped = []
        for j in range(start + step, start + step * 6, step):
            if not 0 <= j < len(tokens):
                break
            word = tokens[j]
            if word in BARRIERS or word in NEGATIVE or j in consumed or j in modified:
                break
            if word in STOPWORDS or word in INTENSITY:
                skipped.append(j)
                continue
            return j, skipped
        return None, skipped

    for i, word in enumerate(tokens):
        if word not in NEGATIVE or i in consumed:
            continue
        direction = 1 if word == 'مت' else -1
        j, skipped = target(i, direction)
        if j is None:
            j, skipped = target(i, -direction)
        if j is None:
            continue  # retain an unresolved marker
        modifiers = [k for k in skipped if tokens[k] in INTENSITY]
        if j > 0 and tokens[j - 1] in INTENSITY and j - 1 not in consumed:
            modifiers.append(j - 1)
        kind = 'notenough' if any(tokens[k] == 'کافی' for k in modifiers) else (
            'nottoo' if any(tokens[k] == 'زیادہ' for k in modifiers) else 'not')
        modified[j] = kind + '_' + tokens[j]
        consumed.add(i)
        consumed.update(modifiers)
        # Absorb the explicit excess qualifier when negating it.
        for k in modifiers:
            if k >= 2 and tokens[k - 2:k] in (['حد', 'سے'], ['ضرورت', 'سے']):
                consumed.update((k - 2, k - 1))

    for i in range(len(tokens) - 3):
        if tokens[i:i + 3] not in (['حد', 'سے', 'زیادہ'], ['ضرورت', 'سے', 'زیادہ']):
            continue
        if any(k in consumed for k in range(i, i + 3)):
            continue
        j, _ = target(i + 2, 1)
        if j is not None:
            modified[j] = 'too_' + tokens[j]
            consumed.update(range(i, i + 3))
    return [modified.get(i, word) for i, word in enumerate(tokens) if i not in consumed]
