import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from collections import Counter
import itertools
import hashlib
import re
import os
import io
import tempfile
import unicodedata
from wordcloud import WordCloud
import matplotlib
from text_analysis import BOUNDARY, extract_phrases, select_terms, response_statistics, translated_weights

try:
    import community as community_louvain  # python-louvain
except Exception:  # fall back to networkx's own Louvain implementation
    community_louvain = None

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Semantic Word Cloud", layout="wide")

# ─── NLTK ───────────────────────────────────────────────────────────────────
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("omw-1.4",  quiet=True)

# ─── Languages ───────────────────────────────────────────────────────────────
# "engine" decides how each language is tokenized/lemmatized:
#   spacy    → spaCy model installed via requirements.txt (loaded lazily, only
#              for the language actually selected)
#   jieba    → Chinese word segmentation (no spaces between words)
#   thai     → PyThaiNLP "newmm" segmentation (Thai has no spaces between
#              words either) + PyThaiNLP's bundled Thai stopword list
#   stanza   → Stanford Stanza (tokenize + mwt + pos + lemma). Used for Arabic
#              and Indonesian. NOT imported at startup: `import stanza` (and
#              torch behind it) only happens the first time one of these two
#              languages is analysed, and its model is downloaded at that
#              moment (first use only, then reused from disk/cache).
LANGUAGES = {
    "English":    {"code": "en", "nltk_stop": "english",    "spacy_model": "en_core_web_sm",  "engine": "spacy"},
    "French":     {"code": "fr", "nltk_stop": "french",     "spacy_model": "fr_core_news_sm", "engine": "spacy"},
    "Spanish":    {"code": "es", "nltk_stop": "spanish",    "spacy_model": "es_core_news_sm", "engine": "spacy"},
    "Italian":    {"code": "it", "nltk_stop": "italian",    "spacy_model": "it_core_news_sm", "engine": "spacy"},
    "German":     {"code": "de", "nltk_stop": "german",     "spacy_model": "de_core_news_sm", "engine": "spacy"},
    "Portuguese": {"code": "pt", "nltk_stop": "portuguese", "spacy_model": "pt_core_news_sm", "engine": "spacy"},
    "Chinese":    {"code": "zh", "nltk_stop": None,         "spacy_model": None,              "engine": "jieba"},
    "Japanese":   {"code": "ja", "nltk_stop": None,         "spacy_model": "ja_core_news_sm", "engine": "spacy"},
    "Thai":       {"code": "th", "nltk_stop": None,         "spacy_model": None,              "engine": "thai"},
    "Indonesian": {"code": "id", "nltk_stop": "indonesian", "spacy_model": None,              "engine": "stanza"},
    "Arabic":     {"code": "ar", "nltk_stop": "arabic",     "spacy_model": None,              "engine": "stanza"},
}

# ─── Negation ────────────────────────────────────────────────────────────────
# A negation is merged with the word it negates into ONE token, so
# "not fresh" is counted, clustered and drawn as a unit. Internally the token
# is "not_<lemma>" (spaCy languages, Chinese, Japanese) or "<negword>_<lemma>"
# (Thai / Indonesian / Arabic); display_label() shows it in the local
# language ("pas frais", "nicht frisch", "強くない", "tidak segar").
#
# spaCy languages: resolved on spaCy's tokens + part-of-speech tags (see
# spacy_negation_lemmas): intensifiers/determiners/auxiliaries between the
# negation and its word are skipped ("not very fresh" → not fresh,
# "ce n'est pas du tout frais" → pas frais), French "ne + verb + pas" negates
# the verb ("je n'aime pas" → pas aimer), and a negation with nothing after it
# negates the preceding verb/adjective ("riecht man nicht" / "j'aime pas").
# Japanese: ない / ません / ず attach BACKWARDS to the word they follow
# (強くない, 感じない, 好きじゃない, 匂いがない, 好きではありません).
NEGATION_WORDS = {
    # spaCy splits "don't" → do + n't, "can't" → ca + n't, "cannot" → can + not
    "en": ["not", "n't", "no", "never", "nor", "neither", "hardly", "don't", "doesn't", "didn't",
           "isn't", "aren't", "wasn't", "weren't", "can't", "won't", "wouldn't", "shouldn't", "couldn't"],
    "fr": ["pas", "jamais", "aucun", "aucune", "non", "sans"],
    "es": ["no", "nunca", "jamás", "ningún", "ninguna", "ninguno", "tampoco", "ni", "sin"],
    "it": ["non", "mai", "nessuno", "nessuna", "nessun", "niente", "né", "senza"],
    "de": ["nicht", "kein", "keine", "keinen", "keinem", "keiner", "keines", "nie", "niemals", "ohne"],
    "pt": ["não", "nunca", "jamais", "nenhum", "nenhuma", "tampouco", "nem", "sem"],
    "zh": ["不", "没", "没有", "别", "无", "非", "不太", "不怎么", "没那么", "不是很", "并不"],
    "th": ["ไม่", "ไม่ได้", "มิ", "ไม่ค่อย", "ไม่มี", "ไร้"],
    "id": ["tidak", "tak", "bukan", "belum", "jangan", "tanpa", "nggak", "gak", "ga", "enggak"],
    "ar": ["لا", "لم", "لن", "ليس", "غير", "بدون"],
}
# French "ne" closers ("ne … pas", "n' … jamais").
FR_NE = {"ne", "n'"}
FR_CLOSERS = {"pas", "plus", "jamais", "rien", "guère", "point", "aucun", "aucune", "personne"}
# Japanese negative endings (lemmas as given by spaCy/SudachiPy): ない,
# ん/ぬ (polite ません = ます + ん), ず.
JA_NEG = {"ない", "ぬ", "ん", "ず", "無い"}
JA_SKIP = {"じゃ", "じゃあ", "で", "では", "は", "が", "も", "く", "て", "ある", "いる", "ます",
           "だ", "です", "の", "そう", "ませ", "でき", "できる", "られる", "れる", "せる", "させる"}

# Words skipped between a negation and the word it negates.
INTENSIFIERS = {
    "en": {"very", "too", "so", "really", "that", "quite", "at", "all", "particularly", "overly",
           "as", "much", "any", "a", "an", "the", "my", "this", "it", "too", "even", "exactly", "super"},
    "fr": {"très", "trop", "si", "assez", "vraiment", "tellement", "super", "hyper", "du", "tout",
           "le", "la", "les", "un", "une", "de", "d'", "des", "particulièrement", "spécialement", "aussi",
           "encore", "forcément", "franchement"},
    "es": {"muy", "tan", "nada", "tanto", "mucho", "realmente", "el", "la", "un", "una", "lo",
           "me", "le", "lo", "se", "te", "es", "está", "parece"},
    "it": {"molto", "troppo", "così", "tanto", "proprio", "affatto", "per", "niente", "il", "la", "un",
           "una", "mi", "ci", "si", "è", "sembra", "davvero"},
    "de": {"sehr", "zu", "so", "wirklich", "besonders", "ganz", "gar", "allzu", "richtig", "der", "die",
           "das", "ein", "eine", "einen", "mehr", "mal", "wirkt", "ist"},
    "pt": {"muito", "tão", "demais", "bem", "nada", "tanto", "realmente", "o", "a", "um", "uma", "me",
           "se", "é", "está", "parece"},
    "zh": {"是", "很", "太", "那么", "这么", "怎么", "够", "特别", "非常", "算", "会", "觉得", "有", "大"},
    "th": {"ค่อย", "ค่อนข้าง", "เท่าไหร่", "เลย", "ได้", "มี", "ถึงกับ", "ค่อยจะ", "ใช่"},
    "id": {"terlalu", "begitu", "sangat", "cukup", "amat", "terlampau", "ada", "sama", "sekali", "begitu",
           "terlalu", "jadi", "punya", "lagi", "terasa"},
    "ar": {"جدا", "كثيرا", "تماما", "وجد", "يوجد", "كان", "يكن", "ب", "ال", "فيه", "له", "هناك", "عند"},
}
# Local-language word shown before a negated lemma (Japanese/Chinese handled separately).
NEG_DISPLAY_PREFIX = {"en": "not", "fr": "pas", "es": "no", "it": "non", "de": "nicht", "pt": "não"}
_CONTENT_POS = {"ADJ", "NOUN", "VERB", "PROPN", "ADV"}
# Linking / sensory verbs: "doesn't smell fresh", "ne sent pas bon", "riecht nicht
# gut" → the negation goes to the adjective, not to the verb.
LINKING_VERBS = {
    "be", "smell", "feel", "seem", "look", "sound", "become", "appear", "remain", "stay", "get",
    "être", "sentir", "sembler", "paraître", "rester", "devenir", "avoir",
    "ser", "estar", "oler", "parecer", "quedar", "sentirse",
    "essere", "sembrare", "sentire", "profumare", "odorare", "restare", "diventare",
    "sein", "riechen", "wirken", "scheinen", "duften", "bleiben", "werden",
    "cheirar", "ficar", "sentir-se",
}
_SKIP_POS = {"DET", "PRON", "AUX", "PART", "ADP", "SCONJ", "CCONJ", "NUM", "SPACE"}
# Languages where negation is merged on the token list (after segmentation/lemmatization).
TOKEN_NEGATION_LANGS = {"zh", "th", "id", "ar"}

# ─── "Not enough" / "too" modifiers ──────────────────────────────────────────
# Besides plain negation ("not_<lemma>"), two intensity modifiers get their own
# token so the cloud keeps the diagnostic apart:
#   notenough_<lemma>  "not fresh enough", "pas assez frais", "不够新鲜", "kurang segar"
#   too_<lemma>        "too strong", "trop fort", "zu stark", "太浓", "terlalu kuat"
#   toomuch_<lemma>    "too much perfume", "trop de parfum", "zu viel Duft" (nouns)
ENOUGH_WORDS = {   # turn a negation into "not enough" (before or after the word)
    "en": {"enough", "sufficiently"},
    "fr": {"assez", "suffisamment"},
    "es": {"suficientemente", "suficiente", "bastante"},
    "it": {"abbastanza", "sufficientemente"},
    "de": {"genug", "genügend", "ausreichend"},
    "pt": {"suficientemente", "suficiente", "bastante"},
}
TOO_WORDS = {
    "en": {"too", "overly", "excessively"},
    "fr": {"trop", "excessivement"},
    "es": {"demasiado", "excesivamente"},
    "it": {"troppo", "eccessivamente"},
    "de": {"zu", "allzu", "übermäßig"},
    "pt": {"demasiado", "excessivamente", "demasiadamente"},
}
TOO_POST_WORDS = {"pt": {"demais"}}                         # "forte demais"
LITTLE_WORDS = {"little", "peu", "wenig", "poco", "pouco"}   # "too little", "trop peu", "zu wenig"
MUCH_WORDS = {"much", "many", "of", "de", "d'", "viel", "viele", "vielen"}
INSUFFICIENT_WORDS = {   # "not enough" without a negation word
    "en": {"insufficiently"},
    "fr": {"insuffisamment"},
    "es": {"poco", "insuficientemente"},
    "it": {"poco", "insufficientemente"},
    "pt": {"pouco", "insuficientemente"},
    "de": set(),
}
A_LITTLE = {"un", "una", "um", "uma", "a"}   # "un poco fresco" = a bit fresh (not a modifier)

# Japanese: X足りない → not enough X ; Xすぎる → too X
JA_ENOUGH_VERBS = {"足りる", "たりる", "足る"}
JA_TOO = {"過ぎる", "すぎる", "過ぎ", "すぎ"}

# Token-list languages (after segmentation / Stanza lemmatization).
TOKEN_MODIFIERS = {
    "zh": {"ne_pre": {"不够", "不太够", "缺乏", "缺少"}, "too_pre": {"太", "过于", "太过"},
           "too_post": set(), "enough_post": set()},
    "th": {"ne_pre": set(), "too_pre": set(),
           "too_post": {"เกินไป", "เกิน"}, "enough_post": {"พอ"}},
    "id": {"ne_pre": {"kurang"}, "too_pre": {"terlalu", "kelewat"},
           "too_post": set(), "enough_post": {"cukup"}},
    "ar": {"ne_pre": set(), "too_pre": {"مفرط"},
           "too_post": {"مفرط", "لازم", "اللازم", "لزوم", "اللزوم"},
           "enough_post": {"كفاية", "الكفاية", "كاف", "كافي", "كافية"}},
}
# Words between an Arabic adjective and "اللازم" ("أكثر من اللازم" = more than necessary).
TOKEN_POST_FILLERS = {"أكثر", "اكثر", "زيادة", "زائد", "من", "عن", "ب", "ما", "في", "فيه", "ه",
                      "มาก", "ไป", "จน"}

# How each token kind is displayed, per language ({} = the word).
MOD_DISPLAY = {
    "notenough": {"en": "not {} enough", "fr": "pas assez {}", "es": "poco {}", "it": "non abbastanza {}",
                  "de": "nicht {} genug", "pt": "pouco {}", "zh": "不够{}", "ja": "{}が足りない",
                  "th": "{}ไม่พอ", "id": "kurang {}", "ar": "{} غير كاف"},
    "too":       {"en": "too {}", "fr": "trop {}", "es": "demasiado {}", "it": "troppo {}",
                  "de": "zu {}", "pt": "demasiado {}", "zh": "太{}", "ja": "{}すぎる",
                  "th": "{}เกินไป", "id": "terlalu {}", "ar": "{} أكثر من اللازم"},
    "toolittle": {"en": "not enough {}", "fr": "pas assez de {}", "es": "poco {}", "it": "poco {}",
                  "de": "zu wenig {}", "pt": "pouco {}", "zh": "不够{}", "ja": "{}が足りない",
                  "th": "{}ไม่พอ", "id": "kurang {}", "ar": "{} غير كاف"},
    "toomuch":   {"en": "too much {}", "fr": "trop de {}", "es": "demasiado {}", "it": "troppo {}",
                  "de": "zu viel {}", "pt": "demasiado {}", "zh": "太多{}", "ja": "{}が多すぎる",
                  "th": "{}มากเกินไป", "id": "terlalu banyak {}", "ar": "{} أكثر من اللازم"},
}
MOD_DISPLAY["nottoo"] = {
    "en": "not too {}", "fr": "pas trop {}", "es": "no demasiado {}",
    "it": "non troppo {}", "de": "nicht zu {}", "pt": "não demasiado {}",
    "zh": "不太{}", "ja": "{}すぎない", "th": "ไม่{}เกินไป",
    "id": "tidak terlalu {}", "ar": "ليس {} بشكل مفرط",
}
MOD_KINDS = ("nottoo", "notenough", "toolittle", "toomuch", "too", "not")
# English wording used when the cloud is translated: (prefix, suffix)
MOD_ENGLISH = {"nottoo": ("not too ", ""), "not": ("not ", ""), "notenough": ("not ", " enough"), "toolittle": ("not enough ", ""), "too": ("too ", ""),
               "toomuch": ("too much ", "")}

# Default exclusion words, translated per language (best-effort — editable
# by the user via the "Extra exclusion words" sidebar field either way).
DEFAULT_EXCLUSIONS_BY_LANG = {
    "English": [
        "product", "smell", "feel", "really", "just", "like", "little",
        "think", "lot", "make", "also", "bit", "quite", "something",
        "seem", "evoke", "find", "remind",
    ],
    "French": [
        "produit", "sentir", "ressentir", "vraiment", "juste", "comme",
        "petit", "penser", "beaucoup", "faire", "aussi", "peu", "assez",
        "quelque", "sembler", "évoquer", "trouver", "rappeler",
    ],
    "Spanish": [
        "producto", "oler", "sentir", "realmente", "solo", "como", "poco",
        "pensar", "mucho", "hacer", "también", "bastante", "algo",
        "parecer", "evocar", "encontrar", "recordar",
    ],
    "Italian": [
        "prodotto", "odore", "sentire", "davvero", "solo", "come", "poco",
        "pensare", "molto", "fare", "anche", "abbastanza", "qualcosa",
        "sembrare", "evocare", "trovare", "ricordare",
    ],
    "German": [
        "produkt", "riechen", "fühlen", "wirklich", "nur", "wie", "wenig",
        "denken", "viel", "machen", "auch", "bisschen", "ziemlich",
        "etwas", "scheinen", "hervorrufen", "finden", "erinnern",
    ],
    "Portuguese": [
        "produto", "cheirar", "sentir", "realmente", "apenas", "como",
        "pouco", "pensar", "muito", "fazer", "também", "bastante",
        "algo", "parecer", "evocar", "encontrar", "lembrar",
    ],
    # Larger than the others on purpose: NLTK has no Chinese stopword corpus
    # at all, so this list also has to cover common grammatical function
    # words (的/了/是/在 etc.), not just filler content words the way the
    # other languages' lists do.
    "Chinese": [
        "产品", "闻起来", "感觉", "真的", "只是", "喜欢", "好像", "一点",
        "认为", "觉得", "很多", "也是", "相当", "什么", "似乎", "引起",
        "唤起", "找到", "提醒", "的", "了", "和", "是", "在", "我", "你",
        "他", "她", "它", "们", "这", "那", "有", "也", "都", "就", "还",
        "而", "或", "与", "及", "着", "吗", "呢", "啊", "吧", "把", "被",
        "让", "给", "从", "到", "对", "于", "上", "下", "中", "后", "前",
        "里", "外", "个", "些", "怎么", "为什么", "因为", "所以", "但是",
        "可是", "如果", "虽然", "一个", "一些", "这个", "那个", "可以",
        "应该", "会", "要", "能",
    ],
    # Same reasoning as Chinese: no NLTK stopword corpus for Japanese, so
    # this list also covers common grammatical particles/function words
    # (の/は/を/に/が etc.), not just filler content words.
    "Japanese": [
        "製品", "感じ", "本当に", "ただ", "少し", "思う", "たくさん",
        "作る", "また", "かなり", "何か", "ような", "引き起こす",
        "見つける", "思い出させる", "の", "は", "を", "に", "が", "で",
        "と", "も", "な", "し", "から", "まで", "より", "へ", "や", "か",
        "ね", "よ", "わ", "これ", "それ", "あれ", "この", "その", "あの",
        "です", "ます", "ある", "いる", "する", "なる", "という", "こと",
        "もの", "ため", "よう", "って", "たり", "とか", "けど", "でも",
        "しかし", "だから", "そして", "また", "など",
    ],
}

# Suggested extra exclusion words, auto-filled into the "Extra exclusion
# words" sidebar box the first time Chinese or Japanese is selected (see the
# language-change handling in the sidebar). Both languages need this far more
# than the others: NLTK has no stopword corpus for either, and jieba/spaCy's
# Japanese tokenizer segment plenty of grammatical particles and fragments
# that the shorter DEFAULT_EXCLUSIONS_BY_LANG lists above don't cover. The
# Chinese list is the user-supplied stopwords_cn.txt (a common, widely-used
# Chinese stopword list) plus a handful of segmentation-fragment/pronoun
# words observed directly in this app's own word clouds (我们, 有点, 那种,
# 这种, 一样, 以及, 还有, 完全, 来说, 以下, 令人, 起来, 感到, 一片, 一股, 一丝).
# Auto-filled once per language, not re-applied on every rerun — the user's
# own edits afterward are preserved (see sidebar).
SUGGESTED_EXTRA_EXCLUSIONS = {
    "Chinese": [
    '我们', '有点', '那种', '这种', '一样', '以及', '还有', '完全', '来说', '以下', '令人', '起来', '感到', '一片', '一股',
    '一丝', '!', '"', '$', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6',
    '7', '8', '9', ':', ';', '<', '>', '?', '[', '\\', ']', '_', '{', '|', '}', '~', '·', '—',
    '‘', '’', '“', '”', '…', '、', '。', '〈', '〉', '《', '》', '【', '】', '一', '七', '万', '三', '上',
    '下', '不', '与', '且', '个', '临', '为', '乃', '么', '之', '乎', '乘', '九', '也', '了', '二', '于', '五',
    '些', '亦', '人', '亿', '今', '仍', '从', '他', '令', '以', '们', '件', '任', '份', '但', '位', '何', '你',
    '使', '依', '便', '俺', '倘', '借', '像', '儿', '全', '八', '六', '兮', '共', '其', '内', '册', '再', '冒',
    '冲', '几', '凡', '凭', '切', '则', '刚', '别', '到', '副', '十', '千', '即', '却', '卷', '去', '又', '及',
    '双', '另', '只', '叫', '可', '台', '各', '同', '名', '后', '向', '吓', '吗', '吧', '吱', '呀', '呃', '呕',
    '呗', '呜', '呢', '呵', '呸', '咋', '和', '咚', '咦', '咧', '咯', '咱', '咳', '哇', '哈', '哉', '哎', '哗',
    '哟', '哦', '哩', '哪', '哼', '唉', '啊', '啐', '啥', '啦', '啰', '喂', '喏', '喔', '喽', '嗡', '嗬', '嗯',
    '嗳', '嘎', '嘘', '嘛', '嘻', '嘿', '噢', '四', '因', '在', '地', '场', '块', '处', '多', '大', '太', '套',
    '她', '好', '如', '宁', '它', '家', '对', '将', '小', '尔', '就', '尽', '层', '己', '已', '巴', '幅', '幕',
    '幢', '并', '座', '张', '归', '当', '彼', '往', '待', '很', '得', '怎', '您', '我', '或', '户', '所', '才',
    '打', '批', '把', '抹', '拿', '按', '挺', '据', '支', '故', '无', '既', '时', '是', '曾', '替', '最', '有',
    '望', '朝', '本', '束', '条', '来', '极', '枚', '架', '某', '栋', '样', '根', '次', '欤', '正', '此', '段',
    '每', '比', '沿', '滴', '点', '焉', '照', '片', '用', '由', '略', '番', '百', '的', '看', '着', '矣', '离',
    '种', '稍', '章', '第', '等', '管', '篇', '类', '粒', '纵', '线', '经', '给', '缕', '群', '者', '而', '能',
    '腾', '自', '至', '致', '般', '艘', '节', '若', '虽', '蛮', '被', '要', '觉', '角', '让', '论', '该', '诸',
    '谁', '赶', '起', '趁', '距', '跟', '较', '辆', '边', '过', '还', '这', '连', '那', '都', '间', '阵', '阿',
    '除', '随', '零', '靠', '页', '顺', '颗', '首', '齐', '﹏', '！', '＂', '＃', '＄', '％', '＆', '＇', '（',
    '）', '＊', '＋', '，', '－', '／', '：', '；', '＜', '＝', '＞', '？', '＠', '［', '］', '＾', '＿', '｀',
    '｛', '｜', '｝', '～', '一下', '一个', '一些', '一件', '一份', '一位', '一何', '一共', '一切', '一则', '一同', '一名',
    '一向', '一场', '一处', '一定', '一层', '一幕', '一并', '一张', '一律', '一批', '一抹', '一旦', '一时', '一条', '一来',
    '一概', '一次', '一段', '一点', '一番', '一直', '一种', '一线', '一缕', '一群', '一致', '一般', '一角', '一起', '一阵',
    '一齐', '万一', '万分', '上下', '上去', '上来', '下去', '下来', '不仅', '不但', '不光', '不单', '不只', '不如', '不妨',
    '不尽', '不得', '不怕', '不惟', '不成', '不拘', '不料', '不是', '不比', '不然', '不特', '不独', '不管', '不若', '不论',
    '不过', '不问', '与其', '与否', '且说', '两者', '个别', '为了', '为何', '为止', '为此', '为着', '乃至', '之一', '之上',
    '之下', '之中', '之内', '之前', '之后', '之外', '之流', '之类', '之间', '之际', '乌乎', '也即', '也好', '也罢', '也许',
    '予以', '二来', '于是', '云云', '云尔', '亦即', '产生', '人们', '人家', '什么', '介于', '仍旧', '从不', '从来', '从此',
    '从而', '他人', '他们', '他的', '以上', '以为', '以便', '以免', '以内', '以外', '以故', '以期', '以来', '以此', '以至',
    '以致', '任何', '任凭', '仿佛', '估计', '似乎', '似的', '但凡', '但是', '位于', '何以', '何况', '何处', '何故', '何时',
    '余外', '作为', '作出', '你们', '使人', '使得', '例如', '依据', '依照', '便于', '俺们', '倘使', '倘或', '倘然', '倘若',
    '借助', '假使', '假如', '假若', '做出', '傥然', '光是', '全体', '全然', '全部', '全都', '关于', '其一', '其中', '其二',
    '其他', '其余', '其它', '其实', '其次', '其间', '兼之', '再则', '再有', '再次', '再者', '再说', '况且', '几时', '凡是',
    '凭借', '出于', '出去', '出来', '分别', '分外', '则甚', '刚刚', '利用', '别人', '别处', '别是', '别的', '别管', '别说',
    '前后', '前此', '前者', '加之', '加以', '十分', '即令', '即使', '即便', '即如', '即将', '即或', '即是', '即若', '又及',
    '及其', '及至', '反之', '反而', '受到', '另外', '另悉', '只当', '只怕', '只是', '只有', '只消', '只要', '只限', '叮咚',
    '可以', '可否', '可是', '可能', '可见', '各个', '各位', '各种', '各自', '同时', '同样', '后者', '向使', '向来', '向着',
    '否则', '吧哒', '听来', '呜呼', '呵呵', '呼哧', '咱们', '哈哈', '哎呀', '哎哟', '哪个', '哪些', '哪儿', '哪天', '哪年',
    '哪怕', '哪样', '哪边', '哪里', '哼唷', '唯有', '啪达', '啷当', '喔唷', '嗡嗡', '嘎登', '嘿嘿', '回来', '因为', '因了',
    '因此', '因着', '因而', '固然', '在下', '在于', '基于', '处于', '处在', '多么', '多少', '大家', '大概', '大约', '她们',
    '她的', '好像', '如上', '如下', '如何', '如其', '如同', '如是', '如果', '如此', '如若', '始终', '始而', '孰料', '孰知',
    '宁可', '宁愿', '宁肯', '它们', '它的', '对于', '对待', '对方', '对比', '导致', '将要', '尔后', '尔尔', '尚且', '尤为',
    '尤其', '就是', '就此', '就算', '就要', '尽管', '展开', '属于', '岂但', '已矣', '已经', '巴巴', '带来', '带给', '常常',
    '并且', '并非', '应当', '应该', '庶乎', '庶几', '开外', '开始', '开展', '引发', '引起', '归齐', '当中', '当地', '当然',
    '当着', '彻底', '彼时', '彼此', '往往', '得了', '必定', '必须', '忆起', '快要', '忽然', '怎么', '怎奈', '怎样', '总之',
    '总是', '惟其', '想来', '想起', '感觉', '慢慢', '慢说', '或则', '或是', '或曰', '或者', '或许', '截至', '所以', '所在',
    '所幸', '所有', '才能', '打从', '抑或', '按照', '据此', '接着', '故此', '故而', '旁人', '无宁', '无论', '既往', '既是',
    '既然', '时候', '时常', '是以', '是否', '是的', '曾经', '替代', '最后', '最终', '有些', '有关', '有及', '有时', '有的',
    '朝着', '期间', '未必', '本人', '本地', '本着', '本身', '来看', '来着', '来自', '来讲', '极为', '极了', '极其', '极度',
    '果然', '果真', '某个', '某些', '某人', '某某', '根据', '格外', '正值', '正在', '正如', '正巧', '正是', '此后', '此地',
    '此处', '此外', '此时', '此次', '此间', '毋宁', '每个', '每当', '比及', '比如', '比方', '沿着', '渐渐', '漫说', '然则',
    '然后', '然而', '照着', '特别', '犹且', '犹自', '甚且', '甚么', '甚或', '甚而', '甚至', '用来', '由于', '由是', '由此',
    '略微', '的确', '的话', '直到', '相似', '相同', '相当', '相继', '省得', '看来', '眨眼', '着呢', '矣乎', '矣哉', '稍微',
    '稍稍', '突然', '立刻', '立即', '竟而', '等到', '等等', '类如', '系列', '纵令', '纵使', '纵然', '纷纷', '终于', '经常',
    '经由', '经过', '结果', '给予', '统统', '继之', '继后', '继而', '综上', '罢了', '而且', '而况', '而后', '而外', '而已',
    '而是', '而言', '肯定', '能否', '自从', '自后', '自家', '自己', '自打', '自身', '至于', '至今', '至若', '般的', '若夫',
    '若是', '若果', '若非', '莫如', '莫若', '虽则', '虽是', '虽然', '虽说', '要不', '要么', '要是', '觉得', '譬喻', '譬如',
    '让人', '记起', '许多', '设使', '设或', '设若', '诚如', '诚然', '说来', '诸位', '诸如', '谁人', '谁料', '谁知', '贼死',
    '赖以', '起见', '趁着', '越是', '较之', '过于', '过分', '过去', '过度', '还是', '还要', '这个', '这么', '这些', '这儿',
    '这时', '这样', '这次', '这般', '这边', '这里', '进去', '进而', '进行', '连同', '逐步', '逐渐', '通常', '通过', '通通',
    '造成', '遵循', '遵照', '那个', '那么', '那些', '那儿', '那时', '那样', '那般', '那边', '那里', '部分', '鄙人', '鉴于',
    '针对', '陆续', '除了', '除外', '除开', '除非', '随后', '随时', '随着', '非但', '非常', '非徒', '非特', '非独', '面对',
    '顺着', '顿时', '首先', '马上', '一切都', '一方面', '一系列', '一转眼', '一部分', '不外乎', '不尽然', '不至于', '与其说',
    '且不说', '为什么', '乃至于', '之所以', '之类的', '也就是', '事实上', '于是乎', '什么样', '什么的', '以至于', '先不先', '再其次',
    '再者说', '反过来', '另一边', '只不过', '同样地', '听起来', '坦白说', '实际上', '就是了', '就是说', '怎么办', '怎么样', '换言之',
    '接下来', '是不是', '更何况', '有一点', '有没有', '比方说', '没奈何', '甚至于', '看起来', '简言之', '算起来', '紧接着', '老实说',
    '自个儿', '自各儿', '莫不然', '要不是', '要不然', '说实话', '说起来', '这一来', '这么些', '这么样', '这会儿', '那么些', '那么样',
    '那会儿', '难道说', '一概而论', '一般来说', '与此同时', '举例来说', '也就是说', '具体地说', '具体说来', '反过来说', '另一方面',
    '如上所述', '就此而言', '尽管如此', '总的来看', '总的来说', '总的说来', '总而言之', '恰恰相反', '打个比方', '换句话说', '由此可见',
    '相对而言', '简而言之', '综上所述', '这么点儿', '这就是说', '除此之外', '就……而言',
    ],
    "Japanese": [
    'ある', 'する', 'なる', 'できる', '感じる', '思う', '見える', '使う', '入る', '行く', '与える', '起こす', '呼ぶ',
    '特に', 'まるで', 'いつも', '瞬間', '時間', '自分', 'たち', 'イメージ', '雰囲気', '場所', '良い', 'いい', '好き', '美しい',
    ],
}

# Default exclusion words for the three new languages (best-effort, editable
# via "Extra exclusion words" in the sidebar).
DEFAULT_EXCLUSIONS_BY_LANG.update({
    "Thai": [
        "ผลิตภัณฑ์", "รู้สึก", "จริงๆ", "แค่", "เหมือน", "นิดหน่อย", "คิด",
        "เยอะ", "ทำ", "ด้วย", "ค่อนข้าง", "อะไร", "ดู", "เจอ", "นึกถึง",
        "ค่ะ", "คะ", "ครับ", "นะ", "จ้ะ", "จ้า",
    ],
    "Indonesian": [
        "produk", "merasa", "rasa", "benar", "hanya", "seperti", "sedikit",
        "pikir", "banyak", "membuat", "buat", "juga", "agak", "cukup",
        "sesuatu", "tampak", "temu", "ingat", "sangat", "sekali", "nya",
    ],
    "Arabic": [
        "منتج", "حقا", "فقط", "مثل", "قليل", "اعتقد", "أعتقد", "كثير",
        "أيضا", "ايضا", "جدا", "شيء", "يبدو", "بدا", "وجد", "ذكر",
    ],
})

# Cluster colours (first 10 = original palette; extended so >10 Louvain
# communities don't recycle colours too early).
CLUSTER_COLORS = [
    "#0085AF", "#E8A838", "#C62F4B", "#6AAB6A", "#8B6BB1",
    "#4BA8B0", "#E07B39", "#B85C8A", "#7B9E3E", "#D4724A",
    "#2E5E8C", "#A3832B", "#7A2E3B", "#3E7D5A", "#5C4A8C",
    "#2C8C8C", "#B0562A", "#8C3E6E", "#55702B", "#9C4A33",
]
UNCLUSTERED_COLOR = "#9AA0A6"

# ─── Color helpers ────────────────────────────────────────────────────────────
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def shade_rgb_str(hex_color, t):
    """t in [0,1] — 0 lightens toward white, 1 darkens toward black."""
    r, g, b = hex_to_rgb(hex_color)
    if t < 0.5:
        f = (0.5 - t) * 2
        r, g, b = (c + (255 - c) * f * 0.65 for c in (r, g, b))
    else:
        f = (t - 0.5) * 2
        r, g, b = (c * (1 - f * 0.55) for c in (r, g, b))
    return f"rgb({int(r)},{int(g)},{int(b)})"

def rgb_str(hex_color):
    r, g, b = hex_to_rgb(hex_color)
    return f"rgb({r},{g},{b})"

# ─── Fonts ───────────────────────────────────────────────────────────────────
# The cloud is rendered SERVER-SIDE (PIL), so what matters is which font
# files exist on the deployed machine. On Streamlit Community Cloud the
# system fonts come from packages.txt (fonts-noto-core gives Noto Sans Thai /
# Noto Sans Arabic / Noto Naskh Arabic; fonts-noto-cjk gives CJK). Any .ttf /
# .otf committed into a repo-level "fonts/" folder is picked up too, and the
# user can upload their own font file from the UI.
def _mpl_bundled_font(filename):
    try:
        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", filename)
        return path if os.path.exists(path) else None
    except Exception:
        return None

def _find_installed_font(pattern, family_name=None):
    """Resolve a fontconfig pattern (e.g. 'Roboto:style=Bold') to a font file.
    Returns None when fontconfig falls back to an unrelated family."""
    family_name = family_name or pattern.split(":")[0]
    try:
        import subprocess
        result = subprocess.run(
            ["fc-match", pattern, "-f", "%{family}|||%{style}|||%{file}"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        fam, style, path = result.stdout.split("|||", 2)
        if family_name.lower() not in fam.lower():
            return None
        if ":style=bold" in pattern.lower() and "bold" not in style.lower():
            return None  # no real bold face installed — avoid a duplicate of Regular
        return path if os.path.exists(path) else None
    except Exception:
        return None

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

# (label, script, resolver) — script tells which writing system a font is
# meant for, so the right default is chosen when a language is selected.
_FONT_SPECS = [
    ("DejaVu Sans",          "latin", lambda: _mpl_bundled_font("DejaVuSans.ttf")),
    ("DejaVu Sans Bold",     "latin", lambda: _mpl_bundled_font("DejaVuSans-Bold.ttf")),
    ("DejaVu Serif",         "latin", lambda: _mpl_bundled_font("DejaVuSerif.ttf")),
    ("DejaVu Serif Bold",    "latin", lambda: _mpl_bundled_font("DejaVuSerif-Bold.ttf")),
    ("STIX General (serif)", "latin", lambda: _mpl_bundled_font("STIXGeneral.ttf")),
    ("Inter",                "latin", lambda: _first_existing(["fonts/Inter-Regular.ttf"]) or _find_installed_font("Inter")),
    ("Inter Bold",           "latin", lambda: _first_existing(["fonts/Inter-Bold.ttf"]) or _find_installed_font("Inter:style=Bold")),
    ("Open Sans",            "latin", lambda: _first_existing(["fonts/OpenSans-Regular.ttf"]) or _find_installed_font("Open Sans")),
    ("Open Sans Bold",       "latin", lambda: _first_existing(["fonts/OpenSans-Bold.ttf"]) or _find_installed_font("Open Sans:style=Bold")),
    ("Roboto",               "latin", lambda: _first_existing(["fonts/Roboto-Regular.ttf"]) or _find_installed_font("Roboto")),
    ("Roboto Bold",          "latin", lambda: _first_existing(["fonts/Roboto-Bold.ttf"]) or _find_installed_font("Roboto:style=Bold")),
    ("Lato",                 "latin", lambda: _first_existing(["fonts/Lato-Regular.ttf"]) or _find_installed_font("Lato")),
    ("Lato Bold",            "latin", lambda: _first_existing(["fonts/Lato-Bold.ttf"]) or _find_installed_font("Lato:style=Bold")),
    ("Montserrat",           "latin", lambda: _first_existing(["fonts/Montserrat-Regular.ttf"]) or _find_installed_font("Montserrat")),
    ("Montserrat Bold",      "latin", lambda: _first_existing(["fonts/Montserrat-Bold.ttf"]) or _find_installed_font("Montserrat:style=Bold")),
    ("Nunito",               "latin", lambda: _first_existing(["fonts/Nunito-Regular.ttf"]) or _find_installed_font("Nunito")),
    ("Noto Sans CJK (Chinese / Japanese)", "cjk", lambda: _first_existing([
        "fonts/NotoSansCJK-Regular.ttc", "fonts/NotoSansCJK-Regular.otf", "fonts/NotoSansSC-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    ]) or _find_installed_font("Noto Sans CJK SC", "Noto Sans CJK")),
    ("Noto Sans CJK Bold (Chinese / Japanese)", "cjk", lambda: _first_existing([
        "fonts/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]) or _find_installed_font("Noto Sans CJK SC:style=Bold", "Noto Sans CJK")),
    ("Noto Serif CJK (Chinese / Japanese)", "cjk", lambda: _first_existing([
        "fonts/NotoSerifCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]) or _find_installed_font("Noto Serif CJK SC", "Noto Serif CJK")),
    ("Noto Sans Thai",       "thai", lambda: _first_existing([
        "fonts/NotoSansThai-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    ]) or _find_installed_font("Noto Sans Thai")),
    ("Noto Sans Thai Bold",  "thai", lambda: _first_existing([
        "fonts/NotoSansThai-Bold.ttf", "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
    ]) or _find_installed_font("Noto Sans Thai:style=Bold")),
    ("Noto Serif Thai",      "thai", lambda: _find_installed_font("Noto Serif Thai")),
    ("Sarabun (Thai)",       "thai", lambda: _first_existing(["fonts/Sarabun-Regular.ttf"]) or _find_installed_font("Sarabun")),
    ("Loma (Thai)",          "thai", lambda: _find_installed_font("Loma")),
    ("Garuda (Thai)",        "thai", lambda: _find_installed_font("Garuda")),
    ("Noto Sans Arabic",     "arabic", lambda: _first_existing([
        "fonts/NotoSansArabic-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    ]) or _find_installed_font("Noto Sans Arabic")),
    ("Noto Sans Arabic Bold", "arabic", lambda: _first_existing([
        "fonts/NotoSansArabic-Bold.ttf", "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
    ]) or _find_installed_font("Noto Sans Arabic:style=Bold")),
    ("Noto Naskh Arabic",    "arabic", lambda: _first_existing([
        "fonts/NotoNaskhArabic-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    ]) or _find_installed_font("Noto Naskh Arabic")),
    ("Noto Kufi Arabic",     "arabic", lambda: _find_installed_font("Noto Kufi Arabic")),
    ("Amiri (Arabic)",       "arabic", lambda: _first_existing(["fonts/Amiri-Regular.ttf"]) or _find_installed_font("Amiri")),
]

LANG_SCRIPT = {"zh": "cjk", "ja": "cjk", "th": "thai", "ar": "arabic"}

@st.cache_resource
def available_fonts():
    """{label: (path, script)} for every font actually present on this machine
    (deduplicated by file path). Computed once per server process."""
    out, seen = {}, set()
    for label, script, resolver in _FONT_SPECS:
        try:
            path = resolver()
        except Exception:
            path = None
        if path and os.path.exists(path) and path not in seen:
            out[label] = (path, script)
            seen.add(path)
    # Any extra font dropped into the repo's fonts/ folder.
    if os.path.isdir("fonts"):
        for fn in sorted(os.listdir("fonts")):
            p = os.path.join("fonts", fn)
            if fn.lower().endswith((".ttf", ".otf", ".ttc")) and p not in seen:
                out[f"{os.path.splitext(fn)[0]} (repo fonts/)"] = (p, "latin")
                seen.add(p)
    return out

@st.cache_resource
def save_uploaded_font(data: bytes, filename: str):
    """Persist an uploaded font to a temp file (keyed by content hash)."""
    ext = os.path.splitext(filename)[1].lower() or ".ttf"
    digest = hashlib.md5(data).hexdigest()[:12]
    path = os.path.join(tempfile.gettempdir(), f"user_font_{digest}{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path

@st.cache_data(show_spinner=False)
def font_codepoints(path):
    """Set of Unicode code points the font has glyphs for (fontTools ships
    with matplotlib, so no extra dependency)."""
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(path, fontNumber=0, lazy=True)
        cmap = f.getBestCmap() or {}
        return frozenset(cmap.keys())
    except Exception:
        return None

def missing_glyph_ratio(path, words):
    cps = font_codepoints(path)
    if not cps:
        return 0.0
    chars = {c for w in words for c in w if not c.isspace() and unicodedata.category(c)[0] == "L"}
    if not chars:
        return 0.0
    return sum(1 for c in chars if ord(c) not in cps) / len(chars)

# ─── Right-to-left (Arabic) rendering ─────────────────────────────────────────
@st.cache_resource
def pil_has_complex_layout():
    """True when Pillow can shape Arabic/Thai itself (libraqm + fribidi)."""
    try:
        from PIL import features
        return bool(features.check_feature("raqm")) and bool(features.check_feature("fribidi"))
    except Exception:
        return False

def shape_for_render(text, lang_code):
    """With libraqm, Pillow shapes Arabic and handles right-to-left itself.
    Without it, letters would come out disconnected and in the wrong order, so
    we pre-shape with arabic-reshaper + python-bidi instead."""
    if lang_code != "ar" or pil_has_complex_layout():
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text

# ─── NLP loaders (all lazy — only the selected language is ever loaded) ──────
@st.cache_resource
def load_lemmatizer():
    return WordNetLemmatizer()

@st.cache_resource
def load_spacy_model(model_name):
    """Returns (model, error). Models are installed via requirements.txt."""
    if model_name is None:
        return None, None
    try:
        import spacy
        return spacy.load(model_name, disable=["parser", "ner"]), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

@st.cache_resource
def load_jieba():
    try:
        import jieba
        return jieba, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

@st.cache_resource
def load_thai():
    """PyThaiNLP: pure-Python word segmentation (newmm) + bundled stopwords.
    Returns (tokenize_fn, stopword_set, error)."""
    try:
        from pythainlp.tokenize import word_tokenize
        try:
            from pythainlp.corpus.common import thai_stopwords
            sw = set(thai_stopwords())
        except Exception:
            sw = set()
        tok = lambda text: word_tokenize(text, engine="newmm", keep_whitespace=False)
        return tok, sw, None
    except Exception as e:
        return None, set(), f"{type(e).__name__}: {e}"

def _stanza_resources_dir():
    """First writable location for Stanza models."""
    candidates = [
        os.environ.get("STANZA_RESOURCES_DIR"),
        os.path.join(os.path.expanduser("~"), "stanza_resources"),
        os.path.join(tempfile.gettempdir(), "stanza_resources"),
    ]
    for d in candidates:
        if not d:
            continue
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".write_test")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            return d
        except Exception:
            continue
    return None

@st.cache_resource(show_spinner=False)
def load_stanza(lang_code):
    """Imports Stanza (and torch) ONLY when called — i.e. only when Arabic or
    Indonesian is actually analysed. Downloads the language model on first use
    (reused afterwards). Returns (pipeline, error)."""
    try:
        import stanza
    except Exception as e:
        return None, f"Stanza is not installed ({type(e).__name__}: {e})"
    try:
        import torch
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except Exception:
        pass
    model_dir = _stanza_resources_dir()
    try:
        from stanza.pipeline.core import DownloadMethod
        dl = DownloadMethod.REUSE_RESOURCES
    except Exception:
        dl = "reuse_resources"
    last_err = None
    # Not every language ships an MWT (multi-word token) model — try with it
    # first (Arabic needs it to split clitics), then without.
    for procs in ("tokenize,mwt,pos,lemma", "tokenize,pos,lemma"):
        try:
            kwargs = dict(lang=lang_code, processors=procs, use_gpu=False,
                          verbose=False, download_method=dl)
            if model_dir:
                kwargs["dir"] = model_dir
            return stanza.Pipeline(**kwargs), None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err

def stanza_lemmatize(texts, nlp, progress_cb=None, batch_size=64):
    """Lemmatize many verbatims with Stanza in batches (much faster than one
    call per row). Returns one lemma list per input text."""
    import stanza
    out = [[] for _ in texts]
    idx = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    total = len(idx)
    for start in range(0, total, batch_size):
        chunk = idx[start:start + batch_size]
        docs = [stanza.Document([], text=texts[i]) for i in chunk]
        try:
            docs = nlp(docs)
        except Exception:
            # Fallback: one at a time, so one bad row doesn't kill the batch.
            docs = []
            for i in chunk:
                try:
                    docs.append(nlp(texts[i]))
                except Exception:
                    docs.append(None)
        for i, d in zip(chunk, docs):
            if d is None:
                continue
            out[i] = [t for sentence in d.sentences
                      for t in [*(w.lemma or w.text for w in sentence.words), BOUNDARY]]
        if progress_cb:
            progress_cb(min(1.0, (start + len(chunk)) / max(1, total)))
    return out

@st.cache_resource
def load_stopwords(nltk_lang):
    if nltk_lang is None:
        return set()
    try:
        return set(stopwords.words(nltk_lang))
    except Exception:
        return set()

# ─── Tokenization ────────────────────────────────────────────────────────────
WORD_PATTERN = re.compile(r"\b[^\W\d_][^\W\d_]+\b", re.UNICODE)
ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")

# Minimum surviving length (len > value). Lower for scripts where meaningful
# words are often 2 characters.
MIN_LEMMA_LEN = {"zh": 1, "ja": 1, "th": 1, "ar": 1}

def clean_token(tok, lang_code):
    tok = tok.lower().strip()
    if lang_code == "ar":
        tok = ARABIC_DIACRITICS.sub("", tok)
    return tok

def is_word(tok):
    """Letters (+ combining marks, needed for Thai/Arabic) only, at least one letter."""
    if not tok:
        return False
    cats = [unicodedata.category(c) for c in tok]
    return any(c.startswith("L") for c in cats) and all(c[0] in "LM" for c in cats)

def merge_negations(tokens, lang_code, stop_words):
    """Token-list modifiers (Chinese, Thai, Indonesian, Arabic).
    • negation + next content token, skipping intensifiers/stopwords
      ("tidak terlalu segar" → tidak_segar, "不是很新鲜" → not_新鲜)
    • 'not enough': prefix words (不够, kurang), an 'enough' word skipped after a
      negation ("tidak cukup segar"), or after the word ("ไม่สดพอ", "สดไม่พอ",
      "ليس قويا بما فيه الكفاية", "قوي غير كاف")
    • 'too': prefix (太, terlalu, مفرط) or suffix (แรงเกินไป, قوي أكثر من اللازم)."""
    neg_words = set(NEGATION_WORDS.get(lang_code, []))
    cfg = TOKEN_MODIFIERS.get(lang_code, {})
    ne_pre, too_pre = cfg.get("ne_pre", set()), cfg.get("too_pre", set())
    too_post, enough_post = cfg.get("too_post", set()), cfg.get("enough_post", set())
    skip = INTENSIFIERS.get(lang_code, set())
    markers = neg_words | ne_pre | too_pre | too_post | enough_post
    neg_prefixes = tuple(f"{w}_" for w in neg_words) + ("not_",)

    def skippable(tok):
        # intensifiers like 太 / terlalu / cukup are skipped after a negation
        # ("不太浓" → not strong, "tidak cukup segar" → not fresh enough)
        return (tok in skip or tok in stop_words or tok in too_pre or tok in enough_post) \
            and tok not in neg_words and tok not in ne_pre

    def next_target(i):
        """Index of the next content token after i (≤3 ahead), and skipped tokens."""
        j, skipped = i + 1, []
        while j < len(tokens) and j <= i + 3 and skippable(tokens[j]):
            skipped.append(tokens[j])
            j += 1
        if j < len(tokens) and j <= i + 3 and tokens[j] != BOUNDARY and tokens[j] not in markers and tokens[j] not in stop_words:
            return j, skipped
        return None, skipped

    def retag_previous(merged, kind):
        """Apply a post-position modifier to the last content token already emitted."""
        p = len(merged) - 1
        while p >= 0 and len(merged) - p <= 7:
            w = merged[p]
            if w == BOUNDARY:
                return False
            if w in neg_words or w in TOKEN_POST_FILLERS or w in skip or w in stop_words:
                p -= 1                           # "สดไม่พอ": bare negation absorbed below
                continue
            if kind == "notenough" and not (
                    w.startswith(neg_prefixes + ("notenough_",))
                    or any(t in neg_words for t in merged[p + 1:])):
                return False
            if w.startswith(("notenough_", "too_", "toomuch_", "nottoo_")):
                return True
            base = w
            for pre in neg_prefixes:
                if w.startswith(pre):
                    base = w[len(pre):]
                    break
            merged[p] = f"{kind}_{base}"
            del merged[p + 1:]                   # drop fillers between word and marker
            return True
        return False

    merged, i, n = [], 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t in enough_post and t not in neg_words and retag_previous(merged, "notenough"):
            i += 1
            continue
        if t in too_post and retag_previous(merged, "too"):
            if lang_code == "th" and i + 1 < n and tokens[i + 1] == "ไป":
                i += 1
            i += 1
            continue
        if t in neg_words or t in ne_pre or t in too_pre:
            j, skipped = next_target(i)
            if j is not None:
                if t in too_pre and t not in neg_words:
                    merged.append(f"too_{tokens[j]}")
                elif t in neg_words and any(s in too_pre for s in skipped):
                    merged.append(f"nottoo_{tokens[j]}")
                elif t in ne_pre or any(s in enough_post for s in skipped):
                    merged.append(f"notenough_{tokens[j]}")
                else:
                    prefix = "not" if lang_code == "zh" else t
                    merged.append(f"{prefix}_{tokens[j]}")
                i = j + 1
                continue
        merged.append(t)
        i += 1
    return merged

def _norm_apostrophes(text):
    return text.replace("’", "'").replace("‘", "'").replace("`", "'")

def spacy_negation_lemmas(doc, lang_code, neg_surfaces=None):
    """Lemmas from a spaCy doc with negation / 'not enough' / 'too' merged into
    one token ('not_<lemma>', 'notenough_<lemma>', 'too_<lemma>',
    'toomuch_<lemma>') using part-of-speech tags. neg_surfaces (optional dict)
    collects the original wording of each such token (used for Japanese display)."""
    toks = list(doc)
    n = len(toks)
    neg_words = set(NEGATION_WORDS.get(lang_code, []))
    skip_words = INTENSIFIERS.get(lang_code, set())
    enough = ENOUGH_WORDS.get(lang_code, set())
    too_words = TOO_WORDS.get(lang_code, set())
    too_post = TOO_POST_WORDS.get(lang_code, set())
    insufficient = INSUFFICIENT_WORDS.get(lang_code, set())
    modifier_words = enough | too_words | too_post | insufficient | LITTLE_WORDS
    consumed, mods = set(), {}            # mods: index → [kind, surface]

    def lower(t):
        return _norm_apostrophes(t.lower_)

    def is_content(t):
        lw = lower(t)
        return (t.pos_ in _CONTENT_POS and (t.is_alpha or lang_code == "ja")
                and lw not in skip_words and lw not in neg_words and lw not in modifier_words
                and t.lemma_ not in JA_NEG)

    def find_forward(i, max_ahead=5):
        """(index of next content token, set of skipped lowercase words)."""
        skipped = set()
        for j in range(i + 1, min(n, i + 1 + max_ahead)):
            t = toks[j]
            lw = lower(t)
            if t.is_punct or "\n" in t.text or t.pos_ in ("CCONJ", "SCONJ") or j in consumed or lw in neg_words:
                return None, skipped
            if lw in skip_words or lw in modifier_words or t.pos_ in _SKIP_POS:
                skipped.add(lw)
                continue
            if is_content(t):
                return j, skipped
            return None, skipped
        return None, skipped

    def find_backward(i, max_back=4):
        for j in range(i - 1, max(-1, i - 1 - max_back), -1):
            t = toks[j]
            if t.is_punct or "\n" in t.text or t.pos_ in ("CCONJ", "SCONJ") or j in mods or j in consumed:
                return None
            if t.pos_ in ("VERB", "ADJ") and is_content(t):
                return j
        return None

    def enough_after(target):
        """'not fresh enough', 'nicht frisch genug', 'o suficiente': consume it."""
        for j in range(target + 1, min(n, target + 3)):
            lw = lower(toks[j])
            if lw in enough:
                consumed.add(j)
                return True
            if lw not in ("o", "lo", "le") and toks[j].pos_ != "DET":
                return False
        return False

    def prefer_adjective(target):
        """Linking verb followed by an adjective → modify the adjective."""
        if target is None or toks[target].pos_ not in ("VERB", "AUX"):
            return target
        if toks[target].lemma_.lower() not in LINKING_VERBS:
            return target
        nxt, _ = find_forward(target)
        return nxt if nxt is not None and toks[nxt].pos_ in ("ADJ", "ADV") else target

    def set_negation(target, skipped, start=None):
        after = enough_after(target)
        kind = "notenough" if (skipped & enough) or after else "not"
        if skipped & too_words:
            kind = "nottoo"
        if kind == "notenough" and toks[target].pos_ in ("NOUN", "PROPN"):
            kind = "toolittle"
        if start is not None and start < target:
            # absorb the words between the negation and its target
            # ("not too strong", "pas assez frais") so they don't modify it again
            for j in range(start + 1, target):
                if lower(toks[j]) in modifier_words or lower(toks[j]) in skip_words:
                    consumed.add(j)
        mods[target] = [kind, None]

    for i, t in enumerate(toks):
        if i in consumed or i in mods:
            continue
        low = lower(t)
        only_words = {"en": {"only", "just"}, "fr": {"seulement", "uniquement"},
                      "es": {"solo", "solamente"}, "it": {"solo", "soltanto"},
                      "de": {"nur"}, "pt": {"só", "somente", "apenas"}}
        if low in neg_words and i + 1 < n and lower(toks[i + 1]) in only_words.get(lang_code, set()):
            consumed.update((i, i + 1))
            continue

        # ── Japanese: endings attach backwards ───────────────────────────────
        if lang_code == "ja":
            is_neg = t.lemma_ in JA_NEG or t.text in JA_NEG
            is_too = t.lemma_ in JA_TOO or t.text in JA_TOO
            if is_neg or is_too:
                consumed.add(i)
                j = i - 1
                while j >= 0 and i - j <= 6 and not toks[j].is_punct and (
                        toks[j].lemma_ in JA_SKIP or toks[j].text in JA_SKIP or toks[j].lemma_ in JA_NEG
                        or toks[j].pos_ in ("ADP", "AUX", "PART", "SCONJ")):
                    j -= 1
                if is_neg and j >= 0 and (toks[j].lemma_ in JA_TOO or toks[j].text in JA_TOO):
                    previous = next((k for k in range(j - 1, max(-1, j - 7), -1)
                                     if k in mods and mods[k][0] == "too"), None)
                    if previous is not None:
                        mods[previous] = ["nottoo", doc.text[toks[previous].idx:t.idx + len(t.text)]]
                        continue
                if j < 0 or j in mods or toks[j].is_punct or toks[j].pos_ not in _CONTENT_POS:
                    continue
                kind = "too" if is_too else "not"
                if is_neg and toks[j].lemma_ in JA_ENOUGH_VERBS:      # 新鮮さが足りない
                    k = j - 1
                    while k >= 0 and j - k <= 3 and (toks[k].lemma_ in JA_SKIP or toks[k].pos_ in ("ADP", "PART")):
                        k -= 1
                    if k >= 0 and toks[k].pos_ in _CONTENT_POS and k not in mods:
                        consumed.update(range(k + 1, i))
                        mods[k] = ["notenough", doc.text[toks[k].idx: t.idx + len(t.text)]]
                    continue
                consumed.update(range(j + 1, i))
                mods[j] = [kind, doc.text[toks[j].idx: t.idx + len(t.text)]]
            continue

        # ── French "ne … pas [assez] X" ───────────────────────────────────────
        if lang_code == "fr" and low in FR_NE:
            k = None
            for candidate in range(i + 1, min(n, i + 6)):
                if toks[candidate].is_punct or "\n" in toks[candidate].text:
                    break
                if lower(toks[candidate]) in FR_CLOSERS:
                    k = candidate
                    break
            if k is not None and k + 1 < n and lower(toks[k + 1]) in {"seulement", "uniquement"}:
                consumed.update((i, k, k + 1))
                continue
            consumed.add(i)
            if k is not None:
                consumed.add(k)
                verbs = [j for j in range(i + 1, k) if toks[j].pos_ == "VERB" and is_content(toks[j])]
                fwd, skipped = find_forward(k)
                target = verbs[-1] if verbs else fwd
                if verbs and toks[target].lemma_.lower() in LINKING_VERBS and fwd is not None \
                        and toks[fwd].pos_ in ("ADJ", "ADV"):
                    target = fwd
                if target is not None:
                    set_negation(target, skipped if target == fwd else set(), start=k if target == fwd else None)
            continue

        # ── too / too much / too little ───────────────────────────────────────
        if low in too_words:
            if lang_code == "de" and low == "zu":
                nxt = toks[i + 1] if i + 1 < n else None
                if nxt is None or not (nxt.pos_ in ("ADJ", "ADV") or lower(nxt) in LITTLE_WORDS | MUCH_WORDS):
                    continue                       # "zu Hause", "zu riechen" — not "too"
            j, kind_hint = i + 1, None
            while j < n and j <= i + 2 and lower(toks[j]) in LITTLE_WORDS | MUCH_WORDS:
                kind_hint = "little" if lower(toks[j]) in LITTLE_WORDS else "much"
                j += 1
            target, _ = find_forward(j - 1, max_ahead=2)
            if target is not None:
                consumed.update(range(i, target))
                if kind_hint == "little":
                    mods[target] = ["toolittle" if toks[target].pos_ in ("NOUN", "PROPN") else "notenough", None]
                elif toks[target].pos_ in ("NOUN", "PROPN"):
                    mods[target] = ["toomuch", None]
                elif toks[target].pos_ in ("ADJ", "ADV") or kind_hint == "much":
                    mods[target] = ["too", None]
            continue

        # ── insufficiently / poco / pouco ─────────────────────────────────────
        if low in insufficient:
            if i > 0 and lower(toks[i - 1]) in A_LITTLE:      # "un poco fresco" = a bit fresh
                continue
            target, _ = find_forward(i, max_ahead=2)
            if target is not None and toks[target].pos_ in ("ADJ", "ADV", "NOUN"):
                consumed.add(i)
                mods[target] = ["toolittle" if toks[target].pos_ == "NOUN" else "notenough", None]
            continue

        # ── post-position "too" (pt "forte demais") ────────────────────────────
        if low in too_post:
            j = find_backward(i, max_back=2)
            if j is not None:
                consumed.add(i)
                mods[j] = ["too", None]
            continue

        # ── plain negation (→ "not enough" if an enough-word is involved) ─────
        if low in neg_words:
            consumed.add(i)
            target, skipped = find_forward(i)
            target = prefer_adjective(target)
            if target is not None and target != find_forward(i)[0]:
                skipped = skipped | find_forward(find_forward(i)[0])[1]
            start = i
            if target is None:
                target, skipped, start = find_backward(i), set(), None
            if target is not None:
                set_negation(target, skipped, start=start)

    lemmas = []
    for i, t in enumerate(toks):
        if i in consumed:
            continue
        lemma = t.lemma_.lower() if t.lemma_ else t.lower_
        if i in mods:
            kind, surface = mods[i]
            tok = f"{kind}_{lemma}"
            lemmas.append(tok)
            if neg_surfaces is not None and surface:
                neg_surfaces.setdefault(tok, Counter())[surface] += 1
        elif t.is_alpha:
            lemmas.append(lemma)
        elif not t.is_space or "\n" in t.text:
            lemmas.append(BOUNDARY)
    return lemmas

def _regex_negation(text, lang_code):
    """Fallback when no spaCy model is loaded (pattern rules on raw text):
    'too/trop X' → too_X, 'neg [assez/enough] X [enough]' → notenough_X,
    'neg [intensifiers] X' → not_X."""
    W = r"([^\W\d_]+)"
    too = TOO_WORDS.get(lang_code, set()) - {"zu"}
    enough = ENOUGH_WORDS.get(lang_code, set())
    neg_words = [w for w in NEGATION_WORDS.get(lang_code, []) if w != "n't"]
    alt = lambda ws: "|".join(re.escape(w) for w in sorted(ws, key=len, reverse=True))
    skip = INTENSIFIERS.get(lang_code, set()) - {""} - enough
    skip_re = (r"(?:(?:" + alt(skip) + r")\s+){0,3}") if skip else ""
    neg_skip = (r"(?:(?:" + alt(skip | too) + r")\s+){0,3}") if (skip or too) else ""
    only = {"en": "only|just", "fr": "seulement|uniquement", "es": "solo|solamente",
            "it": "solo|soltanto", "de": "nur", "pt": "só|somente|apenas"}.get(lang_code)
    if only and neg_words:
        text = re.sub(r"\b(?:" + alt(neg_words) + r")\s+(?:" + only + r")\b", "", text)
    if neg_words and too:
        text = re.sub(r"\b(?:" + alt(neg_words) + r")\s+(?:" + alt(too) + r")\s+" + W,
                      r"nottoo_\1", text)
    if neg_words and enough:      # "not enough fresh", "pas assez frais", "not fresh enough"
        text = re.sub(r"\b(?:" + alt(neg_words) + r")\s+" + skip_re + r"(?:" + alt(enough) + r")\s+" + skip_re + W,
                      r"notenough_\1", text)
        text = re.sub(r"\b(?:" + alt(neg_words) + r")\s+" + skip_re + W + r"\s+(?:" + alt(enough) + r")\b",
                      r"notenough_\1", text)
    if neg_words:                 # plain negation after not-too has been handled
        text = re.sub(r"\b(?:" + alt(neg_words) + r")\s+" + neg_skip + W, r"not_\1", text)
    if lang_code == "de":
        text = re.sub(r"\bzu\s+wenig\s+" + W, r"toolittle_\1", text)
        text = re.sub(r"\bzu\s+viel(?:e|en)?\s+" + W, r"toomuch_\1", text)
    if too:
        text = re.sub(r"\b(?:" + alt(too) + r")\s+(?:peu|little)\s+(?:de\s+|d'|of\s+)" + W, r"toolittle_\1", text)
        text = re.sub(r"\b(?:" + alt(too) + r")\s+(?:peu|little)\s+" + W, r"notenough_\1", text)
        text = re.sub(r"\b(?:" + alt(too) + r")\s+(?:much|de|d')\s*" + W, r"toomuch_\1", text)
        text = re.sub(r"\b(?:" + alt(too) + r")\s+" + W, r"too_\1", text)
    return text

def preprocess(text, lang_code, lemmatizer_en, spacy_nlp, custom_stops, stop_words,
               jieba_mod=None, thai_tok=None, pre_lemmas=None, neg_surfaces=None, retain_structure=False):
    """Returns the filtered lemma list for one verbatim.
    pre_lemmas: lemmas already produced by Stanza (Arabic / Indonesian).
    neg_surfaces: optional dict collecting original wordings of negated tokens."""
    if pre_lemmas is not None:
        lemmas = [clean_token(t, lang_code) for t in pre_lemmas]
        lemmas = [t if is_word(t) else BOUNDARY for t in lemmas]
    else:
        if not isinstance(text, str) or not text.strip():
            return []
        text = _norm_apostrophes(text.lower())
        if lang_code == "zh":
            if jieba_mod is not None:
                lemmas = [t if is_word(t) else BOUNDARY for t in jieba_mod.cut(text) if t.strip() or "\n" in t]
            else:
                lemmas = [c if is_word(c) else BOUNDARY for c in text if c.strip()]
        elif lang_code == "th":
            if thai_tok is not None:
                lemmas = [t.strip() for t in thai_tok(text)]
            else:
                lemmas = re.findall(r"[^\W\d_]+|[^\w\s]", text)
            lemmas = [t if is_word(t) else BOUNDARY for t in lemmas]
        elif spacy_nlp is not None:
            lemmas = spacy_negation_lemmas(spacy_nlp(text), lang_code, neg_surfaces)
        else:
            text = _regex_negation(text, lang_code)
            tokens = re.findall(r"(?:nottoo|notenough|toolittle|toomuch|too|not)_[^\W\d_]+|[^\W\d_]+|\d+|[^\w\s]|[\r\n]+", text, flags=re.UNICODE)
            tokens = [t if any(c.isalpha() for c in t) else BOUNDARY for t in tokens]
            if lang_code == "en":
                lemmas = [t if "_" in t else lemmatizer_en.lemmatize(t) for t in tokens]
            else:
                lemmas = tokens

    if lang_code in TOKEN_NEGATION_LANGS:
        lemmas = merge_negations(lemmas, lang_code, stop_words | custom_stops)

    if retain_structure:
        return lemmas
    return [lemma for lemma in lemmas
            if lemma != BOUNDARY and lemma not in stop_words and lemma not in custom_stops]

def build_subcorpus_mask(texts, keywords, lang_code=None):
    """True if the RAW verbatim contains at least one keyword (case-insensitive).
    Substring matching for scripts without spaces (zh / ja / th), whole-word
    matching otherwise."""
    if not keywords:
        return [True] * len(texts)
    if lang_code in ("zh", "ja", "th"):
        kws = [kw.lower() for kw in keywords]
        return [isinstance(t, str) and t.strip() != "" and any(k in t.lower() for k in kws) for t in texts]
    patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE | re.UNICODE) for kw in keywords]
    return [isinstance(t, str) and t.strip() != "" and any(p.search(t) for p in patterns) for t in texts]

# ─── Spelling correction (optional, English only) ─────────────────────────────
@st.cache_resource
def load_spellchecker(lang_code):
    try:
        from spellchecker import SpellChecker
        return SpellChecker(language=lang_code, distance=1)
    except Exception:
        return None

def build_spelling_corrections(texts, spellchecker):
    word_counts = Counter()
    for t in texts:
        if isinstance(t, str):
            word_counts.update(WORD_PATTERN.findall(t.lower()))
    if not word_counts:
        return {}, 0
    correction_map = {}
    for w in spellchecker.unknown(word_counts.keys()):
        corrected = spellchecker.correction(w)
        if corrected and corrected != w:
            correction_map[w] = corrected
    return correction_map, sum(word_counts[w] for w in correction_map)

def apply_spelling_corrections(text, correction_map):
    if not isinstance(text, str) or not text.strip() or not correction_map:
        return text
    def repl(m):
        word = m.group(0)
        corrected = correction_map.get(word.lower())
        if not corrected:
            return word
        if word.isupper():
            return corrected.upper()
        if word[0].isupper():
            return corrected[:1].upper() + corrected[1:]
        return corrected
    return WORD_PATTERN.sub(repl, text)

def display_label(word, lang_code=None, neg_display=None):
    """Internal token → label shown in the cloud.
    • phrases 'easy_apply' → 'easy apply' (Thai joined without spaces)
    • 'not_frais' → 'pas frais', 'notenough_frais' → 'pas assez frais',
      'too_fort' → 'trop fort', 'not_新鲜' → '不新鲜'; Japanese → most frequent
      original wording ('強くない', '強すぎる')."""
    if neg_display and word in neg_display:
        return neg_display[word]
    parts = word.split("_")
    out, i = [], 0
    while i < len(parts):
        p = parts[i]
        if p in MOD_KINDS and i + 1 < len(parts):
            nxt = parts[i + 1]
            key = f"{p}_{nxt}"
            if neg_display and key in neg_display:
                out.append(neg_display[key])
            elif p == "not":
                if lang_code == "ja":
                    out.append(nxt + "ない")
                elif lang_code == "zh":
                    out.append("不" + nxt)
                else:
                    out.append(f"{NEG_DISPLAY_PREFIX.get(lang_code, 'not')} {nxt}")
            else:
                out.append(MOD_DISPLAY[p].get(lang_code, MOD_DISPLAY[p]["en"]).format(nxt))
            i += 2
        else:
            out.append(p)
            i += 1
    return ("" if lang_code == "th" else " ").join(out)

# ─── TF-IDF ───────────────────────────────────────────────────────────────────
def compute_tfidf_scores(token_lists):
    docs = [" ".join(toks) for toks in token_lists if toks]
    if not docs:
        return {}
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(), preprocessor=lambda x: x,
        token_pattern=None, lowercase=False,
    )
    matrix = vectorizer.fit_transform(docs)
    return dict(zip(vectorizer.get_feature_names_out(), matrix.sum(axis=0).A1))

# ─── Clustering (runs in the background, only used for colouring) ─────────────
def _louvain(G, resolution, seed):
    if community_louvain is not None:
        return community_louvain.best_partition(G, resolution=resolution, random_state=seed)
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    return {n: i for i, c in enumerate(comms) for n in c}

def cluster_to_target(G, target_n, seed=42):
    """Searches Louvain's resolution to land as close as possible to the target
    number of clusters. Disconnected components are a hard floor."""
    n_components = nx.number_connected_components(G)
    resolutions = [
        0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0,
        7.0, 10.0, 15.0, 20.0,
    ]
    best_p, best_diff = None, 10 ** 9
    for res in resolutions:
        p = _louvain(G, res, seed)
        diff = abs(len(set(p.values())) - target_n)
        if diff < best_diff:
            best_diff, best_p = diff, p
        if diff == 0:
            break
    return best_p, best_diff, n_components

def renumber_by_size(partition, word_freq):
    """Cluster 1 = cluster with the highest total frequency, etc."""
    totals = Counter()
    for w, c in partition.items():
        totals[c] += word_freq.get(w, 0)
    order = [c for c, _ in totals.most_common()]
    remap = {old: new for new, old in enumerate(order)}
    return {w: remap[c] for w, c in partition.items()}

# ─── Translation to English (Helsinki-NLP opus-mt, runs locally) ──────────────
# Only the words that end up in the cloud are translated (never the verbatims),
# AFTER the analysis — lemmatization, stopwords, bigrams and clustering all
# stay in the source language. Models run inside the app (nothing is sent to
# an external service); each is ~300 MB, downloaded from huggingface.co on
# first use, and `transformers` is only imported when translation is used.
# Candidates are tried in order: direct pair first, then multilingual fallbacks.
HELSINKI_MODELS = {
    "fr": ["Helsinki-NLP/opus-mt-fr-en", "Helsinki-NLP/opus-mt-ROMANCE-en", "Helsinki-NLP/opus-mt-mul-en"],
    "es": ["Helsinki-NLP/opus-mt-es-en", "Helsinki-NLP/opus-mt-ROMANCE-en", "Helsinki-NLP/opus-mt-mul-en"],
    "it": ["Helsinki-NLP/opus-mt-it-en", "Helsinki-NLP/opus-mt-ROMANCE-en", "Helsinki-NLP/opus-mt-mul-en"],
    "pt": ["Helsinki-NLP/opus-mt-ROMANCE-en", "Helsinki-NLP/opus-mt-mul-en"],
    "de": ["Helsinki-NLP/opus-mt-de-en", "Helsinki-NLP/opus-mt-mul-en"],
    "zh": ["Helsinki-NLP/opus-mt-zh-en", "Helsinki-NLP/opus-mt-mul-en"],
    "ja": ["Helsinki-NLP/opus-mt-ja-en", "Helsinki-NLP/opus-mt-mul-en"],
    "th": ["Helsinki-NLP/opus-mt-th-en", "Helsinki-NLP/opus-mt-mul-en"],
    "id": ["Helsinki-NLP/opus-mt-id-en", "Helsinki-NLP/opus-mt-mul-en"],
    "ar": ["Helsinki-NLP/opus-mt-ar-en", "Helsinki-NLP/opus-mt-mul-en"],
}

@st.cache_resource(show_spinner=False)
def load_translator(lang_code):
    """Returns ((tokenizer, model, model_name), error). Lazy: transformers and
    torch are imported only here."""
    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except Exception as e:
        return None, f"transformers / torch not installed ({type(e).__name__}: {e})"
    last_err = None
    for name in HELSINKI_MODELS.get(lang_code, []):
        try:
            tok = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name)
            model.eval()
            return (tok, model, name), None
        except Exception as e:
            last_err = f"{name}: {type(e).__name__}: {e}"
    return None, last_err or f"No translation model configured for '{lang_code}'"

_TRANSLATION_STRIP = " \t\n.,;:!?\"'«»“”‘’()[]{}…-–—"

def clean_translation(source, text):
    """Tidy a single-word/phrase translation: no trailing punctuation, lower
    case, and a cap on length (sentence-trained models sometimes add words)."""
    t = (text or "").strip().strip(_TRANSLATION_STRIP)
    if not t:
        return source
    words = t.split()
    cap = max(3, 2 * len(source.split()) + 1)
    return " ".join(words[:cap]).lower()

def translate_words(words, lang_code, label_fn):
    """{internal_word: english}. Results are cached per session and per
    language, so changing font/size/scope never re-translates. The model is
    only loaded if some word isn't cached yet. Returns (mapping, error)."""
    cache = st.session_state.setdefault("translation_cache", {}).setdefault(lang_code, {})
    todo = [w for w in words if w not in cache]
    if todo:
        loaded, err = load_translator(lang_code)
        if loaded is None:
            load_translator.clear()  # allow a retry later
            return {}, err
        tok, model, _ = loaded
        import torch
        # Modifier tokens ("not_x", "notenough_x", "too_x", "toomuch_x"):
        # translate the word itself and wrap it ("not … enough", "too …").
        sources = []
        for w in todo:
            kind = next((k for k in MOD_KINDS if w.startswith(k + "_") and "_" not in w[len(k) + 1:]), None)
            if kind:
                pre, suf = MOD_ENGLISH[kind]
                sources.append((pre, suf, label_fn(w[len(kind) + 1:])))
            else:
                sources.append(("", "", label_fn(w)))
        for start in range(0, len(todo), 32):
            batch = sources[start:start + 32]
            inputs = tok([s for _, _, s in batch], return_tensors="pt", padding=True,
                         truncation=True, max_length=32)
            with torch.no_grad():
                out = model.generate(**inputs, num_beams=4, max_new_tokens=16)
            decoded = tok.batch_decode(out, skip_special_tokens=True)
            for w, (prefix, suffix, src), txt in zip(todo[start:start + 32], batch, decoded):
                cache[w] = prefix + clean_translation(src, txt) + suffix
    return {w: cache[w] for w in words}, None

# ─── Word cloud ────────────────────────────────────────────────────────────────
CANVAS_PRESETS = {
    "Landscape (1600 × 800)": (1600, 800),
    "Wide 16:9 (1600 × 900)": (1600, 900),
    "Square (1200 × 1200)":   (1200, 1200),
    "Portrait (900 × 1200)":  (900, 1200),
    "Banner (2000 × 600)":    (2000, 600),
}

def render_wordcloud(freqs, color_of, font_path, width, height, max_words, min_font,
                     max_font, relative_scaling, margin, background, scale, lang_code, label_fn=None):
    """freqs: {internal_word: weight}; color_of: internal_word → 'rgb(...)'.
    Always horizontal (prefer_horizontal=1.0)."""
    disp_to_orig = {}
    disp_freqs = {}
    for w, v in freqs.items():
        d = shape_for_render(label_fn(w) if label_fn else display_label(w, lang_code), lang_code)
        disp_to_orig[d] = w
        disp_freqs[d] = disp_freqs.get(d, 0) + v

    kwargs = dict(
        width=width, height=height, background_color=background,
        prefer_horizontal=1.0, max_words=max_words,
        min_font_size=min_font, max_font_size=max_font,
        relative_scaling=relative_scaling, margin=margin,
        scale=scale, random_state=42, font_step=1, normalize_plurals=False,
    )
    if font_path:
        kwargs["font_path"] = font_path
    wc = WordCloud(**kwargs).generate_from_frequencies(disp_freqs)

    def _color_func(word, font_size, position, orientation, random_state=None, **kw):
        return color_of(disp_to_orig.get(word, word))

    wc.recolor(color_func=_color_func, random_state=42)
    return wc


def analyse_subset(token_lists, min_freq, min_edge, n_clusters):
    """Frequencies, TF-IDF and background Louvain clusters for one set of
    verbatims (the whole corpus, or one product code). Fast — tokens are
    already computed, so switching product only redoes counting/clustering."""
    token_lists = list(token_lists)
    word_freq = Counter(itertools.chain.from_iterable(token_lists))
    response_counts, response_percent = response_statistics(token_lists)
    pair_counts = Counter()
    for tokens in token_lists:
        ut = sorted(set(t for t in tokens if word_freq[t] >= min_freq))
        pair_counts.update(itertools.combinations(ut, 2))
    G = nx.Graph()
    for (u, v), w in pair_counts.items():
        if w >= min_edge:
            G.add_edge(u, v, weight=w)
    if G.number_of_nodes() > 0:
        best_p, best_d, n_components = cluster_to_target(G, n_clusters)
        best_p = renumber_by_size(best_p, word_freq)
    else:
        best_p, best_d, n_components = {}, 0, 0
    return {
        "word_freq": word_freq,
        "response_counts": response_counts,
        "response_percent": response_percent,
        "tfidf_scores": compute_tfidf_scores(token_lists),
        "best_p": best_p,
        "cluster_ids": sorted(set(best_p.values())),
        "cluster_match_diff": best_d,
        "n_components": n_components,
        "n_rows": len(token_lists),
    }

TOTAL_CODES = "Total (all codes)"

# ─── Sidebar (unchanged layout) ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    lang_choice = st.selectbox("🌍 Verbatim language", list(LANGUAGES.keys()), index=0, key="lang_choice")
    lang_info = LANGUAGES[lang_choice]

    if lang_choice in SUGGESTED_EXTRA_EXCLUSIONS and st.session_state.get("_excl_prefilled_for") != lang_choice:
        st.session_state["user_extra_stops"] = ", ".join(SUGGESTED_EXTRA_EXCLUSIONS[lang_choice])
        st.session_state["_excl_prefilled_for"] = lang_choice

    # Only the selected language's tools are touched here. Stanza is NOT
    # loaded on selection — only when "Generate" is clicked.
    if lang_info["engine"] == "spacy":
        _spacy_model_check, _spacy_error = load_spacy_model(lang_info["spacy_model"])
        if _spacy_model_check is None:
            fallback_note = (
                "falling back to NLTK's noun-only lemmatizer"
                if lang_choice == "English"
                else "falling back to basic tokenization with no real lemmatization"
            )
            st.caption(f"⚠️ spaCy model **{lang_info['spacy_model']}** failed to load — {fallback_note}.")
            if _spacy_error:
                st.caption(f"Technical detail: `{_spacy_error}`")
    elif lang_info["engine"] == "jieba":
        _jieba_check, _jieba_error = load_jieba()
        if _jieba_check is None:
            st.caption("⚠️ jieba (Chinese segmentation) failed to load — falling back to character splitting.")
            if _jieba_error:
                st.caption(f"Technical detail: `{_jieba_error}`")
    elif lang_info["engine"] == "thai":
        _thai_tok, _thai_sw, _thai_err = load_thai()
        if _thai_tok is None:
            st.caption("⚠️ PyThaiNLP failed to load — Thai text will only be split on spaces (poor results).")
            if _thai_err:
                st.caption(f"Technical detail: `{_thai_err}`")
    elif lang_info["engine"] == "stanza":
        st.caption(
            f"🧠 {lang_choice} is lemmatized with **Stanza**. It is loaded only when you click "
            "**Generate** — the first run downloads the model (≈ 100–300 MB, ~1 min), later runs reuse it. "
            "Stanza is slower than spaCy: allow roughly 1 min per 1,000 verbatims."
        )
    if lang_choice != "English":
        st.caption("ℹ️ Spelling correction is English-only and is disabled for this language.")
    uploaded_file = st.file_uploader("📂 Upload Excel corpus", type=["xlsx"])

    sheet_choice = None
    df = None
    subtarget_col_choice = "None"
    subtarget_values = []

    if uploaded_file:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        if len(sheet_names) > 1:
            sheet_choice = st.selectbox("Sheet", sheet_names, index=0, key="sheet_choice")
        else:
            sheet_choice = sheet_names[0]
        df = excel_file.parse(sheet_name=sheet_choice)

        st.markdown("---")
        st.caption("🎯 Sub-target filter (optional)")
        subtarget_col_choice = st.selectbox(
            "Restrict the analysis to a sub-target:",
            ["None"] + list(df.columns),
            key="subtarget_col_choice",
            help="Restricts the ENTIRE analysis to only the rows matching the value(s) you pick below.",
        )
        if subtarget_col_choice != "None":
            value_counts = df[subtarget_col_choice].astype(str).value_counts()
            value_options = value_counts.index.tolist()
            subtarget_values = st.multiselect(
                "Value(s) to keep:",
                value_options,
                default=value_options[:1],
                key="subtarget_values",
                format_func=lambda v: f"{v} ({value_counts.get(v, 0)})",
            )
            if subtarget_values:
                n_before_subtarget = len(df)
                df = df[df[subtarget_col_choice].astype(str).isin(subtarget_values)].reset_index(drop=True)
                st.caption(
                    f"🎯 Sub-target active — **{len(df)} of {n_before_subtarget}** rows kept "
                    f"({subtarget_col_choice}: {', '.join(subtarget_values)})."
                )
            else:
                st.caption("⚠️ No value selected — pick at least one, or the whole corpus is used.")

    st.markdown("---")
    min_freq = st.slider("Min word occurrences", 1, 50, 5)
    min_edge = st.slider(
        "Min connection strength", 1, 20, 3,
        help="How many verbatims two words must share before they are linked for the "
             "background clustering that decides the word colours. Higher = stricter "
             "communities; words with no link above this threshold are shown in grey.",
    )
    n_clusters = st.slider(
        "Target number of clusters", 2, 10, 5,
        help="Number of colour communities to aim for (best-effort — fully disconnected "
             "groups of words can't be merged, so they set a floor).",
    )
    st.markdown("---")
    st.caption("🔤 Phrases & weighting")
    term_mode = st.selectbox("What to display", ["Mixed", "Words", "Phrases"], key="term_mode",
                             help="Mixed replaces overlapping words with selected phrases. Words preserves individual-word counts.")
    max_phrase_n = st.selectbox("Maximum phrase length", [2, 3], index=1)
    min_phrase_responses = st.slider("Min responses containing a phrase", 2, 20, 4)
    min_phrase_pmi = st.slider("Min phrase association (PMI)", 0.0, 10.0, 0.0, 0.5)
    weight_mode = st.selectbox("How to size terms", ["Percentage of responses", "Total occurrences", "TF-IDF"])
    use_tfidf = weight_mode == "TF-IDF"
    st.caption("Phrase settings apply after Generate. Display and sizing update instantly.")
    use_spellcheck = st.checkbox(
        "✏️ Correct spelling before analysis (beta, English only)", value=False, key="use_spellcheck",
        disabled=(lang_info["code"] != "en"),
        help="Fixes likely typos (single-character edits only) before tokenizing. Only "
             "affects the analysis, never the original verbatims.",
    )
    st.markdown("---")
    st.caption("ℹ️ After adding words here, click **Generate** again to regenerate the analysis for newly excluded words.")
    user_extra_stops = st.text_area("Extra exclusion words (comma-sep):", "", key="user_extra_stops")
    st.markdown("---")
    st.caption("🎯 Sub-corpus filter (optional)")
    subcorpus_input = st.text_area(
        "Only analyze verbatims containing at least one of these words/phrases:",
        "",
        key="subcorpus_input",
        placeholder="e.g. fresh, clean",
        help="Leave empty to use the full corpus. Matches on the ORIGINAL verbatim text, "
             "case-insensitive (whole word, except Chinese/Japanese/Thai: substring).",
    )
    subcorpus_words = [w.strip().lower() for w in subcorpus_input.split(",") if w.strip()]

_lang_code_now = lang_info["code"]
all_stops = set(
    clean_token(w, _lang_code_now)
    for w in DEFAULT_EXCLUSIONS_BY_LANG.get(lang_choice, [])
    + [w.strip() for w in user_extra_stops.split(",") if w.strip()]
)

# ─── Main ────────────────────────────────────────────────────────────────────
st.title("☁️ Semantic Word Cloud")

if uploaded_file and df is not None:
    tc1, tc2 = st.columns(2)
    col = tc1.selectbox("Text column", df.columns)
    tc2.selectbox(
        "🏷️ Product code column (optional)",
        ["None"] + [c for c in df.columns if c != col],
        key="product_col_choice",
        help="Choose the column holding the product code: a selector then appears above the word "
             "cloud to restrict it to one code's verbatims, or show the total. Can be changed after "
             "generating — no need to regenerate.",
    )

    if subcorpus_words:
        preview_mask = build_subcorpus_mask(df[col].tolist(), subcorpus_words, lang_code=lang_info["code"])
        st.caption(
            f"🎯 Sub-corpus filter active — **{sum(preview_mask)} of {len(df)}** verbatims match "
            f"({', '.join(subcorpus_words)}). The analysis will run on this subset only."
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("🔀 Group comparison")
    group_col_choice = st.sidebar.selectbox(
        "Compare groups by (optional)",
        ["None"] + [c for c in df.columns if c != col],
        key="group_col_choice",
        help="Adds a selector above the word cloud to draw it for one group at a time — "
             "colours stay tied to the same clusters, so groups are directly comparable.",
    )

    if st.button("🚀 Generate word cloud", use_container_width=True):
        code = lang_info["code"]
        engine = lang_info["engine"]
        lemmatizer_en = load_lemmatizer()
        spacy_nlp = load_spacy_model(lang_info["spacy_model"])[0] if engine == "spacy" else None
        jieba_mod = load_jieba()[0] if engine == "jieba" else None
        thai_tok, thai_sw = (None, set())
        if engine == "thai":
            thai_tok, thai_sw, _ = load_thai()
        stop_words = set(clean_token(w, code) for w in load_stopwords(lang_info["nltk_stop"])) | thai_sw

        stanza_nlp = None
        if engine == "stanza":
            with st.spinner(f"Loading Stanza for {lang_choice} (first time: downloading the model)…"):
                stanza_nlp, stanza_err = load_stanza(code)
            if stanza_nlp is None:
                st.error(
                    f"Stanza could not be loaded for {lang_choice} — check that `stanza` and `torch` "
                    f"are in requirements.txt and that the app can reach huggingface.co.\n\n"
                    f"Technical detail: `{stanza_err}`"
                )
                load_stanza.clear()  # allow a retry on the next click
                st.stop()

        with st.spinner("Analysing text and clustering words…"):
            n_before_filter = len(df)
            if subcorpus_words:
                mask = build_subcorpus_mask(df[col].tolist(), subcorpus_words, lang_code=code)
                df = df.loc[mask].reset_index(drop=True)
                if df.empty:
                    st.warning("No verbatims match your sub-corpus filter words.")
                    st.stop()

            analysis_col = col
            spelling_corrections, spelling_corrected_occurrences = {}, 0
            if use_spellcheck and code == "en":
                spellchecker = load_spellchecker(code)
                if spellchecker is None:
                    st.warning("Spelling correction isn't available right now — skipped.")
                else:
                    spelling_corrections, spelling_corrected_occurrences = build_spelling_corrections(
                        df[col].tolist(), spellchecker
                    )
                    if spelling_corrections:
                        df["_analysis_text"] = df[col].apply(
                            lambda t: apply_spelling_corrections(t, spelling_corrections)
                        )
                        analysis_col = "_analysis_text"

            # Tokenise / lemmatise
            neg_surfaces = {}   # Japanese: original wording of each negated token
            if engine == "stanza":
                bar = st.progress(0.0, text=f"Stanza lemmatization ({len(df)} verbatims)…")
                raw = df[analysis_col].tolist()
                stanza_lemmas = stanza_lemmatize(
                    raw, stanza_nlp, progress_cb=lambda p: bar.progress(p, text=f"Stanza lemmatization… {int(p*100)}%")
                )
                bar.empty()
                df["raw_tokens"] = [
                    preprocess(None, code, lemmatizer_en, None, all_stops, stop_words, pre_lemmas=lem, retain_structure=True)
                    for lem in stanza_lemmas
                ]
            else:
                df["raw_tokens"] = df[analysis_col].apply(
                    lambda x: preprocess(x, code, lemmatizer_en, spacy_nlp, all_stops, stop_words,
                                         jieba_mod=jieba_mod, thai_tok=thai_tok, neg_surfaces=neg_surfaces, retain_structure=True)
                )

            phrase_stops = stop_words | all_stops
            phrases = extract_phrases(df["raw_tokens"].tolist(), phrase_stops, code,
                                      min_phrase_responses, max_phrase_n, min_phrase_pmi)
            df["tokens"] = df["raw_tokens"].apply(lambda ts: select_terms(ts, phrases, phrase_stops, "Words"))

            if not any(len(t) for t in df["tokens"]):
                st.warning("No words left after filtering — check the language, text column or exclusions.")
                st.stop()

            # Keep every original column (so the product-code column can be
            # changed afterwards without regenerating); drop helper columns.
            st.session_state["results"] = {
                "phrases": phrases,
                "phrase_stops": phrase_stops,
                "resp_df": df.drop(columns=["_analysis_text"], errors="ignore").copy(),
                "text_col": col,
                "group_col": group_col_choice if group_col_choice != "None" else None,
                "lang_code": code,
                "lang_name": lang_choice,
                "neg_display": {t: c.most_common(1)[0][0] for t, c in neg_surfaces.items()},
                "run_id": st.session_state.get("results", {}).get("run_id", 0) + 1,
                "n_before_filter": n_before_filter,
                "n_after_filter": len(df),
                "subcorpus_words": list(subcorpus_words),
                "subtarget_col": subtarget_col_choice if subtarget_col_choice != "None" else None,
                "subtarget_values": list(subtarget_values),
                "spelling_corrections": spelling_corrections,
                "spelling_corrected_occurrences": spelling_corrected_occurrences,
            }
            # Colours are per cluster rank (C1 = biggest cluster), shared by
            # every product code.
            st.session_state["cluster_colors"] = {i: c for i, c in enumerate(CLUSTER_COLORS)}
            st.session_state["analysis_cache"] = {}
            st.session_state.pop("codes_zip", None)

# ── Render from session_state ────────────────────────────────────────────────
if "results" in st.session_state:
    res = st.session_state["results"]
    full_df      = res["resp_df"]
    group_col    = res.get("group_col")
    lang_code    = res["lang_code"]
    res_min_freq = min_freq          # live sidebar values: re-clustering is instant
    dl = lambda w: display_label(w, lang_code, res.get("neg_display"))
    phrases = res.get("phrases", {})
    with st.expander("Review detected phrases", expanded=False):
        st.caption("Counts describe candidates before overlap resolution. Negation/modifier expressions count as one token. Longer phrases take priority; ties use response support and association.")
        excluded = st.multiselect("Exclude phrases", list(phrases),
                                 format_func=lambda p: dl("_".join(p)),
                                 key=f"excluded_phrases_{res.get('run_id')}")
        review_rows = [{"Phrase": dl("_".join(p)), "Tokens": len(p),
                        "Occurrences": m["occurrences"], "Responses": m["responses"],
                        "PMI": round(m["pmi"], 2),
                        "Examples": " | ".join(str(full_df.iloc[i][res["text_col"]]) for i in m["examples"])}
                       for p, m in sorted(phrases.items(), key=lambda item: (-item[1]["responses"], -len(item[0]), item[0]))]
        if review_rows:
            st.dataframe(pd.DataFrame(review_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No phrases meet the current thresholds. Lower the phrase threshold and Generate again.")
    active_phrases = {p: m for p, m in phrases.items() if p not in excluded}
    full_df = full_df.copy()
    if "raw_tokens" in full_df:
        full_df["tokens"] = full_df["raw_tokens"].apply(
            lambda ts: select_terms(ts, active_phrases, res["phrase_stops"], term_mode))
    view_key = (term_mode, tuple(sorted(excluded)))
    if st.session_state.get("analysis_view_key") != view_key:
        st.session_state["analysis_cache"] = {}
        st.session_state.pop("codes_zip", None)
        st.session_state["analysis_view_key"] = view_key
    analysis_cache = st.session_state.setdefault("analysis_cache", {})

    def get_analysis(product_col, code_value, rows):
        key = (res.get("run_id"), product_col, code_value, min_freq, min_edge, n_clusters)
        if key not in analysis_cache:
            analysis_cache[key] = analyse_subset(rows["tokens"].tolist(), min_freq, min_edge, n_clusters)
        return analysis_cache[key]

    # ── Product code split ───────────────────────────────────────────────────
    product_col = st.session_state.get("product_col_choice", "None")
    if product_col not in full_df.columns:
        product_col = None
    product_value = TOTAL_CODES
    code_counts = None
    if product_col:
        code_counts = full_df[product_col].astype(str).value_counts().sort_index()
        product_value = st.selectbox(
            f"🏷️ Product code ({product_col})",
            [TOTAL_CODES] + code_counts.index.tolist(),
            key=f"product_value_{product_col}",
            format_func=lambda v: (f"{v} — {len(full_df)} verbatims" if v == TOTAL_CODES
                                   else f"{v} — {code_counts.get(v, 0)} verbatims"),
            help="Restricts the whole analysis (frequencies, TF-IDF, colour clusters) to the verbatims "
                 "of this code. 'Total' uses every code.",
        )
    if product_col and product_value != TOTAL_CODES:
        resp_df = full_df[full_df[product_col].astype(str) == product_value].reset_index(drop=True)
    else:
        resp_df = full_df

    with st.spinner("Clustering words…"):
        an = get_analysis(product_col, product_value, resp_df)
    word_freq    = an["word_freq"]
    tfidf_scores = an["tfidf_scores"]
    best_p       = an["best_p"]
    cluster_ids  = an["cluster_ids"]
    res["cluster_match_diff"] = an["cluster_match_diff"]
    res["n_components"] = an["n_components"]
    res["target_n_clusters"] = n_clusters
    if product_col and product_value != TOTAL_CODES and len(resp_df) < 30:
        st.caption(f"ℹ️ Only {len(resp_df)} verbatims for this code — lower 'Min word occurrences' / "
                   "'Min connection strength' in the sidebar if the cloud looks empty or all grey.")

    # English translation switch: the widget itself is drawn just above the
    # word cloud (further down); its current value is read here because the
    # font list depends on it. Off by default → words stay in the local language.
    translate_on = lang_code != "en" and bool(st.session_state.get("translate_to_en", False))

    def cluster_members(cid, freq_source):
        return sorted((w for w, c in best_p.items() if c == cid), key=lambda w: -freq_source.get(w, 0))

    # ── Status captions ──────────────────────────────────────────────────────
    if res.get("spelling_corrections"):
        sc = res["spelling_corrections"]
        with st.expander(
            f"✏️ Spelling correction applied — {len(sc)} unique words changed "
            f"({res['spelling_corrected_occurrences']} occurrences). Click to review."
        ):
            rows = sorted(sc.items())
            st.dataframe(pd.DataFrame(rows, columns=["Original", "Corrected"]),
                         use_container_width=True, height=min(300, 40 + 35 * len(rows)))
    if res.get("subtarget_col"):
        st.caption(f"🎯 Sub-target active — **{res['subtarget_col']}: {', '.join(res['subtarget_values'])}**.")
    if res.get("subcorpus_words"):
        st.caption(
            f"🎯 Sub-corpus filter applied — **{res['n_after_filter']} of {res['n_before_filter']}** verbatims "
            f"matching: {', '.join(res['subcorpus_words'])}."
        )
    if not cluster_ids:
        st.caption("ℹ️ No word pairs reach the 'Min connection strength' threshold — every word is shown in grey. "
                   "Lower that slider (or 'Min word occurrences') and regenerate to get cluster colours.")
    elif res.get("cluster_match_diff"):
        nc = res.get("n_components") or 0
        tgt = res.get("target_n_clusters")
        why = (f"the co-occurrence graph splits into {nc} disconnected groups, which is a hard floor"
               if tgt and tgt < nc else "the closest Louvain could get")
        st.caption(f"Found **{len(cluster_ids)} colour clusters** (target {tgt}) — {why}. "
                   "Adjust 'Min connection strength' / 'Min word occurrences' to reshape them.")

    # ── Cluster colours ──────────────────────────────────────────────────────
    if cluster_ids:
        with st.expander("🎨 Cluster colours", expanded=False):
            per_row = 6
            for row_start in range(0, len(cluster_ids), per_row):
                row_ids = cluster_ids[row_start:row_start + per_row]
                pcols = st.columns(per_row)
                for j, cid in enumerate(row_ids):
                    members = cluster_members(cid, word_freq)
                    label = dl(members[0]) if members else f"Cluster {cid + 1}"
                    picked = pcols[j].color_picker(
                        f"C{cid + 1} · {label}",
                        value=st.session_state["cluster_colors"].get(cid, CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]),
                        key=f"color_cluster_{cid}",
                    )
                    st.session_state["cluster_colors"][cid] = picked
    color_map = st.session_state.get("cluster_colors", {})

    # ── Appearance: font & size ──────────────────────────────────────────────
    fonts = available_fonts()
    script = "latin" if translate_on else LANG_SCRIPT.get(lang_code, "latin")
    font_labels = list(fonts.keys())

    with st.expander("🔠 Font & word size", expanded=True):
        fc1, fc2 = st.columns([2, 1])
        custom_font = fc2.file_uploader(
            "…or upload a font (.ttf / .otf)", type=["ttf", "otf", "ttc"], key="custom_font",
            help="Use any font file (e.g. a brand font). It must contain the glyphs of your language.",
        )
        if custom_font is not None:
            cpath = save_uploaded_font(custom_font.getvalue(), custom_font.name)
            fonts = dict(fonts)
            fonts[f"⬆ {custom_font.name}"] = (cpath, script)
            font_labels = [f"⬆ {custom_font.name}"] + font_labels

        # Script-appropriate fonts first, so the default renders correctly.
        same_script = [l for l in font_labels if fonts[l][1] == script or l.startswith("⬆")]
        others = [l for l in font_labels if l not in same_script]
        ordered = same_script + others
        if not ordered:
            ordered = ["WordCloud default"]
        font_key = f"cloud_font_{script}"
        if st.session_state.get(font_key) not in ordered:
            st.session_state.pop(font_key, None)
        font_choice = fc1.selectbox(
            "Font", ordered, index=0, key=font_key,
            format_func=lambda l: l if (script == "latin" or l.startswith("⬆") or fonts.get(l, (None, ""))[1] == script)
                                  else f"{l}  (may lack {script} glyphs)",
        )
        font_path = fonts.get(font_choice, (None, None))[0]

        sc1, sc2, sc3, sc4 = st.columns(4)
        max_words = sc1.slider("Max number of words", 10, 400, 150, step=10, key="wc_max_words")
        font_range = sc2.slider("Word size range (px)", 4, 300, (10, 130), key="wc_font_range",
                                help="Smallest and largest font size. Words that don't fit at the "
                                     "smallest size are left out.")
        rel_scaling = sc3.slider("Size contrast", 0.0, 1.0, 0.5, 0.05, key="wc_rel_scaling",
                                 help="0 = size follows rank only (flatter cloud); 1 = size strictly "
                                      "proportional to frequency / TF-IDF (big gaps between words).")
        margin = sc4.slider("Spacing between words", 0, 20, 2, key="wc_margin")

        lc1, lc2, lc3 = st.columns(3)
        canvas_choice = lc1.selectbox("Canvas", list(CANVAS_PRESETS.keys()), key="wc_canvas")
        background = lc2.color_picker("Background", "#FFFFFF", key="wc_bg")
        export_scale = lc3.select_slider("PNG export quality", options=[1, 2, 3, 4], value=2, key="wc_scale",
                                         help="Multiplies the PNG resolution (same layout).")
        show_unclustered = st.checkbox(
            "Include words that belong to no cluster (shown in grey)", value=True, key="wc_unclustered",
        )

    # ── Scope & group ────────────────────────────────────────────────────────
    s1, s2 = st.columns(2)
    scope_options = ["Entire sample"] + [f"Cluster {cid + 1}" for cid in cluster_ids]
    cloud_scope = s1.selectbox("Show word cloud for:", scope_options, key="cloud_scope")
    group_value = None
    if group_col and group_col in resp_df.columns:
        gvals = resp_df[group_col].astype(str).value_counts()
        group_value = s2.selectbox(
            f"Group ({group_col})", ["All"] + gvals.index.tolist(), key="cloud_group",
            format_func=lambda v: v if v == "All" else f"{v} ({gvals.get(v, 0)})",
        )
        if group_value == "All":
            group_value = None

    # Frequencies for the selected group (clusters/colours stay global).
    if group_value is not None:
        sub = resp_df[resp_df[group_col].astype(str) == group_value]
        scope_freq = Counter(itertools.chain.from_iterable(sub["tokens"]))
        scope_tfidf = compute_tfidf_scores(sub["tokens"].tolist()) if use_tfidf else {}
    else:
        scope_freq, scope_tfidf = word_freq, tfidf_scores

    scope_rows = sub if group_value is not None else resp_df
    scope_responses, scope_percent = response_statistics(scope_rows["tokens"].tolist())
    eligible = [w for w, n in word_freq.items() if n >= res_min_freq and scope_freq.get(w, 0) > 0]
    if cloud_scope == "Entire sample":
        focus_cid = None
        words = [w for w in eligible if show_unclustered or w in best_p]
    else:
        focus_cid = int(cloud_scope.split(" ")[1]) - 1
        words = [w for w in eligible if best_p.get(w) == focus_cid]

    weights = scope_tfidf if use_tfidf else (scope_percent if weight_mode == "Percentage of responses" else scope_freq)
    weight = lambda w: float(weights.get(w, 0))
    cloud_freqs = {w: weight(w) for w in words if weight(w) > 0}

    if focus_cid is None:
        def color_of(w):
            cid = best_p.get(w)
            return rgb_str(color_map.get(cid, UNCLUSTERED_COLOR)) if cid is not None else rgb_str(UNCLUSTERED_COLOR)
    else:
        base_color = color_map.get(focus_cid, CLUSTER_COLORS[focus_cid % len(CLUSTER_COLORS)])
        vals = list(cloud_freqs.values()) or [0]
        fmin, fmax = min(vals), max(vals)
        def color_of(w):
            f = cloud_freqs.get(w, fmin)
            t = 0.5 if fmax == fmin else (f - fmin) / (fmax - fmin)
            return shade_rgb_str(base_color, 0.3 + t * 0.55)

    if lang_code != "en":
        translate_on = st.toggle(
            "🇬🇧 Translate the word cloud into English",
            value=False, key="translate_to_en",
            help="Off = words in the original language. On = the words drawn in the cloud are "
                 "translated to English with Helsinki-NLP, locally inside the app (nothing is sent "
                 "to an external service). The analysis (lemmas, phrases, clusters) always stays in "
                 f"the original language. First use downloads the {res.get('lang_name', '')}→English "
                 "model (~300 MB, ~1 min). Words that translate to the same English word are merged.",
        )

    label_of = dl
    render_freqs, render_color, render_lang = cloud_freqs, color_of, lang_code
    drawn_src = set(cloud_freqs)
    if translate_on and cloud_freqs:
        top_src = [w for w, _ in sorted(cloud_freqs.items(), key=lambda kv: -kv[1])[:max_words]]
        n_cached = len(st.session_state.get("translation_cache", {}).get(lang_code, {}))
        with st.spinner(
            f"Translating {len(top_src)} words to English"
            + (" (first time: downloading the translation model, ~1 min)…" if n_cached == 0 else "…")
        ):
            trans, trans_err = translate_words(top_src, lang_code, dl)
        if trans_err:
            st.warning(
                "⚠️ English translation unavailable — showing the original words. Check that "
                "`transformers`, `sentencepiece` and `torch` are in requirements.txt and that the app "
                f"can reach huggingface.co.\n\nTechnical detail: `{trans_err}`"
            )
        else:
            # Manual corrections (kept for the session, per language).
            overrides = st.session_state.setdefault("translation_overrides", {}).setdefault(lang_code, {})
            with st.expander("✏️ Check / correct translations", expanded=False):
                st.caption("Edit the English column to fix a translation — the cloud updates immediately. "
                           "Single words are translated without context, so the odd one can be off.")
                edit_df = pd.DataFrame({
                    "Original": [dl(w) for w in top_src],
                    "English": [overrides.get(w, trans[w]) for w in top_src],
                })
                edited = st.data_editor(
                    edit_df, disabled=["Original"], hide_index=True, use_container_width=True,
                    key=f"translation_editor_{lang_code}", height=min(420, 40 + 35 * len(edit_df)),
                )
                for w, new_val in zip(top_src, edited["English"].tolist()):
                    new_val = (str(new_val) if new_val is not None else "").strip().lower()
                    if new_val and new_val != trans[w]:
                        overrides[w] = new_val
                    elif w in overrides and new_val == trans[w]:
                        overrides.pop(w)
            final = {w: overrides.get(w, trans[w]) for w in top_src}
            label_of = lambda w, _f=final: _f.get(w, dl(w))

            # Merge words sharing the same English label; the colour comes from
            # the heaviest source word (its cluster).
            render_freqs, best_src = {}, {}
            for w in top_src:
                lab, v = final[w], cloud_freqs[w]
                render_freqs[lab] = render_freqs.get(lab, 0) + v
                if lab not in best_src or v > cloud_freqs[best_src[lab]]:
                    best_src[lab] = w
            render_freqs = translated_weights(scope_rows["tokens"].tolist(), final, cloud_freqs,
                                               weight_mode == "Percentage of responses")
            render_color = lambda lab, _b=best_src: color_of(_b.get(lab, lab))
            render_lang = "en"
            drawn_src = set(top_src)
            n_merged = len(top_src) - len(render_freqs)
            if n_merged > 0:
                st.caption(f"🇬🇧 Shown in English — {n_merged} words merged with another word that "
                           + ("has the same English translation (responses counted once per label)." if weight_mode == "Percentage of responses"
                            else "has the same English translation (weights added)."))

    if not cloud_freqs:
        st.info("No words to display for this selection.")
    else:
        top_for_check = [w for w, _ in sorted(render_freqs.items(), key=lambda kv: -kv[1])[:max_words]]
        if render_lang == lang_code:
            top_for_check = [dl(w) for w in top_for_check]
        if font_path and missing_glyph_ratio(font_path, top_for_check) > 0.05:
            st.warning(
                f"⚠️ **{font_choice}** doesn't contain all the characters used in these words — "
                "some may render as empty boxes. Pick a font matching your language "
                "(they're listed first), or upload one."
            )
        if script != "latin" and not any(v[1] == script for v in fonts.values()):
            st.caption(
                f"ℹ️ No {script.upper()} font found on this server. On Streamlit Community Cloud keep "
                "`fonts-noto-core` / `fonts-noto-cjk` in packages.txt, or commit a .ttf into `fonts/`."
            )

        w_px, h_px = CANVAS_PRESETS[canvas_choice]
        lo, hi = font_range
        if hi <= lo:
            hi = lo + 1
        with st.spinner("Drawing word cloud…"):
            wc = render_wordcloud(
                render_freqs, render_color, font_path, w_px, h_px, max_words, lo, hi,
                rel_scaling, margin, background, export_scale, render_lang,
                label_fn=dl if render_lang == lang_code else (lambda x: x),
            )
        img = wc.to_image()
        st.image(img, use_container_width=True)
        n_drawn = len(wc.layout_)
        metric_label = weight_mode.lower()
        st.caption(
            f"{n_drawn} words drawn · size = {metric_label} · colour = response co-occurrence cluster"
            + (f" · group: {group_value}" if group_value else "")
            + (" · translated to English" if render_lang != lang_code else "")
            + ("" if n_drawn >= min(max_words, len(render_freqs)) else
               " · some words didn't fit — lower the minimum size or enlarge the canvas")
        )

        # ── Legend ───────────────────────────────────────────────────────────
        if focus_cid is None and cluster_ids:
            chips = []
            for cid in cluster_ids:
                members = [w for w in cluster_members(cid, scope_freq) if w in drawn_src]
                if not members:
                    continue
                c = color_map.get(cid, CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])
                chips.append(
                    f'<div style="display:inline-block;margin:3px 6px 3px 0;padding:6px 10px;border-radius:8px;'
                    f'border-left:6px solid {c};background:rgba(0,0,0,0.03);font-size:12.5px;">'
                    f'<b style="color:{c}">C{cid + 1} · {label_of(members[0])}</b> '
                    f'<span style="opacity:.75">{", ".join(label_of(w) for w in members[1:5])}'
                    f'{"…" if len(members) > 5 else ""} ({len(members)})</span></div>'
                )
            if show_unclustered and any(w not in best_p for w in cloud_freqs):
                chips.append(
                    f'<div style="display:inline-block;margin:3px 6px 3px 0;padding:6px 10px;border-radius:8px;'
                    f'border-left:6px solid {UNCLUSTERED_COLOR};background:rgba(0,0,0,0.03);font-size:12.5px;">'
                    f'<b style="color:{UNCLUSTERED_COLOR}">Not clustered</b></div>'
                )
            st.markdown("".join(chips), unsafe_allow_html=True)

        # ── Downloads ────────────────────────────────────────────────────────
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fname = f"wordcloud_{re.sub(r'[^A-Za-z0-9]+', '_', cloud_scope).lower()}"
        if product_col and product_value != TOTAL_CODES:
            fname = f"wordcloud_{re.sub(r'[^A-Za-z0-9]+', '_', str(product_value))}_" + fname[len("wordcloud_"):]
        if group_value:
            fname += "_" + re.sub(r"[^A-Za-z0-9]+", "_", str(group_value)).lower()
        png_bytes = buf.getvalue()

        rows = []
        for w in sorted(cloud_freqs, key=lambda w: -cloud_freqs[w]):
            cid = best_p.get(w)
            rows.append({
                **({"product_code": product_value} if product_col else {}),
                "word": dl(w),
                **({"english": label_of(w) if w in drawn_src else ""} if render_lang != lang_code else {}),
                "frequency": scope_freq.get(w, 0),
                "responses": scope_responses.get(w, 0),
                "response_percent": round(scope_percent.get(w, 0), 2),
                "tfidf": round(float(scope_tfidf.get(w, 0.0)), 4) if scope_tfidf else None,
                "cluster": f"C{cid + 1}" if cid is not None else "—",
                "colour": color_map.get(cid, UNCLUSTERED_COLOR) if cid is not None else UNCLUSTERED_COLOR,
            })
        words_df = pd.DataFrame(rows)
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            words_df.to_excel(writer, index=False, sheet_name="words")

        d1, d2 = st.columns(2)
        d1.download_button("💾 Download word cloud (PNG)", data=png_bytes, file_name=f"{fname}.png",
                           mime="image/png", use_container_width=True, key="download_wordcloud_png")
        d2.download_button("💾 Download word list + clusters (XLSX)", data=xbuf.getvalue(),
                           file_name=f"{fname}_words.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True, key="download_words_xlsx")
        st.sidebar.markdown("---")
        st.sidebar.download_button("💾 Download word cloud (PNG)", data=png_bytes, file_name=f"{fname}.png",
                                   mime="image/png", use_container_width=True, key="download_wordcloud_png_sb")

        with st.expander("📋 Word table"):
            st.dataframe(words_df, use_container_width=True, hide_index=True,
                         height=min(420, 40 + 35 * len(words_df)))

    # ── Every product code at once (ZIP) ─────────────────────────────────────
    if product_col and code_counts is not None:
        st.markdown("---")
        st.markdown("#### 📦 All product codes")
        st.caption(
            "Builds one word cloud per code (plus the total) with the current settings — font, sizes, "
            "TF-IDF, English translation and your translation corrections — using the entire sample "
            "of each code. Also includes one Excel file with every code's words."
        )

        def build_code_cloud(rows):
            a = get_analysis(product_col, code_v, rows)
            p, wfreq = a["best_p"], a["word_freq"]
            wts = a["tfidf_scores"] if use_tfidf else (a["response_percent"] if weight_mode == "Percentage of responses" else wfreq)
            ws = [w for w, n in wfreq.items() if n >= min_freq and (show_unclustered or w in p)]
            freqs = {w: wts.get(w, 0.0) for w in ws if wts.get(w, 0.0) > 0}
            if not freqs:
                return None, []
            col_of = lambda w: rgb_str(color_map.get(p[w], UNCLUSTERED_COLOR)) if w in p else rgb_str(UNCLUSTERED_COLOR)
            top = [w for w, _ in sorted(freqs.items(), key=lambda kv: -kv[1])[:max_words]]
            r_freqs, r_color, r_lang, eng = freqs, col_of, lang_code, {}
            if translate_on:
                tr, err = translate_words(top, lang_code, dl)
                if not err:
                    ovr = st.session_state.get("translation_overrides", {}).get(lang_code, {})
                    eng = {w: ovr.get(w, tr[w]) for w in top}
                    r_freqs, src = {}, {}
                    for w in top:
                        r_freqs[eng[w]] = r_freqs.get(eng[w], 0) + freqs[w]
                        if eng[w] not in src or freqs[w] > freqs[src[eng[w]]]:
                            src[eng[w]] = w
                    r_freqs = translated_weights(rows["tokens"].tolist(), eng, freqs,
                                                 weight_mode == "Percentage of responses")
                    r_color = lambda lab, _s=src, _c=col_of: _c(_s.get(lab, lab))
                    r_lang = "en"
            lo_, hi_ = font_range[0], max(font_range[1], font_range[0] + 1)
            wc_ = render_wordcloud(r_freqs, r_color, font_path, *CANVAS_PRESETS[canvas_choice], max_words,
                                   lo_, hi_, rel_scaling, margin, background, export_scale, r_lang,
                                   label_fn=dl if r_lang == lang_code else (lambda x: x))
            out_rows = []
            for w in sorted(freqs, key=lambda w: -freqs[w]):
                out_rows.append({
                    "product_code": code_v, "word": dl(w),
                    **({"english": eng.get(w, "")} if eng else {}),
                    "frequency": wfreq.get(w, 0),
                    "responses": a["response_counts"].get(w, 0),
                    "response_percent": round(a["response_percent"].get(w, 0), 2),
                    "tfidf": round(float(a["tfidf_scores"].get(w, 0.0)), 4),
                    "cluster": f"C{p[w] + 1}" if w in p else "—",
                })
            return wc_.to_image(), out_rows

        zip_settings = (res.get("run_id"), view_key, weight_mode, min_freq, min_edge, n_clusters,
                        max_words, tuple(font_range), rel_scaling, margin, background, export_scale,
                        canvas_choice, font_path, show_unclustered, translate_on,
                        repr(st.session_state.get("translation_overrides")), repr(color_map), product_col)
        if st.session_state.get("codes_zip_settings") != zip_settings:
            st.session_state.pop("codes_zip", None)
            st.session_state["codes_zip_settings"] = zip_settings
        if st.button("📦 Build word clouds for every code (ZIP)", use_container_width=True, key="build_codes_zip"):
            import zipfile
            codes = [TOTAL_CODES] + code_counts.index.tolist()
            bar = st.progress(0.0, text="Building word clouds…")
            zbuf, all_rows = io.BytesIO(), []
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, code_v in enumerate(codes):
                    rows_v = full_df if code_v == TOTAL_CODES else \
                        full_df[full_df[product_col].astype(str) == code_v].reset_index(drop=True)
                    img_v, rows_out = build_code_cloud(rows_v)
                    if img_v is not None:
                        b = io.BytesIO()
                        img_v.save(b, format="PNG")
                        safe = "TOTAL" if code_v == TOTAL_CODES else re.sub(r"[^A-Za-z0-9._-]+", "_", code_v)
                        zf.writestr(f"wordcloud_{safe}.png", b.getvalue())
                        all_rows.extend(rows_out)
                    bar.progress((i + 1) / len(codes), text=f"Building word clouds… {code_v}")
                xb = io.BytesIO()
                with pd.ExcelWriter(xb, engine="openpyxl") as writer:
                    pd.DataFrame(all_rows).to_excel(writer, index=False, sheet_name="words_by_code")
                zf.writestr("words_by_code.xlsx", xb.getvalue())
            bar.empty()
            st.session_state["codes_zip"] = zbuf.getvalue()
        if st.session_state.get("codes_zip"):
            st.download_button("💾 Download all codes (ZIP)", data=st.session_state["codes_zip"],
                               file_name=f"wordclouds_by_{re.sub(r'[^A-Za-z0-9]+', '_', product_col)}.zip",
                               mime="application/zip", use_container_width=True, key="download_codes_zip")
