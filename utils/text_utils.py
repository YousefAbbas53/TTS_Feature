import re
from typing import List

_AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_SENT_SPLIT = re.compile(r"(?<=[\\.\\!\\?])\s+|(?<=[\u061F\u06D4])\s+|(?<=\n)\s*")
_SOFT_SPLIT = re.compile(r"(?<=[,;:،؛])\s+")

def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def normalize_punctuation(s: str) -> str:
    s = s.replace("…", "...")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", s)
    return s

def normalize_arabic(s: str) -> str:
    s = s.replace("\u0640", "")            # tatweel
    s = _AR_DIACRITICS.sub("", s)          # diacritics
    s = s.replace("،", "، ").replace("؟", "؟ ").replace("؛", "؛ ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def clean_text(s: str, lang: str = "en") -> str:
    s = normalize_whitespace(s)
    s = normalize_punctuation(s)
    if lang.lower().startswith("ar"):
        s = normalize_arabic(s)
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E]", "", s)  # bidi junk
    return normalize_whitespace(s)

def split_into_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    return [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]

def hard_wrap_by_words(s: str, max_chars: int) -> List[str]:
    words = s.split()
    out, cur = [], []
    for w in words:
        cand = (" ".join(cur + [w])).strip()
        if len(cand) <= max_chars:
            cur.append(w)
        else:
            if cur:
                out.append(" ".join(cur))
                cur = [w]
            else:
                out.append(w[:max_chars])
                rest = w[max_chars:]
                if rest:
                    cur = [rest]
    if cur:
        out.append(" ".join(cur))
    return [x.strip() for x in out if x.strip()]

def chunk_text(text: str, lang: str = "en", max_chars: int = 220, min_chars: int = 20) -> List[str]:
    text = clean_text(text, lang=lang)
    sents = split_into_sentences(text)

    chunks, cur = [], ""

    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = ""

    for sent in sents:
        if len(sent) > max_chars:
            subs = [x.strip() for x in _SOFT_SPLIT.split(sent) if x.strip()]
            if len(subs) == 1:
                subs = hard_wrap_by_words(sent, max_chars=max_chars)
        else:
            subs = [sent]

        for part in subs:
            if not cur:
                cur = part
            elif len(cur) + 1 + len(part) <= max_chars:
                cur = cur + " " + part
            else:
                flush()
                cur = part

    flush()

    merged = []
    for ch in chunks:
        if merged and len(ch) < min_chars:
            merged[-1] = (merged[-1] + " " + ch).strip()
        else:
            merged.append(ch)
    return merged
