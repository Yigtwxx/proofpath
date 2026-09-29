"""Whether a claim reads as English, the only language the models were trained on.

Deliberately small and one-sided: English is the default, and a claim is called
something else only on positive evidence. Decisive on its own: a word in another
script (Cyrillic, CJK...) or a letter only Turkish writes. Otherwise it takes two
signals, where a signal is a distinct foreign function word, or -- once, however many
there are -- a word with an accented Latin letter ("subió", "président"). A wrong
"English" costs a weaker check (and with a judge, ``certainly_english`` sends the
claim to its translation anyway); a wrong "not English" would cost the claim its
search without a judge, so the doubt goes to English.

Capitalised words are never evidence: they are names and acronyms ("MIT", "Le Monde",
"El Niño", "André", "Uğur Şahin"), and English text is full of foreign ones.
"""

from __future__ import annotations

import re
import unicodedata

# Function words of languages a reader is likely to paste. Only lower-case words are
# read, so "Los Angeles", "Le Monde" and "Die Hard" never count. Left out on purpose,
# because English text prints them lower-case too: "et" ("et al."), "de" ("de facto",
# "de Gaulle"), "da" and "del" ("da Vinci", "del Toro"), "est" (the time zone), "y",
# "ama", and the particles "della", "du", "den" ("van den Berg").
_FOREIGN_WORDS = frozenset(
    {
        # Turkish
        "ve", "bir", "bu", "için", "ile", "değil", "çok", "daha", "gibi", "olan",
        "olarak", "şu", "mi", "mı", "mu", "mü", "ki", "yok", "var",  # noqa: RUF001 - real Turkish letters
        # Spanish / Portuguese / Italian
        "el", "la", "los", "las", "que", "es", "por", "con", "una", "un", "para", "não",
        "uma", "il", "ha", "che", "sono",
        # German / Dutch
        "der", "die", "das", "und", "mit", "hat", "ist", "nicht", "ein", "eine", "het",
        "een", "heeft",
        # French
        "le", "les", "des", "une", "pas", "pour", "dans", "hier",
    }
)  # fmt: skip
# One signal proves nothing: English borrows words ("que sera", "a con", "café").
_MIN_SIGNALS = 2
_WORDS = re.compile(r"[^\W\d_]+")

# Letters only Turkish writes. ç, ö and ü are not here: they turn up in names
# ("Müller", "François"), and a name is not a language.
_TURKISH_ONLY = frozenset("ğĞıİşŞ")


def _foreign_word(word: str) -> bool:
    """A word in another script, or with a letter only Turkish uses.

    A Latin letter with an accent ("André", "Beyoncé") is not evidence: English
    borrows names. A single letter of another script is not either: it is a symbol
    ("μ-opioid", "π"), where a sentence in that script has words.
    """
    if any(char in _TURKISH_ONLY for char in word):
        return True
    return len(word) > 1 and any(
        not unicodedata.name(char, "LATIN").startswith("LATIN") for char in word
    )


def _accented(word: str) -> bool:
    """A Latin word with a letter outside ASCII: "subió", "président", "yüzde"."""
    return any(not char.isascii() for char in word)


def looks_english(text: str) -> bool:
    # Only words that do not start with a capital: a script with no case (CJK) is
    # never capitalised, so it is always read.
    words = [word for word in _WORDS.findall(text) if not word[0].isupper()]
    if any(_foreign_word(word) for word in words):
        return False
    signals = len({word.casefold() for word in words} & _FOREIGN_WORDS)
    signals += any(_accented(word) for word in words)
    return signals < _MIN_SIGNALS


def certainly_english(text: str) -> bool:
    """English with no doubt at all: no lower-case word with a letter outside ASCII,
    and no foreign function word, capitalised or not.

    Stricter than :func:`looks_english` on purpose. That one decides whether a claim
    may be searched with no judge, so the doubt goes to English; this one decides
    whether a judge's translation may be set aside, so the doubt goes to the judge
    ("Las vacunas causan autismo." has one capitalised signal and reads as English,
    but it is not certainly English).
    """
    words = _WORDS.findall(text)
    if any(_accented(word) for word in words if not word[0].isupper()):
        return False
    return not {word.casefold() for word in words} & _FOREIGN_WORDS
