"""Whether a claim reads as English, the only language the models were trained on.

Deliberately small and one-sided: English is the default, and a claim is called
something else only on positive evidence -- a letter from another script (real
Turkish, Cyrillic, CJK...), a letter only Turkish writes, or a common function word
of another language. A wrong "English" costs a weaker check; a wrong "not English"
would cost the claim its search without a judge, so the doubt goes to English.
"""

from __future__ import annotations

import re
import unicodedata

# Function words of languages a reader is likely to paste, none of them English words.
_FOREIGN_WORDS = frozenset(
    {
        # Turkish
        "ve", "bir", "bu", "için", "ile", "değil", "çok", "daha", "gibi", "olan",
        "olarak", "ama", "şu", "mi", "mı", "mu", "mü", "da", "de", "ki", "yok", "var",  # noqa: RUF001 - real Turkish letters
        # Spanish / Portuguese / Italian
        "el", "los", "las", "que", "y", "es", "por", "con", "una", "para", "del",
        "não", "uma", "il", "della", "che", "sono",
        # German / Dutch
        "der", "das", "und", "ist", "nicht", "ein", "eine", "mit", "den", "het", "een",
        # French
        "le", "les", "et", "est", "une", "des", "du", "pas", "pour", "dans",
    }
)  # fmt: skip
_WORDS = re.compile(r"[^\W\d_]+")

# Letters only Turkish writes. ç, ö and ü are not here: they turn up in names
# ("Müller", "François"), and a name is not a language.
_TURKISH_ONLY = frozenset("ğĞıİşŞ")


def _foreign_letter(char: str) -> bool:
    """A letter from another script, or one only Turkish uses. A Latin letter with
    an accent ("André", "Beyoncé") is not evidence: English borrows names."""
    if char in _TURKISH_ONLY:
        return True
    return not unicodedata.name(char, "LATIN").startswith("LATIN")


def looks_english(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if any(_foreign_letter(char) for char in letters):
        return False
    return not any(word.casefold() in _FOREIGN_WORDS for word in _WORDS.findall(text))
