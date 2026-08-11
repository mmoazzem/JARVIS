"""Text on its way to the voice — splitting and normalization, pure functions.

WRITTEN FORM AND SPOKEN FORM ARE DIFFERENT JOBS, and one string was doing both.
The transcript and the weather tile want `°F` and `Aug`; the ear wants
"Fahrenheit" and "August". Anything that fixed one by changing what the model
WRITES broke the other, so the fix lives here, at the speech boundary: nothing
in this module is reachable from the transcript, the event log or the tile.

Two passes, in this order:

  1. split_sentences() decides where a clip ends. It guards against breaking on
     an abbreviation, an initial, a URL scheme, or a lowercase continuation —
     without which "Dr. Chen replied re: the Q3 budget." is spoken as three
     clips with a synthesis gap in each seam.
  2. normalize_for_speech() rewrites what a sentence SOUNDS like. It runs after
     the split, per sentence, so an expansion can never move a boundary.

Deliberately NOT normalized: ranges ("12-17") and slashes ("11.2/15.9",
"24/7"). Both mean different things in different sentences and no regex can see
which, and speech that is confidently wrong is worse than speech that is ugly
and right. "12 dash 17" is acceptable; "December 17th" when the text meant a
range is not.
"""
from __future__ import annotations

import re
from typing import Optional

from core.constants import (
    SENTENCE_END_PATTERN,
    SPEECH_ABBREVIATIONS,
    SPEECH_DAY_WORDS,
    SPEECH_DEGREE_BARE,
    SPEECH_DEGREE_UNITS,
    SPEECH_KNOTS_PATTERN,
    SPEECH_KNOTS_REPLACEMENT,
    SPEECH_LAST_WORD_PATTERN,
    SPEECH_MONTH_WORDS,
    SPEECH_RE_PATTERN,
    SPEECH_RE_REPLACEMENT,
    SPEECH_STRIP_CHARS_PATTERN,
    SPEECH_URL_SCHEMES,
)

_SENTENCE_END = re.compile(SENTENCE_END_PATTERN)
_STRIP_CHARS = re.compile(SPEECH_STRIP_CHARS_PATTERN)
_LAST_WORD = re.compile(SPEECH_LAST_WORD_PATTERN)
_KNOTS = re.compile(SPEECH_KNOTS_PATTERN)
_RE_PREFIX = re.compile(SPEECH_RE_PATTERN, re.IGNORECASE)
_WORD_MAP = {**SPEECH_MONTH_WORDS, **SPEECH_DAY_WORDS}
# Capitalized forms only, and never the start of a longer word: \b after "Mon"
# fails inside "Monday", so an already-expanded word is left alone.
_ABBREV_WORDS = re.compile(r"\b(" + "|".join(
    sorted(_WORD_MAP, key=len, reverse=True)) + r")\b\.?")


def clean_for_speech(text: str) -> str:
    """Drop markdown markup characters so they are never read aloud."""
    return _STRIP_CHARS.sub("", text).strip()


# --- splitting ---------------------------------------------------------------

def _is_break(buffer: str, match: re.Match) -> Optional[bool]:
    """Is this punctuation the end of a spoken sentence?

    None means UNDECIDABLE with the text so far — the deciding character has not
    streamed in yet. The caller waits rather than guessing, because a guess here
    is a clip boundary that the finished text would not have had.
    """
    token = match.group(0)
    if token.startswith("\n"):
        return True
    punct = token[0]
    if punct in "!?":
        return True  # never ambiguous, and never worth delaying a clip for

    after = buffer[match.end():].lstrip()
    word = _LAST_WORD.search(buffer[:match.start()])
    # Dots removed so "e.g" compares as "eg" and "U.S" as "us".
    preceding = word.group(0).replace(".", "").lower() if word else ""

    if punct == ".":
        # Decidable from the text BEFORE the mark alone, so a full stop at the
        # end of the buffer still speaks immediately instead of waiting for the
        # next token — that wait would delay the first clip of every turn.
        if preceding in SPEECH_ABBREVIATIONS:
            return False
        # A single capital is an initial ("J. Chen"), not a full stop.
        if len(preceding) == 1 and word.group(0)[0].isupper():
            return False
        return not (after and after[0].islower())

    # ":" and ";" are the genuinely ambiguous marks: "replied re: the budget"
    # continues, "Note: Buffalo is cold" does not, and only the following word
    # tells them apart.
    if preceding in SPEECH_URL_SCHEMES:
        return False
    if not after:
        return None  # the deciding character has not streamed in yet
    return not after[0].islower()


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off every COMPLETE sentence; return (sentences, unfinished remainder)."""
    sentences: list[str] = []
    start = 0   # where the sentence being assembled begins
    search = 0  # where to look for the next candidate break
    while True:
        match = _SENTENCE_END.search(buffer, search)
        if match is None:
            return sentences, buffer[start:]
        verdict = _is_break(buffer, match)
        if verdict is None:
            return sentences, buffer[start:]
        search = match.end()
        if not verdict:
            continue  # not a sentence end — keep scanning inside this sentence
        sentence = buffer[start:match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()


# --- normalization -----------------------------------------------------------

def normalize_for_speech(text: str) -> str:
    """One sentence as it should SOUND. Pure; never alters sentence boundaries.

    Small on purpose. Percentages, decimals, times and ordinary prose already
    read correctly through Piper, and every rule added here is a rule that can
    fire on a sentence it was not written for.
    """
    if not text:
        return text
    spoken = _RE_PREFIX.sub(SPEECH_RE_REPLACEMENT, text)
    for symbol, words in SPEECH_DEGREE_UNITS:
        spoken = spoken.replace(symbol, words)
    spoken = spoken.replace(*SPEECH_DEGREE_BARE)
    spoken = _KNOTS.sub(SPEECH_KNOTS_REPLACEMENT, spoken)
    spoken = _ABBREV_WORDS.sub(lambda m: _expanded(m, spoken), spoken)
    return re.sub(r"[ \t]{2,}", " ", spoken).strip()


def _expanded(match: re.Match, text: str) -> str:
    """"Aug" -> "August", and the abbreviation's own period dropped with it.

    "Aug. 12" must not become "August. 12": "aug" is a guarded abbreviation but
    "august" is not, so keeping that period would introduce a clip break the
    written text never had — a normalizer moving a sentence boundary, which is
    the one thing this pass may not do. The period survives only when it is
    also the sentence's full stop, so "on Mon." still ends with a pause.
    """
    word = _WORD_MAP[match.group(1)]
    if not match.group(0).endswith("."):
        return word
    return word + ("." if not text[match.end():].strip() else "")
