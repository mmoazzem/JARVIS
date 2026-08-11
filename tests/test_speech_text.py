"""Speech normalization — the two failure shapes, on the pure functions only.

No audio here. What can silently go wrong is (a) a sentence fragmenting into
several clips with a synthesis gap in each seam, and (b) a substitution firing
on a sentence it was not written for — a bare "F" becoming Fahrenheit, or the
preposition "in" becoming inches.
"""
from interface.speech_text import normalize_for_speech, split_sentences


def _sentences(text: str) -> list[str]:
    """Split a COMPLETE utterance the way a finished turn is split.

    The trailing space stands in for the `done` flush: mid-stream the splitter
    deliberately waits for the character after the space before deciding, so a
    buffer that ends at the punctuation has nothing to decide on yet.
    """
    sentences, remainder = split_sentences(text + " ")
    if remainder.strip():
        sentences.append(remainder.strip())
    return sentences


# --- 1. the splitter guard ----------------------------------------------------

def test_abbreviations_and_schemes_stay_in_one_clip():
    # Each of these fragmented live: "Dr." / "re:" / "e.g." each opened a new
    # clip, and the gap between clips is the synthesis latency, audibly.
    assert _sentences("Dr. Chen replied re: the Q3 budget.") == [
        "Dr. Chen replied re: the Q3 budget."]
    assert _sentences("You have 3 tasks due, e.g. the flight search.") == [
        "You have 3 tasks due, e.g. the flight search."]
    assert _sentences("See https://openweathermap.org/api for details.") == [
        "See https://openweathermap.org/api for details."]
    # An initial is not a full stop either.
    assert _sentences("J. Chen sent it.") == ["J. Chen sent it."]


def test_real_sentence_ends_still_split():
    # The regression this guard could plausibly cause: nothing splits any more.
    assert _sentences("It is warm. The wind is light.") == [
        "It is warm.", "The wind is light."]
    assert _sentences("Ready? Not yet. Now.") == ["Ready?", "Not yet.", "Now."]
    assert split_sentences("First line\nsecond line") == (
        ["First line"], "second line")


def test_numbers_and_times_are_not_sentence_ends():
    # These were already safe (the pattern needs whitespace after the mark) and
    # are the easiest thing to break while fixing the cases above.
    for text in ("It is 73.8 degrees outside.",
                 "The model scored 24/7 uptime at v0.1.0.",
                 "Your meeting is at 3:30 PM.",
                 "The file is 4.7 MB and took 1.2 s to load."):
        assert _sentences(text) == [text], text


def test_a_pending_decision_waits_instead_of_guessing():
    # Mid-stream: the character that decides "sentence end or abbreviation" has
    # not arrived. Splitting here would create a boundary the finished text does
    # not have, so the text is held back instead.
    assert split_sentences("He replied re: ") == ([], "He replied re: ")
    assert split_sentences("He replied re: the budget.") == (
        [], "He replied re: the budget.")


# --- 2. normalization ---------------------------------------------------------

def test_symbols_and_abbreviations_are_spoken_as_words():
    assert normalize_for_speech("It is 73.8°F outside with winds at 3.5 kn.") == (
        "It is 73.8 degrees Fahrenheit outside with winds at 3.5 knots.")
    assert normalize_for_speech("GPU is at 62°C, VRAM 11.2/15.9 GB.") == (
        "GPU is at 62 degrees Celsius, VRAM 11.2/15.9 GB.")
    assert normalize_for_speech("The forecast runs Aug 12-17 with rain midweek.") == (
        "The forecast runs August 12-17 with rain midweek.")
    assert normalize_for_speech("Your meeting is at 3:30 PM on Mon.") == (
        "Your meeting is at 3:30 PM on Monday.")
    assert normalize_for_speech("Dr. Chen replied re: the Q3 budget.") == (
        "Dr. Chen replied regarding the Q3 budget.")
    # A bare degree sign still needs the word.
    assert normalize_for_speech("Wind from 199°.") == "Wind from 199 degrees."


def test_ordinary_prose_is_left_alone():
    # A bare F is a grade, not Fahrenheit; "in" is a preposition, not inches;
    # "kn" only means knots after a number.
    for text in ("The grade was F for the exam.",
                 "Humidity is 86% with 0.02 in of precipitation.",
                 "The file is 4.7 MB and took 1.2 s to load.",
                 "The model scored 24/7 uptime at v0.1.0.",
                 "He will know the answer in August.",
                 "Monday and September are already words."):
        assert normalize_for_speech(text) == text, text


def test_normalization_never_moves_a_sentence_boundary():
    # "Aug. 12" expands to "August. 12" only if the period is mishandled — that
    # would introduce a clip break the written text never had.
    spoken = normalize_for_speech("The forecast runs Aug. 12 with rain.")
    assert _sentences(spoken) == [spoken]
