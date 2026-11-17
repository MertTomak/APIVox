"""Tests for the rule converter in src/eval_zero_shot.py."""
import pytest

from src.eval_zero_shot import convert, join_spelled_letters, join_symbol_words


@pytest.mark.parametrize("raw, expected", [
    ("Use data frame dot groupby to split the data.", "Use data frame.groupby to split the data"),
    ("when using index dot sort underscore values for datetime",
     "when using index.sort_values for datetime"),
    ("use o s dot path dot join here", "use os.path.join here"),
    ("call read underscore csv on the file", "call read_csv on the file"),
    ("Bug in merge underscore ordered, couldn't handle", "Bug in merge_ordered couldn't handle"),
    ("returns a n d array", "returns a nd array"),
    ("the median absolute error", "the median absolute error"),
])
def test_convert(raw, expected):
    assert convert(raw) == expected


def test_symbol_word_at_edges_is_kept():
    assert join_symbol_words(["dot", "product"]) == ["dot", "product"]
    assert join_symbol_words(["the", "dot"]) == ["the", "dot"]


def test_symbol_word_is_case_insensitive():
    assert join_symbol_words(["np", "Dot", "reshape"]) == ["np.reshape"]


def test_single_letter_alone_is_not_joined():
    assert join_spelled_letters(["a", "frame"]) == ["a", "frame"]


def test_letters_at_end_are_joined():
    assert join_spelled_letters(["read", "c", "s", "v"]) == ["read", "csv"]


def test_existing_code_tokens_are_untouched():
    assert convert("use DataFrame.groupby here") == "use DataFrame.groupby here"
