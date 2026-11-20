"""Tests for src/codeswitch.py on hand-made lexicons."""
import gzip
import json
from collections import Counter

import pytest

from src import codeswitch, data_io
from src.codeswitch import Lexicons


@pytest.fixture
def lex() -> Lexicons:
    # "және" stands in for the rest of a realistic Kazakh corpus, so rare Russian words reach the ratio.
    kz = Counter({"мен": 500, "бастадым": 40, "ойлана": 10, "бүгін": 80, "келдім": 30, "жақсы": 60, "от": 20,
                  "балалар": 25, "кітап": 30, "проблема": 3000, "және": 100000})
    ru = Counter({"уже": 300, "вообще": 50, "как": 400, "будто": 30, "настроение": 12, "проблема": 40,
                  "от": 200, "работа": 60, "конечно": 30})
    return Lexicons(kz=kz, ru=ru)


@pytest.mark.parametrize("word, tag", [
    ("уже", "ru"), ("вообще", "ru"), ("как", "ru"), ("будто", "ru"),
    ("работа", "ru"),
    ("ойлана", "kz"), ("бастадым", "kz"), ("бүгін", "kz"),
    ("настроениеміз", "mix"), ("работада", "mix"), ("настроениелерің", "mix"),
    ("от", "amb"), ("там", "amb"), ("ну", "amb"),
    ("проблема", "amb"),
    ("белгісіз", "kz"),
    ("неизвестное", "amb"),
])
def test_tag_word(lex, word, tag):
    assert codeswitch.tag_word(word, lex) == tag


def test_homograph_from_is_never_russian(lex):
    # Kazakh "от" (fire) is far more frequent in the Russian lexicon as "from", but must stay ambiguous.
    assert codeswitch.tag_word("от", lex) == "amb"
    assert not codeswitch.is_cs(codeswitch.tag_text("мен от жақсы", lex))


def test_short_stem_is_not_mixed(lex):
    lex.ru["как"] = 400
    assert codeswitch.tag_word("какта", lex) != "mix"


@pytest.mark.parametrize("word", ["я", "у", "с", "и", "в", "о"])
def test_single_letters_and_kazakh_o_are_ambiguous(lex, word):
    lex.ru[word] = 5000
    assert codeswitch.tag_word(word, lex) == "amb"


def test_established_loan_is_not_russian(lex):
    lex.kz["кодекс"], lex.ru["кодекс"] = 40, 900
    lex.kz["орган"], lex.ru["орган"] = 25, 300
    assert codeswitch.tag_word("кодекс", lex) == "amb"
    assert codeswitch.tag_word("органынан", lex) not in ("ru", "mix")
    lex.kz["уже"] = 50
    assert codeswitch.tag_word("уже", lex) == "ru"


def test_known_kazakh_word_is_not_mixed(lex):
    assert codeswitch.tag_word("балалар", lex) == "kz"


def test_pure_kazakh_sentence_is_not_cs(lex):
    tags = codeswitch.tag_text("мен бүгін келдім", lex)
    assert tags == ["kz", "kz", "kz"]
    assert not codeswitch.is_cs(tags)


def test_code_switched_sentence_is_cs(lex):
    tags = codeswitch.tag_text("мен уже ойлана бастадым", lex)
    assert tags == ["kz", "ru", "kz", "kz"]
    assert codeswitch.is_cs(tags)


def test_russian_only_sentence_is_not_cs(lex):
    assert not codeswitch.is_cs(codeswitch.tag_text("как будто уже", lex))


def test_tag_text_matches_word_count(lex):
    text = "мен уже ойлана бастадым как будто настроениеміз"
    assert len(codeswitch.tag_text(text, lex)) == len(text.split())


@pytest.mark.parametrize("words, expected", [
    ("в этой связи фракция просит принять меры".split(), False),
    ("это очень важно для нас".split(), True),
    ("мен уже бүгін келдім".split(), False),
    ("это важно".split(), False),
])
def test_is_russian_sentence(words, expected):
    assert codeswitch.is_russian_sentence(words) is expected


def test_build_lexicons_uses_only_the_right_sources(tmp_path):
    rows = [
        {"id": "a", "source": "crowdsourced", "orig_split": "train", "text": "мен бүгін келдім мен бүгін келдім"},
        {"id": "b", "source": "crowdsourced", "orig_split": "test", "text": "тест тест сөз сөз"},
        {"id": "c", "source": "parliament", "orig_split": "train", "text": "это очень важно для нас это очень важно для нас"},
        {"id": "d", "source": "tv_news", "orig_split": "dev", "text": "это уже все и это уже все и"},
        {"id": "e", "source": "podcasts", "orig_split": "train", "text": "это очень важно это очень важно"},
        {"id": "f", "source": "parliament", "orig_split": "test", "text": "мен уже бүгін мен уже бүгін"},
    ]
    path = tmp_path / "all.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    lex = codeswitch.build_lexicons(path)
    assert set(lex.kz) == {"мен", "бүгін", "келдім"}
    assert "тест" not in lex.kz
    assert {"это", "очень", "важно", "для", "нас", "уже", "все", "и"} == set(lex.ru)
    assert "бүгін" not in lex.ru


def test_lexicons_round_trip(lex, tmp_path):
    codeswitch.write_lexicons(lex, tmp_path)
    back = codeswitch.load_lexicons(tmp_path)
    assert back.kz == lex.kz
    assert back.ru == lex.ru
    first = (tmp_path / "ru_words.txt").read_text(encoding="utf-8").splitlines()[0]
    assert first == "как\t400"


def test_audit_is_deterministic_and_marks_switches(lex):
    records = []
    for i in range(30):
        text = "мен уже ойлана бастадым" if i % 2 else "мен бүгін келдім"
        tags = codeswitch.tag_text(text, lex)
        records.append({"id": f"podcasts_02_00_{i:03d}", "text": text, "lang_tags": tags, "cs": data_io.is_cs(tags)})
    a = codeswitch.audit(records, n=5, seed=7)
    assert a == codeswitch.audit(records, n=5, seed=7)
    assert len(a) == 5
    assert all("[уже|ru]" in line for line in a)
