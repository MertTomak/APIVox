"""Build Kazakh and Russian lexicons from KSC2 transcripts and tag each word as kz, ru, mix or amb."""
import argparse
import gzip
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src import data_io

ROOT = Path(__file__).resolve().parents[1]
ALL_TEXT = ROOT / "data" / "raw" / "text" / "ksc2_all_text.jsonl.gz"
LEXICON_DIR = ROOT / "data" / "processed" / "lexicon"

KAZAKH_SPECIFIC = set(data_io.KAZAKH_LETTERS)
# Unambiguous Russian function words; used to find Russian sentences and always tagged ru.
RU_FUNCTION = frozenset("""
что чтобы это этот эта эти этого этой этом как для или если когда только уже еще очень тоже также
был была были было будет будут есть нет его ее их они она мы вы меня мне тебя нас вам него ней ним
который которая которые которых которого котором потому почему сейчас сегодня вот даже свой своей
своих все всех весь вся можно нужно надо чем чего где здесь тут из при но просто вообще конечно
значит например типа кстати то есть так как будто хорошо спасибо пожалуйста ничего
""".split())
# Words that are real Kazakh words as well as Russian ones; they are never evidence of a switch.
HOMOGRAPHS = frozenset("а о от там тот ну том не да же он ал бал бар тек жар".split())
# Kazakh inflectional suffixes that attach to Russian stems ("настроение" + "міз").
KZ_SUFFIXES = tuple(sorted(set("""
лар лер дар дер тар тер ға ге қа ке на не да де та те нда нде дан ден тан тен нан нен ды ді ты ті ны ні
мен бен пен дың дің тың тің ның нің ым ім м ың ің ң сы сі ы і мыз міз ымыз іміз ңыз ңіз ыңыз іңіз
ша ше дай дей тай тей сыз сіз лық лік дық дік тық тік шы ші сын сің
""".split()), key=len, reverse=True))
# KSC2 sources that none of our splits uses, so no evaluation text enters a lexicon.
RU_LEXICON_SOURCES = ("parliament", "tv_news")
MIN_COUNT = 2
RATIO = 20.0
# A word read this often in Kazakh text is an established loan ("кодекс", "офис"), not a switch.
LOAN_MIN_KZ = 10
MIN_STEM = 5
MAX_SUFFIXES = 3


@dataclass(frozen=True)
class Lexicons:
    """Word counts of the Kazakh and the Russian reference text."""
    kz: Counter
    ru: Counter

    @property
    def kz_total(self) -> int:
        return sum(self.kz.values()) or 1

    @property
    def ru_total(self) -> int:
        return sum(self.ru.values()) or 1


def _read_all_text(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_russian_sentence(words: list[str]) -> bool:
    """No Kazakh-specific letter and at least two unambiguous Russian function words."""
    if any(set(w) & KAZAKH_SPECIFIC for w in words):
        return False
    return sum(w in RU_FUNCTION for w in words) >= 2


def build_lexicons(all_text_path: Path = ALL_TEXT) -> Lexicons:
    """Kazakh counts from KSC train text; Russian counts from Russian sentences of sources we never evaluate on."""
    kz, ru = Counter(), Counter()
    for row in _read_all_text(all_text_path):
        words = data_io.normalize_text(row["text"]).split()
        if row["source"] == "crowdsourced" and row["orig_split"] == "train":
            kz.update(words)
        elif row["source"] in RU_LEXICON_SOURCES and is_russian_sentence(words):
            ru.update(words)
    return Lexicons(
        kz=Counter({w: c for w, c in kz.items() if c >= MIN_COUNT}),
        ru=Counter({w: c for w, c in ru.items() if c >= MIN_COUNT}),
    )


def _log_ratio(word: str, lex: Lexicons) -> float:
    """log of P(word | Russian) / P(word | Kazakh) with add-0.5 smoothing."""
    p_ru = (lex.ru.get(word, 0) + 0.5) / lex.ru_total
    p_kz = (lex.kz.get(word, 0) + 0.5) / lex.kz_total
    return math.log(p_ru / p_kz)


def _russian_word(word: str, lex: Lexicons) -> bool:
    """Function word, or only in the Russian lexicon, or in both but RATIO times more frequent in Russian; never an established loan."""
    if word in HOMOGRAPHS:
        return False
    if word in RU_FUNCTION:
        return True
    if word not in lex.ru or lex.kz.get(word, 0) >= LOAN_MIN_KZ:
        return False
    return word not in lex.kz or _log_ratio(word, lex) >= math.log(RATIO)


def _kazakh_word(word: str, lex: Lexicons) -> bool:
    """Only in the Kazakh lexicon, or in both but RATIO times more frequent in Kazakh."""
    if word not in lex.kz:
        return False
    return word not in lex.ru or _log_ratio(word, lex) <= -math.log(RATIO)


def _suffix_chain(rest: str, depth: int = 0) -> bool:
    """True if rest is a concatenation of at most MAX_SUFFIXES Kazakh suffixes."""
    if not rest:
        return depth > 0
    if depth == MAX_SUFFIXES:
        return False
    return any(rest.startswith(s) and _suffix_chain(rest[len(s):], depth + 1) for s in KZ_SUFFIXES)


def is_mixed_word(word: str, lex: Lexicons) -> bool:
    """A Russian stem of at least MIN_STEM letters followed by Kazakh suffixes, not itself a known Kazakh word."""
    if word in lex.kz or word in HOMOGRAPHS:
        return False
    for cut in range(len(word) - 1, MIN_STEM - 1, -1):
        stem = word[:cut]
        if stem in lex.ru and _russian_word(stem, lex) and _suffix_chain(word[cut:]):
            return True
    return False


def tag_word(word: str, lex: Lexicons) -> str:
    """kz, ru, mix or amb for one normalized word; single letters are always amb (spelled letters, initials)."""
    if word in HOMOGRAPHS or len(word) == 1:
        return "amb"
    if is_mixed_word(word, lex):
        return "mix"
    if set(word) & KAZAKH_SPECIFIC:
        return "kz"
    if _russian_word(word, lex):
        return "ru"
    if _kazakh_word(word, lex):
        return "kz"
    return "amb"


def tag_text(text: str, lex: Lexicons) -> list[str]:
    """One tag per word of an already normalized text."""
    return [tag_word(w, lex) for w in text.split()]


def is_cs(tags: list[str]) -> bool:
    return data_io.is_cs(tags)


def write_lexicons(lex: Lexicons, out_dir: Path = LEXICON_DIR) -> None:
    """Write kz_words.txt and ru_words.txt as word TAB count, most frequent first."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, counts in (("kz_words.txt", lex.kz), ("ru_words.txt", lex.ru)):
        rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        (out_dir / name).write_text("".join(f"{w}\t{c}\n" for w, c in rows), encoding="utf-8")


def load_lexicons(directory: Path = LEXICON_DIR) -> Lexicons:
    def read(name: str) -> Counter:
        rows = (line.split("\t") for line in (directory / name).read_text(encoding="utf-8").splitlines() if line)
        return Counter({w: int(c) for w, c in rows})
    return Lexicons(kz=read("kz_words.txt"), ru=read("ru_words.txt"))


def audit(records: list[dict], n: int = 100, seed: int = 7) -> list[str]:
    """n random code-switched records with ru and mix words marked as [word|tag], for a manual precision check."""
    cs = sorted((r for r in records if r["cs"]), key=lambda r: r["id"])
    sample = random.Random(seed).sample(cs, min(n, len(cs)))
    lines = []
    for r in sorted(sample, key=lambda r: r["id"]):
        marked = [f"[{w}|{t}]" if t in data_io.CS_TAGS else w for w, t in zip(r["text"].split(), r["lang_tags"])]
        lines.append(f"{r['id']}\t{' '.join(marked)}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="print 100 code-switched dev sentences for a manual check")
    parser.add_argument("--tag", metavar="TEXT", help="print the tags of one text with the written lexicons")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tag:
        lex = load_lexicons()
        text = data_io.normalize_text(args.tag)
        print(" ".join(f"{w}|{t}" for w, t in zip(text.split(), tag_text(text, lex))))
    if args.audit:
        print("\n".join(audit(data_io.load_split("dev"))))


if __name__ == "__main__":
    main()
