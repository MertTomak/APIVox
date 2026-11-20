"""Train and evaluate the GMM-HMM baseline: char units with 3-state HMMs on per-utterance normalized MFCCs, Viterbi training and a char bigram LM."""
import argparse
import contextlib
import hashlib
import inspect
import itertools
import multiprocessing as mp
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from src import data_io, features, metrics

MODEL_NAME = "gmm_hmm"
N_CHARS = len(data_io.CHARS)
SIL = N_CHARS
N_UNITS = N_CHARS + 1
N_SUB = 3
N_STATES = N_UNITS * N_SUB
SIL_START, SIL_END = N_CHARS, N_CHARS + 1
STAGES = (1, 2, 4, 8)
ITERS_PER_STAGE = 3
VAR_FLOOR = 0.01
MIN_OCC = 1.0
SPLIT_EPS = 0.2
LM_SMOOTHING = 0.1
LM_WEIGHTS = (1.0, 2.0, 4.0, 8.0)
PENALTIES = (0.0, 1.0, 2.0, 3.0)
CHUNK = 100
NEG = -np.inf
FINAL_EVAL = ("test", "test_talkshow", "test_read")
DEFAULT_CACHE = "/tmp/aralas_gmm"


@dataclass
class Model:
    """Diagonal GMM per HMM state plus log self-loop and log advance probabilities per state."""

    means: np.ndarray
    var: np.ndarray
    log_w: np.ndarray
    log_self: np.ndarray
    log_adv: np.ndarray

    @property
    def n_components(self) -> int:
        return self.means.shape[1]

    def compile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the quadratic-form parameters (A, B, c) so log-densities become two matrix products."""
        s, k, d = self.means.shape
        prec = 1.0 / self.var
        const = -0.5 * (d * np.log(2 * np.pi) + np.log(self.var).sum(-1) + (self.means**2 * prec).sum(-1)) + self.log_w
        return (-0.5 * prec).reshape(s * k, d), (self.means * prec).reshape(s * k, d), const.reshape(-1)


def initial_model(dim: int) -> Model:
    """One standard-normal component per state with even transitions."""
    half = np.full(N_STATES, np.log(0.5))
    return Model(np.zeros((N_STATES, 1, dim)), np.ones((N_STATES, 1, dim)), np.zeros((N_STATES, 1)), half, half.copy())


def component_logprobs(compiled: tuple, x: np.ndarray, k: int) -> np.ndarray:
    """Weighted log-density of every frame under every component, shape [T, S, K]."""
    a, b, c = compiled
    x = x.astype(np.float64)
    return ((x * x) @ a.T + x @ b.T + c).reshape(len(x), N_STATES, k)


def logsumexp_last(lp: np.ndarray) -> np.ndarray:
    """Log-sum-exp over the last axis."""
    m = lp.max(-1, keepdims=True)
    return (m + np.log(np.exp(lp - m).sum(-1, keepdims=True)))[..., 0]


def cmvn(feats: np.ndarray) -> np.ndarray:
    """Per-utterance mean and variance normalization of every feature dimension."""
    return ((feats - feats.mean(0)) / (feats.std(0) + 1e-8)).astype(np.float32)


def cache_subdir(cache_dir: str | Path) -> Path:
    """Cache folder keyed by the source of features.mfcc so a changed extractor never reuses old files."""
    try:
        key = hashlib.sha1(inspect.getsource(features.mfcc).encode()).hexdigest()[:10]
    except (OSError, TypeError):
        key = "nosource"
    path = Path(cache_dir) / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_feats(cache: Path, utt_id: str) -> np.ndarray:
    """Read the cached normalized MFCCs of one utterance."""
    return np.load(cache / f"{utt_id}.npy")


def _prepare_chunk(task: tuple) -> list[int]:
    records, cache = task
    frames = []
    for rec in records:
        path = cache / f"{rec['id']}.npy"
        if not path.is_file():
            np.save(path, cmvn(features.mfcc(data_io.load_audio(rec))))
        frames.append(len(np.load(path, mmap_mode="r")))
    return frames


def chunks(items: Sequence, size: int = CHUNK) -> list[list]:
    """Fixed-size chunks in input order, independent of the worker count."""
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


@contextlib.contextmanager
def mapper(workers: int) -> Iterator[Callable]:
    """Ordered map over tasks, in-process for one worker and in a process pool otherwise."""
    if workers <= 1:
        yield lambda fn, tasks: [fn(t) for t in tasks]
        return
    with mp.get_context("spawn").Pool(workers) as pool:
        yield lambda fn, tasks: pool.map(fn, tasks, chunksize=1)


def state_sequence(text: str) -> np.ndarray:
    """HMM state ids of an utterance: silence, one unit per character, silence."""
    units = np.array([SIL] + [data_io.CHAR2ID[c] for c in text] + [SIL])
    return (units[:, None] * N_SUB + np.arange(N_SUB)).ravel()


def forced_align(emit: np.ndarray, seq: np.ndarray, log_self: np.ndarray, log_adv: np.ndarray) -> tuple[np.ndarray, float]:
    """Viterbi path through the left-to-right chain seq: returns the chain position of every frame and the path score."""
    e = emit[:, seq]
    n_frames, n = e.shape
    stay, adv = log_self[seq], log_adv[seq]
    delta = np.full(n, NEG)
    delta[0] = e[0, 0]
    back = np.zeros((n_frames, n), dtype=bool)
    for t in range(1, n_frames):
        moved = np.concatenate(([NEG], delta[:-1] + adv[:-1]))
        kept = delta + stay
        back[t] = moved > kept
        delta = np.maximum(kept, moved) + e[t]
    path = np.empty(n_frames, dtype=np.int64)
    pos = n - 1
    path[-1] = pos
    for t in range(n_frames - 1, 0, -1):
        pos -= int(back[t, pos])
        path[t - 1] = pos
    return path, float(delta[-1])


def flat_path(n_frames: int, n_states: int) -> np.ndarray:
    """Even segmentation of the frames over the chain, used for the flat start."""
    return (np.arange(n_frames) * n_states) // n_frames


@dataclass
class Stats:
    """Sufficient statistics of one pass over the aligned frames."""

    occ: np.ndarray
    sum_x: np.ndarray
    sum_xx: np.ndarray
    n_self: np.ndarray
    n_adv: np.ndarray
    loglik: float = 0.0
    frames: int = 0

    @classmethod
    def zeros(cls, k: int, dim: int) -> "Stats":
        return cls(np.zeros((N_STATES, k)), np.zeros((N_STATES, k, dim)), np.zeros((N_STATES, k, dim)), np.zeros(N_STATES), np.zeros(N_STATES))

    def add(self, other: "Stats") -> None:
        for name in ("occ", "sum_x", "sum_xx", "n_self", "n_adv", "loglik", "frames"):
            setattr(self, name, getattr(self, name) + getattr(other, name))


def accumulate(x: np.ndarray, lp: np.ndarray, states: np.ndarray) -> Stats:
    """Responsibility-weighted statistics of the frames assigned to their aligned states."""
    n_frames, _, k = lp.shape
    own = lp[np.arange(n_frames), states]
    gamma = np.exp(own - logsumexp_last(own)[:, None])
    onehot = np.zeros((n_frames, N_STATES))
    onehot[np.arange(n_frames), states] = 1.0
    x = x.astype(np.float64)
    st = Stats.zeros(k, x.shape[1])
    st.occ = onehot.T @ gamma
    for j in range(k):
        w = onehot * gamma[:, j : j + 1]
        st.sum_x[:, j] = w.T @ x
        st.sum_xx[:, j] = w.T @ (x * x)
    same = states[:-1] == states[1:]
    st.n_self = np.bincount(states[:-1][same], minlength=N_STATES).astype(np.float64)
    st.n_adv = np.bincount(states[:-1][~same], minlength=N_STATES).astype(np.float64)
    st.frames = n_frames
    return st


def _train_chunk(task: tuple) -> Stats:
    model, cache, items, flat = task
    compiled, k = model.compile(), model.n_components
    total = None
    for utt_id, text in items:
        x = load_feats(cache, utt_id)
        seq = state_sequence(text)
        lp = component_logprobs(compiled, x, k)
        if flat:
            states, score = seq[flat_path(len(x), len(seq))], 0.0
        else:
            path, score = forced_align(logsumexp_last(lp), seq, model.log_self, model.log_adv)
            states = seq[path]
        st = accumulate(x, lp, states)
        st.loglik = score
        if total is None:
            total = st
        else:
            total.add(st)
    return total


def m_step(model: Model, st: Stats) -> Model:
    """Maximum-likelihood update of GMMs and transitions; states or components without data keep their values."""
    occ = st.occ[..., None]
    seen = occ >= MIN_OCC
    safe = np.where(seen, occ, 1.0)
    means = np.where(seen, st.sum_x / safe, model.means)
    var = np.where(seen, np.maximum(st.sum_xx / safe - means**2, VAR_FLOOR), model.var)
    total = st.occ.sum(1, keepdims=True)
    weights = np.maximum(st.occ / np.maximum(total, 1e-12), 1e-4)
    log_w = np.where(total >= MIN_OCC, np.log(weights / weights.sum(1, keepdims=True)), model.log_w)
    p_self = (st.n_self + 1.0) / (st.n_self + st.n_adv + 2.0)
    return Model(means, var, log_w, np.log(p_self), np.log1p(-p_self))


def split_model(model: Model, rng: np.random.Generator) -> Model:
    """Double every GMM by moving each component by plus and minus 0.2 sigma along a random sign pattern."""
    sign = rng.choice([-1.0, 1.0], size=model.means.shape)
    delta = SPLIT_EPS * np.sqrt(model.var) * sign
    return Model(
        np.concatenate([model.means - delta, model.means + delta], axis=1),
        np.concatenate([model.var, model.var], axis=1),
        np.concatenate([model.log_w, model.log_w], axis=1) - np.log(2),
        model.log_self,
        model.log_adv,
    )


def run_pass(model: Model, chunked: list, cache: Path, flat: bool, run_map: Callable) -> Stats:
    """One pass over all utterances; chunk statistics are summed in fixed chunk order."""
    parts = run_map(_train_chunk, [(model, cache, c, flat) for c in chunked])
    total = parts[0]
    for part in parts[1:]:
        total.add(part)
    return total


def train(
    items: list[tuple[str, str]], cache: Path, seed: int, workers: int = 1, stages: Sequence[int] | None = None, iters: int | None = None
) -> tuple[Model, list[float]]:
    """Flat start, then Viterbi iterations per mixture stage; returns the model and the mean log-likelihood per frame of every iteration."""
    stages, iters = stages or STAGES, iters or ITERS_PER_STAGE
    rng = np.random.default_rng(seed)
    chunked = chunks(items)
    dim = load_feats(cache, items[0][0]).shape[1]
    history: list[float] = []
    with mapper(workers) as run_map:
        model = m_step(initial_model(dim), run_pass(initial_model(dim), chunked, cache, True, run_map))
        for stage, k in enumerate(stages):
            while model.n_components < k:
                model = split_model(model, rng)
            for it in range(iters):
                st = run_pass(model, chunked, cache, False, run_map)
                history.append(st.loglik / st.frames)
                print(f"stage {k} comps  iter {it + 1}/{iters}  mean log-likelihood per frame {history[-1]:.4f}", flush=True)
                model = m_step(model, st)
    return model, history


def build_lm(texts: Sequence[str]) -> np.ndarray:
    """Log char-bigram table: rows are <s> then the characters, columns are the characters then </s>."""
    counts = np.full((N_CHARS + 1, N_CHARS + 1), LM_SMOOTHING)
    for text in texts:
        prev = 0
        for ch in text:
            counts[prev, data_io.CHAR2ID[ch]] += 1
            prev = data_io.CHAR2ID[ch] + 1
        counts[prev, N_CHARS] += 1
    return np.log(counts / counts.sum(1, keepdims=True))


SRC_UNITS = [SIL_START] + list(range(N_CHARS))
DST_UNITS = list(range(N_CHARS)) + [SIL_END]
UNIT_STATES = np.array([u * N_SUB if u < N_CHARS else SIL * N_SUB for u in range(N_UNITS + 1)])[:, None] + np.arange(N_SUB)


def decode(emit: np.ndarray, model: Model, lm: np.ndarray, weight: float, penalty: float, beam: float) -> str:
    """Exact Viterbi over a loop of character units between two silences, scored with the bigram LM."""
    n_frames = len(emit)
    e = emit[:, UNIT_STATES]
    stay, adv = model.log_self[UNIT_STATES], model.log_adv[UNIT_STATES]
    lm_scores = weight * lm
    lm_scores[:, :N_CHARS] -= penalty
    lm_scores[0, N_CHARS] = NEG
    n_units = N_UNITS + 1
    alpha = np.full((n_units, N_SUB), NEG)
    alpha[SIL_START, 0] = e[0, SIL_START, 0]
    moved_flag = np.zeros((n_frames, n_units, N_SUB), dtype=bool)
    from_src = np.zeros((n_frames, N_CHARS + 1), dtype=np.int64)
    cols = np.arange(N_CHARS + 1)
    for t in range(1, n_frames):
        cand = (alpha[SRC_UNITS, 2] + adv[SRC_UNITS, 2])[:, None] + lm_scores
        arg = cand.argmax(0)
        moved = np.full((n_units, N_SUB), NEG)
        moved[:, 1:] = alpha[:, :-1] + adv[:, :-1]
        moved[DST_UNITS, 0] = cand[arg, cols]
        kept = alpha + stay
        moved_flag[t], from_src[t] = moved > kept, arg
        alpha = np.maximum(kept, moved) + e[t]
        if np.isfinite(beam):
            alpha[alpha < alpha.max() - beam] = NEG
    if not np.isfinite(alpha[SIL_END, 2]):
        return ""
    return _traceback(moved_flag, from_src)


def _traceback(moved_flag: np.ndarray, from_src: np.ndarray) -> str:
    """Characters on the best path, read backwards from the final silence."""
    unit, sub, chars = SIL_END, 2, []
    for t in range(len(moved_flag) - 1, 0, -1):
        if not moved_flag[t, unit, sub]:
            continue
        if sub > 0:
            sub -= 1
            continue
        if unit < N_CHARS:
            chars.append(data_io.ID2CHAR[unit])
        unit, sub = SRC_UNITS[from_src[t, unit if unit < N_CHARS else N_CHARS]], 2
    return "".join(reversed(chars))


def _decode_chunk(task: tuple) -> list[list[str]]:
    model, lm, cache, ids, settings, beam = task
    compiled, k = model.compile(), model.n_components
    out: list[list[str]] = [[] for _ in settings]
    for utt_id in ids:
        emit = logsumexp_last(component_logprobs(compiled, load_feats(cache, utt_id), k))
        for slot, (weight, penalty) in enumerate(settings):
            out[slot].append(decode(emit, model, lm, weight, penalty, beam))
    return out


def decode_split(
    model: Model, lm: np.ndarray, cache: Path, ids: list[str], settings: list[tuple[float, float]], beam: float, run_map: Callable
) -> list[list[str]]:
    """Hypotheses of every utterance for every (LM weight, insertion penalty) setting."""
    parts = run_map(_decode_chunk, [(model, lm, cache, c, settings, beam) for c in chunks(ids)])
    return [list(itertools.chain.from_iterable(p[s] for p in parts)) for s in range(len(settings))]


def prepare(records: list[dict], cache: Path, run_map: Callable) -> dict[str, int]:
    """Extract and cache the features of the records; returns the frame count per id."""
    parts = run_map(_prepare_chunk, [(c, cache) for c in chunks(records)])
    return dict(zip((r["id"] for r in records), itertools.chain.from_iterable(parts)))


def select_train(records: list[dict], hours: float | None, seed: int) -> list[dict]:
    """A seeded random subset of about the given hours, or all records; returned sorted by id."""
    if hours is None:
        return sorted(records, key=lambda r: r["id"])
    chosen, total = [], 0.0
    for i in np.random.default_rng(seed).permutation(len(records)):
        if total >= hours * 3600:
            break
        chosen.append(records[i])
        total += records[i]["duration"]
    return sorted(chosen, key=lambda r: r["id"])


def cpu_name() -> str:
    """CPU model string of this machine."""
    try:
        if platform.system() == "Darwin":
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        else:
            name = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration, subprocess.CalledProcessError):
        name = platform.processor() or platform.machine()
    return f"{name} ({mp.cpu_count()} cores)"


def tune(model: Model, lm: np.ndarray, cache: Path, dev: list[dict], beam: float, run_map: Callable) -> tuple[tuple[float, float], list[list[str]]]:
    """Score the LM weight x insertion penalty grid on dev and return the best setting with its dev hypotheses."""
    settings = list(itertools.product(LM_WEIGHTS, PENALTIES))
    hyps = decode_split(model, lm, cache, [r["id"] for r in dev], settings, beam, run_map)
    wers = [metrics.evaluate(dev, h)["wer"] for h in hyps]
    print("LM weight  insertion penalty  dev WER")
    for (w, p), wer in zip(settings, wers):
        print(f"{w:9.1f}  {p:17.1f}  {wer:.4f}")
    best = int(np.argmin(wers))
    print(f"best: LM weight {settings[best][0]}, insertion penalty {settings[best][1]}, dev WER {wers[best]:.4f}", flush=True)
    return settings[best], hyps[best]


def evaluate_final(model, lm, cache, setting, beam, run_map, seed, dev, dev_hyps):
    """Decode the final splits with the chosen setting; returns metrics per split and writes all predictions."""
    scores = {}
    for name, records, hyps in [("dev", dev, dev_hyps)] + [(n, *_decode_final(n, model, lm, cache, setting, beam, run_map)) for n in FINAL_EVAL]:
        preds = [{"id": r["id"], "hyp": h} for r, h in zip(records, hyps)]
        metrics.write_preds(MODEL_NAME, seed, name, preds, final=name in FINAL_EVAL)
        scores[name] = metrics.evaluate(records, hyps)
        print(f"{name}: WER {scores[name]['wer']:.4f}  CER {scores[name]['cer']:.4f}", flush=True)
    return scores


def _decode_final(name, model, lm, cache, setting, beam, run_map):
    records = data_io.load_split(name, final=True)
    prepare(records, cache, run_map)
    return records, decode_split(model, lm, cache, [r["id"] for r in records], [setting], beam, run_map)[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13, help="seed for the train subset and the mixture split directions")
    parser.add_argument("--final", action="store_true", help="also evaluate test, test_talkshow and test_read and write predictions and the run file")
    parser.add_argument("--train-hours", type=float, default=None, help="train on a random subset of about this many hours (default: all of train)")
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()), help="worker processes; 1 runs in-process")
    parser.add_argument("--beam", type=float, default=200.0, help="log-score beam of the decoder; inf turns pruning off")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE, help="feature cache outside the repository")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start = time.time()
    with mapper(args.workers) as run_map:
        cache = cache_subdir(args.cache_dir)
        train_recs = select_train(data_io.load_split("train"), args.train_hours, args.seed)
        frames = prepare(train_recs, cache, run_map)
        usable = [r for r in train_recs if frames[r["id"]] >= N_SUB * (len(r["text"]) + 2)]
        hours = sum(r["duration"] for r in usable) / 3600
        print(f"train: {len(usable)} utterances, {hours:.3f} h, {len(train_recs) - len(usable)} skipped as too short for their text", flush=True)
        model, history = train([(r["id"], r["text"]) for r in usable], cache, args.seed, args.workers)
        train_time = time.time() - start
        lm = build_lm([r["text"] for r in usable])
        dev = data_io.load_split("dev")
        prepare(dev, cache, run_map)
        setting, dev_hyps = tune(model, lm, cache, dev, args.beam, run_map)
        print(f"training time {train_time:.0f} s (features and training), total {time.time() - start:.0f} s")
        if not args.final:
            return
        scores = evaluate_final(model, lm, cache, setting, args.beam, run_map, args.seed, dev, dev_hyps)
    config = {
        "components": list(STAGES),
        "iterations_per_stage": ITERS_PER_STAGE,
        "lm_weight_grid": list(LM_WEIGHTS),
        "insertion_penalty_grid": list(PENALTIES),
        "lm_weight": setting[0],
        "insertion_penalty": setting[1],
        "beam": args.beam,
        "train_hours_requested": args.train_hours,
        "train_hours_used": round(hours, 3),
        "train_utterances": len(usable),
        "skipped_utterances": len(train_recs) - len(usable),
        "final_train_loglik_per_frame": history[-1],
    }
    n_gauss = int(np.prod(model.means.shape[:2]))
    metrics.write_result(
        model=MODEL_NAME, seed=args.seed, checkpoint=None, revision=None, config=config, train_time_s=train_time,
        hardware=cpu_name(), n_params=n_gauss, best_epoch=None, splits=scores,
    )


if __name__ == "__main__":
    main()
