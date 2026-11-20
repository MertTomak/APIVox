"""Define the shared encoder and the CTC, attention encoder-decoder and RNN-T models trained from scratch."""
import argparse

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src import data_io

BLANK = 0
EOS = 0
HEADS = ("ctc", "aed", "rnnt")
HIDDEN = 256
LAYERS = 4
DROPOUT = 0.2
ATTENTION_DIM = 256
LABEL_SMOOTHING = 0.1
JOINT_DIM = 256
MAX_SYMBOLS_PER_FRAME = 5
FEATURE_DIMS = {"logmel": 80, "mfcc": 39}


def ctc_collapse(ids: list[int], blank: int = BLANK) -> list[int]:
    """Merge consecutive repeats, then drop blanks: [a, a, -, a, b, b, -] -> [a, a, b]."""
    out: list[int] = []
    prev = None
    for i in ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def encode_text(text: str) -> list[int]:
    """Map text to CTC targets CHAR2ID[c] + 1 so that index 0 stays the blank."""
    unknown = sorted({c for c in text if c not in data_io.CHAR2ID})
    if unknown:
        raise ValueError(f"characters outside data_io.CHARS in {text!r}: {unknown}")
    return [data_io.CHAR2ID[c] + 1 for c in text]


def decode_ids(ids: list[int]) -> str:
    """Map collapsed CTC ids back to text with CHARS[id - 1]."""
    return "".join(data_io.CHARS[i - 1] for i in ids)


def subsampled_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Frame counts after two stride-2 convolutions: ceil(ceil(L / 2) / 2) = ceil(L / 4)."""
    return (lengths + 3) // 4


def _zero_padding(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None].to(x.device)
    return x * mask[:, :, None]


class Subsampler(nn.Module):
    """Two Conv1d layers (kernel 3, stride 2) with ReLU: [B, T, D] -> [B, ceil(T / 4), C]."""

    def __init__(self, in_dim: int, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for conv in (self.conv1, self.conv2):
            lengths = (lengths + 1) // 2
            # Padded frames are zeroed so that a batch gives the same output as a single utterance.
            x = _zero_padding(torch.relu(conv(x.transpose(1, 2))).transpose(1, 2), lengths)
        return x, lengths


class Encoder(nn.Module):
    """Subsampler then a bidirectional LSTM over the valid frames: H = BiLSTM(Conv(x)), shape [B, ceil(T / 4), 2 * hidden]."""

    def __init__(self, in_dim: int, hidden: int = HIDDEN, layers: int = LAYERS, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.subsample = Subsampler(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out_dim = 2 * hidden

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, lengths = self.subsample(feats, lengths)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return h, lengths


class CTCModel(nn.Module):
    """Shared encoder plus a linear layer to n_chars + 1 classes, index 0 being the CTC blank."""

    def __init__(self, in_dim: int, n_chars: int, hidden: int = HIDDEN, layers: int = LAYERS,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.encoder = Encoder(in_dim, hidden, layers, dropout)
        self.output = nn.Linear(self.encoder.out_dim, n_chars + 1)
        self.ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=False)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, out_lengths = self.encoder(feats, lengths)
        return self.output(h).float().log_softmax(dim=-1), out_lengths

    def loss(self, feats: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor,
             target_lengths: torch.Tensor) -> torch.Tensor:
        """CTC loss -log p(y | x) summed over alignments, divided by target length and averaged over the batch."""
        log_probs, out_lengths = self(feats, lengths)
        return self.ctc_loss(log_probs.transpose(0, 1), targets, out_lengths, target_lengths)

    @torch.no_grad()
    def decode(self, feats: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        """Greedy decoding: argmax per frame, then ctc_collapse."""
        log_probs, out_lengths = self(feats, lengths)
        best = log_probs.argmax(dim=-1).cpu()
        return [ctc_collapse(best[i, :n].tolist()) for i, n in enumerate(out_lengths.tolist())]


def ctc_min_frames(target: list[int]) -> int:
    """Fewest encoder frames a CTC alignment needs: len(y) plus one blank between each repeated pair."""
    return len(target) + sum(a == b for a, b in zip(target, target[1:]))


class AdditiveAttention(nn.Module):
    """Bahdanau attention: e_t = v^T tanh(W_s s + W_h h_t), alpha = softmax(e) over valid frames, c = sum_t alpha_t h_t."""

    def __init__(self, state_dim: int, enc_dim: int, attn_dim: int = ATTENTION_DIM) -> None:
        super().__init__()
        self.query = nn.Linear(state_dim, attn_dim, bias=False)
        self.key = nn.Linear(enc_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, state: torch.Tensor, h: torch.Tensor, keys: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = self.score(torch.tanh(keys + self.query(state)[:, None, :])).squeeze(-1)
        weights = energy.float().masked_fill(~mask, float("-inf")).softmax(dim=-1)
        return torch.bmm(weights[:, None, :].to(h.dtype), h).squeeze(1), weights


class AEDModel(nn.Module):
    """Shared encoder plus a 1-layer LSTM decoder with additive attention that predicts <eos> (0) or a character (1..n)."""

    def __init__(self, in_dim: int, n_chars: int, max_len: int, hidden: int = HIDDEN, layers: int = LAYERS,
                 dropout: float = DROPOUT) -> None:
        super().__init__()
        self.encoder = Encoder(in_dim, hidden, layers, dropout)
        enc_dim, dec_dim = self.encoder.out_dim, 2 * hidden
        self.sos = n_chars + 1
        self.max_len = max_len
        self.embed = nn.Embedding(n_chars + 2, hidden)
        self.attention = AdditiveAttention(dec_dim, enc_dim)
        self.cell = nn.LSTMCell(hidden + enc_dim, dec_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dec_dim + enc_dim, n_chars + 1)

    def _encode(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h, out_lengths = self.encoder(feats, lengths)
        mask = torch.arange(h.shape[1], device=h.device)[None, :] < out_lengths[:, None].to(h.device)
        zeros = h.new_zeros(h.shape[0], self.cell.hidden_size)
        return h, self.attention.key(h), mask, (zeros, zeros)

    def _step(self, prev: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor], h: torch.Tensor,
              keys: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, tuple, torch.Tensor]:
        context, weights = self.attention(state[0], h, keys, mask)
        state = self.cell(torch.cat([self.embed(prev), context], dim=-1), state)
        logits = self.output(self.dropout(torch.cat([state[0], context], dim=-1)))
        return logits.float(), state, weights

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits [B, S + 1, n_chars + 1] for the inputs <sos> y_1 .. y_S."""
        h, keys, mask, state = self._encode(feats, lengths)
        inputs = torch.cat([targets.new_full((targets.shape[0], 1), self.sos), targets], dim=1)
        logits = []
        for i in range(inputs.shape[1]):
            step_logits, state, _ = self._step(inputs[:, i], state, h, keys, mask)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)

    def loss(self, feats: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor,
             target_lengths: torch.Tensor) -> torch.Tensor:
        """Cross-entropy with label smoothing 0.1 over the tokens y_1 .. y_S <eos>, averaged over tokens."""
        logits = self(feats, lengths, targets)
        positions = torch.arange(logits.shape[1], device=logits.device)[None, :]
        target_lengths = target_lengths.to(logits.device)
        gold = torch.cat([targets, targets.new_zeros(targets.shape[0], 1)], dim=1)
        gold = gold.masked_fill(positions == target_lengths[:, None], EOS)
        valid = positions <= target_lengths[:, None]
        return F.cross_entropy(logits[valid], gold[valid], label_smoothing=LABEL_SMOOTHING)

    @torch.no_grad()
    def decode(self, feats: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        """Greedy decoding from <sos> until <eos> or max_len characters."""
        h, keys, mask, state = self._encode(feats, lengths)
        prev = torch.full((h.shape[0],), self.sos, dtype=torch.long, device=h.device)
        out = [[] for _ in range(h.shape[0])]
        done = torch.zeros(h.shape[0], dtype=torch.bool, device=h.device)
        for _ in range(self.max_len):
            logits, state, _ = self._step(prev, state, h, keys, mask)
            prev = logits.argmax(dim=-1)
            for b in torch.nonzero(~done & (prev != EOS)).flatten().tolist():
                out[b].append(int(prev[b]))
            done |= prev == EOS
            if done.all():
                break
        return out


class RNNTModel(nn.Module):
    """Shared encoder, a 1-layer LSTM prediction network over previous non-blank characters and a joint network."""

    def __init__(self, in_dim: int, n_chars: int, hidden: int = HIDDEN, layers: int = LAYERS,
                 dropout: float = DROPOUT, joint_dim: int = JOINT_DIM) -> None:
        super().__init__()
        self.encoder = Encoder(in_dim, hidden, layers, dropout)
        pred_dim = 2 * hidden
        self.embed = nn.Embedding(n_chars + 1, hidden)
        self.predictor = nn.LSTM(hidden, pred_dim, batch_first=True)
        self.enc_proj = nn.Linear(self.encoder.out_dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim, bias=False)
        self.output = nn.Linear(joint_dim, n_chars + 1)

    def joint(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Joint network z = W_out tanh(W_enc h_t + W_pred g_u), on already projected h_t and g_u."""
        return self.output(torch.tanh(enc + pred)).float()

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor,
                targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint logits [B, T, U + 1, n_chars + 1] for the predictor inputs <blank> y_1 .. y_U."""
        h, out_lengths = self.encoder(feats, lengths)
        inputs = torch.cat([targets.new_full((targets.shape[0], 1), BLANK), targets], dim=1)
        g, _ = self.predictor(self.embed(inputs))
        return self.joint(self.enc_proj(h)[:, :, None, :], self.pred_proj(g)[:, None, :, :]), out_lengths

    def loss(self, feats: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor,
             target_lengths: torch.Tensor) -> torch.Tensor:
        """RNN-T loss -log p(y | x) summed over alignments, divided by target length and averaged over the batch."""
        targets = targets[:, :int(target_lengths.max())]
        logits, out_lengths = self(feats, lengths, targets)
        per_utt = torchaudio.functional.rnnt_loss(
            logits, targets.int(), out_lengths.to(logits.device).int(), target_lengths.to(logits.device).int(),
            blank=BLANK, reduction="none")
        return (per_utt / target_lengths.to(logits.device).clamp(min=1)).mean()

    def _predict(self, tokens: torch.Tensor, state: tuple | None) -> tuple[torch.Tensor, tuple]:
        g, state = self.predictor(self.embed(tokens)[:, None, :], state)
        return self.pred_proj(g[:, 0]), state

    @torch.no_grad()
    def decode(self, feats: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        """Greedy transducer decoding: per frame emit argmax symbols until blank, at most 5 per frame."""
        h, out_lengths = self.encoder(feats, lengths)
        enc = self.enc_proj(h)
        batch = h.shape[0]
        pred, state = self._predict(torch.full((batch,), BLANK, dtype=torch.long, device=h.device), None)
        out = [[] for _ in range(batch)]
        valid = torch.arange(h.shape[1], device=h.device)[None, :] < out_lengths[:, None].to(h.device)
        for t in range(h.shape[1]):
            active = valid[:, t]
            for _ in range(MAX_SYMBOLS_PER_FRAME):
                best = self.joint(enc[:, t], pred).argmax(dim=-1)
                emit = active & (best != BLANK)
                if not emit.any():
                    break
                for b in torch.nonzero(emit).flatten().tolist():
                    out[b].append(int(best[b]))
                new_pred, new_state = self._predict(best, state)
                # Only utterances that emitted a symbol advance their prediction network.
                pred = torch.where(emit[:, None], new_pred, pred)
                state = tuple(torch.where(emit[None, :, None], n, s) for n, s in zip(new_state, state))
                active = emit
        return out


def build_model(head: str, in_dim: int, n_chars: int | None = None, max_len: int | None = None,
                hidden: int = HIDDEN, layers: int = LAYERS, dropout: float = DROPOUT) -> nn.Module:
    """Create the model for a head on top of the shared encoder; n_chars defaults to len(data_io.CHARS)."""
    n_chars = n_chars or len(data_io.CHARS)
    if head == "ctc":
        return CTCModel(in_dim, n_chars, hidden, layers, dropout)
    if head == "aed":
        if not max_len:
            raise ValueError("the aed head needs max_len (2 x the longest train target)")
        return AEDModel(in_dim, n_chars, max_len, hidden, layers, dropout)
    if head == "rnnt":
        return RNNTModel(in_dim, n_chars, hidden, layers, dropout)
    raise ValueError(f"unknown head {head!r}, expected one of {HEADS}")


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the parameter count of a from-scratch model.")
    parser.add_argument("--head", choices=HEADS, default="ctc", help="output head (default: ctc)")
    parser.add_argument("--features", choices=tuple(FEATURE_DIMS), default="logmel",
                        help="input features (default: logmel)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(args.head, FEATURE_DIMS[args.features], max_len=1)
    print(f"{args.head} on {args.features}: {count_params(model):,} parameters, "
          f"{model.output.out_features} output classes")


if __name__ == "__main__":
    main()
