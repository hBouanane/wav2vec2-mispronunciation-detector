"""Data pipeline for the reference-anchored MDD model (v2).

Every training example is (audio, reference phonemes, spoken phonemes):
  * Iqra_TTS        reference = phoneme_ref, spoken = phoneme_mis (synthetic errors)
  * Iqra_train      spoken = phoneme_ref (native, correct). With probability `corrupt_p`
                    the *reference* is corrupted with realistic confusions so that the
                    real human audio no longer matches it -> errors on real speech.

From (reference, spoken) we derive per-slot labels by Levenshtein alignment:
  slot_labels[i]  what was said for reference phone i: a phone id, or DEL
  ins_labels[j]   phones inserted after reference position j-1 (j=0: before the first):
                  up to MAX_INS phones in spoken order, then NONE as the stop marker
"""
import json
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Audio, interleave_datasets, load_dataset

SR = 16000
MAX_INS = 3  # insertions per gap; covers 99.99% of Iqra_TTS (1: 98.7%, 2: 99.88%)


# ---------------------------------------------------------------- vocabulary
class PhoneVocab:
    """68 IqraEval phones; slot classes add DEL, insertion classes add NONE (same index)."""

    def __init__(self, vocab_file):
        with open(vocab_file, encoding="utf-8") as f:
            self.phones = [l.strip("\r\n") for l in f if l.strip()]
        self.idx = {p: i for i, p in enumerate(self.phones)}
        n = len(self.phones)
        self.DEL = self.NONE = n            # extra class for slot / insertion heads
        self.UNK = n + 1                    # unknown input phone
        self.PAD = n + 2                    # padding in reference input
        self.BOS = n + 3                    # prepended to reference (gives an insertion slot 0)
        self.BLANK = n                      # CTC blank (CTC head has n+1 outputs)
        self.n_phones = n
        self.n_input = n + 4

    def encode(self, phones):
        return [self.idx.get(p, self.UNK) for p in phones]


def clean(seq):
    return seq.replace("<sil>", " ").split()


# ---------------------------------------------------------------- alignment
def align_labels(ref, spoken, DEL, NONE, max_ins=MAX_INS):
    """Levenshtein-align ref -> spoken (lists of ids). Returns slot labels (len(ref)) and
    insertion labels (len(ref)+1 lists: inserted phones in order + NONE, at most max_ins
    phones). Ties prefer match/substitution."""
    n, m = len(ref), len(spoken)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (ri != spoken[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    slot = [DEL] * n
    ins = [[] for _ in range(n + 1)]
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != spoken[j - 1]):
            slot[i - 1] = spoken[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            slot[i - 1] = DEL
            i -= 1
        else:
            ins[i].append(spoken[j - 1])  # collected right-to-left
            j -= 1
    ins = [g[::-1][:max_ins] + ([NONE] if len(g) < max_ins else []) for g in ins]
    return slot, ins


def labels_to_phones(ref_ids, slot_pred, ins_pred, vocab):
    out = []
    for i in range(len(ref_ids) + 1):
        for c in ins_pred[i]:
            if c == vocab.NONE:
                break
            out.append(vocab.phones[c])
        if i < len(ref_ids) and slot_pred[i] != vocab.DEL:
            out.append(vocab.phones[slot_pred[i]])
    return out


# ---------------------------------------------------------------- corruption
class RefCorrupter:
    """Turns a correctly-spoken phone sequence into a plausible *reference* that differs
    from it, using ref->spoken confusion counts measured on Iqra_TTS.

    The speaker said X at a position; we pick a reference phone Y with P(Y | spoken X)
    so that "reference Y, spoken X" is a realistic mispronunciation."""

    def __init__(self, conf_file, rate_range=(0.02, 0.12)):
        c = json.load(open(conf_file, encoding="utf-8"))
        rev = {}
        for y, row in c["sub"].items():          # y = reference phone, x = spoken phone
            for x, k in row.items():
                rev.setdefault(x, {})[y] = rev.get(x, {}).get(y, 0) + k
        self.rev = {x: (list(d), np.array(list(d.values()), dtype=float) / sum(d.values())) for x, d in rev.items()}
        self.del_phones, w = list(c["del"]), np.array(list(c["del"].values()), dtype=float)
        self.del_w = w / w.sum()
        r = c["rate"]
        tot = r["sub"] + r["del"] + r["ins"]
        self.mix = np.array([r["sub"], r["del"], r["ins"]]) / tot
        self.rate_range = rate_range

    def __call__(self, spoken, rng):
        rate = rng.uniform(*self.rate_range)
        ref = []
        dropped = 0  # at most MAX_INS drops in a row: the insertion head holds MAX_INS phones per gap
        for x in spoken:
            if rng.random() >= rate:
                ref.append(x)
                dropped = 0
                continue
            kind = rng.choice(3, p=self.mix)
            if kind == 2 and dropped >= MAX_INS:
                kind = 0
            dropped = dropped + 1 if kind == 2 else 0
            if kind == 0 and x in self.rev:                    # speaker substituted: ref has Y
                ys, p = self.rev[x]
                ref.append(ys[rng.choice(len(ys), p=p)])
            elif kind == 1:                                    # speaker deleted Y: ref has an extra phone
                ref.append(x)
                ref.append(self.del_phones[rng.choice(len(self.del_phones), p=self.del_w)])
            elif kind == 2:                                    # speaker inserted x: ref lacks it
                continue
            else:
                ref.append(x)
        return ref if ref else list(spoken)


# ---------------------------------------------------------------- datasets
TTS_SHARDS = 13
TTS_VAL_SHARD = 12  # held out


def tts_files(train=True):
    ids = [i for i in range(TTS_SHARDS) if (i != TTS_VAL_SHARD) == train]
    return [f"data/train-{i:05d}-of-{TTS_SHARDS:05d}.parquet" for i in ids]


def _std_tts(ex):
    return {"audio": ex["audio"], "ref": ex["phoneme_ref"], "spoken": ex["phoneme_mis"], "native": False}


def _std_native(ex):
    return {"audio": ex["audio"], "ref": ex["phoneme_ref"], "spoken": ex["phoneme_ref"], "native": True}


def build_train_stream(seed, p_native=0.55, buffer=500):
    tts = load_dataset("IqraEval/Iqra_TTS", data_files=tts_files(True), split="train", streaming=True)
    nat = load_dataset("IqraEval/Iqra_train", split="train", streaming=True)
    tts = tts.cast_column("audio", Audio(sampling_rate=SR)).map(_std_tts, remove_columns=list(tts.column_names))
    nat = nat.cast_column("audio", Audio(sampling_rate=SR)).map(_std_native, remove_columns=list(nat.column_names))
    mixed = interleave_datasets([nat, tts], probabilities=[p_native, 1 - p_native], seed=seed,
                                stopping_strategy="all_exhausted")
    return mixed.shuffle(seed=seed, buffer_size=buffer)


VAL_VERSION = 2  # bump when the validation set changes, so "best F1" is reset on resume


def build_val_sets(corrupter, n_native=1500, n_tts=1500, seed=123):
    """Fixed validation lists (no test data), built to resemble the test set (QuranMB.v2: errors
    in 93% of utterances, ~7% of phonemes): EVERY utterance gets a deterministically corrupted
    reference (2-12% of phones). Audio is always correct speech:
      nat*  Iqra_train dev (native speakers)
      tts*  Iqra_TTS shard 12, which holds only label=="original" (correct) recordings and is
            never used for training.
    v1 left half the native set and all of shard 12 error-free, so only ~1.6% of phonemes were
    errors and precision was not comparable to the leaderboard."""
    rng = np.random.default_rng(seed)
    # data_files: only fetch the dev shard (split="dev" alone would download all 12 GB of train)
    nat = load_dataset("parquet", split="train",
                       data_files="hf://datasets/IqraEval/Iqra_train/data/dev-00000-of-00001.parquet")
    nat = nat.cast_column("audio", Audio(sampling_rate=SR))
    nat = nat.shuffle(seed=seed).select(range(min(n_native, len(nat))))
    out = []
    for k, ex in enumerate(nat):
        spoken = clean(ex["phoneme_ref"])
        ref = corrupter(spoken, rng)
        out.append({"id": f"nat{k:05d}", "audio": np.asarray(ex["audio"]["array"], dtype=np.float32), "ref": ref, "spoken": spoken})
    tts = load_dataset("parquet", split="train",
                       data_files=[f"hf://datasets/IqraEval/Iqra_TTS/{f}" for f in tts_files(False)])
    tts = tts.cast_column("audio", Audio(sampling_rate=SR)).shuffle(seed=seed).select(range(n_tts))
    for k, ex in enumerate(tts):
        spoken = clean(ex["phoneme_mis"])  # == phoneme_ref for these "original" recordings
        out.append({"id": f"tts{k:05d}", "audio": np.asarray(ex["audio"]["array"], dtype=np.float32),
                    "ref": corrupter(spoken, rng), "spoken": spoken})
    return out


# ---------------------------------------------------------------- example -> tensors
class Featurizer:
    def __init__(self, vocab, corrupter, corrupt_p=0.5, max_seconds=15.0, seed=0):
        self.v, self.corrupt, self.corrupt_p = vocab, corrupter, corrupt_p
        self.max_len = int(max_seconds * SR)
        self.rng = np.random.default_rng(seed)

    def from_lists(self, wave, ref, spoken):
        wave = np.asarray(wave, dtype=np.float32)
        if len(wave) > self.max_len or not ref or not spoken:
            return None
        wave = (wave - wave.mean()) / (wave.std() + 1e-7)
        r, s = self.v.encode(ref), self.v.encode(spoken)
        slot, ins = align_labels(r, s, self.v.DEL, self.v.NONE)
        return {"wave": wave, "ref": r, "slot": slot, "ins": ins, "ctc": [t for t in s if t < self.v.n_phones]}

    def __call__(self, ex):
        spoken = clean(ex["spoken"])
        ref = clean(ex["ref"])
        if ex["native"] and self.rng.random() < self.corrupt_p:
            ref = self.corrupt(spoken, self.rng)
        return self.from_lists(ex["audio"]["array"], ref, spoken)


@dataclass
class Collator:
    vocab: PhoneVocab

    def __call__(self, items):
        items = [x for x in items if x is not None]
        if not items:
            return None
        B = len(items)
        T = max(len(x["wave"]) for x in items)
        S = max(len(x["ref"]) for x in items)
        L = max(max(len(x["ctc"]) for x in items), 1)
        wave = torch.zeros(B, T)
        wmask = torch.zeros(B, T, dtype=torch.long)
        ref = torch.full((B, S + 1), self.vocab.PAD, dtype=torch.long)   # +1 for BOS
        slot = torch.full((B, S), -100, dtype=torch.long)
        ins = torch.full((B, S + 1, MAX_INS), -100, dtype=torch.long)
        ctc = torch.full((B, L), -100, dtype=torch.long)
        for b, x in enumerate(items):
            n, s = len(x["wave"]), len(x["ref"])
            wave[b, :n] = torch.from_numpy(x["wave"])
            wmask[b, :n] = 1
            ref[b, 0] = self.vocab.BOS
            ref[b, 1:s + 1] = torch.tensor(x["ref"])
            slot[b, :s] = torch.tensor(x["slot"])
            for j, g in enumerate(x["ins"]):
                ins[b, j, :len(g)] = torch.tensor(g)
            if x["ctc"]:
                ctc[b, :len(x["ctc"])] = torch.tensor(x["ctc"])
        return {"wave": wave, "wave_mask": wmask, "ref": ref, "slot": slot, "ins": ins, "ctc": ctc}
