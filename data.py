"""Training / dev data read directly from the released parquet files (no streaming, no
datasets cache, no modification of references or labels).

  Iqra_train (native)  target = phoneme_ref (== phoneme_aug: native speakers read correctly)
  Iqra_TTS             target = phoneme_mis (what the TTS voice actually says), reference
                       = phoneme_ref (kept only for the optional reference penalty and dev MDD)

Shuffling: each epoch the list of all row groups (~100 rows each, from every file) is
shuffled, split across GPUs and data workers, and each worker mixes `shuffle_pool` row
groups at a time, so a batch mixes random files, speakers and both datasets.
"""
import glob
import io
import math
import os
import zlib

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import IterableDataset, get_worker_info

SR = 16000
COLUMNS = {"native": ("audio", "phoneme_ref", "phoneme_ref"),   # audio, reference, target
           "tts": ("audio", "phoneme_ref", "phoneme_mis")}


def expand(pattern):
    """Glob with ** support (Kaggle mounts notebook outputs at varying depths under /kaggle/input)."""
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        root = pattern.split("*")[0] or "."
        found = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))[:10]
        raise FileNotFoundError(f"no parquet files match {pattern}; parquet files under {root}: {found or 'none'}")
    unique = {}
    for f in files:  # the same data attached twice must not be read twice
        unique.setdefault(os.path.basename(f), f)
    return sorted(unique.values())


def decode_audio(audio):
    wave, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(1)
    if sr != SR:
        g = math.gcd(sr, SR)
        wave = resample_poly(wave, SR // g, sr // g).astype(np.float32)
    return wave


def is_tts_dev(path, rg, every):
    """Deterministic hold-out of whole TTS row groups (stable across machines and runs)."""
    name = path.replace("\\", "/").split("/")[-1]
    return zlib.crc32(f"{name}:{rg}".encode()) % every == 0


def row_groups(files, kind, keep):
    units = []
    for f in files:
        for rg in range(pq.ParquetFile(f).metadata.num_row_groups):
            if keep(f, rg):
                units.append((f, rg, kind))
    return units


def augment(wave, cfg, rng):
    if rng.random() < cfg.speed_p:
        factor = 1.0 + rng.uniform(-cfg.speed_range, cfg.speed_range)
        up, down = 100, int(round(100 * factor))            # factor > 1 -> faster -> shorter
        wave = resample_poly(wave, up, down).astype(np.float32)
    if rng.random() < cfg.noise_p:
        snr = rng.uniform(*cfg.noise_snr_db)
        power = float(np.mean(wave ** 2)) + 1e-10
        wave = wave + rng.standard_normal(len(wave)).astype(np.float32) * math.sqrt(power / 10 ** (snr / 10))
    return wave


class RowGroupShuffle(IterableDataset):
    """Infinite iterator over training examples (each rank and worker gets its own share of
    every epoch; training stops by step count, so ranks never run out at different times)."""

    def __init__(self, units, vocab, cfg, rank=0, world=1, seed=0, train=True):
        self.units, self.vocab, self.cfg = units, vocab, cfg
        self.rank, self.world, self.seed, self.train = rank, world, seed, train
        self.qps = None
        if cfg.target == "qps":  # targets for muaalem's own phoneme + tafkheem heads
            from muaalem import QPSVocab
            self.qps = QPSVocab()

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, self.rank, wid])
        handles = {}
        epoch = 0
        while True:
            order = np.random.default_rng([self.seed, epoch]).permutation(len(self.units))
            mine = order[self.rank::self.world][wid::nw]
            for start in range(0, len(mine), self.cfg.shuffle_pool):
                pool = []
                for u in mine[start:start + self.cfg.shuffle_pool]:
                    path, rg, kind = self.units[u]
                    if path not in handles:
                        handles[path] = pq.ParquetFile(path)
                    a, r, t = COLUMNS[kind]
                    keep_label = self.cfg.tts_labels if kind == "tts" and self.cfg.tts_labels != "all" else None
                    cols = list(dict.fromkeys([a, r, t] + (["label"] if keep_label else [])))
                    tab = handles[path].read_row_group(rg, columns=cols)
                    name = path.replace("\\", "/").split("/")[-1]
                    for row in tab.to_pylist():
                        if keep_label and row["label"] != keep_label:
                            continue  # e.g. --tts-labels original: skip clips with injected errors
                        pool.append((row[a], row[r], row[t], f"{kind}:{name}:{rg}"))
                rng.shuffle(pool)
                for audio, ref, tgt, src in pool:
                    ex = self.featurize(audio, ref, tgt, rng)
                    if ex is not None:
                        ex["src"] = src  # dataset:file:row_group, for diagnostics only
                        yield ex
            epoch += 1

    def featurize(self, audio, ref, tgt, rng):
        label = self.vocab.encode(tgt)
        if not label:
            return None
        wave = decode_audio(audio)
        if self.train:
            wave = augment(wave, self.cfg, rng)
        if len(wave) > self.cfg.max_seconds * SR:
            return None
        ex = {"wave": wave, "label": label, "ref": self.vocab.encode(ref)}
        if self.qps is not None:
            ex["qps"], ex["tafk"] = self.qps.encode(self.vocab.decode(label))
            if not ex["qps"]:
                return None
        return ex


def load_dev(cfg, vocab, tts_units):
    """Fixed dev list, unmodified: Iqra_train dev (native) + held-out TTS row groups."""
    rng = np.random.default_rng(123)
    out = []
    for path in expand(cfg.native_dev):
        rows = pq.read_table(path, columns=["audio", "phoneme_ref"]).to_pylist()
        for k in rng.permutation(len(rows))[:cfg.n_dev_native]:
            p = vocab.encode(rows[k]["phoneme_ref"])
            out.append({"id": f"nat{k:05d}", "wave": decode_audio(rows[k]["audio"]), "ref": p, "label": p})
    tts_rows = []
    for path, rg, _ in tts_units:
        tab = pq.ParquetFile(path).read_row_group(rg, columns=["audio", "phoneme_ref", "phoneme_mis", "label"])
        tts_rows += tab.to_pylist()
    for k in rng.permutation(len(tts_rows))[:cfg.n_dev_tts]:
        r = tts_rows[k]
        out.append({"id": f"tts{k:05d}", "wave": decode_audio(r["audio"]), "ref": vocab.encode(r["phoneme_ref"]),
                    "label": vocab.encode(r["phoneme_mis"]), "tts_label": r["label"]})
    return [x for x in out if x["label"] and len(x["wave"]) <= 60 * SR]


def build(cfg, vocab):
    """Training units (row groups) and the held-out TTS units."""
    native = row_groups(expand(cfg.native_glob), "native", lambda f, rg: True)
    tts_all = row_groups(expand(cfg.tts_glob), "tts", lambda f, rg: True)
    tts_dev = [u for u in tts_all if is_tts_dev(u[0], u[1], cfg.tts_dev_every)]
    tts_train = [u for u in tts_all if not is_tts_dev(u[0], u[1], cfg.tts_dev_every)]
    return native + tts_train, tts_dev


class Collator:
    """Pads a batch with the encoder's own feature extractor (raw waveform for wav2vec2,
    log-mel features for w2v-BERT)."""

    def __init__(self, extractor):
        self.fx = extractor

    def __call__(self, items):
        feats = self.fx([x["wave"] for x in items], sampling_rate=SR, return_tensors="pt", padding=True,
                        return_attention_mask=True)
        key = "input_values" if "input_values" in feats else "input_features"
        L = max(len(x["label"]) for x in items)
        R = max(max((len(x["ref"]) for x in items), default=0), 1)
        labels = torch.full((len(items), L), -100, dtype=torch.long)
        refs = torch.full((len(items), R), -100, dtype=torch.long)
        for i, x in enumerate(items):
            labels[i, :len(x["label"])] = torch.tensor(x["label"])
            if x["ref"]:
                refs[i, :len(x["ref"])] = torch.tensor(x["ref"])
        batch = {"inputs": feats[key], "attention_mask": feats["attention_mask"], "labels": labels, "refs": refs}
        for k in ("qps", "tafk"):  # muaalem targets (target == "qps")
            if all(k in x for x in items):
                t = torch.full((len(items), max(max(len(x[k]) for x in items), 1)), -100, dtype=torch.long)
                for i, x in enumerate(items):
                    t[i, :len(x[k])] = torch.tensor(x[k])
                batch[k] = t
        return batch
