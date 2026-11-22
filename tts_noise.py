"""How noisy are the Iqra_TTS training labels? A trained model (here xlsr step 9k) is used as
a judge on a sample of TTS *training* clips:

  1. per speaker / per label type: how far the model's transcription is from the label (PER)
  2. injected errors: for every substitution the label claims (reference Y -> spoken X), does
     the model hear X (error realised), the reference Y (TTS probably did not say it), or
     something else?
  3. per clip: the distribution of disagreement, to choose a cleaning threshold

  python tts_noise.py --logits results/logits_ctc_step9000
"""
import argparse
import collections
import os
import pickle

import jiwer
import numpy as np

from utils import greedy_ctc


def align(a, b):
    """Levenshtein alignment; returns for each index of `a` the aligned token of `b` (or None)."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (a[i - 1] != b[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    out, i, j = [None] * n, n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            out[i - 1] = b[j - 1]; i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    a = ap.parse_args()
    d = pickle.load(open(os.path.join(a.logits, "tts_train.pkl"), "rb"))
    blank, phones = d["blank"], d["phones"]
    rows = d["rows"]
    for r in rows:
        r["pred"] = [phones[i] for i in greedy_ctc(r["logp"].astype(np.float32).argmax(-1), len(r["logp"]), blank)]
        r["per"] = jiwer.wer(" ".join(r["label"]), " ".join(r["pred"]) or "<empty>")

    print("1) model vs label, by speaker and label type (PER, lower = label and audio agree)")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["speaker"], r["tts_label"])].append(r["per"])
    for (spk, lab), v in sorted(groups.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"   {spk[:18]:18s} {lab:9s} n={len(v):4d}  mean PER {np.mean(v):.3f}  median {np.median(v):.3f}")

    print("\n2) injected substitutions in 'augmented' clips: what does the model hear?")
    by_spk = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["tts_label"] != "augmented":
            continue
        ref_at_label = align(r["label"], r["ref"])      # reference phone aligned to each label position
        heard = align(r["label"], r["pred"])            # model output aligned to each label position
        for j, (x, y) in enumerate(zip(r["label"], ref_at_label)):
            if y is not None and y != x:                 # label says: reference y was spoken as x
                h = heard[j]
                by_spk[r["speaker"]]["error heard (label right)" if h == x else
                                     "reference heard (TTS likely did not say it)" if h == y else "something else"] += 1
    tot = collections.Counter()
    for spk, c in sorted(by_spk.items()):
        n = sum(c.values()); tot += c
        print(f"   {spk[:18]:18s} n={n:4d}  " + "  ".join(f"{k}: {v / n:.0%}" for k, v in c.most_common()))
    n = sum(tot.values())
    print(f"   {'ALL':18s} n={n:4d}  " + "  ".join(f"{k}: {v / n:.0%}" for k, v in tot.most_common()))

    print("\n3) per-clip disagreement (PER) distribution")
    pers = np.array([r["per"] for r in rows])
    for q in (50, 75, 90, 95, 99):
        print(f"   {q}th percentile: {np.percentile(pers, q):.3f}")
    for thr in (0.2, 0.3, 0.5):
        print(f"   clips with PER > {thr}: {np.mean(pers > thr):.1%}")


if __name__ == "__main__":
    main()
