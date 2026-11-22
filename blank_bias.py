"""Decoding-time precision knob for a CTC model: add a bias to the blank log-probability
before greedy decoding. A higher bias makes the model emit a phoneme only when it is more
confident, which removes spurious (inserted) phonemes -> fewer false alarms, some lost recall.
The bias is chosen on dev, then applied once to test.

  python blank_bias.py --logits results/logits_ctc_step9000 sweep
  python blank_bias.py --logits results/logits_ctc_step9000 test --bias 1.0
"""
import argparse
import os
import pickle

import jiwer
import numpy as np

from mdd_metric import official_mdd
from utils import greedy_ctc

BIASES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
MDD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interspeech_IqraEval", "mdd_eval")


def decode(d, bias):
    blank, phones = d["blank"], d["phones"]
    preds = []
    for r in d["rows"]:
        lp = r["logp"].astype(np.float32)
        lp[:, blank] += bias
        preds.append([phones[i] for i in greedy_ctc(lp.argmax(-1), len(lp), blank)])
    return preds


def score(rows, preds, workdir):
    res = {"per": jiwer.wer([" ".join(r["label"]) for r in rows], [" ".join(p) or "<empty>" for p in preds])}
    if any(r["ref"] != r["label"] for r in rows):
        res.update(official_mdd([r["id"] for r in rows], [r["ref"] for r in rows], [r["label"] for r in rows],
                                preds, MDD, workdir=workdir))
    return res


def fmt(r):
    s = f"PER {r['per']:.4f}"
    if "f1" in r:
        s += f"  F1 {r['f1']:.4f}  P {r['precision']:.3f}  R {r['recall']:.3f}  FR {r['FR']:.3f}  FA {r['FA']:.3f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("sweep")
    t = sp.add_parser("test"); t.add_argument("--bias", type=float, required=True)
    a = ap.parse_args()
    tmp = os.path.join(a.logits, "tmp")
    if a.cmd == "sweep":
        d = pickle.load(open(os.path.join(a.logits, "dev.pkl"), "rb"))
        best = None
        for b in BIASES:
            preds = decode(d, b)
            r = score(d["rows"], preds, tmp)
            nat = [i for i, x in enumerate(d["rows"]) if x["id"].startswith("nat")]
            r_nat = jiwer.wer([" ".join(d["rows"][i]["label"]) for i in nat], [" ".join(preds[i]) or "<empty>" for i in nat])
            print(f"dev bias {b:3.1f}: {fmt(r)} | native PER {r_nat:.4f}", flush=True)
            if best is None or r["f1"] > best[1]:
                best = (b, r["f1"])
        print(f"best blank bias on dev: {best[0]}")
    else:
        d = pickle.load(open(os.path.join(a.logits, "test.pkl"), "rb"))
        for b in (0.0, a.bias):
            tag = "no bias (plain greedy)" if b == 0 else "bias chosen on dev"
            print(f"TEST bias {b:3.1f}: {fmt(score(d['rows'], decode(d, b), tmp))}  [{tag}]", flush=True)


if __name__ == "__main__":
    main()
