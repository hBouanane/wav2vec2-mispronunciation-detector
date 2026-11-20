"""Does the v2 model listen, or does it detect corrupted references from the text alone?

Scores the same validation utterances (corrupted references) three ways:
  real   - the correct audio
  swap   - another utterance's audio (the reference no longer matches what is heard at all)
  silent - near-silent audio
If F1/recall stay high with swapped or silent audio, the model is reading the reference text,
not the speech.

  python v2/probe_shortcut.py --run runs/hub_step4000 --n 300
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_v2 import Collator, Featurizer, PhoneVocab, RefCorrupter, build_val_sets  # noqa: E402
from mdd_metric import official_mdd  # noqa: E402
from model_v2 import RefAnchoredMDD, decode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=300, help="utterances per subset")
    ap.add_argument("--mdd-eval", default=os.path.join(HERE, "..", "..", "interspeech_IqraEval", "mdd_eval"))
    a = ap.parse_args()

    best = json.load(open(os.path.join(a.run, "best.json")))
    vocab = PhoneVocab(os.path.join(HERE, "..", "sws_arabic.txt"))
    model = RefAnchoredMDD(vocab, encoder_name=best["args"]["encoder"], grad_checkpointing=False,
                           encoder_config_only=True)
    model.load_state_dict(torch.load(os.path.join(a.run, "best_model.pt"), map_location="cpu"))
    dev = torch.device("cuda")
    model.to(dev).eval()
    tau = best["best_tau"]

    val = build_val_sets(RefCorrupter(os.path.join(HERE, "confusions.json")), n_native=a.n, n_tts=a.n)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(val))
    feat, collate = Featurizer(vocab, None, max_seconds=60), Collator(vocab)

    for mode in ("real", "swap", "silent"):
        ids, refs, anns, preds = [], [], [], []
        for k, x in enumerate(val):
            if mode == "real":
                wave = x["audio"]
            elif mode == "swap":
                wave = val[perm[k] if perm[k] != k else (k + 1) % len(val)]["audio"]
            else:
                wave = (rng.standard_normal(len(x["audio"])) * 1e-4).astype(np.float32)
            f = feat.from_lists(wave, x["ref"], x["spoken"])
            b = {kk: vv.to(dev) for kk, vv in collate([f]).items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                out = model(b["wave"], b["wave_mask"], b["ref"])
            ids.append(x["id"]); refs.append(x["ref"]); anns.append(x["spoken"])
            preds.append(decode(out, b["ref"], vocab, tau=tau)[0])
        r = official_mdd(ids, refs, anns, preds, a.mdd_eval)
        print(f"{mode:6s} audio: F1 {r['f1']:.4f}  P {r['precision']:.3f}  R {r['recall']:.3f}", flush=True)


if __name__ == "__main__":
    main()
