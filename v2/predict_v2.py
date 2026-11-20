"""Predict the IqraEval test set (QuranMB.v2) with a trained v2 model.

The decision threshold tau is read from <run>/best.json, i.e. chosen on the 2025
validation data, never on the test set.

  python v2/predict_v2.py --run runs/run1 --out results/v2_run1 --fp16
  python v2/predict_v2.py --hub-repo Haitam03/iqraeval-mdd-v2 --run runs/from_hub --out results/v2 --fp16
  python score_testset.py --pred results/v2_run1/predictions.csv --mdd-eval ../interspeech_IqraEval/mdd_eval
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_v2 import Collator, Featurizer, PhoneVocab, clean  # noqa: E402
from model_v2 import RefAnchoredMDD, decode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training output dir with best_model.pt and best.json")
    ap.add_argument("--hub-repo", default=None, help="download best_model.pt/best.json from this HF repo into --run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=None, help="override tau from best.json")
    ap.add_argument("--tau-ins", type=float, default=0.5)
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()

    if a.hub_repo:
        from huggingface_hub import snapshot_download
        snapshot_download(a.hub_repo, local_dir=a.run, allow_patterns=["best_model.pt", "best.json"])
    best = json.load(open(os.path.join(a.run, "best.json")))
    targs = best["args"]
    tau = a.tau if a.tau is not None else best["best_tau"]
    vocab = PhoneVocab(os.path.join(HERE, "..", "sws_arabic.txt"))
    smoke = targs.get("smoke", False)
    model = RefAnchoredMDD(vocab, encoder_name=targs["encoder"], d_model=64 if smoke else 512,
                           n_layers=1 if smoke else 4, n_heads=2 if smoke else 8,
                           grad_checkpointing=False, encoder_config_only=True)
    model.load_state_dict(torch.load(os.path.join(a.run, "best_model.pt"), map_location="cpu"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    audio = load_dataset("IqraEval/QuranMB.v2", split="test").cast_column("audio", Audio(sampling_rate=16000))
    refs = load_dataset("IqraEval/test_references", split="test").to_pandas()
    ref_by_id = dict(zip(refs["ID"], refs["Reference_phn"]))
    feat, collate = Featurizer(vocab, None, max_seconds=600), Collator(vocab)

    rows = []
    for ex in tqdm(audio, desc=f"Predicting (tau={tau})"):
        ref = clean(ref_by_id[ex["ID"]])
        f = feat.from_lists(ex["audio"]["array"], ref, ref)  # spoken unknown: labels unused
        b = {k: v.to(dev) for k, v in collate([f]).items()}
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=a.fp16):
            out = model(b["wave"], b["wave_mask"], b["ref"])
        pred = decode(out, b["ref"], vocab, tau=tau, tau_ins=a.tau_ins)[0]
        rows.append({"ID": ex["ID"], "Prediction": " ".join(pred)})

    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    df.rename(columns={"Prediction": "Labels"}).to_csv(os.path.join(a.out, "submission.csv"), index=False)
    json.dump({"tau": tau, "tau_ins": a.tau_ins, "run": a.run}, open(os.path.join(a.out, "decode.json"), "w"))
    print(f"wrote {len(df)} predictions to {a.out} (tau={tau})")


if __name__ == "__main__":
    main()
