"""Transcribe the Iqra'Eval test set (QuranMB.v2, 1642 utterances) with a trained model.

The model only hears the audio; the reference is not used here. score_testset.py then
compares prediction, reference and human annotation with the official scripts.

  python predict_testset.py --run runs/myrun --out results/myrun            # local checkpoint
  python predict_testset.py --hub-repo USER/REPO --run runs/hub --out results/hub
  python score_testset.py --pred results/myrun/predictions.csv --mdd-eval ../interspeech_IqraEval/mdd_eval

Writes predictions.csv (ID,Prediction) and submission.csv (ID,Labels).
The released v1 model (reference-conditioned) is available with `git checkout v1`.
"""
import argparse
import json
import os

import pandas as pd
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from config import Config
from data import Collator
from model import frame_lengths, greedy_decode, load_trained, to_device
from utils import Vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with best_model.pt and best.json")
    ap.add_argument("--hub-repo", default=None, help="first download best_model.pt/best.json from this repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if a.hub_repo:
        from huggingface_hub import snapshot_download
        snapshot_download(a.hub_repo, local_dir=a.run, allow_patterns=["best_model.pt", "best.json"])
    best = json.load(open(os.path.join(a.run, "best.json")))
    cfg = Config(**{k: v for k, v in best["config"].items() if k in Config.__dataclass_fields__})
    cfg.grad_checkpointing = False
    vocab = Vocab(cfg.vocab_file)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, extractor = load_trained(cfg, vocab, os.path.join(a.run, "best_model.pt"), dev)
    collate = Collator(extractor)
    print(f"model from step {best['step']}, dev: {best['dev'].get('all')}")

    audio = load_dataset("IqraEval/QuranMB.v2", split="test").cast_column("audio", Audio(sampling_rate=16000))
    n = len(audio) if a.limit is None else min(a.limit, len(audio))
    rows = []
    for i in tqdm(range(n), desc="Transcribing"):
        ex = audio[i]
        b = to_device(collate([{"wave": ex["audio"]["array"], "label": [0], "ref": []}]), dev)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(b["inputs"], attention_mask=b["attention_mask"]).logits
        rows.append({"ID": ex["ID"], "Prediction": " ".join(greedy_decode(logits, frame_lengths(model, b["attention_mask"]), vocab, model)[0])})

    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    df.rename(columns={"Prediction": "Labels"}).to_csv(os.path.join(a.out, "submission.csv"), index=False)
    print(f"wrote {len(df)} predictions to {a.out}")


if __name__ == "__main__":
    main()
