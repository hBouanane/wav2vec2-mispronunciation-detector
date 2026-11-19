"""Run the detector on the Iqra'Eval test set (ArabicNLP 2025; reused by IqraEval.2 2026).

Audio:      IqraEval/QuranMB.v2  (split "test", 1642 utterances; formerly IqraEval/open_testset)
References: IqraEval/test_references (canonical phonemes the model conditions on)

Writes two CSVs with the same predictions:
  <out>/predictions.csv  -> ID,Prediction  (input to mdd_eval/align_data.py)
  <out>/submission.csv   -> ID,Labels      (challenge submission format)

Example:
  python predict_testset.py --checkpoint hf_model/best_model.pt --out results/best_model
"""
import argparse
import os

import pandas as pd
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from config import Config
from model import Wav2Vec2MispronunciationDetector
from utils import create_processor, create_vocabulary


def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    # strip prefixes left by accelerate / DDP / torch.compile
    return {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in ckpt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N utterances")
    ap.add_argument("--fp16", action="store_true", help="autocast to fp16 (low-VRAM GPUs)")
    args = ap.parse_args()

    config = Config()
    create_vocabulary(config.vocab_file)  # writes vocab.json used by the processor
    processor = create_processor(config.sampling_rate)
    tok = processor.tokenizer

    model = Wav2Vec2MispronunciationDetector(
        vocab_size=len(tok),
        pad_token_id=tok.pad_token_id,
        blank_token_id=tok.pad_token_id,  # training used [PAD] as the CTC blank
        config=config,
    )
    missing, unexpected = model.load_state_dict(load_state_dict(args.checkpoint), strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch\n missing: {missing}\n unexpected: {unexpected}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    audio = load_dataset("IqraEval/QuranMB.v2", split="test")
    audio = audio.cast_column("audio", Audio(sampling_rate=config.sampling_rate))
    refs = load_dataset("IqraEval/test_references", split="test").to_pandas()
    ref_by_id = dict(zip(refs["ID"], refs["Reference_phn"]))

    n = len(audio) if args.limit is None else min(args.limit, len(audio))
    rows = []
    for i in tqdm(range(n), desc="Predicting"):
        ex = audio[i]
        uid = ex["ID"]
        ref = ref_by_id[uid].replace("<sil>", "").strip()

        inputs = processor(ex["audio"]["array"], sampling_rate=config.sampling_rate,
                           return_tensors="pt").input_values.to(device)
        with processor.as_target_processor():
            ref_ids = torch.tensor([processor(ref).input_ids], device=device)

        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=args.fp16):
            logits = model(inputs, ref_ids)["logits"]
        pred_ids = logits.argmax(-1).cpu().numpy()
        pred = tok.batch_decode(pred_ids, group_tokens=True)[0]
        pred = " ".join(p for p in pred.split() if p not in ("[BLANK]", "[PAD]", "[UNK]"))
        rows.append({"ID": uid, "Prediction": pred})

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "predictions.csv"), index=False)
    df.rename(columns={"Prediction": "Labels"}).to_csv(os.path.join(args.out, "submission.csv"), index=False)
    print(f"wrote {len(df)} predictions to {args.out}")


if __name__ == "__main__":
    main()
