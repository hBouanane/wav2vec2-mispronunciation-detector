"""Probe whether the model actually uses the reference phonemes.

For N test utterances, decode with:
  correct   - the true reference
  shuffled  - another utterance's reference
  blank     - the reference replaced by [UNK] tokens of the same length
and report how often/much the prediction changes, plus PER vs the human annotation.
"""
import argparse
import random

import jiwer
import pandas as pd
import torch
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download

from config import Config
from model import Wav2Vec2MispronunciationDetector
from predict_testset import load_state_dict
from utils import create_processor, create_vocabulary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="hf_model/best_mode_full_data.pt")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    random.seed(0)

    cfg = Config()
    create_vocabulary(cfg.vocab_file)
    proc = create_processor(cfg.sampling_rate)
    tok = proc.tokenizer
    model = Wav2Vec2MispronunciationDetector(len(tok), tok.pad_token_id, tok.pad_token_id, cfg)
    model.load_state_dict(load_state_dict(args.checkpoint))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    audio = load_dataset("IqraEval/QuranMB.v2", split="test").cast_column("audio", Audio(sampling_rate=16000))
    gt = pd.read_csv(hf_hub_download("IqraEval/IqraEval_Test_GT", "labels_test.csv", repo_type="dataset"))
    gt = gt.set_index("ID")
    idx = random.sample(range(len(audio)), args.n)
    all_refs = gt["Reference_phn"].str.strip().tolist()

    def decode(wave, ref):
        x = proc(wave, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        r = torch.tensor([tok.convert_tokens_to_ids(ref.split())], device=dev)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16):
            out = model(x, r)
        logits = out["logits"].float()
        pred = tok.batch_decode(logits.argmax(-1).cpu().numpy(), group_tokens=True)[0]
        return " ".join(p for p in pred.split() if p not in ("[BLANK]", "[PAD]", "[UNK]")), out["attention_weights"][0].float()  # [T_audio, S_ref], head-averaged

    rows = []
    for i in idx:
        ex = audio[i]
        ref = gt.loc[ex["ID"], "Reference_phn"].strip()
        wrong = random.choice([r for r in all_refs if r != ref])
        blank = " ".join(["[UNK]"] * len(ref.split()))
        p_ok, att = decode(ex["audio"]["array"], ref)
        p_wrong, _ = decode(ex["audio"]["array"], wrong)
        p_blank, _ = decode(ex["audio"]["array"], blank)
        # attention entropy relative to uniform (1.0 = uniform = not attending to anything specific)
        ent = -(att.clamp_min(1e-9) * att.clamp_min(1e-9).log()).sum(-1).mean().item()
        ent_rel = ent / torch.log(torch.tensor(float(att.shape[-1]))).item()
        rows.append(dict(ann=gt.loc[ex["ID"], "Annotation_phn"].strip(), ref=ref,
                         ok=p_ok, wrong=p_wrong, blank=p_blank, wrongref=wrong, ent_rel=ent_rel))

    df = pd.DataFrame(rows)
    per = lambda col: jiwer.wer(df.ann.tolist(), df[col].tolist())
    print(f"utterances: {len(df)}")
    for c in ("ok", "wrong", "blank"):
        print(f"PER vs annotation  [{c:6s} ref]: {per(c):.4f}")
    print(f"identical output, wrong vs correct ref: {(df.ok == df.wrong).mean():.1%}")
    print(f"identical output, blank vs correct ref: {(df.ok == df.blank).mean():.1%}")
    print(f"edit distance correct-ref output -> wrong-ref output (PER): {jiwer.wer(df.ok.tolist(), df.wrong.tolist()):.4f}")
    print(f"PER of wrong-ref output vs the WRONG ref (does it copy?): {jiwer.wer(df.wrongref.tolist(), df.wrong.tolist()):.4f}")
    print(f"PER of copying the reference (floor):  {jiwer.wer(df.ann.tolist(), df.ref.tolist()):.4f}")
    print(f"cross-attention entropy / uniform: {df.ent_rel.mean():.3f}")


if __name__ == "__main__":
    main()
