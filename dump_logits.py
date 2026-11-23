"""Run a trained CTC model once and save its frame log-probabilities, for analyses that
need many decodings (decoding bias sweeps, training-data noise checks).

  python dump_logits.py --run runs/myrun --set dev        # held-out TTS + Iqra_train dev
  python dump_logits.py --run runs/myrun --set test       # QuranMB.v2 + released labels
  python dump_logits.py --run runs/myrun --set tts_train  # sample of TTS TRAINING rows

Output: results/logits_<run>/<set>.pkl  (list of dicts: id, ref, label, logp[T,V] fp16, ...)
Uses locally cached TTS shards 0 (augmented) and 12 (original); held-out row groups follow
the same hash rule as training (data.is_tts_dev).
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from config import Config
from data import Collator, decode_audio, is_tts_dev
from model import load_trained, to_device
from utils import Vocab

TTS_SHARDS = (0, 12)


def tts_rows(held_out, n_per_shard, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for shard in TTS_SHARDS:
        name = f"train-{shard:05d}-of-00013.parquet"
        pf = pq.ParquetFile(hf_hub_download("IqraEval/Iqra_TTS", f"data/{name}", repo_type="dataset"))
        rows = []
        for rg in range(pf.metadata.num_row_groups):
            if is_tts_dev(name, rg, 20) == held_out:
                for k, r in enumerate(pf.read_row_group(rg, columns=["audio", "phoneme_ref", "phoneme_mis",
                                                                     "speaker", "label"]).to_pylist()):
                    rows.append((f"tts{shard:02d}_{rg:02d}_{k:03d}", r))
        for i in rng.permutation(len(rows))[:n_per_shard]:
            uid, r = rows[i]
            out.append({"id": uid, "audio": r["audio"], "ref": r["phoneme_ref"], "label": r["phoneme_mis"],
                        "speaker": r["speaker"], "tts_label": r["label"]})
    return out


def native_dev(n=1000):
    t = pq.read_table(hf_hub_download("IqraEval/Iqra_train", "data/dev-00000-of-00001.parquet", repo_type="dataset"),
                      columns=["audio", "phoneme_ref"]).to_pylist()[:n]
    return [{"id": f"nat{k:05d}", "audio": r["audio"], "ref": r["phoneme_ref"], "label": r["phoneme_ref"]}
            for k, r in enumerate(t)]


def test_rows():
    from datasets import Audio, load_dataset
    gt = pd.read_csv(hf_hub_download("IqraEval/IqraEval_Test_GT", "labels_test.csv", repo_type="dataset")).set_index("ID")
    ds = load_dataset("IqraEval/QuranMB.v2", split="test").cast_column("audio", Audio(sampling_rate=16000))
    return [{"id": ex["ID"], "wave": ex["audio"]["array"], "ref": gt.loc[ex["ID"], "Reference_phn"],
             "label": gt.loc[ex["ID"], "Annotation_phn"]} for ex in ds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", choices=["dev", "test", "tts_train"], required=True)
    ap.add_argument("--n-per-shard", type=int, default=1500)
    a = ap.parse_args()
    best = json.load(open(os.path.join(a.run, "best.json")))
    cfg = Config(**{k: v for k, v in best["config"].items() if k in Config.__dataclass_fields__})
    cfg.grad_checkpointing = False
    vocab = Vocab(cfg.vocab_file)
    dev = torch.device("cuda")
    model, fx = load_trained(cfg, vocab, os.path.join(a.run, "best_model.pt"), dev)
    collate = Collator(fx)
    rows = {"dev": lambda: tts_rows(True, 10_000) + native_dev(),
            "test": test_rows,
            "tts_train": lambda: tts_rows(False, a.n_per_shard)}[a.set]()
    out = []
    for i, r in enumerate(rows):
        wave = r.pop("wave", None)
        wave = decode_audio(r.pop("audio")) if wave is None else np.asarray(wave, dtype=np.float32)
        b = to_device(collate([{"wave": wave, "label": [0], "ref": []}]), dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(b["inputs"], attention_mask=b["attention_mask"]).logits[0].float()
        T = int(model._get_feat_extract_output_lengths(b["attention_mask"].sum(-1))[0])
        r["ref"] = r["ref"].replace("<sil>", " ").split()
        r["label"] = r["label"].replace("<sil>", " ").split()
        r["logp"] = F.log_softmax(logits[:T], -1).cpu().numpy().astype(np.float16)
        out.append(r)
        if i % 250 == 0:
            print(f"{a.set}: {i}/{len(rows)}", flush=True)
    d = os.path.join("results", "logits_" + os.path.basename(a.run.rstrip("/\\")))
    os.makedirs(d, exist_ok=True)
    pickle.dump({"blank": vocab.blank, "phones": vocab.phones, "rows": out}, open(os.path.join(d, f"{a.set}.pkl"), "wb"))
    print(f"saved {len(out)} to {d}/{a.set}.pkl")


if __name__ == "__main__":
    main()
