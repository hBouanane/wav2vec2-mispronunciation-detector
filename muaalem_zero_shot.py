"""Zero-shot evaluation of the Quran-pretrained muaalem model on IqraEval (no training).

muaalem's Quran Phonetic Script output (+ its tafkheem head for emphatic vowels) is converted
to IqraEval phonemes with muaalem.to_iqraeval. The emphasis variant is chosen on dev data
(held-out Iqra_TTS row groups — Quranic verses — and Iqra_train dev), then applied once to test.

  python muaalem_zero_shot.py dump --set dev
  python muaalem_zero_shot.py dump --set test
  python muaalem_zero_shot.py score          # dev: all variants; picks the best
  python muaalem_zero_shot.py test --emphasis head_no_raa
"""
import argparse
import os
import pickle

import jiwer
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from data import decode_audio, is_tts_dev
from mdd_metric import official_mdd
from muaalem import Muaalem, to_iqraeval

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "muaalem_zero_shot")
MDD = os.path.join(HERE, "..", "interspeech_IqraEval", "mdd_eval")
VARIANTS = ["none", "head", "head_no_raa"]
clean = lambda s: s.replace("<sil>", " ").split()


def dev_rows(n_native=1000):
    """Held-out TTS row groups (same hash rule as training, locally cached shards 0 and 12)
    + Iqra_train dev. Nothing here is used for training."""
    rows = []
    for shard in (0, 12):
        name = f"train-{shard:05d}-of-00013.parquet"
        pf = pq.ParquetFile(hf_hub_download("IqraEval/Iqra_TTS", f"data/{name}", repo_type="dataset"))
        for rg in range(pf.metadata.num_row_groups):
            if is_tts_dev(name, rg, 20):
                for k, r in enumerate(pf.read_row_group(rg, columns=["audio", "phoneme_ref", "phoneme_mis"]).to_pylist()):
                    rows.append((f"tts{shard:02d}_{rg:02d}_{k:03d}", r["audio"], clean(r["phoneme_ref"]), clean(r["phoneme_mis"])))
    nat = pq.read_table(hf_hub_download("IqraEval/Iqra_train", "data/dev-00000-of-00001.parquet", repo_type="dataset"),
                        columns=["audio", "phoneme_ref"]).to_pylist()[:n_native]
    rows += [(f"nat{k:05d}", r["audio"], clean(r["phoneme_ref"]), clean(r["phoneme_ref"])) for k, r in enumerate(nat)]
    return rows


def test_rows():
    from datasets import load_dataset
    gt = pd.read_csv(hf_hub_download("IqraEval/IqraEval_Test_GT", "labels_test.csv", repo_type="dataset")).set_index("ID")
    ds = load_dataset("IqraEval/QuranMB.v2", split="test")
    ds = ds.cast_column("audio", __import__("datasets").Audio(sampling_rate=16000))
    return [(ex["ID"], ex["audio"]["array"], gt.loc[ex["ID"], "Reference_phn"].split(),
             gt.loc[ex["ID"], "Annotation_phn"].split()) for ex in ds]


def dump(args):
    model = Muaalem().eval().cuda()
    rows = dev_rows() if args.set == "dev" else test_rows()
    out = []
    for i, (uid, audio, ref, ann) in enumerate(rows):
        wave = decode_audio(audio) if isinstance(audio, dict) else audio
        r = model.transcribe(wave, "cuda")
        out.append({"id": uid, "ref": ref, "ann": ann, "qps": "".join(t for t, _ in r["phonemes"]),
                    "tafkheem": [t for t, _ in r["tafkheem_or_taqeeq"]]})
        if i % 250 == 0:
            print(f"{args.set}: {i}/{len(rows)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(out, open(os.path.join(OUT, f"{args.set}.pkl"), "wb"))
    print(f"saved {len(out)}")


def evaluate(rows, variant, workdir):
    preds = [to_iqraeval(r["qps"], r["tafkheem"], variant) or ["<unk>"] for r in rows]
    res = {"per": jiwer.wer([" ".join(r["ann"]) for r in rows], [" ".join(p) for p in preds])}
    if any(r["ref"] != r["ann"] for r in rows):
        res.update(official_mdd([r["id"] for r in rows], [r["ref"] for r in rows], [r["ann"] for r in rows],
                                preds, MDD, workdir=workdir))
    return res, preds


def fmt(r):
    s = f"PER {r['per']:.4f}"
    if "f1" in r:
        s += f"  F1 {r['f1']:.4f}  P {r['precision']:.3f}  R {r['recall']:.3f}  FA {r['FA']:.3f}  FR {r['FR']:.3f}"
    return s


def score(args):
    rows = pickle.load(open(os.path.join(OUT, "dev.pkl"), "rb"))
    subsets = {"tts (Quran verses, TTS voices)": [r for r in rows if r["id"].startswith("tts")],
               "native (MSA, no errors)": [r for r in rows if r["id"].startswith("nat")]}
    best = None
    for v in VARIANTS:
        for name, sub in subsets.items():
            res, _ = evaluate(sub, v, os.path.join(OUT, "tmp"))
            print(f"dev {name:32s} emphasis={v:12s} {fmt(res)}", flush=True)
            if name.startswith("tts") and (best is None or res["f1"] > best[1]):
                best = (v, res["f1"])
    print(f"best emphasis variant on dev (TTS F1): {best[0]}")


def test(args):
    rows = pickle.load(open(os.path.join(OUT, "test.pkl"), "rb"))
    res, preds = evaluate(rows, args.emphasis, os.path.join(OUT, "tmp"))
    print(f"TEST emphasis={args.emphasis}: {fmt(res)}")
    pd.DataFrame({"ID": [r["id"] for r in rows], "Labels": [" ".join(p) for p in preds]}).to_csv(
        os.path.join(OUT, f"submission_{args.emphasis}.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump"); d.add_argument("--set", choices=["dev", "test"], required=True)
    sp.add_parser("score")
    t = sp.add_parser("test"); t.add_argument("--emphasis", choices=VARIANTS, required=True)
    a = ap.parse_args()
    {"dump": dump, "score": score, "test": test}[a.cmd](a)
