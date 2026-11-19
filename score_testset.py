"""Score test-set predictions with the official IqraEval MDD scripts.

Needs a checkout of https://github.com/Iqra-Eval/interspeech_IqraEval (for mdd_eval/)
and access to the gated IqraEval/IqraEval_Test_GT dataset (`huggingface-cli login`).

Example:
  python score_testset.py --pred results/best_model/predictions.csv \
      --mdd-eval ../interspeech_IqraEval/mdd_eval
"""
import argparse
import os
import subprocess
import sys

import jiwer
import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV with ID,Prediction")
    ap.add_argument("--mdd-eval", required=True, help="path to interspeech_IqraEval/mdd_eval")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.pred))
    truth = hf_hub_download("IqraEval/IqraEval_Test_GT", "labels_test.csv", repo_type="dataset")
    aligned = os.path.join(out_dir, "aligned") + os.sep  # official script concatenates folder + filename
    metrics_file = os.path.join(out_dir, "metrics.txt")
    if os.path.exists(metrics_file):
        os.remove(metrics_file)

    mdd = os.path.abspath(args.mdd_eval)
    subprocess.run([sys.executable, "align_data.py", "--truth-file", truth, "--pred-file",
                    os.path.abspath(args.pred), "--output-dir", aligned], cwd=mdd, check=True)
    subprocess.run([sys.executable, "ins_del_cor_sub_analysis.py", "--align-folder", aligned,
                    "--out-file", metrics_file], cwd=mdd, check=True)

    # PER of the prediction against the human annotation (leaderboard "PER" column)
    gt = pd.read_csv(truth)
    pred = pd.read_csv(args.pred).fillna("")
    df = gt.merge(pred, on="ID")
    per = jiwer.wer(df["Annotation_phn"].str.strip().tolist(), df["Prediction"].str.strip().tolist())
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(f"PER (vs. human annotation): {per:.4f}\nUtterances scored: {len(df)}/{len(gt)}\n")
    print(open(metrics_file, encoding="utf-8").read())


if __name__ == "__main__":
    main()
