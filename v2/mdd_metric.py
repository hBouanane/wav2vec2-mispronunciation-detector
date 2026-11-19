"""Run the official IqraEval MDD scoring (mdd_eval/ from Iqra-Eval/interspeech_IqraEval)
on in-memory predictions and return the metrics as a dict."""
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd

KEYS = {"Recall": "recall", "Precision": "precision", "f1-score": "f1", "True Acception": "TA",
        "False Rejection": "FR", "False Acceptance": "FA", "Correct Diagnosis": "CD",
        "Correct Rate": "correct_rate", "Accuracy": "accuracy"}


def official_mdd(ids, refs, annotations, predictions, mdd_dir, workdir=None):
    """refs/annotations/predictions: lists of phone lists (or space-joined strings)."""
    j = lambda x: x if isinstance(x, str) else " ".join(x)
    workdir = os.path.abspath(workdir or tempfile.mkdtemp(prefix="mdd_"))
    os.makedirs(workdir, exist_ok=True)
    truth = os.path.join(workdir, "truth.csv")
    pred = os.path.join(workdir, "pred.csv")
    pd.DataFrame({"ID": ids, "Reference_phn": [j(r) for r in refs],
                  "Annotation_phn": [j(a) for a in annotations]}).to_csv(truth, index=False)
    # the official reader turns "" into NaN and crashes; an empty prediction becomes one unknown token
    preds = [j(p).strip() or "<unk>" for p in predictions]
    pd.DataFrame({"ID": ids, "Prediction": preds}).to_csv(pred, index=False)
    aligned = os.path.join(workdir, "aligned") + os.sep
    out_file = os.path.join(workdir, "metrics.txt")
    if os.path.exists(out_file):
        os.remove(out_file)
    mdd_dir = os.path.abspath(mdd_dir)
    r1 = subprocess.run([sys.executable, "align_data.py", "--truth-file", truth, "--pred-file", pred,
                         "--output-dir", aligned], cwd=mdd_dir, capture_output=True, text=True, check=True)
    subprocess.run([sys.executable, "ins_del_cor_sub_analysis.py", "--align-folder", aligned,
                    "--out-file", out_file], cwd=mdd_dir, capture_output=True, text=True, check=True)
    text = r1.stdout + open(out_file, encoding="utf-8").read()
    res = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z0-9\- ]+?):\s*([0-9.]+)", line)
        if m and m.group(1) in KEYS:
            res[KEYS[m.group(1)]] = float(m.group(2))
    return res


if __name__ == "__main__":
    # self-check: re-score saved predictions against the released test ground truth
    import argparse
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mdd-eval", required=True)
    a = ap.parse_args()
    gt = pd.read_csv(hf_hub_download("IqraEval/IqraEval_Test_GT", "labels_test.csv", repo_type="dataset"))
    df = gt.merge(pd.read_csv(a.pred).fillna(""), on="ID")
    print(official_mdd(df.ID.tolist(), df.Reference_phn.str.split().tolist(),
                       df.Annotation_phn.str.split().tolist(), df.Prediction.str.split().tolist(), a.mdd_eval))
