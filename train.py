"""Train the CTC phoneme recogniser on the released 2025 IqraEval data.

Kaggle (2 GPUs):  accelerate launch --multi_gpu --num_processes 2 train.py --out /kaggle/working/run \
                      --mdd-eval interspeech_IqraEval/mdd_eval --hub-repo USER/REPO --wandb-project P
Local smoke test: python train.py --smoke --out runs/smoke --mdd-eval ../interspeech_IqraEval/mdd_eval

Every field of config.Config is a flag (--encoder, --max-steps, --ref-penalty, ...).
The best checkpoint is chosen by dev PER (released data, unmodified); the test set is not used.
"""
import argparse
import dataclasses
import json
import math
import os
import time
from datetime import timedelta

import jiwer
import numpy as np
import psutil
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from huggingface_hub import HfApi, snapshot_download
from torch.utils.data import DataLoader

from config import Config
from data import Collator, RowGroupShuffle, build, load_dev
from mdd_metric import official_mdd
from model import build_model, frame_lengths, greedy_decode, reference_penalty, to_device
from utils import Vocab

HUB_FILES = ["last/*", "best_model.pt", "best.json", "eval_log.jsonl"]


def get_args():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type in (bool, "bool"):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        elif f.type in (tuple, "tuple"):
            ap.add_argument(flag, type=float, nargs=2, default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mdd-eval", required=True, help="path to interspeech_IqraEval/mdd_eval")
    ap.add_argument("--hub-repo", default=None, help="private HF repo for checkpoints; resumes from it")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny random encoder + a few local files")
    a = ap.parse_args()
    cfg = Config(**{f.name: getattr(a, f.name) for f in dataclasses.fields(Config)})
    return a, cfg


class HubSync:
    """Checkpoints in a private HF model repo; uploads in the background, history squashed."""

    def __init__(self, repo, out):
        self.api, self.repo, self.out, self.future = HfApi(), repo, out, None
        self.api.create_repo(repo, private=True, exist_ok=True, repo_type="model")

    def pull(self):
        if "last/train_state.json" not in self.api.list_repo_files(self.repo):
            return False
        snapshot_download(self.repo, local_dir=self.out, allow_patterns=HUB_FILES)
        return True

    def wait(self):
        if self.future is not None:
            try:
                self.future.result()
            except Exception as e:
                print(f"[hub] upload failed: {e}", flush=True)
            self.future = None

    def push(self, msg, block=False):
        self.wait()
        try:
            self.api.super_squash_history(repo_id=self.repo)
        except Exception as e:
            print(f"[hub] squash skipped: {e}", flush=True)
        self.future = self.api.upload_folder(repo_id=self.repo, folder_path=self.out, allow_patterns=HUB_FILES,
                                             commit_message=msg, run_as_future=True)
        if block:
            self.wait()


def smoke_setup(a, cfg):
    """Point the data globs at a few locally cached parquet files and shrink everything."""
    from huggingface_hub import hf_hub_download
    get = lambda repo, f: hf_hub_download(repo, f"data/{f}", repo_type="dataset")
    cfg.native_glob = get("IqraEval/Iqra_train", "train-00000-of-00029.parquet")
    cfg.native_dev = get("IqraEval/Iqra_train", "dev-00000-of-00001.parquet")
    t0 = get("IqraEval/Iqra_TTS", "train-00000-of-00013.parquet")
    get("IqraEval/Iqra_TTS", "train-00012-of-00013.parquet")
    cfg.tts_glob = os.path.join(os.path.dirname(t0), "train-000*-of-00013.parquet")
    if cfg.target == "qps":   # real muaalem (no tiny version exists); freeze most of it to fit small GPUs
        cfg.encoder, cfg.freeze_layers, cfg.grad_checkpointing = "obadx/muaalem-model-v3_2", 22, True
    else:
        cfg.encoder = "hf-internal-testing/tiny-random-wav2vec2"
        cfg.grad_checkpointing = False
    cfg.max_steps, cfg.eval_every, cfg.save_every, cfg.log_every = 6, 6, 3, 2
    cfg.batch_size, cfg.grad_accum, cfg.warmup, cfg.num_workers = 2, 1, 2, 0
    cfg.n_dev_native, cfg.n_dev_tts, cfg.shuffle_pool = 6, 6, 2
    cfg.tts_dev_every = 5


@torch.no_grad()
def evaluate(model, dev, collate, vocab, acc, a):
    model.eval()
    order = sorted(range(len(dev)), key=lambda i: len(dev[i]["wave"]))
    preds = [None] * len(dev)
    for s in range(0, len(order), 8):
        idx = order[s:s + 8]
        b = to_device(collate([dev[i] for i in idx]), acc.device)
        with acc.autocast():
            logits = model(b["inputs"], attention_mask=b["attention_mask"]).logits
        for i, p in zip(idx, greedy_decode(logits, frame_lengths(model, b["attention_mask"]), vocab, model)):
            preds[i] = p
    model.train()
    res = {}
    for name in ("nat", "tts", "all"):
        sel = [i for i, x in enumerate(dev) if name == "all" or x["id"].startswith(name)]
        if not sel:
            continue
        lab = [" ".join(vocab.decode(dev[i]["label"])) for i in sel]
        hyp = [" ".join(preds[i]) or "<empty>" for i in sel]
        r = {"per": jiwer.wer(lab, hyp)}
        if any(dev[i]["ref"] != dev[i]["label"] for i in sel):  # MDD needs at least one real error
            m = official_mdd([dev[i]["id"] for i in sel], [vocab.decode(dev[i]["ref"]) for i in sel],
                             [vocab.decode(dev[i]["label"]) for i in sel], [preds[i] for i in sel], a.mdd_eval,
                             workdir=os.path.join(a.out, "dev_tmp"))
            r.update({k: m[k] for k in ("f1", "precision", "recall", "FA", "FR") if k in m})
        res[name] = r
    return res


def main():
    a, cfg = get_args()
    if a.smoke:
        smoke_setup(a, cfg)
    acc = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no",
                      gradient_accumulation_steps=cfg.grad_accum,
                      kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=1))])
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    vocab = Vocab(cfg.vocab_file)
    if cfg.target == "qps" and cfg.ref_penalty > 0:
        raise ValueError("--ref-penalty is only implemented for the IqraEval CTC head (target iqraeval)")

    model, extractor = build_model(cfg, vocab)
    if cfg.init_from:
        path = cfg.init_from
        if not os.path.exists(path):  # Hub repo id
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(cfg.init_from, "best_model.pt")
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        print(f"initialised weights from {cfg.init_from}")
    is_head = lambda n: n.startswith(("lm_head", "level_to_lm_head"))
    head = [p for n, p in model.named_parameters() if is_head(n) and p.requires_grad]
    body = [p for n, p in model.named_parameters() if not is_head(n) and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": cfg.lr}, {"params": head, "lr": cfg.lr * cfg.head_lr_mult}],
                            weight_decay=cfg.weight_decay)

    def lr_lambda(s):
        if s < cfg.warmup:
            return s / max(1, cfg.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - cfg.warmup) / max(1, cfg.max_steps - cfg.warmup))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)  # stepped manually per optimizer step

    model, opt = acc.prepare(model, opt)
    last_dir = os.path.join(a.out, "last")
    state = {"step": 0, "best_per": 1e9}
    hub = HubSync(a.hub_repo, a.out) if a.hub_repo and acc.is_main_process else None
    if hub is not None and hub.pull():
        acc.print(f"[hub] pulled checkpoint from {a.hub_repo}")
    acc.wait_for_everyone()
    if os.path.exists(os.path.join(last_dir, "train_state.json")):
        acc.load_state(last_dir)
        state = json.load(open(os.path.join(last_dir, "train_state.json")))
        for _ in range(state["step"]):
            sched.step()
        acc.print(f"resumed at step {state['step']}")

    wb = None
    if a.wandb_project and acc.is_main_process:
        import wandb
        wb = wandb.init(project=a.wandb_project, name=a.wandb_name, id=state.get("wandb_id"), resume="allow",
                        config={**cfg.to_dict(), "out": a.out, "hub_repo": a.hub_repo})
        state["wandb_id"] = wb.id
        for k in ("train/*", "dev_nat/*", "dev_tts/*", "dev_all/*", "sys/*"):
            wb.define_metric(k, step_metric="train/step")

    train_units, tts_dev_units = build(cfg, vocab)
    data = RowGroupShuffle(train_units, vocab, cfg, rank=acc.process_index, world=acc.num_processes,
                           seed=cfg.seed + state["step"])
    collate = Collator(extractor)
    loader = DataLoader(data, batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate,
                        persistent_workers=cfg.num_workers > 0)
    dev = load_dev(cfg, vocab, tts_dev_units) if acc.is_main_process else None
    acc.print(f"train row groups: {len(train_units)} | dev utterances: {len(dev) if dev else 0} "
              f"(TTS dev row groups: {len(tts_dev_units)})")

    t0, log = time.time(), []
    model.train()
    unwrapped = acc.unwrap_model(model)
    for batch in loader:
        stop = torch.tensor(float(state["step"] >= cfg.max_steps or (time.time() - t0) / 3600 > cfg.time_limit_h),
                            device=acc.device)
        if acc.reduce(stop, reduction="max").item() > 0:
            break
        batch = to_device(batch, acc.device)
        with acc.accumulate(model):
            labels = ({"phonemes": batch["qps"], "tafkheem_or_taqeeq": batch["tafk"]} if cfg.target == "qps"
                      else batch["labels"])
            out = model(batch["inputs"], attention_mask=batch["attention_mask"], labels=labels)
            loss, pen, n_subs = out.loss, torch.zeros((), device=acc.device), 0
            if cfg.ref_penalty > 0:
                pen, n_subs = reference_penalty(out.logits, frame_lengths(model, batch["attention_mask"]),
                                                batch["labels"], batch["refs"], vocab.blank)
                loss = loss + cfg.ref_penalty * pen
            acc.backward(loss)
            if acc.sync_gradients:
                acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if not acc.sync_gradients:
            continue
        sched.step()
        state["step"] += 1
        s = state["step"]
        log.append([out.loss.item(), float(pen), n_subs])
        if s % cfg.log_every == 0:
            m = np.mean(log, 0); log = []
            ram = psutil.virtual_memory()
            used = (ram.total - ram.available) / 1e9
            acc.print(f"step {s} ctc {m[0]:.4f} ref_pen {m[1]:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / 60:.1f} min | RAM {used:.1f}/{ram.total / 1e9:.0f} GB", flush=True)
            if wb is not None:
                wb.log({"train/step": s, "train/ctc": m[0], "train/ref_penalty": m[1], "train/subs_per_batch": m[2],
                        "train/lr": sched.get_last_lr()[0], "train/hours": (time.time() - t0) / 3600,
                        "sys/ram_used_gb": used})
        if s % cfg.eval_every == 0 and acc.is_main_process:
            res = evaluate(unwrapped, dev, collate, vocab, acc, a)
            acc.print(f"[dev step {s}] " + " | ".join(
                f"{k}: PER {r['per']:.4f}" + (f" F1 {r['f1']:.4f} P {r['precision']:.3f} R {r['recall']:.3f} "
                                              f"FA {r['FA']:.3f}" if "f1" in r else "") for k, r in res.items()), flush=True)
            if wb is not None:
                wb.log({"train/step": s, **{f"dev_{k}/{m}": v for k, r in res.items() for m, v in r.items()}})
            with open(os.path.join(a.out, "eval_log.jsonl"), "a") as f:
                f.write(json.dumps({"step": s, "dev": res}) + "\n")
            if res["all"]["per"] < state["best_per"]:
                state["best_per"] = res["all"]["per"]
                if hub is not None:
                    hub.wait()
                torch.save(unwrapped.state_dict(), os.path.join(a.out, "best_model.pt"))
                json.dump({"step": s, "dev": res, "config": cfg.to_dict()}, open(os.path.join(a.out, "best.json"), "w"),
                          indent=1)
                acc.print(f"  new best dev PER {res['all']['per']:.4f}")
                if wb is not None:
                    wb.summary.update({"best_dev_per": res["all"]["per"], "best_step": s})
        acc.wait_for_everyone()
        if s % cfg.save_every == 0:
            if hub is not None:
                hub.wait()
            acc.save_state(last_dir)
            if acc.is_main_process:
                json.dump(state, open(os.path.join(last_dir, "train_state.json"), "w"))
                if hub is not None:
                    hub.push(f"step {s}")

    if hub is not None:
        hub.wait()
    acc.save_state(last_dir)
    if acc.is_main_process:
        json.dump(state, open(os.path.join(last_dir, "train_state.json"), "w"))
        if hub is not None:
            hub.push(f"stopped at step {state['step']}", block=True)
    if wb is not None:
        wb.finish()
    acc.print(f"stopped at step {state['step']} | best dev PER {state['best_per']:.4f}")


if __name__ == "__main__":
    main()
