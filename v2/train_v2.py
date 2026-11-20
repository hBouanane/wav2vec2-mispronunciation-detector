"""Train the reference-anchored MDD model (v2) on 2025 IqraEval data only.

Single GPU:   python v2/train_v2.py --out /kaggle/working/run1 --mdd-eval interspeech_IqraEval/mdd_eval
Two GPUs:     accelerate launch --multi_gpu --num_processes 2 v2/train_v2.py ...
Resume:       add --resume (continues from <out>/last/), or use --hub-repo: checkpoints are
              uploaded to a private Hugging Face repo and a new session resumes from it automatically
Local smoke:  python v2/train_v2.py --smoke --out runs/smoke --mdd-eval ../interspeech_IqraEval/mdd_eval
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta

import numpy as np
import psutil
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets.distributed import split_dataset_by_node
from huggingface_hub import HfApi, snapshot_download
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_v2 import (Collator, Featurizer, PhoneVocab, RefCorrupter, build_train_stream,  # noqa: E402
                     build_val_sets)
from mdd_metric import official_mdd  # noqa: E402
from model_v2 import RefAnchoredMDD, decode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAUS = [0.3, 0.5, 0.7, 0.85, 0.95]


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mdd-eval", required=True, help="path to interspeech_IqraEval/mdd_eval")
    ap.add_argument("--encoder", default="elgeish/wav2vec2-large-xlsr-53-arabic")
    ap.add_argument("--vocab", default=os.path.join(HERE, "..", "sws_arabic.txt"))
    ap.add_argument("--confusions", default=os.path.join(HERE, "confusions.json"))
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr-encoder", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--freeze-encoder-steps", type=int, default=1000,
                    help="train only the new layers first so random heads don't wreck the encoder")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--p-native", type=float, default=0.55, help="share of native Iqra_train in the mix")
    ap.add_argument("--corrupt-p", type=float, default=0.5, help="P(corrupt reference) for native audio")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--n-val", type=int, default=1500, help="validation utterances per subset")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="data workers per GPU; each holds ~2.5 GB RAM (Kaggle has ~30 GB in total)")
    ap.add_argument("--shuffle-buffer", type=int, default=500,
                    help="streaming shuffle buffer per worker; 2000 used ~3 GB RAM per worker and OOM-killed Kaggle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hub-repo", default=None,
                    help="private HF model repo (e.g. user/iqraeval-mdd-v2) for checkpoints; auto-resume from it")
    ap.add_argument("--hub-every", type=int, default=1000, help="upload checkpoint to the Hub every N steps")
    ap.add_argument("--wandb-project", default=None, help="log to this wandb project (needs WANDB_API_KEY)")
    ap.add_argument("--wandb-name", default=None, help="wandb run name; the run id is kept in the checkpoint "
                                                         "so resumed sessions continue the same run")
    ap.add_argument("--time-limit-h", type=float, default=11.3, help="save and stop before the Kaggle 12 h limit")
    ap.add_argument("--smoke", action="store_true", help="tiny random encoder, a few steps, CPU/GPU sanity check")
    return ap.parse_args()


HUB_FILES = ["last/*", "best_model.pt", "best.json", "eval_log.jsonl"]


class HubSync:
    """Checkpoint persistence through a private Hugging Face model repo (main process only).
    Uploads run in the background; history is squashed so storage stays ~one checkpoint."""

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
            except Exception as e:  # a failed upload must not kill training
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


class StreamCollate:
    def __init__(self, collate):
        self.collate = collate

    def __call__(self, xs):
        return self.collate([x["f"] for x in xs])


def evaluate(model, val, vocab, collate, acc, args, step):
    model.eval()
    feat = Featurizer(vocab, None, max_seconds=60)
    items = [(x, feat.from_lists(x["audio"], x["ref"], x["spoken"])) for x in val]
    items = [(x, f) for x, f in items if f is not None]
    items.sort(key=lambda t: len(t[1]["wave"]))
    probs = []  # keep logits once, decode with several thresholds
    with torch.no_grad():
        for i in range(0, len(items), 8):
            chunk = items[i:i + 8]
            b = collate([f for _, f in chunk])
            b = {k: v.to(acc.device) for k, v in b.items()}
            with acc.autocast():
                out = model(b["wave"], b["wave_mask"], b["ref"])
            probs.append(({"slot_logits": out["slot_logits"].float().cpu(), "ins_logits": out["ins_logits"].float().cpu()},
                          b["ref"].cpu(), chunk))
    results = {}
    for tau in TAUS:
        ids, refs, anns, preds = [], [], [], []
        for out, ref, chunk in probs:
            for (x, _), p in zip(chunk, decode(out, ref, vocab, tau=tau, tau_ins=0.5)):
                ids.append(x["id"]); refs.append(x["ref"]); anns.append(x["spoken"]); preds.append(p)
        results[tau] = official_mdd(ids, refs, anns, preds, args.mdd_eval,
                                    workdir=os.path.join(args.out, "val_tmp"))
    model.train()
    best_tau = max(results, key=lambda t: results[t].get("f1", 0))
    return best_tau, results


def main():
    args = get_args()
    if args.smoke:
        args.encoder = "hf-internal-testing/tiny-random-wav2vec2"
        args.max_steps, args.eval_every, args.save_every = 6, 6, 3
        args.batch_size, args.grad_accum, args.n_val, args.warmup = 2, 1, 6, 2
        args.freeze_encoder_steps, args.num_workers, args.hub_every = 2, 0, 3
        args.log_every = 2
    acc = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no",
                      gradient_accumulation_steps=args.grad_accum,
                      # validation runs on rank 0 only for several minutes; don't let rank 1 time out
                      kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=1))])
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    vocab = PhoneVocab(args.vocab)
    corrupter = RefCorrupter(args.confusions)
    model = RefAnchoredMDD(vocab, encoder_name=args.encoder, d_model=64 if args.smoke else 512,
                           n_layers=1 if args.smoke else 4, n_heads=2 if args.smoke else 8,
                           grad_checkpointing=not args.smoke)

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr_encoder},
                             {"params": head_params, "lr": args.lr_head}], weight_decay=0.01)

    def head_lr(s):
        if s < args.warmup:
            return s / max(1, args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / max(1, args.max_steps - args.warmup))))

    def enc_lr(s):  # encoder stays frozen (lr 0) first, then its own warm-up
        f = args.freeze_encoder_steps
        return 0.0 if s < f else head_lr(s - f) if s - f < args.warmup else head_lr(s)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [enc_lr, head_lr])  # stepped manually per optimizer step

    state = {"step": 0, "best_f1": -1.0, "best_tau": 0.5}
    last_dir = os.path.join(args.out, "last")
    model, opt = acc.prepare(model, opt)
    hub = HubSync(args.hub_repo, args.out) if args.hub_repo and acc.is_main_process else None
    if hub is not None and hub.pull():
        acc.print(f"[hub] pulled checkpoint from {args.hub_repo}")
    acc.wait_for_everyone()
    if (args.resume or args.hub_repo) and os.path.exists(os.path.join(last_dir, "train_state.json")):
        acc.load_state(last_dir)
        state = json.load(open(os.path.join(last_dir, "train_state.json")))
        for _ in range(state["step"]):
            sched.step()
        acc.print(f"resumed at step {state['step']}")

    wb = None
    if args.wandb_project and acc.is_main_process:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_name, id=state.get("wandb_id"),
                        resume="allow", config=vars(args))
        state["wandb_id"] = wb.id
        wb.define_metric("val/*", step_metric="train/step")
        wb.define_metric("train/*", step_metric="train/step")

    # streaming data; reseed on resume so we don't replay the same order
    stream = build_train_stream(seed=args.seed + state["step"], p_native=args.p_native, buffer=args.shuffle_buffer)
    stream = split_dataset_by_node(stream, rank=acc.process_index, world_size=acc.num_processes)
    feat = Featurizer(vocab, corrupter, corrupt_p=args.corrupt_p, max_seconds=args.max_seconds,
                      seed=args.seed + state["step"] + 1000 * acc.process_index)
    stream = stream.map(lambda ex: {"f": feat(ex)}, remove_columns=["audio", "ref", "spoken", "native"])
    collate = Collator(vocab)
    loader = DataLoader(stream, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=StreamCollate(collate))

    val = build_val_sets(corrupter, n_native=args.n_val, n_tts=args.n_val) if acc.is_main_process else None
    acc.print(f"validation utterances: {len(val) if val else 0}")

    t0, log = time.time(), []
    model.train()
    unwrapped = acc.unwrap_model(model)
    for batch in loader:
        # every rank must stop on the same step, or the other one hangs in a collective
        stop = torch.tensor(float(state["step"] >= args.max_steps or (time.time() - t0) / 3600 > args.time_limit_h),
                            device=acc.device)
        if acc.reduce(stop, reduction="max").item() > 0:
            break
        if batch is None:
            continue
        batch = {k: v.to(acc.device, non_blocking=True) for k, v in batch.items()}
        frozen = state["step"] < args.freeze_encoder_steps
        with acc.accumulate(model):
            out = model(batch["wave"], batch["wave_mask"], batch["ref"], batch["slot"], batch["ins"], batch["ctc"])
            acc.backward(out["loss"])
            if acc.sync_gradients:
                acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if not acc.sync_gradients:
            continue
        sched.step()
        state["step"] += 1
        s = state["step"]
        log.append([out["loss"].item(), out["l_slot"].item(), out["l_ins"].item(), out["l_ctc"].item()])
        if s % args.log_every == 0:
            m = np.mean(log, 0); log = []
            ram = psutil.virtual_memory()
            ram_used, ram_total = (ram.total - ram.available) / 1e9, ram.total / 1e9
            if wb is not None:
                lrs = sched.get_last_lr()
                wb.log({"train/step": s, "train/loss": m[0], "train/loss_slot": m[1], "train/loss_ins": m[2],
                        "train/loss_ctc": m[3], "train/lr_head": lrs[1], "train/lr_encoder": lrs[0],
                        "train/hours": (time.time() - t0) / 3600, "sys/ram_used_gb": ram_used})
            acc.print(f"step {s} loss {m[0]:.4f} slot {m[1]:.4f} ins {m[2]:.4f} ctc {m[3]:.4f} "
                      f"lr {sched.get_last_lr()[1]:.2e}/{sched.get_last_lr()[0]:.2e} {'(encoder frozen)' if frozen else ''} "
                      f"{(time.time() - t0) / 60:.1f} min | RAM {ram_used:.1f}/{ram_total:.0f} GB", flush=True)
        if s % args.save_every == 0:
            if hub is not None:
                hub.wait()  # don't overwrite files an upload is still reading
            acc.save_state(last_dir)
            if acc.is_main_process:
                json.dump(state, open(os.path.join(last_dir, "train_state.json"), "w"))
                if hub is not None and s % args.hub_every == 0:
                    hub.push(f"step {s}")
        if s % args.eval_every == 0 and acc.is_main_process:
            tau, res = evaluate(unwrapped, val, vocab, collate, acc, args, s)
            f1 = res[tau].get("f1", 0)
            acc.print(f"[eval step {s}] " + " | ".join(f"tau {t}: F1 {r.get('f1', 0):.4f} P {r.get('precision', 0):.3f} "
                                                       f"R {r.get('recall', 0):.3f}" for t, r in res.items()), flush=True)
            if wb is not None:
                row = {"train/step": s, "val/f1": f1, "val/best_tau": tau}
                for t, r in res.items():
                    row.update({f"val/{k}@tau{t}": v for k, v in r.items()})
                row.update({f"val/{k}": v for k, v in res[tau].items()})  # metrics at the best tau
                wb.log(row)
            with open(os.path.join(args.out, "eval_log.jsonl"), "a") as f:
                f.write(json.dumps({"step": s, "results": {str(k): v for k, v in res.items()}}) + "\n")
            if f1 > state["best_f1"]:
                state.update(best_f1=f1, best_tau=tau)
                if hub is not None:
                    hub.wait()
                torch.save(unwrapped.state_dict(), os.path.join(args.out, "best_model.pt"))
                json.dump({**state, "args": vars(args)}, open(os.path.join(args.out, "best.json"), "w"), indent=1)
                acc.print(f"  new best F1 {f1:.4f} at tau {tau}")
                if wb is not None:
                    wb.summary.update({"best_val_f1": f1, "best_tau": tau, "best_step": s})
        acc.wait_for_everyone()

    if hub is not None:
        hub.wait()
    acc.save_state(last_dir)
    if acc.is_main_process:
        json.dump(state, open(os.path.join(last_dir, "train_state.json"), "w"))
        if hub is not None:
            hub.push(f"stopped at step {state['step']}", block=True)
            print(f"[hub] final checkpoint uploaded to {args.hub_repo}", flush=True)
    if wb is not None:
        wb.finish()
    acc.print(f"stopped at step {state['step']} best val F1 {state['best_f1']:.4f} (tau {state['best_tau']})")


if __name__ == "__main__":
    main()
