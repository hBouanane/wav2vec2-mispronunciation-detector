"""Training defaults. Every field is also a command-line flag of train.py
(e.g. --encoder facebook/w2v-bert-2.0 --max-steps 20000)."""
from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- data: parquet files from the released IqraEval datasets (Kaggle input or HF cache)
    native_glob: str = "/kaggle/input/**/iqra_train/data/train-*.parquet"
    native_dev: str = "/kaggle/input/**/iqra_train/data/dev-*.parquet"
    tts_glob: str = "/kaggle/input/**/iqra_tts/data/train-*.parquet"
    vocab_file: str = "sws_arabic.txt"
    tts_dev_every: int = 20          # every 20th TTS row group is held out for dev (~5%)
    n_dev_native: int = 1500
    n_dev_tts: int = 1500
    max_seconds: float = 15.0
    shuffle_pool: int = 16           # row groups (~100 rows each) mixed together per worker
    # which Iqra_TTS clips to train on: "all", "original" (correctly read verses only, no injected
    # errors) or "augmented". The dev set is unaffected (it always holds both kinds).
    tts_labels: str = "all"

    # --- augmentation (training only)
    speed_p: float = 0.5             # probability of speed perturbation
    speed_range: float = 0.1         # factor in [1 - r, 1 + r]
    noise_p: float = 0.3             # probability of added white noise
    noise_snr_db: tuple = (10.0, 30.0)

    # --- model
    encoder: str = "elgeish/wav2vec2-large-xlsr-53-arabic"
    mask_time_prob: float = 0.05
    final_dropout: float = 0.1
    grad_checkpointing: bool = True
    freeze_layers: int = 0           # freeze the first N transformer layers (e.g. 12 for w2v-BERT on a T4)
    target: str = "iqraeval"         # "qps": keep muaalem's phoneme + tafkheem heads (needs the muaalem encoder)
    tafkheem_weight: float = 0.3     # weight of the tafkheem CTC loss when target == "qps"

    # --- optimisation
    batch_size: int = 8              # per GPU
    grad_accum: int = 2
    lr: float = 5e-5
    head_lr_mult: float = 10.0       # the new output layer learns faster than the pretrained encoder
    warmup: int = 1000
    max_steps: int = 20000
    weight_decay: float = 0.01
    num_workers: int = 2
    seed: int = 42

    # --- reference penalty (phase 2, off by default): where the speaker said X but the reference
    # has Y, add  w * -log(1 - p(Y))  on the frames aligned to X
    ref_penalty: float = 0.0
    # start from trained weights (fresh optimizer / schedule / step count), e.g. for the penalty
    # fine-tune: a local best_model.pt or a Hub repo id holding best_model.pt. Empty = pretrained encoder.
    init_from: str = ""

    # --- bookkeeping
    eval_every: int = 1000
    save_every: int = 1000
    log_every: int = 50
    time_limit_h: float = 11.3

    def to_dict(self):
        return asdict(self)
