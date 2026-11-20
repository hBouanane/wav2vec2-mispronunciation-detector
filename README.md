# Wav2Vec2 Arabic Mispronunciation Detector

> **Work in progress on this branch (`ctc`):** a plain CTC phoneme recogniser (audio only, no
> reference input) trained on all released 2025 data — see `train.py`, `kaggle_train.ipynb`.
> The model and results described below are **v1**; its code is at git tag
> [`v1`](https://github.com/hBouanane/wav2vec2-mispronunciation-detector/tree/v1).
> The reference-anchored experiment (test F1 0.27) is kept on branch `v2`.

A reference-conditioned phoneme recogniser for **Modern Standard Arabic (MSA) mispronunciation detection and diagnosis (MDD)**, built for the [IqraEval](https://huggingface.co/IqraEval) shared task.

The model listens to a learner reading a fully-vowelised sentence and outputs the phoneme sequence that was actually spoken. Comparing it with the canonical (reference) phonemes shows where the reader substituted, deleted or inserted a sound.

| | |
|---|---|
| 🤗 **Model** | [`Haitam03/Wav2Vec2-mispronunciation-detector`](https://huggingface.co/Haitam03/Wav2Vec2-mispronunciation-detector) |
| 🎙️ **Demo** | [`Haitam03/arabic-pronunciation-checker`](https://huggingface.co/spaces/Haitam03/arabic-pronunciation-checker) (Gradio Space) |
| 🏁 **Benchmark** | [Iqra'Eval 2025 shared task (ArabicNLP 2025)](https://huggingface.co/spaces/IqraEval/SharedTask_ArabicNLP2025), test set `QuranMB.v2` |

```
Reference : < y a t a H a d d a v u n n aa s u ...
Predicted : < y a t a H a d d a s u n n aa s u ...
                              ^ v → s  (substitution flagged)
```

---

## Results on the Iqra'Eval 2025 test set

This model was **not submitted** to the challenge. The numbers below come from running it on the test set after the organisers released the ground truth (`IqraEval/IqraEval_Test_GT`). The scoring uses the [official MDD scripts](https://github.com/Iqra-Eval/interspeech_IqraEval/tree/main/mdd_eval) (see [Reproducing](#reproducing-the-test-set-results)). The comparison rows come from the [ArabicNLP 2025 leaderboard](https://huggingface.co/spaces/IqraEval/Leaderboard_ArabicNLP25).

Test set: `IqraEval/QuranMB.v2` (published in 2025 as `IqraEval/open_testset`) · 1,642 utterances (98 verses × 18 speakers, ≈ 2 h) · human-annotated mispronunciations.

| System | F1 ↑ | Precision | Recall | Correct Rate | Accuracy | TA | FR | FA | CD |
|---|---|---|---|---|---|---|---|---|---|
| Baic (1st) | 0.4726 | 0.3713 | 0.6501 | 0.8985 | 0.8701 | 0.9209 | 0.0791 | 0.3499 | 0.6873 |
| Hafs2Vec (2nd) | 0.4650 | 0.3292 | 0.7920 | 0.8655 | 0.8488 | 0.8840 | 0.1160 | 0.2080 | 0.6252 |
| Ghalib (3rd) | 0.4477 | 0.3218 | 0.7353 | 0.8667 | 0.8506 | 0.8886 | 0.1114 | 0.2647 | 0.5925 |
| Official baseline 1 | 0.4414 | 0.3093 | 0.7707 | 0.8361 | 0.8234 | 0.8763 | 0.1237 | 0.2293 | 0.6120 |
| Metapseud | 0.4236 | 0.2879 | 0.8012 | 0.8397 | 0.8213 | 0.8575 | 0.1425 | 0.1988 | 0.6030 |
| **This model** (`best_mode_full_data.pt`) | **0.4190** | **0.3003** | **0.6928** | **0.8621** | **0.7966** | **0.8839** | **0.1161** | **0.3072** | **0.6187** |
| Official baseline 2 | 0.4042 | 0.2715 | 0.7908 | 0.8093 | 0.7955 | 0.8474 | 0.1526 | 0.2092 | 0.5847 |

Read as a leaderboard entry, the model would have placed **6th of 12 teams**. Its F1 is just below the official HuBERT baseline. Its correct rate and true-acceptance rate are among the higher ones on the board, which means it rarely flags correctly-read phonemes. It misses more real errors (recall 0.69) than most systems, because it was trained only on synthetic TTS errors and has never heard a real learner mistake.

Phoneme error rate against the human annotation is **0.203**: 5,038 substitutions, 2,269 deletions and 3,358 insertions over 52,432 phonemes. Many of the insertions are split geminates (`n n` for `nn`), a clear target for improvement.

TA/FR = true acceptance / false rejection (correct phonemes), FA = false acceptance (missed errors), CD = correct diagnosis. Full output: [`results/best_mode_full_data/metrics.txt`](results/best_mode_full_data/metrics.txt). Challenge-format predictions: [`submission.csv`](results/best_mode_full_data/submission.csv).

---

## How it works

The task gives the model the **reference phonemes** as well as the audio, so it does not have to recognise speech from scratch. It needs to find where the reader departed from the text. The architecture exploits this with cross-attention from audio frames to the reference:

```
 audio (16 kHz)                         reference phonemes
      │                                         │
 Wav2Vec2-large-XLSR-53 (Arabic)         Embedding (1024)
      │                                         │
  LayerNorm                              BiLSTM (2×512)
      │                                         │
      └──────────► Multi-head cross-attention ◄─┘
                   (query = audio, key/value = phonemes, 8 heads)
                              │
                  residual + LayerNorm
                              │
                Linear 1024→1024 + ReLU + Dropout(0.1)
                              │
                      BiLSTM (2×512)
                              │
                   Linear → 71 phoneme logits
                              │
                     CTC (greedy decoding)
```

| Component | Detail |
|---|---|
| Acoustic encoder | [`elgeish/wav2vec2-large-xlsr-53-arabic`](https://huggingface.co/elgeish/wav2vec2-large-xlsr-53-arabic), CNN feature extractor frozen, transformer fine-tuned |
| Phoneme set | 68 IqraEval MSA phonemes ([`sws_arabic.txt`](sws_arabic.txt)) + `[UNK]`, `[PAD]`, `[BLANK]` = 71 tokens. Gemination is a separate symbol (`b` vs `bb`) |
| Loss | CTC, `[PAD]` used as blank |
| Parameters | ~330 M (all trainable except the CNN front-end) |

## Training data

Trained on [`IqraEval/Iqra_TTS`](https://huggingface.co/datasets/IqraEval/Iqra_TTS), a synthetic corpus in which errors were injected into the text before TTS generation. Each example provides:

- `phoneme_ref`: canonical phonemes of the correct sentence → **model input**
- `phoneme_aug`: phonemes of what was actually synthesised (with errors) → **CTC target**

The training split is shuffled with seed 42 and split 80/20 into train and validation.

| Hyper-parameter | Value |
|---|---|
| Optimiser | AdamW, lr 1e-4, weight decay 0.01 |
| Schedule | linear with 10 % warm-up |
| Batch size | 4 |
| Epochs | 25 (best checkpoint by validation PER) |
| Grad clipping | 1.0 |
| Framework | 🤗 Transformers + Accelerate |

## Installation

```bash
git clone https://github.com/hBouanane/wav2vec2-mispronunciation-detector.git
cd wav2vec2-mispronunciation-detector
pip install -r requirements.txt
```

## Usage

### Inference on your own audio

Download the released weights and set the paths in [`inference.py`](inference.py):

```bash
huggingface-cli download Haitam03/Wav2Vec2-mispronunciation-detector \
    best_mode_full_data.pt --local-dir hf_model
```

```python
from inference import predict_phonemes
pred = predict_phonemes(model, processor, "recording.wav",
                        reference_phonemes="< a l H a m d u l i ll a h i")
```

Reference phonemes for new text can be produced with the official [MSA phonetiser](https://github.com/Iqra-Eval/MSA_phonetiser). It needs fully diacritised input.

### Training

```bash
wandb login        # optional, set use_wandb=False in config.py to disable
python train.py
```

All hyper-parameters are in [`config.py`](config.py). Set `total_samples` / `train_size` to train on a streamed subset.

### Validation

```bash
python evaluate_model.py   # PER on the held-out 20 % of Iqra_TTS
```

## Reproducing the test-set results

```bash
# 1. official scoring scripts
git clone https://github.com/Iqra-Eval/interspeech_IqraEval ../interspeech_IqraEval

# 2. ground truth is gated: accept the terms on the dataset page, then
huggingface-cli login

# 3. predict (≈15 min on a 4 GB GPU with --fp16)
python predict_testset.py --checkpoint hf_model/best_mode_full_data.pt \
    --out results/best_mode_full_data --fp16

# 4. score
python score_testset.py --pred results/best_mode_full_data/predictions.csv \
    --mdd-eval ../interspeech_IqraEval/mdd_eval
```

`predict_testset.py` writes `predictions.csv` (`ID,Prediction`, for scoring) and `submission.csv` (`ID,Labels`, the challenge submission format). `score_testset.py` writes `metrics.txt` next to them.

## Released checkpoints

| File | Architecture | Notes |
|---|---|---|
| `best_mode_full_data.pt` | BiLSTM + cross-attention (this repo) | **Final model**, used by the demo Space and for the results above |
| `best_model.pt` | Transformer phoneme encoder + 2 cross-attention blocks | Earlier version, not compatible with current `model.py` |
| `epoch_2_full_data.pt` | – | Intermediate training checkpoint (includes optimiser state) |

## Project structure

```
├── config.py            # all hyper-parameters
├── model.py             # Wav2Vec2MispronunciationDetector
├── data.py              # dataset loading, preprocessing, CTC collator
├── train.py             # training loop (Accelerate + wandb)
├── evaluate_model.py    # validation PER
├── inference.py         # single-file inference
├── predict_testset.py   # IqraEval.2 test-set predictions
├── score_testset.py     # official MDD metrics + PER
├── utils.py             # vocab / processor helpers
├── sws_arabic.txt       # IqraEval phoneme inventory
└── results/             # test-set predictions and metrics
```

## Acknowledgements

- [IqraEval](https://huggingface.co/IqraEval) organisers for the datasets, phoneme set and evaluation scripts
- [elgeish/wav2vec2-large-xlsr-53-arabic](https://huggingface.co/elgeish/wav2vec2-large-xlsr-53-arabic) for the pre-trained encoder

## Citation

```bibtex
@software{bouanane2025wav2vec2mdd,
  title  = {Wav2Vec2 Arabic Mispronunciation Detector},
  author = {Bouanane, Haitam},
  year   = {2025},
  url    = {https://github.com/hBouanane/wav2vec2-mispronunciation-detector}
}
```

## License

MIT
