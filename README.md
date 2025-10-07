# Wav2Vec2 Arabic Mispronunciation Detector

A deep learning model for detecting mispronunciations in Arabic speech using Wav2Vec2 and cross-attention mechanisms.

## Overview

This project implements a mispronunciation detection system that:

* Uses a pre-trained Wav2Vec2 model as an audio encoder
* Employs transformer-based cross-attention to align audio features with reference phoneme sequences
* Predicts phoneme sequences using CTC loss
* Evaluates pronunciation quality by comparing predicted vs. reference phonemes

## Model Architecture

```
┌─────────────────────────────────────────┐
│  Audio Input (16kHz)                    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Wav2Vec2 Encoder 			  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Audio Features (1024-dim)              │
└────────────┬────────────────────────────┘
             │
             │    ┌──────────────────────┐
             │    │ Reference Phonemes   │
             │    └──────────┬───────────┘
             │               │
             │               ▼
             │    ┌──────────────────────┐
             │    │ Phoneme Embeddings   │
             │    └──────────┬───────────┘
             │               │
             │               ▼
             │    ┌──────────────────────┐
             │    │ Transformer Encoder  │
             │    └──────────┬───────────┘
             │               │
             └───────┬───────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Cross-Attention Fusion    │
        │  (Transformer Blocks x2)   │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Projection + Dropout      │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Phoneme Classifier        │
        │  (CTC Loss)                │
        └────────────────────────────┘
```

## Features

* **Pre-trained Wav2Vec2** : Uses `elgeish/wav2vec2-large-xlsr-53-arabic` for robust audio feature extraction
* **Cross-Attention Fusion** : Aligns audio features with reference phoneme sequences
* **CTC Loss** : Handles variable-length alignments between audio and phonemes
* **Efficient Training** : Uses Hugging Face Accelerate for distributed training
* **WER Evaluation** : Measures Word Error Rate on phoneme sequences

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wav2vec2-mispronunciation-detector.git
cd wav2vec2-mispronunciation-detector

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the **IqraEval** dataset:

* `IqraEval/Iqra_TTS`: Text-to-speech samples with reference phonemes
* Contains Arabic speech with phoneme-level annotations

### Required Files

Place your `sws_arabic.txt` vocabulary file in the project root. This file should contain one phoneme per line.

## Usage

### Training

```bash
python train.py
```

The training script will:

1. Load and preprocess the dataset (3000 samples by default)
2. Create vocabulary from `sws_arabic.txt`
3. Train for 25 epochs with batch size 4
4. Save the model to `./Wav2Vec2_mispronunciation_detector/`

### Evaluation

```bash
python evaluate.py
```

This will:

1. Load the trained model
2. Evaluate on the validation set
3. Compute WER (Word Error Rate)
4. Save results to `evaluation_results.json`

### Configuration

Edit `config.py` to customize:

```python
# Data
dataset_name: str = "IqraEval/Iqra_train"
vocab_file: str = "sws_arabic.txt"

# Model
pretrained_model: str = "elgeish/wav2vec2-large-xlsr-53-arabic"
freeze_encoder: bool = False
hidden_size: int = 1024

# Training
batch_size: int = 4
num_epochs: int = 25
learning_rate: float = 1e-4
```

## Project Structure

```
wav2vec2-mispronunciation-detector/
├── config.py              # Configuration settings
├── model.py               # Model architecture
├── data.py                # Data loading & preprocessing
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── utils.py               # Helper functions
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Model Details

### Parameters

* **Total Parameters** : ~367M
* **Trainable Parameters** : ~367M (if encoder not frozen)
* **Vocabulary Size** : 71 phonemes (including special tokens)

### Hyperparameters

* **Learning Rate** : 1e-4
* **Warmup Ratio** : 10%
* **Batch Size** : 4
* **Max Gradient Norm** : 1.0
* **Optimizer** : AdamW
* **Scheduler** : Linear with warmup

### Special Tokens

* `[UNK]`: Unknown phoneme
* `[PAD]`: Padding token
* `[BLANK]`: CTC blank token

## Results

Training on 2000 samples, validating on 1000 samples:

| Epoch | Train Loss |
| ----- | ---------- |
| 1     | 0.0971     |
| 2     | 0.0871     |
| 3     | 0.0878     |
| 4     | 0.0786     |

Validation WER will be available after running `evaluate.py`.

## Hardware Requirements

* **GPU** : NVIDIA GPU with 16GB+ VRAM (T4, V100, A100)
* **RAM** : 16GB+ system memory
* **Storage** : ~5GB for model and dataset cache

## Citation

If you use this code, please cite:

```bibtex
@software{wav2vec2_mispronunciation,
  title={Wav2Vec2 Arabic Mispronunciation Detector},
  author={BOUANANE Haitam},
  year={2025},
  url={https://github.com/yourusername/wav2vec2-mispronunciation-detector}
}
```

## Acknowledgments

* Pre-trained model: [elgeish/wav2vec2-large-xlsr-53-arabic](https://huggingface.co/elgeish/wav2vec2-large-xlsr-53-arabic)
* Dataset: [IqraEval](https://huggingface.co/datasets/IqraEval)
* Framework: Hugging Face Transformers

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub or contact [bouananehaitam03@gmail.com]
