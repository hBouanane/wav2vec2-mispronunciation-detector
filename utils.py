import json
import numpy as np
from transformers import (
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor
)


def create_vocabulary(vocab_file):
    """Create vocabulary dictionary from file."""
    print("Creating vocabulary...")
    
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab_list = [line.strip("\r\n") for line in f]

    vocab_dict = {v: idx for idx, v in enumerate(vocab_list)}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    vocab_dict["[BLANK]"] = len(vocab_dict)
    vocab_size = len(vocab_dict)

    # Save vocab
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    PAD_TOKEN_ID = vocab_dict["[PAD]"]
    BLANK_TOKEN_ID = vocab_dict["[BLANK]"]

    print(f"Vocabulary size: {vocab_size}")
    print(f"PAD token ID: {PAD_TOKEN_ID}")
    print(f"BLANK token ID: {BLANK_TOKEN_ID}")
    
    return vocab_dict, vocab_size, PAD_TOKEN_ID, BLANK_TOKEN_ID


def create_processor(sampling_rate):
    """Create Wav2Vec2 processor with tokenizer and feature extractor."""
    print("Creating processor...")
    
    tokenizer = Wav2Vec2PhonemeCTCTokenizer(
        "./vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        do_phonemize=False
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    return processor


def compute_metrics(pred_logits, label_ids, processor, wer_metric):
    """
    Compute WER for a batch.

    pred_logits: np.array of shape (batch, seq_len, vocab_size)
    label_ids: np.array of shape (batch, seq_len)
    processor: Wav2Vec2 processor (feature_extractor + tokenizer)
    wer_metric: a WER metric object
    """
    # Take argmax to get predicted token ids
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 in labels with pad_token_id for decoding
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids, group_tokens=True)
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    pred_str = [p.replace("[BLANK] ", "").replace("[BLANK]", "") for p in pred_str]

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}, pred_str, label_str