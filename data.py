from dataclasses import dataclass
from typing import List, Dict, Union
import torch
from datasets import load_dataset, Dataset, Audio
from transformers import Wav2Vec2Processor


def load_and_prepare_dataset(config):
    """Load dataset from HuggingFace and prepare it."""
    print("Loading dataset...")
    
    if config.total_samples:
        dataset_with_reference_phoneme = load_dataset(
            config.dataset_with_reference_name,
            split="train"
        )
    else:
        # Streaming dataset
        dataset_with_reference_phoneme = load_dataset(
            config.dataset_with_reference_name,
            split="train",
            streaming=True
        )
        
        # Shuffle with a buffer
        dataset_with_reference_phoneme = dataset_with_reference_phoneme.shuffle(buffer_size=10_000, seed=42)
        
        # Take samples
        dataset_with_reference_phoneme = Dataset.from_generator(
            dataset_with_reference_phoneme.take(config.total_samples).__iter__
        )
    
    # Cast audio
    dataset_with_reference_phoneme = dataset_with_reference_phoneme.cast_column(
        "audio", Audio(sampling_rate=config.sampling_rate)
    )
    
    return dataset_with_reference_phoneme


def prepare_dataset(batch, processor, config):
    """
    Batch process dataset with error handling and validation.
    """
    # Extract audio arrays
    audio_arrays = [audio["array"] for audio in batch["audio"]]

    # Clean phoneme sequences with validation
    phoneme_aug_cleaned = []
    phoneme_ref_cleaned = []
    valid_indices = []

    for idx, (aug, ref) in enumerate(zip(batch['phoneme_aug'], batch['phoneme_ref'])):
        aug_clean = aug.replace('<sil>', '').strip()
        ref_clean = ref.replace('<sil>', '').strip()

        # Only keep examples with valid phonemes
        if aug_clean and ref_clean:
            phoneme_aug_cleaned.append(aug_clean)
            phoneme_ref_cleaned.append(ref_clean)
            valid_indices.append(idx)

    # Filter audio arrays to match valid phonemes
    audio_arrays = [audio_arrays[idx] for idx in valid_indices]

    # Handle empty batch
    if not audio_arrays:
        return {
            "input_values": [],
            "labels": [],
            "phoneme_ref_ids": []
        }

    # Batch process audio
    processed_audio = processor(
        audio_arrays,
        sampling_rate=config.sampling_rate,
        return_tensors="np"
    )

    # Batch process phonemes
    with processor.as_target_processor():
        labels_batch = processor(phoneme_aug_cleaned)
        phoneme_ref_batch = processor(phoneme_ref_cleaned)

    return {
        "input_values": processed_audio.input_values,
        "labels": labels_batch.input_ids,
        "phoneme_ref_ids": phoneme_ref_batch.input_ids
    }


@dataclass
class DataCollatorCTCWithReference:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate features
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        phoneme_ref_features = [{"input_ids": f["phoneme_ref_ids"]} for f in features]

        # Pad input audio
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt"
            )
            phoneme_ref_batch = self.processor.pad(
                phoneme_ref_features,
                padding=self.padding,
                return_tensors="pt"
            )

        # Replace padding tokens in labels with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        batch["phoneme_ref_ids"] = phoneme_ref_batch["input_ids"]

        return batch