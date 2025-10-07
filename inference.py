import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor

from config import Config
from model import Wav2Vec2MispronunciationDetector
from utils import create_vocabulary


def load_audio(audio_path, sampling_rate=16000):
    """Load audio file and resample to target sampling rate."""
    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    return audio


def predict_phonemes(model, processor, audio_path, reference_phonemes, device='cuda'):
    """
    Predict phonemes from audio file.
    
    Args:
        model: Trained Wav2Vec2MispronunciationDetector
        processor: Wav2Vec2Processor
        audio_path: Path to audio file
        reference_phonemes: String of reference phonemes (space-separated)
        device: 'cuda' or 'cpu'
    
    Returns:
        predicted_phonemes: Predicted phoneme sequence (string)
    """
    # Load audio
    audio = load_audio(audio_path, processor.feature_extractor.sampling_rate)
    
    # Process audio
    input_values = processor(
        audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt"
    ).input_values.to(device)
    
    # Process reference phonemes
    with processor.as_target_processor():
        ref_phoneme_ids = processor(reference_phonemes).input_ids
    
    ref_phoneme_ids = torch.tensor([ref_phoneme_ids]).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_values, ref_phoneme_ids)
        logits = output['logits']
        
        # Get predictions
        pred_ids = torch.argmax(logits, dim=-1)
        
        # Decode
        predicted_phonemes = processor.batch_decode(pred_ids, group_tokens=True)[0]
        predicted_phonemes = predicted_phonemes.replace("[BLANK] ", "").replace("[BLANK]", "")
    
    return predicted_phonemes


def main():
    # Initialize config
    config = Config()
    
    # Create vocabulary
    vocab_dict, vocab_size, PAD_TOKEN_ID, BLANK_TOKEN_ID = create_vocabulary(config.vocab_file)
    
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(config.save_path)
    
    # Initialize model
    print("Loading model...")
    model = Wav2Vec2MispronunciationDetector(
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        blank_token_id=processor.tokenizer.pad_token_id,
        config=config
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(f"{config.save_path}/best_model.pt"))
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Model loaded on device: {device}")
    
    # Example usage
    audio_path = "path/to/your/audio.wav"  # Replace with actual path
    reference_phonemes = "aa b t"  # Replace with actual reference phonemes
    
    print(f"\nProcessing audio: {audio_path}")
    print(f"Reference phonemes: {reference_phonemes}")
    
    # Predict
    predicted = predict_phonemes(model, processor, audio_path, reference_phonemes, device)
    
    print(f"Predicted phonemes: {predicted}")
    
    # Compare
    print("\n=== Comparison ===")
    print(f"Reference: {reference_phonemes}")
    print(f"Predicted: {predicted}")
    
    # Calculate simple accuracy (character-level)
    ref_chars = reference_phonemes.replace(" ", "")
    pred_chars = predicted.replace(" ", "")
    
    if len(ref_chars) > 0:
        matches = sum(1 for r, p in zip(ref_chars, pred_chars) if r == p)
        accuracy = matches / max(len(ref_chars), len(pred_chars))
        print(f"Character-level accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()