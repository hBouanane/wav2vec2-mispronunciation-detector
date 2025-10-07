import torch
import numpy as np
from torch.utils.data import DataLoader
import evaluate
import json
import os

from config import Config
from model import Wav2Vec2MispronunciationDetector
from data import load_and_prepare_dataset, prepare_dataset, DataCollatorCTCWithReference
from utils import create_vocabulary, create_processor, compute_metrics


def evaluate_model():
    # Initialize config
    config = Config()
    
    # Create vocabulary
    vocab_dict, vocab_size, PAD_TOKEN_ID, BLANK_TOKEN_ID = create_vocabulary(config.vocab_file)
    
    # Create processor
    processor = create_processor(config.sampling_rate)
    
    # Load dataset
    dataset = load_and_prepare_dataset(config)
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor, config),
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=32,
        desc="Processing audio and phonemes"
    )
    
    # Filter out empty batches
    dataset = dataset.filter(lambda x: len(x["input_values"]) > 0)
    
    # Get validation split
    dataset = dataset.shuffle(seed=42)
    val_dataset = dataset.select(range(config.train_size, config.train_size + config.val_size))
    
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data collator
    data_collator = DataCollatorCTCWithReference(processor=processor, padding=True)
    
    # Create dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    
    # Initialize model
    print("Loading model...")
    model = Wav2Vec2MispronunciationDetector(
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        blank_token_id=processor.tokenizer.pad_token_id,
        config=config
    )
    
    # Load trained weights
    if os.path.exists(f"{config.save_path}/best_model.pt"):
        model.load_state_dict(torch.load(f"{config.save_path}/best_model.pt"))
        print(f"Loaded model from {config.save_path}/best_model.pt")
    else:
        print("Warning: No trained model found. Using randomly initialized model.")
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load WER metric
    wer_metric = evaluate.load("wer")
    
    # Evaluation loop
    print("Starting evaluation...")
    all_metrics = []
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            input_values = batch["input_values"].to(device)
            phoneme_ref_ids = batch["phoneme_ref_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            output = model(input_values, phoneme_ref_ids)
            logits = output['logits']
            
            logits_np = logits.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            metrics, preds, refs = compute_metrics(logits_np, labels_np, processor, wer_metric)
            all_metrics.append(metrics)
            all_predictions.extend(preds)
            all_references.extend(refs)
            
            # Print first few examples from first batch
            if batch_idx == 0:
                print("\n=== Sample Predictions ===")
                for i in range(min(3, len(preds))):
                    print(f"\nExample {i+1}:")
                    print(f"Predicted: {preds[i]}")
                    print(f"Reference: {refs[i]}")
    
    # Aggregate metrics
    avg_wer = np.mean([m["wer"] for m in all_metrics])
    print(f"\n=== Evaluation Results ===")
    print(f"Validation WER: {avg_wer:.4f}")
    print(f"Total examples evaluated: {len(all_predictions)}")
    
    # Save results
    results = {
        "wer": float(avg_wer),
        "num_examples": len(all_predictions),
        "sample_predictions": [
            {"prediction": pred, "reference": ref} 
            for pred, ref in zip(all_predictions[:10], all_references[:10])
        ]
    }
    
    os.makedirs(config.save_path, exist_ok=True)
    with open(f"{config.save_path}/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {config.save_path}/evaluation_results.json")


if __name__ == "__main__":
    evaluate_model()