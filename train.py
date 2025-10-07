import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
import os
import json

from config import Config
from model import Wav2Vec2MispronunciationDetector
from data import load_and_prepare_dataset, prepare_dataset, DataCollatorCTCWithReference
from utils import create_vocabulary, create_processor


def train():
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
    
    # Split dataset
    dataset = dataset.shuffle(seed=42)
    if config.total_samples is None:
        config.train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(config.train_size))
    val_dataset = dataset.select(range(config.train_size, len(dataset)))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data collator
    data_collator = DataCollatorCTCWithReference(processor=processor, padding=True)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Initialize model
    print("Initializing model...")
    model = Wav2Vec2MispronunciationDetector(
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        blank_token_id=processor.tokenizer.pad_token_id,
        config=config
    )
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # Calculate training steps
    num_training_steps = config.num_epochs * len(train_dataloader)
    num_warmup_steps = int(config.warmup_ratio * num_training_steps)
    
    # Initialize scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    print(f"Model initialized on device: {accelerator.device}")
    print(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            input_values = batch["input_values"]
            phoneme_ref_ids = batch["phoneme_ref_ids"]
            labels = batch["labels"]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(input_values=input_values, phoneme_ref_ids=phoneme_ref_ids)
            logits = output['logits']
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, C)
            
            batch_size, max_time, vocab_size = logits.size()
            input_lengths = torch.full(
                (batch_size,),
                fill_value=max_time,
                dtype=torch.long,
                device=logits.device,
            )
            
            targets = []
            target_lengths = []
            for i in range(labels.size(0)):
                valid_labels = labels[i][labels[i] != -100]
                targets.append(valid_labels)
                target_lengths.append(len(valid_labels))
            
            targets = torch.cat(targets)
            target_lengths = torch.tensor(
                target_lengths, dtype=torch.long, device=logits.device
            )
            
            # Compute CTC loss
            loss = model.module.ctc_loss(log_probs, targets, input_lengths, target_lengths)
            
            # Backward pass
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{config.num_epochs} - Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    
    # # Save model
    # print("Saving model...")
    # os.makedirs(config.save_path, exist_ok=True)
    
    # # Save config
    # with open(f"{config.save_path}/config.json", "w", encoding="utf-8") as f:
    #     json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    
    # # Save processor
    # processor.save_pretrained(config.save_path)
    
    # # Save model weights
    # unwrapped_model = accelerator.unwrap_model(model)
    # torch.save(unwrapped_model.state_dict(), f"{config.save_path}/best_model.pt")
    
    # print(f"Model saved to {config.save_path}")


if __name__ == "__main__":
    train()