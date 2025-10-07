import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
import os
import json
import numpy as np
from tqdm import tqdm
import evaluate

from config import Config
from model import Wav2Vec2MispronunciationDetector
from data import load_and_prepare_dataset, prepare_dataset, DataCollatorCTCWithReference
from utils import create_vocabulary, create_processor, compute_metrics


def evaluate_model(model, dataloader, processor, wer_metric, device):
    """Evaluate model on validation set."""
    model.eval()
    all_metrics = []
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch["input_values"]
            phoneme_ref_ids = batch["phoneme_ref_ids"]
            labels = batch["labels"]
            
            # Forward pass
            output = model(input_values=input_values, phoneme_ref_ids=phoneme_ref_ids)
            logits = output['logits']
            
            logits_np = logits.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            metrics, preds, refs = compute_metrics(logits_np, labels_np, processor, wer_metric)
            all_metrics.append(metrics)
            all_predictions.extend(preds)
            all_references.extend(refs)
    
    # Aggregate metrics
    avg_wer = np.mean([m["wer"] for m in all_metrics])
    
    return {
        "eval_wer": avg_wer,
        "eval_samples": len(all_predictions)
    }


def save_checkpoint(model, optimizer, lr_scheduler, epoch, step, config, checkpoint_path, is_best=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
    }
    
    checkpoint_file = os.path.join(checkpoint_path, f"checkpoint-epoch-{epoch}.pt")
    torch.save(checkpoint, checkpoint_file)
    
    if is_best:
        best_file = os.path.join(checkpoint_path, "best_model.pt")
        torch.save(model.state_dict(), best_file)
    
    # Keep only last N checkpoints
    checkpoints = sorted([f for f in os.listdir(checkpoint_path) if f.startswith("checkpoint-epoch")])
    if len(checkpoints) > config.save_total_limit:
        for old_checkpoint in checkpoints[:-config.save_total_limit]:
            os.remove(os.path.join(checkpoint_path, old_checkpoint))
    
    return checkpoint_file


def train():
    # Initialize config
    config = Config()
    
    # Initialize wandb
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                entity=config.wandb_entity,
                config=config.to_dict()
            )
            print("✓ Wandb initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize wandb: {e}")
            config.use_wandb = False
    
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
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
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
    
    # Load WER metric
    wer_metric = evaluate.load("wer")
    
    # Training tracking
    global_step = 0
    best_metric = float('inf') if not config.greater_is_better else float('-inf')
    
    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print(f"Evaluation strategy: {config.eval_strategy}")
    print(f"Save strategy: {config.save_strategy}")
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(epoch_iterator):
            input_values = batch["input_values"]
            phoneme_ref_ids = batch["phoneme_ref_ids"]
            labels = batch["labels"]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(input_values=input_values, phoneme_ref_ids=phoneme_ref_ids)
            logits = output['logits']
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, C)
            
            batch_size, max_time, vocab_size_dim = logits.size()
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
            loss = model.ctc_loss(log_probs, targets, input_lengths, target_lengths)
            
            # Backward pass
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            epoch_iterator.set_postfix({'loss': loss.item(), 'lr': lr_scheduler.get_last_lr()[0]})
            
            # Logging
            if global_step % config.logging_steps == 0:
                avg_loss = total_loss / num_batches
                log_dict = {
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": global_step
                }
                
                if config.use_wandb:
                    wandb.log(log_dict, step=global_step)
                
                print(f"\nStep {global_step}: Loss={loss.item():.4f}, Avg Loss={avg_loss:.4f}, LR={lr_scheduler.get_last_lr()[0]:.2e}")
            
            # Evaluation by steps
            if config.eval_strategy == "steps" and global_step % config.eval_steps == 0:
                print(f"\n{'='*50}")
                print(f"Evaluating at step {global_step}...")
                eval_metrics = evaluate_model(model, val_dataloader, processor, wer_metric, accelerator.device)
                
                print(f"Evaluation WER: {eval_metrics['eval_wer']:.4f}")
                
                if config.use_wandb:
                    wandb.log({
                        "eval/wer": eval_metrics['eval_wer'],
                        "eval/step": global_step
                    }, step=global_step)
                
                # Check if best model
                current_metric = eval_metrics['eval_wer']
                is_best = (current_metric < best_metric) if not config.greater_is_better else (current_metric > best_metric)
                
                if is_best:
                    best_metric = current_metric
                    print(f"✓ New best model! WER: {best_metric:.4f}")
                
                # Save checkpoint
                if config.save_strategy == "steps" and global_step % config.save_steps == 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_checkpoint(
                        unwrapped_model, optimizer, lr_scheduler,
                        epoch, global_step, config, config.save_path, is_best=is_best
                    )
                    print(f"✓ Checkpoint saved at step {global_step}")
                
                model.train()
                print(f"{'='*50}\n")
        
        # End of epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"\nEpoch {epoch+1}/{config.num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Evaluation by epoch
        if config.eval_strategy == "epoch":
            print(f"\n{'='*50}")
            print(f"Evaluating at end of epoch {epoch+1}...")
            eval_metrics = evaluate_model(model, val_dataloader, processor, wer_metric, accelerator.device)
            
            print(f"Evaluation WER: {eval_metrics['eval_wer']:.4f}")
            
            if config.use_wandb:
                wandb.log({
                    "eval/wer": eval_metrics['eval_wer'],
                    "eval/epoch": epoch + 1,
                    "train/epoch_loss": avg_loss
                }, step=global_step)
            
            # Check if best model
            current_metric = eval_metrics['eval_wer']
            is_best = (current_metric < best_metric) if not config.greater_is_better else (current_metric > best_metric)
            
            if is_best:
                best_metric = current_metric
                print(f"✓ New best model! WER: {best_metric:.4f}")
            
            # Save checkpoint
            if config.save_strategy == "epoch":
                unwrapped_model = accelerator.unwrap_model(model)
                save_checkpoint(
                    unwrapped_model, optimizer, lr_scheduler,
                    epoch + 1, global_step, config, config.save_path, is_best=is_best
                )
                print(f"✓ Checkpoint saved at epoch {epoch+1}")
            
            print(f"{'='*50}\n")
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best WER: {best_metric:.4f}")
    print("="*50)
    
    # Save final model
    print("\nSaving final model...")
    os.makedirs(config.save_path, exist_ok=True)
    
    # Save config
    with open(f"{config.save_path}/config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    
    # Save processor
    processor.save_pretrained(config.save_path)
    
    # Save final model weights
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), f"{config.save_path}/final_model.pt")
    
    print(f"Model saved to {config.save_path}")
    
    # Finish wandb
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()