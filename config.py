from dataclasses import dataclass, asdict

@dataclass
class Config:
    # Data
    dataset_name: str = "IqraEval/Iqra_train"
    dataset_with_reference_name: str = "IqraEval/Iqra_TTS"
    dataset_split: str = "dev"
    vocab_file: str = "sws_arabic.txt"
    
    # Model
    pretrained_model: str = "elgeish/wav2vec2-large-xlsr-53-arabic"
    freeze_encoder: bool = False
    hidden_size: int = 1024
    lstm_hidden_size: int = 512
    num_transformer_layers: int = 2
    
    # Training
    batch_size: int = 4
    num_epochs: int = 25
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Evaluation & Logging (HuggingFace Trainer style)
    eval_strategy: str = "epoch"  # "no", "steps", "epoch"
    eval_steps: int = 500  # Evaluate every N steps if eval_strategy="steps"
    save_strategy: str = "epoch"  # "no", "steps", "epoch"
    save_steps: int = 500  # Save every N steps if save_strategy="steps"
    save_total_limit: int = 3  # Only keep last N checkpoints
    logging_steps: int = 100  # Log every N steps
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False  # For WER, lower is better
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "wav2vec2-mispronunciation"
    wandb_run_name: str = "arabic-ctc-v1"
    wandb_entity: str = None  # Your wandb username/team
    
    # Audio
    sampling_rate: int = 16000
    
    # Paths
    save_path: str = "./Wav2Vec2_mispronunciation_detector"
    
    # Data size
    total_samples: int = 10
    train_size: int = 6
    
    def to_dict(self):
        return asdict(self)

# Usage example
if __name__ == "__main__":
    config = Config()
    config_dict = config.to_dict()
    print(config_dict)