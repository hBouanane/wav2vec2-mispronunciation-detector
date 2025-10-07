from dataclasses import dataclass, asdict

@dataclass
class Config:
    # Data
    dataset_name: str = "IqraEval/Iqra_train"
    dataset_with_reference_name: str = "IqraEval/Iqra_TTS"
    dataset_split: str = "dev"
    vocab_file: str = "sws_arabic.txt"  # Changed from /content/sws_arabic.txt for portability
    
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
    warmup_ratio: float = 0.1  # Alternative to warmup_steps
    max_grad_norm: float = 1.0
    
    # Audio
    sampling_rate: int = 16000
    
    # Paths
    save_path: str = "./Wav2Vec2_mispronunciation_detector"
    
    # Data size (just for testing; set to None to use full dataset)
    train_size: int = 6
    total_samples: int = 10
    
    def to_dict(self):
        return asdict(self)

# Usage example
if __name__ == "__main__":
    config = Config()
    config_dict = config.to_dict()
    print(config_dict)