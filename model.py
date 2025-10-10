import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class Wav2Vec2MispronunciationDetector(nn.Module):
    def __init__(self, vocab_size, pad_token_id, blank_token_id, config):
        super().__init__()

        # Pretrained wav2vec2 encoder
        self.encoder = Wav2Vec2Model.from_pretrained(config.pretrained_model)
        self.encoder.feature_extractor._freeze_parameters()  # freeze conv layers

        # Freeze all encoder parameters (optional)
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Phoneme embedding + phoneme BiLSTM encoder
        self.phoneme_embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.phoneme_encoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            batch_first=True
        )

        # Layer norms
        self.ln_audio = nn.LayerNorm(config.hidden_size)
        self.ln_cross = nn.LayerNorm(config.hidden_size)

        # Projection layer + dropout
        self.projector = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)
        self.dropout = nn.Dropout(p=0.1)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Classifier
        self.classifier = nn.Linear(config.lstm_hidden_size * 2, vocab_size)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=blank_token_id, zero_infinity=True)
        self.pad_token_id = pad_token_id

    def _init_weights(self):
        """Initialize weights for better training stability"""
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_values, phoneme_ref_ids, attention_mask=None):
        """
        Args:
            input_values: Audio features [batch, time]
            phoneme_ref_ids: Target phoneme IDs [batch, seq_len]
            attention_mask: Audio attention mask [batch, time]
        Returns:
            logits: [batch, time, vocab_size]
            attention_weights: cross-attention weights
        """

        # 1. Extract audio features
        audio_features = self.encoder(
            input_values,
            attention_mask=attention_mask
        ).last_hidden_state  # [batch, time, hidden]
        audio_features = self.ln_audio(audio_features)

        # 2. Embed and contextualize phonemes
        phoneme_embed = self.phoneme_embedding(phoneme_ref_ids)
        phoneme_embed, _ = self.phoneme_encoder(phoneme_embed)  # [batch, seq_len, hidden]

        # 3. Cross-attention: audio attends to phonemes
        attended_features, attention_weights = self.cross_attention(
            query=audio_features,
            key=phoneme_embed,
            value=phoneme_embed
        )

        # 4. Residual + layer norm
        x = self.ln_cross(audio_features + attended_features)

        # 5. Project + dropout
        x = self.dropout(F.relu(self.projector(x)))

        # 6. BiLSTM
        x, _ = self.bilstm(x)

        # 7. Classifier
        logits = self.classifier(x)

        output = {
            'logits': logits,
            'attention_weights': attention_weights
        }

        return output
