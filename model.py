import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_size)
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        # Cross-attention if context is provided
        if context is not None:
            attn_out, attn_weights = self.attn(query=x, key=context, value=context)
        else:
            attn_out, attn_weights = self.attn(query=x, key=x, value=x)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x, attn_weights

class Wav2Vec2MispronunciationDetector(nn.Module):
    def __init__(self, vocab_size, pad_token_id, blank_token_id, config, num_transformer_layers=2):
        super().__init__()

        # Wav2Vec2 encoder
        self.encoder = Wav2Vec2Model.from_pretrained(config.pretrained_model)
        self.encoder.feature_extractor._freeze_parameters()
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Phoneme embeddings + small transformer encoder
        self.phoneme_embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.phoneme_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=num_transformer_layers
        )

        # Audio-phoneme fusion Transformer blocks
        self.fusion_blocks = nn.ModuleList([
            TransformerBlock(hidden_size=config.hidden_size, num_heads=8, ff_dim=config.hidden_size*4)
            for _ in range(num_transformer_layers)
        ])

        # Projection + dropout
        self.projector = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)
        self.dropout = nn.Dropout(0.1)

        # Final classifier
        self.classifier = nn.Linear(config.hidden_size, vocab_size)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=blank_token_id, zero_infinity=True)
        self.pad_token_id = pad_token_id

    def forward(self, input_values, phoneme_ref_ids, attention_mask=None):
        # 1. Audio features
        audio_features = self.encoder(input_values, attention_mask=attention_mask).last_hidden_state
        audio_features = F.layer_norm(audio_features, audio_features.shape[-1:])

        # 2. Phoneme embeddings
        phoneme_embed = self.phoneme_embedding(phoneme_ref_ids)
        phoneme_embed = self.phoneme_encoder(phoneme_embed)

        # 3. Cross-attention fusion
        attn_weights_list = []
        x = audio_features
        for block in self.fusion_blocks:
            x, attn_weights = block(x, context=phoneme_embed)
            attn_weights_list.append(attn_weights)

        # 4. Projection + dropout
        x = self.dropout(F.relu(self.projector(x)))

        # 5. Classifier
        logits = self.classifier(x)

        return {
            'logits': logits,
            'attention_weights': attn_weights_list
        }
