"""Reference-anchored mispronunciation detector (v2).

v1 let audio frames query the reference without position information, so attention
was content-based and nearly uniform and the reference acted as noise. v2 inverts this:
each reference phone, with its position, queries the audio through a transformer
decoder and predicts what was actually said in its slot. "Same as reference" becomes
a single class whose probability can be thresholded (precision/recall knob).

Heads
  slot : per reference phone -> one of 68 phones or DEL
  ins  : per gap (before first, between, after last) -> up to K inserted phones, in order,
         terminated by NONE (NONE is the "stop / no more insertions" marker)
  ctc  : auxiliary free phoneme recognition on the audio (keeps the encoder grounded)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model


def sinusoid(n, d, device):
    """Fixed sinusoidal positions for audio frames (no maximum length)."""
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class RefAnchoredMDD(nn.Module):
    def __init__(self, vocab, encoder_name="elgeish/wav2vec2-large-xlsr-53-arabic", d_model=512,
                 n_layers=4, n_heads=8, max_ref_len=512, dropout=0.1, ctc_weight=0.3,
                 grad_checkpointing=True, encoder_config_only=False, max_ins=3):
        super().__init__()
        self.v = vocab
        if encoder_config_only:  # for loading a fine-tuned checkpoint without downloading weights twice
            self.encoder = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(encoder_name))
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(encoder_name)
        self.encoder.config.mask_time_prob = 0.05  # SpecAugment-style masking while training
        self.encoder.config.layerdrop = 0.0         # LayerDrop leaves params without grads -> DDP crash
        self.encoder.feature_extractor._freeze_parameters()
        if grad_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        H = self.encoder.config.hidden_size

        self.mem_proj = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, d_model))
        self.d_model = d_model
        self.ref_emb = nn.Embedding(vocab.n_input, d_model, padding_idx=vocab.PAD)
        self.ref_pos = nn.Embedding(max_ref_len, d_model)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers, norm=nn.LayerNorm(d_model))
        self.slot_head = nn.Linear(d_model, vocab.n_phones + 1)
        self.max_ins = max_ins
        self.ins_head = nn.Linear(d_model, max_ins * (vocab.n_phones + 1))
        self.ctc_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(H, vocab.n_phones + 1))
        self.ctc_weight = ctc_weight

    def forward(self, wave, wave_mask, ref, slot=None, ins=None, ctc=None):
        enc = self.encoder(wave, attention_mask=wave_mask).last_hidden_state          # [B, T, H]
        frame_len = self.encoder._get_feat_extract_output_lengths(wave_mask.sum(-1))
        T = enc.shape[1]
        frame_mask = torch.arange(T, device=enc.device)[None] < frame_len[:, None]   # True = valid

        mem = self.mem_proj(enc) + sinusoid(T, self.d_model, enc.device).to(enc.dtype)[None]
        S1 = ref.shape[1]
        tgt = self.ref_emb(ref) + self.ref_pos(torch.arange(S1, device=ref.device))[None]
        h = self.decoder(tgt, mem, tgt_key_padding_mask=ref.eq(self.v.PAD),
                         memory_key_padding_mask=~frame_mask)                         # [B, S+1, d]
        out = {"slot_logits": self.slot_head(h[:, 1:]), "ins_logits": self.ins_head(h).unflatten(-1, (self.max_ins, self.v.n_phones + 1)),  # [B,S+1,K,C]
               "ctc_logits": self.ctc_head(enc), "frame_len": frame_len}

        if slot is not None:
            l_slot = F.cross_entropy(out["slot_logits"].flatten(0, 1).float(), slot.flatten(), ignore_index=-100)
            l_ins = F.cross_entropy(out["ins_logits"].flatten(0, 2).float(), ins.flatten(), ignore_index=-100)
            logp = F.log_softmax(out["ctc_logits"].float(), -1).transpose(0, 1)
            tgt_len = (ctc >= 0).sum(-1)
            l_ctc = F.ctc_loss(logp, ctc.clamp_min(0), frame_len, tgt_len, blank=self.v.BLANK,
                               zero_infinity=True)
            out.update(loss=l_slot + 0.5 * l_ins + self.ctc_weight * l_ctc,
                       l_slot=l_slot.detach(), l_ins=l_ins.detach(), l_ctc=l_ctc.detach())
        return out


@torch.no_grad()
def decode(out, ref, vocab, tau=0.5, tau_ins=0.5):
    """Keep the reference phone unless its probability falls below tau. In each gap, start
    inserting only if P(NONE) at the first insertion position falls below tau_ins, then keep
    taking the argmax until NONE (the stop marker) or K phones. Returns lists of phones."""
    p_slot = out["slot_logits"].float().softmax(-1)
    p_ins = out["ins_logits"].float().softmax(-1)
    preds = []
    for b in range(ref.shape[0]):
        r = ref[b, 1:]
        S = int((r != vocab.PAD).sum())
        seq = []
        for i in range(S + 1):
            pi = p_ins[b, i]                                   # [K, C]
            if pi[0, vocab.NONE] < tau_ins:
                q = pi[0].clone(); q[vocab.NONE] = -1
                seq.append(vocab.phones[int(q.argmax())])
                for k in range(1, pi.shape[0]):
                    c = int(pi[k].argmax())
                    if c == vocab.NONE:
                        break
                    seq.append(vocab.phones[c])
            if i < S:
                rid = int(r[i])
                ps = p_slot[b, i]
                if rid < vocab.n_phones and ps[rid] >= tau:
                    seq.append(vocab.phones[rid])
                else:
                    q = ps.clone()
                    if rid < vocab.n_phones:
                        q[rid] = -1
                    c = int(q.argmax())
                    if c != vocab.DEL:
                        seq.append(vocab.phones[c])
        preds.append(seq)
    return preds
