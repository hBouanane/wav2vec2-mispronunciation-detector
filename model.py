"""Plain CTC phoneme recogniser: audio in, phonemes out. The reference is NOT a model input;
mispronunciations are found afterwards by comparing the output with the reference
(official IqraEval scoring).

Supported encoders (chosen by their config's model_type):
  wav2vec2       e.g. elgeish/wav2vec2-large-xlsr-53-arabic, facebook/wav2vec2-xls-r-300m
  wav2vec2-bert  e.g. facebook/w2v-bert-2.0

With --target qps the Quran-pretrained muaalem model keeps its own phoneme and tafkheem heads
(MuaalemCTC); IqraEval labels are converted to its Quran Phonetic Script for training.
"""
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoFeatureExtractor, Wav2Vec2BertForCTC, Wav2Vec2BertModel, Wav2Vec2ForCTC)

from utils import ctc_forced_align, greedy_ctc, substitutions

CLASSES = {"wav2vec2": Wav2Vec2ForCTC, "wav2vec2-bert": Wav2Vec2BertForCTC}

# Encoders published with a custom architecture whose encoder is a standard w2v-BERT 2.0:
# we build Wav2Vec2BertForCTC from the base config and copy only the encoder weights.
#   obadx/muaalem-model-v3_2: w2v-bert-2.0 fine-tuned on 850+ h of Quran recitation
#   (Wav2Vec2BertForMultilevelCTC, MIT; arXiv:2509.00094). Its 22 head tensors
#   (level_to_lm_head.*) are dropped; all 794 encoder tensors match by name and shape.
W2VBERT_DERIVED = {"obadx/muaalem-model-v3_2": "facebook/w2v-bert-2.0"}


def _load_derived_encoder(model, repo):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = {k: v for k, v in load_file(hf_hub_download(repo, "model.safetensors")).items()
          if k.startswith("wav2vec2_bert.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("lm_head.") and k != "wav2vec2_bert.masked_spec_embed"]
    if missing or unexpected:
        raise RuntimeError(f"{repo}: encoder weights do not fit (missing {missing[:5]}, unexpected {unexpected[:5]})")
    print(f"loaded {len(sd)} encoder tensors from {repo}")


def freeze_lower_layers(model, n):
    """Freeze the input projection and the first n transformer layers (saves optimizer memory)."""
    base = getattr(model, "wav2vec2_bert", None) or getattr(model, "wav2vec2")
    for p in base.feature_projection.parameters():
        p.requires_grad_(False)
    for layer in base.encoder.layers[:n]:
        for p in layer.parameters():
            p.requires_grad_(False)


class MuaalemCTC(nn.Module):
    """muaalem's encoder + its own `phonemes` (Quran Phonetic Script) and `tafkheem_or_taqeeq`
    heads, both initialised from the pretrained model and trained with CTC on IqraEval labels
    converted by muaalem.from_iqraeval. Output -> IqraEval phonemes via muaalem.to_iqraeval.
    Parameter names follow the original checkpoint (wav2vec2_bert.*, level_to_lm_head.*)."""

    LEVELS = ("phonemes", "tafkheem_or_taqeeq")

    def __init__(self, cfg, pretrained=True):
        super().__init__()
        from muaalem import QPSVocab
        self.qps = QPSVocab()
        enc_cfg = AutoConfig.from_pretrained("facebook/w2v-bert-2.0", add_adapter=True, layerdrop=0.0,
                                             mask_time_prob=cfg.mask_time_prob,
                                             apply_spec_augment=cfg.mask_time_prob > 0)
        self.wav2vec2_bert = Wav2Vec2BertModel(enc_cfg)
        self.dropout = nn.Dropout(cfg.final_dropout)
        self.level_to_lm_head = nn.ModuleDict({lvl: nn.Linear(enc_cfg.hidden_size, self.qps.size[lvl])
                                               for lvl in self.LEVELS})
        self.weights = {"phonemes": 1.0, "tafkheem_or_taqeeq": cfg.tafkheem_weight}
        if pretrained:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            sd = load_file(hf_hub_download("obadx/muaalem-model-v3_2", "model.safetensors"))
            sd = {k: v for k, v in sd.items() if k.startswith("wav2vec2_bert.")
                  or any(k.startswith(f"level_to_lm_head.{lvl}.") for lvl in self.LEVELS)}
            missing, unexpected = self.load_state_dict(sd, strict=False)
            if unexpected or set(missing) - {"wav2vec2_bert.masked_spec_embed"}:
                raise RuntimeError(f"muaalem weights do not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
            print(f"loaded {len(sd)} muaalem tensors (encoder + {', '.join(self.LEVELS)} heads)")

    def gradient_checkpointing_enable(self):
        self.wav2vec2_bert.gradient_checkpointing_enable()

    def _get_feat_extract_output_lengths(self, input_lengths):
        return self.wav2vec2_bert._get_feat_extract_output_lengths(input_lengths)

    def forward(self, input_features, attention_mask=None, labels=None):
        h = self.dropout(self.wav2vec2_bert(input_features, attention_mask=attention_mask).last_hidden_state)
        logits = {lvl: head(h) for lvl, head in self.level_to_lm_head.items()}
        loss = None
        if labels is not None:
            lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).long()
            loss = 0.0
            for lvl, y in labels.items():
                logp = F.log_softmax(logits[lvl].float(), -1).transpose(0, 1)
                loss = loss + self.weights[lvl] * F.ctc_loss(
                    logp, y.clamp_min(0), lengths, (y >= 0).sum(-1), blank=0, reduction="mean", zero_infinity=True)
        return SimpleNamespace(loss=loss, logits=logits)


def build_model(cfg, vocab, pretrained=True):
    if cfg.target == "qps":
        if cfg.encoder != "obadx/muaalem-model-v3_2":
            raise ValueError("--target qps needs --encoder obadx/muaalem-model-v3_2")
        model = MuaalemCTC(cfg, pretrained)
        if cfg.freeze_layers > 0:
            freeze_lower_layers(model, cfg.freeze_layers)
        if cfg.grad_checkpointing:
            model.gradient_checkpointing_enable()
        return model, AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    base_name = W2VBERT_DERIVED.get(cfg.encoder, cfg.encoder)
    enc_cfg = AutoConfig.from_pretrained(base_name)
    cls = CLASSES.get(enc_cfg.model_type)
    if cls is None:
        raise ValueError(f"unsupported encoder type {enc_cfg.model_type}")
    kw = dict(vocab_size=vocab.size, pad_token_id=vocab.blank, ctc_loss_reduction="mean", ctc_zero_infinity=True,
              layerdrop=0.0,  # LayerDrop leaves parameters without gradients -> DDP crash
              mask_time_prob=cfg.mask_time_prob, final_dropout=cfg.final_dropout,
              # w2v-bert-2.0 ships with apply_spec_augment=False: with mask_time_prob > 0 the model would
              # create masked_spec_embed but never use it, and DDP stops on the unused parameter
              apply_spec_augment=cfg.mask_time_prob > 0)
    if enc_cfg.model_type == "wav2vec2-bert":
        kw["add_adapter"] = True
    if pretrained and cfg.encoder in W2VBERT_DERIVED:
        model = cls(AutoConfig.from_pretrained(base_name, **kw))
        _load_derived_encoder(model, cfg.encoder)
    elif pretrained:
        model = cls.from_pretrained(cfg.encoder, ignore_mismatched_sizes=True, **kw)
    else:  # architecture only (weights come from our own checkpoint)
        model = cls(AutoConfig.from_pretrained(base_name, **kw))
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    if cfg.freeze_layers > 0:
        freeze_lower_layers(model, cfg.freeze_layers)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
    extractor = AutoFeatureExtractor.from_pretrained(base_name)  # muaalem uses the base preprocessing unchanged
    return model, extractor


def frame_lengths(model, attention_mask):
    m = model.module if hasattr(model, "module") else model
    return m._get_feat_extract_output_lengths(attention_mask.sum(-1)).long()


@torch.no_grad()
def greedy_decode(logits, lengths, vocab, model=None):
    """-> list of IqraEval phone lists. `logits` is a tensor (IqraEval CTC head) or, for
    MuaalemCTC, a dict of level -> logits decoded in Quran Phonetic Script and converted."""
    if isinstance(logits, dict):
        from muaalem import to_iqraeval
        m = model.module if hasattr(model, "module") else model
        ph = logits["phonemes"].argmax(-1).cpu().numpy()
        tf = logits["tafkheem_or_taqeeq"].argmax(-1).cpu().numpy()
        out = []
        for b in range(len(ph)):
            L = int(lengths[b])
            qps = "".join(m.qps.id2tok["phonemes"][i] for i in greedy_ctc(ph[b], L, 0))
            taf = [m.qps.id2tok["tafkheem_or_taqeeq"][i] for i in greedy_ctc(tf[b], L, 0)]
            out.append(to_iqraeval(qps, taf, emphasis="head"))
        return out
    ids = logits.argmax(-1).cpu().numpy()
    return [vocab.decode(greedy_ctc(ids[b], int(lengths[b]), vocab.blank)) for b in range(len(ids))]


def reference_penalty(logits, lengths, labels, refs, blank):
    """Unlikelihood penalty against 'imagining' the reference.

    For every substitution (speaker said X where the reference has Y), the label is
    force-aligned to the frames, and on the frames of X we add -log(1 - p(Y)).
    Returns (mean penalty over penalised frames, number of substitutions)."""
    logp = F.log_softmax(logits.float(), -1)
    terms, n_subs = [], 0
    for b in range(logits.shape[0]):
        lab = labels[b][labels[b] >= 0].tolist()
        ref = refs[b][refs[b] >= 0].tolist()
        if not ref or lab == ref:
            continue
        subs = substitutions(ref, lab)
        if not subs:
            continue
        T = int(lengths[b])
        frames = ctc_forced_align(logp[b, :T].detach().cpu().numpy(), lab, blank)
        if frames is None:
            continue
        for j, y in subs:
            if frames[j]:
                p_y = logp[b, frames[j], y].exp().clamp(max=1 - 1e-4)
                terms.append(-torch.log1p(-p_y))
                n_subs += 1
    if not terms:
        return logits.new_zeros(()), 0
    return torch.cat(terms).mean(), n_subs


def load_trained(cfg, vocab, state_dict_path, device):
    model, extractor = build_model(cfg, vocab, pretrained=False)
    sd = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
    return model.to(device).eval(), extractor


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

