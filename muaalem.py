"""Run the Quran-pretrained muaalem model (obadx/muaalem-model-v3_2) without its package.

Architecture (from quran-muaalem 0.3.0, modeling_multi_level_ctc.py): a standard
Wav2Vec2BertModel (w2v-bert-2.0 config with the adapter) followed by one linear CTC head
per level on the same frames; blank = [PAD] = id 0. Levels: `phonemes` (43 Quran Phonetic
Script symbols) and 10 sifat (articulation attributes). Each sifa level emits one label per
*phoneme group*, so the i-th sifa label belongs to the i-th group of the phoneme output.
"""
import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2BertModel

REPO = "obadx/muaalem-model-v3_2"
BASE = "facebook/w2v-bert-2.0"


class Muaalem(nn.Module):
    def __init__(self, repo=REPO):
        super().__init__()
        cfg = json.load(open(hf_hub_download(repo, "config.json"), encoding="utf-8"))
        self.encoder = Wav2Vec2BertModel(AutoConfig.from_pretrained(BASE, add_adapter=True, layerdrop=0.0))
        self.heads = nn.ModuleDict({lvl: nn.Linear(1024, n) for lvl, n in cfg["level_to_vocab_size"].items()})
        sd = load_file(hf_hub_download(repo, "model.safetensors"))
        enc = {k.removeprefix("wav2vec2_bert."): v for k, v in sd.items() if k.startswith("wav2vec2_bert.")}
        missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
        assert not unexpected and set(missing) <= {"masked_spec_embed"}, (missing, unexpected)
        self.heads.load_state_dict({k.removeprefix("level_to_lm_head."): v for k, v in sd.items()
                                    if k.startswith("level_to_lm_head.")})
        vocab = json.load(open(hf_hub_download(repo, "vocab.json"), encoding="utf-8"))
        self.id2tok = {lvl: {i: t for t, i in d.items()} for lvl, d in vocab.items()}
        self.extractor = AutoFeatureExtractor.from_pretrained(BASE)

    def forward(self, input_features, attention_mask):
        h = self.encoder(input_features, attention_mask=attention_mask).last_hidden_state
        return {lvl: head(h) for lvl, head in self.heads.items()}

    @torch.no_grad()
    def transcribe(self, wave, device):
        """-> {level: [(token, prob), ...]} after greedy CTC decoding (one utterance)."""
        f = self.extractor([wave], sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        out = self(f["input_features"].to(device, next(self.parameters()).dtype), f["attention_mask"].to(device))
        res = {}
        for lvl, logits in out.items():
            p = logits[0].float().softmax(-1)
            best, ids = p.max(-1)
            seq, prev = [], None
            for t, (i, pr) in enumerate(zip(ids.tolist(), best.tolist())):
                if i != prev and i != 0:
                    seq.append((self.id2tok[lvl][i], pr))
                prev = i
            res[lvl] = seq
        return res


# ------------------------------------------------------------------ QPS -> IqraEval phonemes
# Symbol sets from quran-transcript 0.6.4 (alphabet.py, phonetic_groups), read as text.
CONSONANT = {"ء": "<", "ب": "b", "ت": "t", "ث": "^", "ج": "j", "ح": "H", "خ": "x", "د": "d", "ذ": "*",
             "ر": "r", "ز": "z", "س": "s", "ش": "$", "ص": "S", "ض": "D", "ط": "T", "ظ": "Z", "ع": "E",
             "غ": "g", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w",
             "ي": "y",
             "ٲ": "<",   # hamza mosahala (eased hamza)
             "ں": "n",   # noon mokhfah (nasalised noon)
             "۾": "m"}   # meem mokhfah
MADD = {"ا": "a", "ۥ": "u", "ۦ": "i", "ـ": "a"}   # long-vowel cores (alif, small waw/yaa, alif momala)
SHORT = {"َ": "a", "ُ": "u", "ِ": "i", "ؙ": "u", "۪": "a"}   # fatha, damma, kasra, dama mokhtalasa, fatha momala
IGNORED = {"ڇ", "ۜ"}                                             # qalqala and sakt marks (no IqraEval symbol)
CORE = "".join(CONSONANT) + "".join(MADD)
RESID = "".join(SHORT) + "".join(IGNORED)


def chunk(qps):
    """quran_transcript.chunck_phonemes: a run of one core symbol + at most one residual."""
    import re
    core = "|".join(f"{re.escape(c)}+" for c in CORE)
    return re.findall(f"((?:{core})[{re.escape(RESID)}]?)", qps)


def to_iqraeval(qps, tafkheem=None, emphasis="head_no_raa"):
    """Convert a Quran Phonetic Script string (+ per-group tafkheem labels) to IqraEval phones.

    emphasis: "none"        never emphatic vowels
              "head"        vowel of a group the tafkheem head calls mofakham -> A / U / I / AA ...
              "head_no_raa" same, but not after raa (IqraEval does not mark vowels after r as emphatic)
    """
    out = []
    groups = chunk(qps)
    for g_idx, g in enumerate(groups):
        core, n, res = g[0], len(g.rstrip(RESID)), (g[-1] if g[-1] in RESID else "")
        heavy = False
        if tafkheem is not None and g_idx < len(tafkheem) and emphasis != "none":
            heavy = tafkheem[g_idx] in ("[مفخم]", "[أدنى المفخم]")
            if emphasis == "head_no_raa":
                prev_core = groups[g_idx - 1][0] if (core in MADD and g_idx > 0) else core
                heavy = heavy and prev_core != "ر"
        if core in CONSONANT:
            sym = CONSONANT[core]
            out.append(sym * 2 if n >= 2 else sym)        # doubled letter = gemination
            if res in SHORT:
                v = SHORT[res]
                out.append(v.upper() if heavy else v)
        elif core in MADD:                                # long vowel: lengthen the preceding short vowel
            v = MADD[core]
            if out and out[-1].lower() == v:
                out.pop()
            out.append((v * 2).upper() if heavy else v * 2)
    return out
