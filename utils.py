"""Phoneme vocabulary and alignment helpers."""
import numpy as np


class Vocab:
    """The 68 IqraEval phones (sws_arabic.txt) + the CTC blank as the last id."""

    def __init__(self, vocab_file):
        with open(vocab_file, encoding="utf-8") as f:
            self.phones = [l.strip("\r\n") for l in f if l.strip()]
        self.idx = {p: i for i, p in enumerate(self.phones)}
        self.blank = len(self.phones)
        self.size = len(self.phones) + 1

    def encode(self, text):
        """Space-separated phones -> ids. <sil> and unknown symbols are dropped."""
        return [self.idx[p] for p in text.replace("<sil>", " ").split() if p in self.idx]

    def decode(self, ids):
        return [self.phones[i] for i in ids]


def greedy_ctc(frame_ids, length, blank):
    """Collapse repeats and drop blanks."""
    out, prev = [], None
    for t in frame_ids[:length]:
        if t != prev and t != blank:
            out.append(int(t))
        prev = t
    return out


def substitutions(ref, label):
    """Levenshtein-align ref -> label (id lists). Returns [(label_index, ref_id)] for every
    position where the speaker said label[j] instead of the reference phone ref_id."""
    n, m = len(ref), len(label)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (ref[i - 1] != label[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    subs, i, j = [], n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != label[j - 1]):
            if ref[i - 1] != label[j - 1]:
                subs.append((j - 1, ref[i - 1]))
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return subs[::-1]


def ctc_forced_align(logp, target, blank):
    """Viterbi CTC alignment. logp: [T, C] log-probs (numpy), target: label ids.
    Returns a list (one per target position) of the frames assigned to that token,
    or None if the target cannot fit in T frames."""
    T, L = logp.shape[0], len(target)
    if L == 0:
        return []
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = target
    # a state may skip the preceding blank if it is a token different from the token two back
    can_skip = np.zeros(S, dtype=bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]
    NEG = -1e30
    alpha = np.full(S, NEG)
    alpha[0] = logp[0, blank]
    alpha[1] = logp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, T):
        stay = alpha
        prev1 = np.concatenate(([NEG], alpha[:-1]))
        prev2 = np.where(can_skip, np.concatenate(([NEG, NEG], alpha[:-2])), NEG)
        stack = np.stack([stay, prev1, prev2])
        back[t] = stack.argmax(0)
        alpha = stack.max(0) + logp[t, ext]
    end = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    if alpha[end] <= NEG / 2:
        return None
    frames = [[] for _ in range(L)]
    s = end
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            frames[s // 2].append(t)
        s -= int(back[t, s]) if t > 0 else 0
    return [f[::-1] for f in frames]
