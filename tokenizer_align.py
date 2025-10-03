# tokenizer_align.py
import re, torch
from pathlib import Path

RE_NOTE = re.compile(r'\{(\d)\}([1-8]h\[(\d+):(\d+)\])')
RE_SLIDE = re.compile(r'([1-8])-([1-8])\[(\d+):(\d+)\]')
RE_HOLD = re.compile(r'([1-8])h\[(\d+):(\d+)\]')
RE_BRK  = re.compile(r'([1-8])b')
TOK_SPEC = {'q':'Q', 'p':'P', 'v':'V', '/':'SLASH', ',':'COMMA', 'b':'BRK'}

def tokenize_line(line: str) -> list[str]:
    tokens, i = [], 0
    while i < len(line):
        if line[i] == '{':
            m = RE_NOTE.match(line[i:])
            if m:
                track, hold, a, b = m.groups()
                tokens.append(f'C{track}_{hold}'); i += m.end(); continue
        elif line[i].isdigit() and i+1<len(line) and line[i+1]=='h':
            m = RE_HOLD.match(line[i:])
            if m:
                note, a, b = m.groups()
                tokens.append(f'H{note}_{a}:{b}'); i += m.end(); continue
        elif line[i].isdigit() and i+1<len(line) and line[i+1]=='-':
            m = RE_SLIDE.match(line[i:])
            if m:
                n1, n2, a, b = m.groups()
                tokens.append(f'S{n1}-{n2}_{a}:{b}'); i += m.end(); continue
        elif line[i] in TOK_SPEC:
            tokens.append(TOK_SPEC[line[i]]); i += 1; continue
        elif line[i] == ' ':
            i += 1; continue
        else:
            i += 1
    return tokens

def align_to_grid(chart_txt, bpm_pt_path, steps_per_beat=32):
    bpm_data = torch.load(bpm_pt_path)
    T = bpm_data['feat'].shape[0]
    beat_per_frame = 1 / steps_per_beat
    lines = [l.rstrip() for l in Path(chart_txt).read_text().splitlines() if l.strip()]
    frame_tokens = [[] for _ in range(T)]
    cur_beat = 0.0
    for line in lines:
        if line.startswith('&') or line == 'E': continue
        tail_comma = len(line) - len(line.rstrip(','))
        tokens = tokenize_line(line)
        frame_idx = int(round(cur_beat * steps_per_beat))
        if 0 <= frame_idx < T:
            frame_tokens[frame_idx].extend(tokens)
        cur_beat += (1 + tail_comma) * beat_per_frame
    for f in range(T):
        if not frame_tokens[f]:
            frame_tokens[f] = ['PAD']
    return frame_tokens

def build_vocab(all_tokens):
    vocab = {'PAD': 0, 'BOS': 1, 'EOS': 2}
    for frame in all_tokens:
        for tok in frame:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def encode_tokens(frame_tokens, vocab):
    return [[vocab[tok] for tok in frame] for frame in frame_tokens]

