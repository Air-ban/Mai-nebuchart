from pathlib import Path
import torch

all_ids = []
for p in Path('data').rglob('*.tok.pt'):
    all_ids.extend([i for frm in torch.load(p)['ids'] for i in frm])
vocab = {'PAD': 0, 'BOS': 1, 'EOS': 2}
for tok in set(all_ids):
    if tok not in vocab: vocab[tok] = len(vocab)
torch.save(vocab, 'data/vocab.pt')
print('vocab size:', len(vocab))

