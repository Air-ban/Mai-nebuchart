import math, torch, torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class ChartTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.feat_enc  = nn.Linear(512, d_model)
        self.diff_emb  = nn.Embedding(5, 128)        # 0-4 难度
        self.cond_proj = nn.Linear(128, d_model)     # 映射到模型维度
        self.pos_enc   = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers)
        self.tgt_emb   = nn.Embedding(vocab_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder   = nn.TransformerDecoder(dec_layer, num_layers)
        self.head      = nn.Linear(d_model, vocab_size)

    def forward(self, feat, tgt, diff_idx, tgt_mask=None):
        # diff_idx: (B,)  → (B,1,D) 加在每帧
        diff_vec = self.cond_proj(self.diff_emb(diff_idx)).unsqueeze(1)
        mem = self.encoder(self.pos_enc(self.feat_enc(feat) + diff_vec))
        tgt_emb = self.pos_enc(self.tgt_emb(tgt))
        out = self.decoder(tgt_emb, mem, tgt_mask=tgt_mask)
        return self.head(out)

