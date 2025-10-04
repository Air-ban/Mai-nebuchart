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

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning"""
    def __init__(self, cond_dim, d_model):
        super().__init__()
        self.scale_proj = nn.Linear(cond_dim, d_model)
        self.shift_proj = nn.Linear(cond_dim, d_model)
        
    def forward(self, x, cond):
        # cond: (B, cond_dim) -> (B, 1, d_model) for broadcasting
        scale = self.scale_proj(cond).unsqueeze(1)
        shift = self.shift_proj(cond).unsqueeze(1)
        return x * (1 + scale) + shift

class MultiScaleFeatureProcessor(nn.Module):
    """Process features at multiple temporal scales"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Process different temporal scales
        self.scale_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.scale_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, T, d_model)
        # Apply temporal conv to capture multi-scale patterns
        x_permuted = x.permute(0, 2, 1)  # (B, d_model, T)
        x_scaled = self.scale_conv(x_permuted)  # Apply conv across time
        x_scaled = x_scaled.permute(0, 2, 1)  # (B, T, d_model)
        x = x + x_scaled  # Residual connection
        return self.scale_norm(x)

class ChartTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.feat_enc = nn.Linear(512, d_model)
        self.diff_emb = nn.Embedding(5, 128)  # 0-4 难度
        self.diff_proj = nn.Linear(128, d_model * 2)  # 为FiLM准备
        
        # Multi-scale processor for audio features
        self.multi_scale_processor = MultiScaleFeatureProcessor(d_model)
        
        # FiLM conditioning layers for each transformer layer
        self.film_layers = nn.ModuleList([
            FiLMLayer(d_model, d_model) for _ in range(num_layers)
        ])
        
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, 
                                               norm_first=True)  # Use pre-norm
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True,
                                               norm_first=True)  # Use pre-norm
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        
        self.head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize embedding and linear layers
        nn.init.normal_(self.tgt_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.feat_enc.weight)
        nn.init.constant_(self.feat_enc.bias, 0)
        nn.init.xavier_uniform_(self.diff_proj.weight)
        nn.init.constant_(self.diff_proj.bias, 0)

    def forward(self, feat, tgt, diff_idx, tgt_mask=None):
        batch_size = feat.size(0)
        
        # Process audio features
        feat_emb = self.feat_enc(feat)  # (B, T, d_model)
        
        # Process difficulty conditioning
        diff_emb = self.diff_emb(diff_idx)  # (B, 128)
        diff_proj = self.diff_proj(diff_emb)  # (B, d_model * 2)
        
        # Apply multi-scale processing to audio features
        feat_emb = self.multi_scale_processor(feat_emb)
        
        # Apply positional encoding
        feat_emb = self.pos_enc(feat_emb)
        
        # Encode audio features with difficulty conditioning via FiLM
        memory = feat_emb
        for i, film_layer in enumerate(self.film_layers):
            # Apply FiLM conditioning
            memory = film_layer(memory, diff_proj)
            # Pass through transformer layer
            if i == 0:
                memory = self.encoder.layers[i](memory, src_key_padding_mask=None)
            else:
                memory = self.encoder.layers[i](memory)
        
        # Apply layer norm after encoder
        memory = self.encoder.norm(memory)
        
        # Process target sequence
        tgt_emb = self.tgt_emb(tgt)  # (B, S, d_model)
        tgt_emb = self.pos_enc(tgt_emb)
        
        # Decode with cross-attention to memory
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        # Generate final output
        logits = self.head(output)
        return logits

