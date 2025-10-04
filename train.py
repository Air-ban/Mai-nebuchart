import torch, os, math
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from model import ChartTransformer

# ---------- 超参 ----------
MAX_LEN   = 1024
BATCH_SZ  = 16  # Reduce batch size due to more complex model
EPOCHS    = 50
LR        = 3e-4
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)

# ---------- 数据集（同音频 5 难度） ----------
class ChartDataset(Dataset):
    def __init__(self, split='train'):
        self.feat_files = sorted(Path('data', split).glob('*.bpm32.pt'))
        self.vocab      = torch.load('data/vocab.pt')
    def __len__(self): return len(self.feat_files) * 5   # 5 难度
    def __getitem__(self, idx):
        file_idx = idx // 5
        diff     = idx % 5          # 0-4
        d_feat   = torch.load(self.feat_files[file_idx])
        d_tok    = torch.load(self.feat_files[file_idx].with_suffix('.tok.pt'))
        feat = d_feat['feat'][:MAX_LEN]                  # (T,512)
        ids  = [i for frm in d_tok['ids'][diff] for i in frm][:MAX_LEN-2]
        ids  = [1] + ids + [2]                           # BOS/EOS
        tgt_in, tgt_out = torch.tensor(ids[:-1]), torch.tensor(ids[1:])
        return feat, tgt_in, tgt_out, torch.tensor(diff)

def pad_collate(batch):
    feat, tgt_in, tgt_out, diff = zip(*batch)
    feat = torch.nn.utils.rnn.pad_sequence(feat, batch_first=True)
    tgt_in  = torch.nn.utils.rnn.pad_sequence(tgt_in,  batch_first=True, padding_value=0)
    tgt_out = torch.nn.utils.rnn.pad_sequence(tgt_out, batch_first=True, padding_value=0)
    diff = torch.stack(diff)
    return feat, tgt_in, tgt_out, diff

# ---------- 训练 & 验证 ----------
def run_epoch(model, loader, optimizer=None):
    model.train() if optimizer else model.eval()
    total_loss, n = 0, 0
    for feat, tgt_in, tgt_out, diff in loader:
        feat, tgt_in, tgt_out, diff = feat.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE), diff.to(DEVICE)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(DEVICE)
        logits = model(feat, tgt_in, diff, mask)
        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(
            logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * feat.size(0)
        n += feat.size(0)
    return total_loss / n

def main():
    os.makedirs('ckpt', exist_ok=True)
    vocab = torch.load('data/vocab.pt')
    train_ds = ChartDataset('train')
    val_ds   = ChartDataset('val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SZ, shuffle=True,
                              num_workers=4, collate_fn=pad_collate)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SZ, shuffle=False,
                              num_workers=4, collate_fn=pad_collate)

    model = ChartTransformer(len(vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)  # Using AdamW with weight decay

    for epoch in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader, None)
        print(f'epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}')
        
        # Save best model based on validation loss
        if epoch == 1 or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'ckpt/chart_best.pth')
            print(f'Saved best model with val_loss={val_loss:.4f}')
    
    # Save final model
    torch.save(model.state_dict(), 'ckpt/chart_final.pth')
    print('saved ckpt/chart_final.pth')

if __name__ == '__main__':
    main()

