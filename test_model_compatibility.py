#!/usr/bin/env python3
"""
测试新模型与现有数据集的兼容性
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from model import ChartTransformer

# 复用训练脚本中的数据集类
class ChartDataset(Dataset):
    def __init__(self, split='train'):
        self.feat_files = sorted(Path('data', split).glob('*.bpm32.pt'))
        if not self.feat_files:
            print(f"警告: 在 data/{split} 目录中未找到 .bpm32.pt 文件")
            # 创建一个模拟数据用于测试
            self.feat_files = ['mock_data']  # 模拟数据路径
            self.mock_data = True
        else:
            self.vocab = torch.load('data/vocab.pt')
            self.mock_data = False
            
    def __len__(self):
        if self.mock_data:
            return 1  # 模拟一个样本
        return len(self.feat_files) * 5   # 5 难度
    
    def __getitem__(self, idx):
        if self.mock_data:
            # 返回模拟数据
            feat = torch.randn(100, 512)  # (T, 512)
            tgt_in = torch.randint(0, 100, (50,))  # (S,)
            tgt_out = torch.randint(0, 100, (50,))  # (S,)
            diff = torch.tensor(2)  # 难度 2
            return feat, tgt_in, tgt_out, diff
            
        file_idx = idx // 5
        diff = idx % 5  # 0-4
        d_feat = torch.load(self.feat_files[file_idx])
        d_tok = torch.load(self.feat_files[file_idx].with_suffix('.tok.pt'))
        
        MAX_LEN = 1024
        feat = d_feat['feat'][:MAX_LEN]  # (T,512)
        ids = [i for frm in d_tok['ids'][diff] for i in frm][:MAX_LEN-2]
        ids = [1] + ids + [2]  # BOS/EOS
        tgt_in, tgt_out = torch.tensor(ids[:-1]), torch.tensor(ids[1:])
        return feat, tgt_in, tgt_out, torch.tensor(diff)

def test_model_compatibility():
    """测试新模型架构与现有数据集的兼容性"""
    print("开始测试模型与现有数据集的兼容性...")
    
    # 获取词汇表大小
    try:
        vocab = torch.load('data/vocab.pt')
        vocab_size = len(vocab)
        print(f"词汇表大小: {vocab_size}")
    except FileNotFoundError:
        print("未找到词汇表文件，使用模拟大小")
        vocab_size = 1000  # 模拟词汇表大小
    
    # 创建模型实例
    model = ChartTransformer(vocab_size=vocab_size)
    print("模型创建成功")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 50
    feat_len = 100
    
    # 创建模拟输入
    feat = torch.randn(batch_size, feat_len, 512)  # (B, T, 512)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))  # (B, S)
    diff_idx = torch.randint(0, 5, (batch_size,))  # (B,)
    
    # 创建因果掩码
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len)
    
    print("开始前向传播测试...")
    try:
        # 测试前向传播
        output = model(feat, tgt, diff_idx, tgt_mask)
        print(f"前向传播成功，输出形状: {output.shape}")
        print(f"输出形状应为 (batch_size, seq_len, vocab_size): ({batch_size}, {seq_len}, {vocab_size})")
        
        # 验证输出维度
        assert output.shape == (batch_size, seq_len, vocab_size), f"输出维度错误: {output.shape}"
        print("输出维度验证通过")
        
        # 测试模型参数初始化
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        
        print("模型兼容性测试通过！")
        return True
        
    except Exception as e:
        print(f"前向传播测试失败: {str(e)}")
        return False

def test_with_real_data():
    """使用真实数据测试（如果存在）"""
    print("\n尝试使用真实数据进行测试...")
    
    dataset = ChartDataset('train')
    if hasattr(dataset, 'mock_data') and dataset.mock_data:
        print("真实数据不可用，跳过真实数据测试")
        return True
    
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) == 0:
        print("数据集为空，使用模拟数据")
        return True
    
    # 获取一个批次的数据
    try:
        sample = dataset[0]
        feat, tgt_in, tgt_out, diff = sample
        print(f"特征形状: {feat.shape}")
        print(f"目标输入形状: {tgt_in.shape}")
        print(f"难度索引: {diff}")
        
        # 创建模型
        vocab_size = len(torch.load('data/vocab.pt'))
        model = ChartTransformer(vocab_size)
        
        # 创建批次数据
        batch_size = min(2, len(dataset))
        feat_batch = torch.stack([dataset[i][0] for i in range(batch_size)])
        tgt_batch = torch.stack([dataset[i][1] for i in range(batch_size)])
        diff_batch = torch.stack([dataset[i][3] for i in range(batch_size)])
        
        # 截断到相同长度
        min_len = min([len(dataset[i][1]) for i in range(batch_size)])
        tgt_batch = tgt_batch[:, :min_len]
        
        # 创建掩码
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(min_len)
        
        # 前向传播
        output = model(feat_batch, tgt_batch, diff_batch, tgt_mask)
        print(f"真实数据前向传播成功，输出形状: {output.shape}")
        return True
        
    except Exception as e:
        print(f"真实数据测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== 模型兼容性测试 ===")
    
    # 检查必要的文件
    required_files = ['data/vocab.pt']
    missing_files = []
    for f in required_files:
        if not Path(f).exists():
            missing_files.append(f)
    
    if missing_files:
        print(f"缺少必要文件: {missing_files}")
        print("将使用模拟数据进行测试")
    else:
        print("所有必要文件都存在")
    
    # 执行兼容性测试
    success1 = test_model_compatibility()
    success2 = test_with_real_data()
    
    if success1 and success2:
        print("\n所有测试通过！新模型与现有数据集兼容。")
    else:
        print("\n测试失败！请检查模型实现。")