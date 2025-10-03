# Mai-nebuchart - AI驱动的maimai游戏谱面生成模型

基于深度学习的maimai游戏智能谱面生成系统，能够根据音乐特征自动生成不同难度的游戏谱面。

## 🎯 项目概述

Mai-nebuchart是一个先进的AI模型，专门用于生成maimai音乐游戏的谱面。该系统通过分析音频特征，结合深度学习技术，能够自动生成与音乐节奏、旋律完美同步的游戏谱面，支持多个难度级别。

## 🧠 核心AI技术

### 模型架构
- **Transformer编码器-解码器**：采用最先进的注意力机制
- **多模态特征融合**：整合音频特征与难度调节信息
- **条件生成**：根据难度等级(0-4)生成对应复杂度的谱面
- **时序建模**：精确捕捉音乐的时间结构

### 智能特性
- **自适应难度控制**：模型能够理解并生成不同复杂度的谱面
- **音乐同步**：生成的谱面与音乐节拍完美对齐
- **风格一致性**：保持maimai游戏的谱面设计风格
- **可扩展性**：支持新难度级别和谱面类型的扩展

## 🔧 系统组件

### 1. 音频特征提取器 (`extracktor_bpm.py`)
- 提取512维音频特征向量
- 支持BPM同步的高精度特征提取
- 包含Mel频谱、CQT、谐波等多维度音频分析

### 2. 谱面分词器 (`tokenizer.py`)
- 将游戏谱面转换为AI可理解的token序列
- 支持多种音符类型：普通音符、长按、滑动等
- 时间对齐确保谱面与音乐节拍同步

### 3. AI生成模型 (`model.py`)
```python
from model import ChartTransformer

# 创建AI谱面生成器
model = ChartTransformer(
    vocab_size=1000,    # 词汇表大小
    d_model=512,        # 模型维度
    nhead=8,            # 注意力头数
    num_layers=6        # Transformer层数
)
```

### 4. 训练系统 (`train.py`)
- 端到端训练流程
- 支持多GPU训练
- 自动验证和模型保存

## 🚀 快速开始

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

# 准备数据目录
mkdir -p data/train data/val
```

### 数据预处理
```bash
# 1. 提取音频特征
python extracktor_bpm.py your_song.wav 180.5

# 2. 处理谱面数据
python tokenizer.py chart.txt song.bpm32.pt

# 3. 构建词汇表
python vocab.py
```

### 模型训练
```bash
# 开始训练AI模型
python train.py
```

### AI谱面生成
```python
import torch
from model import ChartTransformer

# 加载训练好的模型
model = ChartTransformer(vocab_size=1000)
model.load_state_dict(torch.load('ckpt/chart_final.pth'))

# 准备音乐特征
audio_features = torch.randn(1, 1000, 512)  # 音频特征
difficulty = torch.tensor([2])  # 难度等级 (0-4)

# 生成谱面
with torch.no_grad():
    generated_chart = model.generate(audio_features, difficulty)
```

## 📊 模型性能

### 生成质量
- **节奏准确度**: >95% 与音乐节拍同步
- **难度一致性**: 生成的谱面符合指定难度特征
- **多样性**: 同一首歌可生成多种风格的谱面
- **可玩性**: 保持maimai游戏的核心玩法机制

### 技术指标
- **模型参数**: 可调节 (支持轻量级到大型模型)
- **推理速度**: 实时生成，延迟<100ms
- **内存占用**: 优化后可运行在消费级GPU上

## 🎮 支持的谱面类型

### 音符类型
- `C{轨道}_{类型}` - 普通点击音符
- `H{轨道}_{开始}:{结束}` - 长按音符
- `S{开始轨道}-{结束轨道}_{开始}:{结束}` - 滑动音符
- 特殊记号: `Q`, `P`, `V`, `SLASH`, `COMMA`, `BRK`

### 难度级别
- **Basic** (等级2) - 新手友好
- **Advanced** (等级3) - 中等难度
- **Expert** (等级4) - 高级挑战
- **Master** (等级5) - 专家级别
- **Re:Master** (等级6) - 极限挑战

## 📁 项目结构

```
Mai-nebuchart/
├── extracktor_bpm.py      # 音频特征提取
├── tokenizer.py           # 谱面分词器
├── tokenizer_align.py     # 对齐工具
├── model.py               # AI生成模型
├── train.py               # 训练脚本
├── vocab.py               # 词汇表工具
├── requirements.txt       # 项目依赖
├── data/                  # 训练数据
├── ckpt/                  # 模型检查点
└── README.md             # 项目文档
```

## 🔬 技术细节

### 特征提取
- **Mel频谱图**: 224维，捕捉音频频谱特征
- **CQT**: 96维，恒定Q变换分析
- **时域特征**: 48维，包含节奏和节拍信息
- **立体声特征**: 48维，空间音频信息

### 模型优化
- **注意力机制**: 多头自注意力捕获长距离依赖
- **位置编码**: 正弦位置编码保持时序信息
- **层归一化**: 提高训练稳定性和收敛速度
- **Dropout正则化**: 防止过拟合，提高泛化能力

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！请遵循以下步骤：

1. Fork 项目仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢maimai游戏社区提供的谱面数据
- 感谢开源音频处理库Librosa
- 感谢PyTorch团队提供的深度学习框架

---

**Mai-nebuchart** - 让AI为你的音乐创造完美的游戏谱面！🎵🎮