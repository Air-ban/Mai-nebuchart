# Mai-nebuchart 项目说明

## 项目概述

这是一个音乐游戏谱面处理工具集，主要用于将音乐游戏的谱面文件转换为机器学习模型可以处理的特征数据。项目包含两个核心功能模块：

1. **BPM特征提取器** (`extracktor_bpm.py`) - 从音频文件中提取与BPM同步的高精度音频特征
2. **谱面分词器** (`tokenizer.py`) - 将游戏谱面文件转换为时间对齐的token序列

## 技术栈

- **Python 3.3+** - 主要编程语言
- **PyTorch** - 深度学习框架，用于数据保存和处理
- **Librosa** - 音频处理库，用于特征提取
- **NumPy** - 数值计算库

## 核心模块

### 1. BPM特征提取器 (`extracktor_bpm.py`)

**功能**：从音频文件中提取与BPM同步的多维度音频特征

**输入**：
- 音频文件路径（支持WAV等格式）
- BPM（每分钟节拍数）值

**输出**：
- `.pt`文件，包含：
  - `feat`: 形状为(T, 512)的特征矩阵，T为时间帧数
  - `bpm`: BPM值

**提取的特征维度**（共512维）：
- Mel频谱图 (224维)
- CQT（恒定Q变换） (96维)
- 谐波谱 (24维)
- 瞬时频率 (48维)
- 色度+调式 (24维)
- Tonnetz (6维)
- 谱对比度+谷底 (14维)
- 带宽+滚降 (2维)
- RMS能量 (1维)
- 过零率 (1维)
- 立体声差异 (48维)

**使用方法**：
```bash
python extracktor_bpm.py <音频文件路径> <BPM值>
# 示例
python extracktor_bpm.py song.wav 190.15
```

### 2. 谱面分词器 (`tokenizer.py`)

**功能**：将游戏谱面文件按难度级别分割并转换为时间对齐的token序列

**输入**：
- 谱面文件（`.txt`格式，包含多个难度级别的谱面数据）
- BPM特征文件（`.pt`格式，由extracktor_bpm.py生成）

**输出**：
- 每个难度级别生成一个`.tok.pt`文件，包含：
  - `tokens`: 时间对齐的token列表
  - `ids`: token对应的ID序列
  - `vocab`: 词汇表
  - `steps_per_beat`: 每拍步数
  - `lv`: 难度级别

**支持的难度级别**：
- basic (2级)
- advanced (3级)
- expert (4级)
- master (5级)
- remaster (6级)

**谱面记号说明**：
- `C{轨道}_{类型}` - 普通音符
- `H{轨道}_{开始}:{结束}` - 长按音符
- `S{开始轨道}-{结束轨道}_{开始}:{结束}` - 滑动音符
- `Q`, `P`, `V`, `SLASH`, `COMMA`, `BRK` - 特殊记号

**使用方法**：
```bash
python tokenizer.py <谱面文件.txt> <BPM数据.pt>
# 示例
python tokenizer.py full_chart.txt song.bpm32.pt
```

### 3. 辅助模块 (`tokenizer_align.py`)

提供谱面解析和token化的基础功能：
- `tokenize_line()` - 解析单行谱面数据
- `align_to_grid()` - 将谱面数据对齐到时间网格
- `build_vocab()` - 构建词汇表
- `encode_tokens()` - 将token转换为ID序列

## 工作流程

1. **音频处理阶段**：
   - 使用`extracktor_bpm.py`提取音频的BPM同步特征
   - 生成包含512维特征的`.pt`文件

2. **谱面处理阶段**：
   - 使用`tokenizer.py`处理谱面文件
   - 按难度级别分割谱面数据
   - 将谱面记号转换为token序列
   - 与音频特征时间对齐
   - 生成每个难度级别的`.tok.pt`文件

## 依赖安装

```bash
pip install -r requirements.txt
```

依赖包：
- librosa>=0.10.0
- numpy>=1.21.0
- torch>=1.12.0

## 文件结构

```
Mai-nebuchart/
├── extracktor_bpm.py      # BPM特征提取器
├── tokenizer.py           # 谱面分词器
├── tokenizer_align.py     # 谱面对齐工具
├── requirements.txt       # 项目依赖
└── __pycache__/          # Python缓存目录
```

## 注意事项

- 确保音频文件的采样率为44100Hz（项目默认）
- 谱面文件需要包含所有难度级别的数据，格式为`&inote_难度级别=...`
- BPM值需要准确，以确保音频特征与谱面正确对齐
- 输出的token文件可用于训练音乐游戏AI模型