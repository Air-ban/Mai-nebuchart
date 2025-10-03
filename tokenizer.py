#!/usr/bin/env python3
"""
一次解析全部难度
用法：python multi_diff_tokenizer.py full_chart.txt song.bpm32.pt
输出：song.bpm32.{basic,advanced,expert,master,remaster}.tok.pt
"""
import re, torch
from tokenizer_align import align_to_grid, build_vocab, encode_tokens
from pathlib import Path

DIFF_MAP = {'2': 'basic', '3': 'advanced', '4': 'expert', '5': 'master', '6': 'remaster'}

def split_inotes(text):
    """返回 dict lv->原始行列表"""
    chunks = {}
    for lv in DIFF_MAP:
        # 懒惰正则：&inote_3= 到下一个 & 或 E
        m = re.search(rf'&inote_{lv}=(.*?)($|&)', text, re.S)
        if m:
            lines = [l.rstrip() for l in m.group(1).splitlines() if l.strip()]
            chunks[lv] = lines
    return chunks

def main(chart_file, bpm_pt):
    bpm_data = torch.load(bpm_pt)
    steps = bpm_data.get('steps_per_beat', 32)
    all_text = Path(chart_file).read_text()
    chunks = split_inotes(all_text)
    base_out = bpm_pt.replace('.pt', '')

    # 先合并所有 token 建一个公共 vocab
    all_tokens = []
    for lv, lines in chunks.items():
        tokens = []
        cur_beat = 0.0
        beat_per_frame = 1 / steps
        for line in lines:
            if line == 'E': break
            tail_comma = len(line) - len(line.rstrip(','))
            from tokenizer_align import tokenize_line
            tokens.extend(tokenize_line(line))
            cur_beat += (1 + tail_comma) * beat_per_frame
        all_tokens.append(tokens)
    vocab = build_vocab(all_tokens)

    # 再逐难度正式对齐
    for lv, lines in chunks.items():
        T = bpm_data['feat'].shape[0]
        frame_tokens = [[] for _ in range(T)]
        cur_beat = 0.0
        for line in lines:
            if line == 'E': break
            tail_comma = len(line) - len(line.rstrip(','))
            tokens = tokenize_line(line)
            frame_idx = int(round(cur_beat * steps))
            if 0 <= frame_idx < T:
                frame_tokens[frame_idx].extend(tokens)
            cur_beat += (1 + tail_comma) * beat_per_frame
        for f in range(T):
            if not frame_tokens[f]:
                frame_tokens[f] = ['PAD']
        ids = encode_tokens(frame_tokens, vocab)
        out = f"{base_out}.{DIFF_MAP[lv]}.tok.pt"
        torch.save({'tokens': frame_tokens, 'ids': ids, 'vocab': vocab,
                    'steps_per_beat': steps, 'lv': lv}, out)
        print(f"saved {out}  frames={T}")

if __name__ == '__main__':
    import sys
    
    # 参数验证
    if len(sys.argv) < 3:
        print("错误：参数不足！")
        print("用法：python tokenizer.py <谱面文件.txt> <BPM数据.pt>")
        print("示例：python tokenizer.py full_chart.txt song.bpm32.pt")
        sys.exit(1)
    
    chart_file = sys.argv[1]
    bpm_pt = sys.argv[2]
    
    # 检查参数是否为空字符串
    if not chart_file or not bpm_pt:
        print("错误：参数不能为空！")
        print("请提供有效的谱面文件(.txt)和BPM数据文件(.pt)路径")
        sys.exit(1)
    
    main(chart_file, bpm_pt)

