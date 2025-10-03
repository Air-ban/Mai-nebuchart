#!/usr/bin/env python3
"""
高精度 BPM 驱动特征提取
usage: python extractor_bpm.py some_song.wav 190.15
输出：some_song.bpm190.pt  (dict{'feat':(T,512), 'bpm':float})
"""
import librosa, numpy as np, torch, math, sys
from pathlib import Path

class BPMGridExtractor:
    def __init__(self, sr=44100, n_mels=224, steps_per_beat=32,
                 min_hop=64, max_hop=8192):
        self.sr = sr
        self.n_mels = n_mels
        self.steps_per_beat = steps_per_beat
        self.min_hop = min_hop
        self.max_hop = max_hop

    def _make_grid(self, y, bpm):
        beat_dur = 60 / bpm                      # 秒
        hop_float = self.sr * beat_dur / self.steps_per_beat
        hop_float = np.clip(hop_float, self.min_hop, self.max_hop)
        n_frames = int(np.floor((len(y) - self.min_hop) / hop_float))
        # 采样点级累加，防漂移
        positions = np.round(np.arange(n_frames) * hop_float).astype(int)
        return positions, hop_float

    def _frame_feat(self, y, positions, n_fft=2048):
        """在指定采样点位置提取帧级特征"""
        stft = librosa.stft(y, n_fft=n_fft, hop_length=None,
                            center=False, window='hann')
        # 取对应列
        cols = np.searchsorted(np.arange(stft.shape[1]), positions, side='right') - 1
        cols = np.clip(cols, 0, stft.shape[1]-1)
        return stft[:, cols]                        # (freq, T)

    def __call__(self, path, bpm):
        y, _ = librosa.load(path, sr=self.sr, mono=False)
        if y.ndim == 2:
            y_mono = librosa.to_mono(y)
            left, right = y[0], y[1]
        else:
            y_mono = left = right = y

        pos, hop_float = self._make_grid(y_mono, bpm)
        T = len(pos)
        print(f"BPM={bpm:.2f}  hop={hop_float:.2f} samples  frames={T}")

        # 1) 高分辨率 mel (224)
        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y_mono, sr=self.sr,
                                           hop_length=math.ceil(hop_float),
                                           n_mels=self.n_mels, fmax=11000))
        mel = mel[:, :T]                            # 对齐长度

        # 2) CQT (96 bins/oct × 8 oct = 768 → 降采样到 96)
        cqt = librosa.amplitude_to_db(
            np.abs(librosa.cqt(y_mono, sr=self.sr,
                               hop_length=math.ceil(hop_float),
                               fmin=32.7, n_bins=96*8,
                               bins_per_octave=96)), ref=np.max)
        cqt = cqt[::8, :T]                          # 每 8 bin 取均值 → 96

        # 3) 谐波谱 (48)
        f0, _, _ = librosa.pyin(y_mono, fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C7'), sr=self.sr,
                                hop_length=math.ceil(hop_float))
        f0 = np.nan_to_num(f0, nan=0.0)[:T]
        harm_spec = librosa.f0_harmonics(
            self._frame_feat(y_mono, pos), f0=f0,
            harmonics=np.arange(1, 25), sr=self.sr)
        harm = librosa.power_to_db(harm_spec + 1e-8)        # (24, T)

        # 4) 瞬时频率 (48)
        stft_phase = np.angle(self._frame_feat(y_mono, pos))
        inst_f = np.diff(np.unwrap(stft_phase, axis=1), axis=1, prepend=0)
        inst_f = inst_f / (2*np.pi) * self.sr / hop_float
        # 取 CQT 对应频率数
        inst_f = inst_f[:96, :T]
        inst_f = inst_f[::2, :]                     # 48

        # 5) 色度 + 调式 (24)
        chroma = librosa.feature.chroma_cqt(C=cqt, sr=self.sr,
                                            hop_length=math.ceil(hop_float),
                                            n_chroma=12)
        mode = np.correlate(chroma.mean(1), np.roll(chroma.mean(1), 3), mode='same')
        mode = np.tile(mode[:12, None], (1, T))
        chroma_mode = np.vstack([chroma[:, :T], mode])

        # 6) Tonnetz (6)
        tonnetz = librosa.feature.tonnetz(chroma=chroma[:, :T])

        # 7) 谱对比度 + 谷底 (14)
        contrast, valleys = librosa.feature.spectral_contrast(y=y_mono, sr=self.sr,
                                                              hop_length=math.ceil(hop_float),
                                                              aggregate=None)
        contrast = contrast[:, :T]
        valleys = valleys[:, :T]

        # 8) 带宽 + 滚降 (2)
        bandwidth = librosa.feature.spectral_bandwidth(y=y_mono, sr=self.sr,
                                                       hop_length=math.ceil(hop_float))[:, :T]
        rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=self.sr,
                                                   hop_length=math.ceil(hop_float))[:, :T]

        # 9) RMS (1)
        rms = librosa.feature.rms(y=y_mono, hop_length=math.ceil(hop_float))[:, :T]

        # 10) ZCR (1)
        zcr = librosa.feature.zero_crossing_rate(y_mono, hop_length=math.ceil(hop_float))[:, :T]

        # 11) 立体声差异 (48)
        if y.ndim == 2:
            mel_l = librosa.power_to_db(
                librosa.feature.melspectrogram(y=left, sr=self.sr,
                                               hop_length=math.ceil(hop_float),
                                               n_mels=self.n_mels, fmax=11000))
            mel_r = librosa.power_to_db(
                librosa.feature.melspectrogram(y=right, sr=self.sr,
                                               hop_length=math.ceil(hop_float),
                                               n_mels=self.n_mels, fmax=11000))
            stereo_diff = np.abs(mel_l - mel_r)
        else:
            stereo_diff = np.zeros((self.n_mels, T))  # 单声道时返回零数组

        # 拼接所有特征
        features = np.vstack([mel, cqt, harm, inst_f, chroma_mode, tonnetz,
                              contrast, valleys, bandwidth, rolloff, rms, zcr,
                              stereo_diff])
        features = features.T  # (T, n_features)

        # 限制特征维度为 512
        if features.shape[1] > 512:
            features = features[:, :512]
        elif features.shape[1] < 512:
            # 如果特征维度不足512，则用0填充
            pad_width = 512 - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')

        return {'feat': torch.from_numpy(features).float(), 'bpm': float(bpm)}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python extracktor_bpm.py <音频文件路径> <BPM值>")
        print("示例: python extracktor_bpm.py song.wav 120.0")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    try:
        bpm = float(sys.argv[2])
    except ValueError:
        print("错误: BPM值必须是数字")
        sys.exit(1)
    
    if not Path(audio_path).exists():
        print(f"错误: 音频文件 '{audio_path}' 不存在")
        sys.exit(1)
    
    extractor = BPMGridExtractor()
    result = extractor(audio_path, bpm)
    
    output_path = f"{Path(audio_path).stem}.bpm{int(bpm)}.pt"
    torch.save(result, output_path)
    print(f"特征提取完成，保存为: {output_path}")
             
