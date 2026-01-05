"""
MPU6050 手势识别模型训练脚本
用法: python train_model.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 参数
WIN_SEC = 0.5    # 窗口长度（秒）
HOP_SEC = 0.1    # 滑动步长（秒）
FS = 50          # 采样率（Hz）
WIN_N = int(WIN_SEC * FS)  # 窗口样本数 = 25
HOP_N = int(HOP_SEC * FS)  # 步长样本数 = 5

# 类别
LABELS = ['still', 'tilt', 'shake', 'circle', 'tap', 'flip']


def load_csv_data(csv_path):
    """加载 CSV 数据"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Labels in data: {df['label'].unique()}")
    return df


def make_windows(data, labels, win_n, hop_n):
    """将连续数据切成窗口"""
    X_list, y_list = [], []
    
    # 按标签分组处理
    for label in labels['label'].unique():
        mask = labels['label'] == label
        indices = np.where(mask)[0]
        
        # 找连续的段
        segments = []
        start = indices[0]
        for i in range(1, len(indices)):
            if indices[i] != indices[i-1] + 1:
                segments.append((start, indices[i-1] + 1))
                start = indices[i]
        segments.append((start, indices[-1] + 1))
        
        # 对每个段切窗口
        for seg_start, seg_end in segments:
            seg_data = data[seg_start:seg_end]
            if len(seg_data) < win_n:
                continue
            
            for i in range(0, len(seg_data) - win_n + 1, hop_n):
                window = seg_data[i:i+win_n]
                X_list.append(window)
                y_list.append(label)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    return X, y


def extract_features(X):
    """提取特征: 每个通道的 mean, std, energy"""
    mean = X.mean(axis=1)           # [N, 6]
    std = X.std(axis=1)             # [N, 6]
    energy = (X * X).mean(axis=1)   # [N, 6]
    F = np.concatenate([mean, std, energy], axis=1)  # [N, 18]
    return F.astype(np.float32)


def main():
    # 加载数据
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in data/")
        return
    
    # 合并所有 CSV
    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"Total samples: {len(df)}")
    print(f"Labels: {df['label'].value_counts().to_dict()}")
    
    # 转换数据格式
    # 原始数据是 int16，转成物理单位
    data = np.zeros((len(df), 6), dtype=np.float32)
    data[:, 0] = df['ax'].values / 16384.0 * 9.81  # m/s²
    data[:, 1] = df['ay'].values / 16384.0 * 9.81
    data[:, 2] = df['az'].values / 16384.0 * 9.81
    data[:, 3] = df['gx'].values / 131.0 * 0.01745  # rad/s
    data[:, 4] = df['gy'].values / 131.0 * 0.01745
    data[:, 5] = df['gz'].values / 131.0 * 0.01745
    
    # 切窗口
    X, y = make_windows(data, df, WIN_N, HOP_N)
    print(f"\nWindows: {X.shape}")  # [N, WIN_N, 6]
    
    # 计算归一化参数（用于 MCU）
    norm_mean = X.mean(axis=(0, 1))
    norm_std = X.std(axis=(0, 1))
    norm_std = np.maximum(norm_std, 1e-6)
    print(f"Norm mean: {norm_mean}")
    print(f"Norm std: {norm_std}")
    
    # 归一化
    X_norm = (X - norm_mean) / norm_std
    
    # 提取特征
    F = extract_features(X_norm)
    print(f"Features: {F.shape}")  # [N, 18]
    
    # 标签编码
    label_to_id = {name: i for i, name in enumerate(LABELS)}
    y_encoded = np.array([label_to_id.get(label, -1) for label in y])
    
    # 过滤掉未知标签
    valid_mask = y_encoded >= 0
    F = F[valid_mask]
    y_encoded = y_encoded[valid_mask]
    
    print(f"\nValid samples: {len(F)}")
    for i, name in enumerate(LABELS):
        count = (y_encoded == i).sum()
        print(f"  {name}: {count}")
    
    # 划分训练/测试集
    F_train, F_test, y_train, y_test = train_test_split(
        F, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain: {len(F_train)}, Test: {len(F_test)}")
    
    # 训练
    clf = LogisticRegression(max_iter=2000)
    clf.fit(F_train, y_train)
    
    # 评估
    y_pred = clf.predict(F_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))
    
    # 导出模型参数
    export_model_params(clf, norm_mean, norm_std, LABELS)


def export_model_params(clf, norm_mean, norm_std, labels):
    """导出模型参数为 C 头文件"""
    coef = clf.coef_.flatten()
    intercept = clf.intercept_
    
    output = f'''#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

/*
 * Auto-generated model parameters for gesture recognition
 * Model: LogisticRegression with 18-dim features (mean/std/energy per channel)
 * Labels: {labels}
 * Window: {WIN_SEC}s ({WIN_N} samples @ {FS}Hz)
 * Hop: {HOP_SEC}s ({HOP_N} samples)
 */

#define NUM_CLASSES {len(labels)}
#define NUM_FEATURES 18
#define NUM_CHANNELS 6
#define WIN_N {WIN_N}
#define HOP_N {HOP_N}

// Label names
static const char* LABEL_NAMES[NUM_CLASSES] = {{{", ".join(f'"{l}"' for l in labels)}}};

// Normalization parameters (per channel, for raw window data)
static const float NORM_MEAN[6] = {{{", ".join(f"{v:.6f}" for v in norm_mean)}}}; // shape: (6,)
static const float NORM_STD[6] = {{{", ".join(f"{v:.6f}" for v in norm_std)}}}; // shape: (6,)

// LogisticRegression weights
static const float LR_COEF[{len(coef)}] = {{{", ".join(f"{v:.6f}" for v in coef)}}}; // shape: ({len(labels)}, 18)
static const float LR_INTERCEPT[{len(labels)}] = {{{", ".join(f"{v:.6f}" for v in intercept)}}}; // shape: ({len(labels)},)

/*
 * Feature extraction (18-dim):
 *   For each of 6 channels: mean, std, energy (mean of x^2)
 *   features[0:6]   = mean of each channel
 *   features[6:12]  = std of each channel  
 *   features[12:18] = energy of each channel
 *
 * Prediction:
 *   logits[k] = sum(features[i] * LR_COEF[k*18 + i]) + LR_INTERCEPT[k]
 *   pred = argmax(logits)
 */

#endif // MODEL_PARAMS_H
'''
    
    out_path = Path("Inc/model_params.h")
    out_path.write_text(output)
    print(f"\nModel exported to: {out_path}")


if __name__ == "__main__":
    main()
