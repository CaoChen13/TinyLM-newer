# ml/04_export_for_mcu.py
"""
导出模型参数为 C 头文件，用于 STM32 部署
"""
from pathlib import Path
import numpy as np
import json

OUT_DIR = Path("ml/outputs")
FIRMWARE_DIR = Path("firmware/Core/Inc")


def load_sklearn_model():
    """加载 sklearn LogisticRegression 的权重"""
    baseline = OUT_DIR / "baseline_sklearn"
    coef = np.load(baseline / "coef.npy")        # [3, 18]
    intercept = np.load(baseline / "intercept.npy")  # [3]
    return coef.astype(np.float32), intercept.astype(np.float32)


def load_normalizer():
    """加载归一化参数"""
    d = np.load(OUT_DIR / "dataset_win0p5_hop0p1.npz", allow_pickle=True)
    mean = d["mean"]  # [6]
    std = d["std"]    # [6]
    label_names = d["label_names"].tolist()
    win_sec = float(d["win_sec"][0])
    hop_sec = float(d["hop_sec"][0])
    return mean.astype(np.float32), std.astype(np.float32), label_names, win_sec, hop_sec


def array_to_c(name: str, arr: np.ndarray, fmt: str = ".6f") -> str:
    """将 numpy 数组转为 C 数组定义"""
    flat = arr.flatten()
    values = ", ".join(f"{v:{fmt}}" for v in flat)
    shape_comment = f"// shape: {arr.shape}"
    return f"static const float {name}[{len(flat)}] = {{{values}}}; {shape_comment}"


def generate_header():
    """生成 C 头文件"""
    coef, intercept = load_sklearn_model()
    mean, std, label_names, win_sec, hop_sec = load_normalizer()
    
    # 计算窗口参数 (假设 fs ≈ 52Hz)
    fs_est = 52.0
    win_n = int(round(fs_est * win_sec))
    hop_n = int(round(fs_est * hop_sec))
    
    header = f'''#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

/*
 * Auto-generated model parameters for gesture recognition
 * Model: LogisticRegression with 18-dim features (mean/std/energy per channel)
 * Labels: {label_names}
 * Window: {win_sec}s ({win_n} samples @ ~{fs_est}Hz)
 * Hop: {hop_sec}s ({hop_n} samples)
 */

#define NUM_CLASSES {len(label_names)}
#define NUM_FEATURES 18
#define NUM_CHANNELS 6
#define WIN_N {win_n}
#define HOP_N {hop_n}

// Label names
static const char* LABEL_NAMES[NUM_CLASSES] = {{"{label_names[0]}", "{label_names[1]}", "{label_names[2]}"}};

// Normalization parameters (per channel, for raw window data)
{array_to_c("NORM_MEAN", mean)}
{array_to_c("NORM_STD", std)}

// LogisticRegression weights
{array_to_c("LR_COEF", coef)}  // [NUM_CLASSES, NUM_FEATURES]
{array_to_c("LR_INTERCEPT", intercept)}

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
    return header


def main():
    header = generate_header()
    
    # 保存到 ml/outputs
    out_path = OUT_DIR / "model_params.h"
    out_path.write_text(header, encoding="utf-8")
    print(f"Saved: {out_path}")
    
    # 如果 firmware 目录存在，也复制一份
    if FIRMWARE_DIR.exists():
        fw_path = FIRMWARE_DIR / "model_params.h"
        fw_path.write_text(header, encoding="utf-8")
        print(f"Saved: {fw_path}")
    else:
        print(f"Note: {FIRMWARE_DIR} not found, skipped firmware copy")
    
    # 打印摘要
    coef, intercept = load_sklearn_model()
    mean, std, label_names, _, _ = load_normalizer()
    print(f"\nModel summary:")
    print(f"  Classes: {label_names}")
    print(f"  Coef shape: {coef.shape}")
    print(f"  Intercept: {intercept}")
    print(f"  Norm mean: {mean}")
    print(f"  Norm std: {std}")


if __name__ == "__main__":
    main()
