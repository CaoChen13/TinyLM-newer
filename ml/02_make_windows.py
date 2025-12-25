# ml/02_make_windows.py
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

EXPORT_DIR = Path("ml/data_raw/exports")
OUT_DIR = Path("ml/outputs")


def load_csv_from_zip(zf: zipfile.ZipFile, csv_name: str) -> pd.DataFrame:
    """Read a CSV inside zip into a DataFrame."""
    candidates = [n for n in zf.namelist() if n.endswith(csv_name)]
    if not candidates:
        raise FileNotFoundError(f"Cannot find {csv_name} in zip. Available files: {zf.namelist()[:20]}...")
    with zf.open(candidates[0]) as f:
        df = pd.read_csv(f)
    needed = ["seconds_elapsed", "x", "y", "z"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_name} missing columns {missing}. Have: {list(df.columns)}")
    df = df[needed].copy()
    df = df.sort_values("seconds_elapsed").reset_index(drop=True)
    return df


def estimate_fs(t: np.ndarray) -> float:
    dt = np.diff(t.astype(np.float64))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 10:
        raise ValueError("Too few valid dt to estimate sampling rate.")
    fs = 1.0 / np.median(dt)
    return float(fs)


def load_six_axis_from_zip(zip_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, data) where data shape is [N, 6]"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        acc = load_csv_from_zip(zf, "TotalAcceleration.csv")
        gyro = load_csv_from_zip(zf, "Gyroscope.csv")

    n = min(len(acc), len(gyro))
    acc = acc.iloc[:n]
    gyro = gyro.iloc[:n]

    t = acc["seconds_elapsed"].to_numpy(dtype=np.float32)
    acc_xyz = acc[["x", "y", "z"]].to_numpy(dtype=np.float32)
    gyro_xyz = gyro[["x", "y", "z"]].to_numpy(dtype=np.float32)
    data = np.concatenate([acc_xyz, gyro_xyz], axis=1)
    return t, data


def make_windows(data: np.ndarray, win_n: int, hop_n: int) -> np.ndarray:
    n = data.shape[0]
    if win_n <= 1 or hop_n <= 0:
        raise ValueError(f"Bad win_n/hop_n: win_n={win_n}, hop_n={hop_n}")
    if n < win_n:
        raise ValueError(f"Sequence too short: N={n} < win_n={win_n}")
    windows = []
    for start in range(0, n - win_n + 1, hop_n):
        windows.append(data[start : start + win_n])
    X = np.stack(windows, axis=0).astype(np.float32)
    return X


def fit_normalizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=(0, 1), dtype=np.float64)
    std = X.std(axis=(0, 1), dtype=np.float64)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def infer_label(zip_path: Path) -> str:
    """从文件名提取标签，如 circle1.zip -> circle"""
    stem = zip_path.stem.lower()
    # 去掉末尾的数字和连字符
    for label in ["circle", "shake", "still"]:
        if stem.startswith(label):
            return label
    return stem


def split_files_by_segment(zip_files: list[Path], train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    按段落（文件）划分 train/val/test，确保同一段的窗口不会跨集合
    每个标签的文件独立划分
    """
    rng = np.random.default_rng(seed)
    
    # 按标签分组
    label_to_files = {}
    for zf in zip_files:
        label = infer_label(zf)
        if label not in label_to_files:
            label_to_files[label] = []
        label_to_files[label].append(zf)
    
    train_files, val_files, test_files = [], [], []
    
    for label, files in label_to_files.items():
        files = sorted(files)
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # 确保 test 至少有 1 个
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])
        
        print(f"[{label}] total={n}, train={n_train}, val={n_val}, test={n - n_train - n_val}")
    
    return train_files, val_files, test_files


def process_files(files: list[Path], label_to_id: dict, win_sec: float, hop_sec: float):
    """处理一组文件，返回 X, y"""
    X_list, y_list = [], []
    
    for zf in files:
        label = infer_label(zf)
        t, data = load_six_axis_from_zip(zf)
        fs = estimate_fs(t)
        
        win_n = int(round(fs * win_sec))
        hop_n = max(1, int(round(fs * hop_sec)))
        
        try:
            X = make_windows(data, win_n=win_n, hop_n=hop_n)
        except ValueError as e:
            print(f"  Skip {zf.name}: {e}")
            continue
            
        y = np.full((X.shape[0],), label_to_id[label], dtype=np.int64)
        X_list.append(X)
        y_list.append(y)
    
    if not X_list:
        return np.array([]), np.array([])
    
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win_sec", type=float, default=0.5)
    ap.add_argument("--hop_sec", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(EXPORT_DIR.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in {EXPORT_DIR.resolve()}")

    # 确定标签
    labels = sorted(list(set(infer_label(z) for z in zip_files)))
    label_to_id = {name: i for i, name in enumerate(labels)}
    print(f"Found labels: {labels}")
    print(f"Total files: {len(zip_files)}")
    print(f"Using win_sec={args.win_sec}, hop_sec={args.hop_sec}")
    print()

    # 按段落划分文件
    train_files, val_files, test_files = split_files_by_segment(
        zip_files, train_ratio=0.6, val_ratio=0.2, seed=args.seed
    )
    
    print(f"\nTrain files: {[f.name for f in train_files]}")
    print(f"Val files: {[f.name for f in val_files]}")
    print(f"Test files: {[f.name for f in test_files]}")
    print()

    # 处理各集合
    X_train, y_train = process_files(train_files, label_to_id, args.win_sec, args.hop_sec)
    X_val, y_val = process_files(val_files, label_to_id, args.win_sec, args.hop_sec)
    X_test, y_test = process_files(test_files, label_to_id, args.win_sec, args.hop_sec)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 只用训练集计算归一化参数
    mean, std = fit_normalizer(X_train)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # 保存
    out_base = f"dataset_win{args.win_sec}_hop{args.hop_sec}".replace(".", "p")
    out_npz = OUT_DIR / f"{out_base}.npz"
    out_meta = OUT_DIR / f"{out_base}_meta.json"

    np.savez_compressed(
        out_npz,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        mean=mean,
        std=std,
        label_names=np.array(labels),
        win_sec=np.array([args.win_sec], dtype=np.float32),
        hop_sec=np.array([args.hop_sec], dtype=np.float32),
    )

    meta = {
        "labels": labels,
        "label_to_id": label_to_id,
        "train_files": [f.name for f in train_files],
        "val_files": [f.name for f in val_files],
        "test_files": [f.name for f in test_files],
        "win_sec": args.win_sec,
        "hop_sec": args.hop_sec,
        "split_method": "by_segment",
        "note": "Train/val/test split by recording segment to avoid data leakage.",
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved: {out_npz}")
    print(f"Saved: {out_meta}")
    print("Done.")


if __name__ == "__main__":
    main()
