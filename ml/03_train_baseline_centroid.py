# ml/03_train_baseline_centroid.py
from __future__ import annotations
from pathlib import Path
import numpy as np


def load_dataset(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    return (
        d["X_train"], d["y_train"],
        d["X_val"], d["y_val"],
        d["X_test"], d["y_test"],
        d["label_names"].tolist(),
    )


def extract_features(X: np.ndarray) -> np.ndarray:
    """X: [N, T, C]  (T=26, C=6)
    return F: [N, C*3]  where each channel has mean/std/energy
    """
    # mean over time
    mean = X.mean(axis=1)                         # [N, C]
    std = X.std(axis=1)                           # [N, C]
    energy = (X * X).mean(axis=1)                 # [N, C]  mean(x^2)
    F = np.concatenate([mean, std, energy], axis=1).astype(np.float32)  # [N, 18]
    return F


def standardize(F_train: np.ndarray, F_other: np.ndarray):
    mu = F_train.mean(axis=0, dtype=np.float64)
    sd = F_train.std(axis=0, dtype=np.float64)
    sd = np.maximum(sd, 1e-6)
    return ((F_other - mu) / sd).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def fit_centroids(F: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    centroids = np.zeros((num_classes, F.shape[1]), dtype=np.float32)
    for k in range(num_classes):
        idx = np.where(y == k)[0]
        if idx.size == 0:
            raise ValueError(f"class {k} has no samples")
        centroids[k] = F[idx].mean(axis=0)
    return centroids


def predict_centroid(F: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # 欧氏距离平方：||x-c||^2 = sum((x-c)^2)
    diff = F[:, None, :] - centroids[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.argmin(dist2, axis=1).astype(np.int64)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    acc = float((y_true == y_pred).mean())
    return cm, acc


def main():
    npz_path = Path("ml/outputs/dataset_win0p5_hop0p1.npz")
    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_dataset(npz_path)
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1

    print("Labels:", label_names)
    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    # 特征
    F_train = extract_features(X_train)
    F_val = extract_features(X_val)
    F_test = extract_features(X_test)

    # 只用训练集统计做标准化，避免信息泄露
    F_train_std, mu, sd = standardize(F_train, F_train)
    F_val_std, _, _ = standardize(F_train, F_val)
    F_test_std, _, _ = standardize(F_train, F_test)

    # 拟合类中心
    centroids = fit_centroids(F_train_std, y_train, num_classes)

    # 评估
    y_pred_val = predict_centroid(F_val_std, centroids)
    cm_val, acc_val = confusion_matrix(y_val, y_pred_val, num_classes)

    y_pred_test = predict_centroid(F_test_std, centroids)
    cm_test, acc_test = confusion_matrix(y_test, y_pred_test, num_classes)

    print(f"\nVal accuracy:  {acc_val:.4f}")
    print("Val CM (rows=true, cols=pred):\n", cm_val)
    print(f"\nTest accuracy: {acc_test:.4f}")
    print("Test CM (rows=true, cols=pred):\n", cm_test)

    out_dir = Path("ml/outputs/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "centroids.npy", centroids)
    np.save(out_dir / "feature_mu.npy", mu)
    np.save(out_dir / "feature_sd.npy", sd)
    np.save(out_dir / "cm_val.npy", cm_val)
    np.save(out_dir / "cm_test.npy", cm_test)
    np.save(out_dir / "y_pred_test.npy", y_pred_test)
    print("\nSaved baseline artifacts to:", out_dir)


if __name__ == "__main__":
    main()
