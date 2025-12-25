# ml/03_train_baseline_sklearn.py
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def extract_features(X: np.ndarray) -> np.ndarray:
    """X: [N, T, C]  (T=26, C=6)
    return F: [N, 18]  where each channel has mean/std/energy
    """
    mean = X.mean(axis=1)                         # [N, C]
    std = X.std(axis=1)                           # [N, C]
    energy = (X * X).mean(axis=1)                 # [N, C]
    F = np.concatenate([mean, std, energy], axis=1).astype(np.float32)  # [N, 18]
    return F


def main():
    d = np.load(Path("ml/outputs/dataset_win0p5_hop0p1.npz"), allow_pickle=True)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    label_names = d["label_names"].tolist()

    # 提取 18 维特征 (mean/std/energy for 6 channels)
    F_train = extract_features(X_train)
    F_val = extract_features(X_val)
    F_test = extract_features(X_test)

    print(f"Feature shape: {F_train.shape}")  # [N, 18]

    clf = LogisticRegression(max_iter=2000)
    clf.fit(F_train, y_train)

    for name, F, y in [("VAL", F_val, y_val), ("TEST", F_test, y_test)]:
        y_pred = clf.predict(F)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        print(f"\n[{name}] accuracy={acc:.4f}")
        print("Confusion matrix (rows=true, cols=pred):\n", cm)
        print(classification_report(y, y_pred, target_names=label_names, digits=4))

    out_dir = Path("ml/outputs/baseline_sklearn")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coef.npy", clf.coef_)
    np.save(out_dir / "intercept.npy", clf.intercept_)
    print("\nSaved weights to:", out_dir)


if __name__ == "__main__":
    main()
