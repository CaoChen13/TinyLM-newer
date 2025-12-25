# ml/01_plot_raw.py
import os
import glob
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXPORT_DIR = os.path.join("ml", "data_raw", "exports")
OUT_DIR = os.path.join("ml", "outputs")


def estimate_fs(seconds_elapsed: np.ndarray) -> float:
    t = np.asarray(seconds_elapsed, dtype=float)
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if len(dt) == 0:
        return float("nan")
    med = np.median(dt)
    return 1.0 / med


def load_sensor_from_zip(zip_path: str, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        if csv_name not in z.namelist():
            raise FileNotFoundError(f"{csv_name} not found in {os.path.basename(zip_path)}")
        df = pd.read_csv(z.open(csv_name))
        # 你的数据列名是 time, seconds_elapsed, z, y, x（注意顺序）
        need_cols = {"seconds_elapsed", "x", "y", "z"}
        if not need_cols.issubset(set(df.columns)):
            raise ValueError(f"Unexpected columns in {csv_name}: {df.columns.tolist()}")
        # 统一成 x,y,z 顺序
        df = df[["seconds_elapsed", "x", "y", "z"]].copy()
        return df


def plot_xyz(df: pd.DataFrame, title: str, out_path: str, seconds: float = 10.0):
    t0 = df["seconds_elapsed"].iloc[0]
    d = df[df["seconds_elapsed"] <= t0 + seconds].copy()
    plt.figure()
    plt.plot(d["seconds_elapsed"] - t0, d["x"], label="x")
    plt.plot(d["seconds_elapsed"] - t0, d["y"], label="y")
    plt.plot(d["seconds_elapsed"] - t0, d["z"], label="z")
    plt.xlabel("time (s)")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    zip_files = sorted(glob.glob(os.path.join(EXPORT_DIR, "*.zip")))
    if not zip_files:
        print(f"No zip found in {EXPORT_DIR}. Put still.zip/shake.zip/circle.zip there.")
        return

    for zp in zip_files:
        name = os.path.splitext(os.path.basename(zp))[0]  # still / shake / circle
        acc = load_sensor_from_zip(zp, "TotalAcceleration.csv")
        gyr = load_sensor_from_zip(zp, "Gyroscope.csv")

        fs_acc = estimate_fs(acc["seconds_elapsed"].to_numpy())
        fs_gyr = estimate_fs(gyr["seconds_elapsed"].to_numpy())
        print(f"[{name}] TotalAcc: N={len(acc)} fs≈{fs_acc:.2f}Hz | Gyro: N={len(gyr)} fs≈{fs_gyr:.2f}Hz")

        plot_xyz(
            acc,
            title=f"{name} - TotalAcceleration (first 10s)",
            out_path=os.path.join(OUT_DIR, f"{name}_totalacc.png"),
        )
        plot_xyz(
            gyr,
            title=f"{name} - Gyroscope (first 10s)",
            out_path=os.path.join(OUT_DIR, f"{name}_gyro.png"),
        )

    print(f"Done. Check plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
