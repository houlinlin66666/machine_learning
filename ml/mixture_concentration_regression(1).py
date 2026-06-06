# -*- coding: utf-8 -*-
"""
EEM (Excitation-Emission Matrix) Fluorescence Spectroscopy
Mixture Antibiotic Concentration Regression Pipeline
==================================================================

任务: 从 Excel 文件读取三维荧光光谱 (EEM), 分别回归预测:
        target 1 = 氧氟沙星 (Ofloxacin)  浓度 (µg/L)
        target 2 = 环丙沙星 (Ciprofloxacin) 浓度 (µg/L)

文件命名: <ofloxacin>-<ciprofloxacin>-<other>.xlsx
        前两段分别为两种抗生素的浓度。

----------------------------------------------------------------
环境依赖 (建议版本):
    python>=3.9
    numpy>=1.24
    pandas>=2.0
    scipy>=1.11
    scikit-learn>=1.3
    xgboost>=2.0
    matplotlib>=3.7
    seaborn>=0.12
    joblib>=1.3
    openpyxl>=3.1
    torch>=2.0

安装:
    pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn \
                joblib openpyxl torch
----------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
import glob
import time
import warnings
import traceback
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (
    train_test_split, KFold, RandomizedSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, mean_absolute_percentage_error,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ============================================================
# 全局配置
# ============================================================
DATA_DIR   = r"C:\Users\yafex\Desktop\hunhe"
FIGURE_DIR = os.path.join(DATA_DIR, "figure")
MODEL_DIR  = os.path.join(DATA_DIR, "models")
RESULT_CSV = os.path.join(DATA_DIR, "results.csv")

TARGETS = ["Ofloxacin", "Ciprofloxacin"]    # 目标 1 / 目标 2

SEED          = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5
N_ITER_SEARCH = 20
LOG_TARGET    = True             # 浓度跨多个数量级时取 log1p
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Nature 风格 matplotlib 配置
# ============================================================
CM = 1.0 / 2.54
SINGLE_COL = 8.5 * CM
DOUBLE_COL = 17.5 * CM


def set_nature_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "axes.linewidth":     1.0,
        "grid.linewidth":     0.5,
        "grid.color":         "#cccccc",
        "lines.linewidth":    1.2,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "savefig.dpi":        600,
        "figure.dpi":         120,
        "savefig.bbox":       "tight",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    """PDF (矢量) + PNG (600 dpi)."""
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.png"), dpi=600)
    plt.close(fig)


# ============================================================
# 任务 1: 数据读取
# ============================================================
def parse_concentrations(fname: str) -> Tuple[float, float] | None:
    """文件名前两段 -> (ofloxacin, ciprofloxacin)。失败返回 None。"""
    stem = os.path.splitext(fname)[0]
    parts = stem.split("-")
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def load_and_parse_data(folder_path: str
                        ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame,
                                   np.ndarray, np.ndarray]:
    """
    返回:
        X (N, n_ex, n_em), Y (N, 2)  [Ofloxacin, Ciprofloxacin],
        meta_df, ex_axis, em_axis
    """
    files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")))
    print(f"[Task 1] 发现 {len(files)} 个 Excel 文件, 开始加载 ...")
    if not files:
        raise FileNotFoundError(f"{folder_path} 内未找到 .xlsx 文件。")

    X_list, Y_list, meta_list = [], [], []
    ex_axis = em_axis = None
    expected = None
    skipped = 0

    for i, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        c = parse_concentrations(fname)
        if c is None:
            print(f"  [WARN] 名称解析失败: {fname}")
            skipped += 1
            continue
        try:
            df = pd.read_excel(fp, header=None)
            if ex_axis is None:
                em_axis = df.iloc[0, 1:].to_numpy(dtype=np.float32)
                ex_axis = df.iloc[1:, 0].to_numpy(dtype=np.float32)
            mat = df.iloc[1:, 1:].to_numpy(dtype=np.float32)
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            if expected is None:
                expected = mat.shape
            elif mat.shape != expected:
                print(f"  [WARN] 形状 {mat.shape} ≠ {expected}: {fname}")
                skipped += 1
                continue
        except Exception as e:
            print(f"  [WARN] 读取失败 {fname}: {e}")
            skipped += 1
            continue

        X_list.append(mat)
        Y_list.append(c)
        meta_list.append({"file": fname,
                          "Ofloxacin": c[0], "Ciprofloxacin": c[1]})

        if i % 200 == 0 or i == len(files):
            print(f"  - 已处理 {i}/{len(files)}")

    X = np.stack(X_list, axis=0)
    Y = np.asarray(Y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta_list)
    print(f"[Task 1] 完成。有效样本: {len(Y)} | 跳过: {skipped} "
          f"| EEM 形状: {expected}")
    for j, t in enumerate(TARGETS):
        print(f"          {t}: [{Y[:, j].min():.4g}, {Y[:, j].max():.4g}] µg/L")
    return X, Y, meta_df, ex_axis, em_axis


# ============================================================
# 任务 2: 划分 + 标准化 (用双目标分箱组合做近似分层)
# ============================================================
def stratified_split_2d(Y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """两个目标各自分位数分箱, 组合为分层标签。"""
    labels = []
    for j in range(Y.shape[1]):
        nb = min(n_bins, max(2, len(np.unique(Y[:, j]))))
        try:
            b = pd.qcut(Y[:, j], q=nb, labels=False, duplicates="drop")
        except ValueError:
            b = pd.cut(Y[:, j], bins=nb, labels=False)
        labels.append(np.asarray(b))
    return labels[0] * (labels[1].max() + 1) + labels[1]


def preprocess_data(X: np.ndarray, Y: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray,
                               StandardScaler]:
    X_flat = X.reshape(X.shape[0], -1)
    strat = stratified_split_2d(Y)
    # 分层桶若过稀疏会失败 -> 回退到随机划分
    try:
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X_flat, Y, test_size=TEST_SIZE,
            random_state=SEED, stratify=strat)
    except ValueError:
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X_flat, Y, test_size=TEST_SIZE, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    print(f"[Task 2] Train: {X_tr_s.shape} | Test: {X_te_s.shape}")
    return X_tr_s, X_te_s, Y_tr, Y_te, scaler


# ============================================================
# 任务 3a: 经典回归 + 超参数随机搜索
# ============================================================
def _model_search_space() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    space: Dict[str, Tuple[Any, Dict[str, Any]]] = {
        "RandomForest": (
            RandomForestRegressor(random_state=SEED, n_jobs=-1),
            {"n_estimators": [200, 400, 600],
             "max_depth":    [None, 10, 20, 40],
             "min_samples_split": [2, 4, 8],
             "max_features": ["sqrt", 0.3, 0.5]},
        ),
        "SVR": (
            SVR(),
            {"C":      [0.1, 1, 10, 100],
             "gamma":  ["scale", 1e-3, 1e-4],
             "kernel": ["rbf"],
             "epsilon": [0.01, 0.05, 0.1]},
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=SEED),
            {"n_estimators": [100, 200, 400],
             "max_depth":    [2, 3, 4],
             "learning_rate":[0.03, 0.05, 0.1],
             "subsample":    [0.7, 1.0]},
        ),
        "KernelRidge": (
            KernelRidge(kernel="rbf"),
            {"alpha": [1e-3, 1e-2, 1e-1, 1],
             "gamma": [1e-4, 1e-3, 1e-2, "scale"]},
        ),
        "ElasticNet": (
            ElasticNet(random_state=SEED, max_iter=10000),
            {"alpha":    [1e-3, 1e-2, 1e-1, 1, 10],
             "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        ),
    }
    if HAVE_XGB:
        space["XGBoost"] = (
            XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0,
                         tree_method="hist"),
            {"n_estimators":     [200, 400, 600],
             "max_depth":        [3, 5, 7],
             "learning_rate":    [0.03, 0.05, 0.1],
             "subsample":        [0.7, 0.9, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0]},
        )
    else:
        print("[Note] 未检测到 xgboost, 跳过 XGBoost。")
    return space


def _fix_gamma_for_kr(grid: Dict[str, Any], n_features: int) -> Dict[str, Any]:
    g = grid.get("gamma", [])
    grid = dict(grid)
    grid["gamma"] = [(1.0 / n_features) if v == "scale" else v for v in g]
    return grid


def train_ml_models_for_target(X_train: np.ndarray, y_train: np.ndarray,
                               target_name: str
                               ) -> Tuple[Dict[str, Any], Dict[str, float]]:
    print(f"[Task 3a] 训练经典回归 ({target_name}) ...")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    space = _model_search_space()
    trained: Dict[str, Any] = {}
    times: Dict[str, float] = {}
    y_fit = np.log1p(y_train) if LOG_TARGET else y_train

    for name, (est, grid) in space.items():
        if name == "KernelRidge":
            grid = _fix_gamma_for_kr(grid, X_train.shape[1])
        print(f"  - {name}: RandomizedSearchCV ...", flush=True)
        t0 = time.time()
        search = RandomizedSearchCV(
            est, grid, n_iter=N_ITER_SEARCH, cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1, random_state=SEED, refit=True)
        search.fit(X_train, y_fit)
        dt = time.time() - t0
        trained[name] = search.best_estimator_
        times[name] = dt
        joblib.dump(search.best_estimator_,
                    os.path.join(MODEL_DIR, f"{target_name}__{name}.joblib"))
        print(f"      best={search.best_params_}")
        print(f"      cv RMSE(log)={-search.best_score_:.4f}  time={dt:.1f}s")
    return trained, times


# ============================================================
# 任务 3b: 深度学习模型 (MLP + 2D-CNN)
# ============================================================
class MLPReg(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),    nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN2DReg(nn.Module):
    """把 EEM 当成 1 通道 2D 图像处理。"""
    def __init__(self, n_ex: int, n_em: int):
        super().__init__()
        self.n_ex, self.n_em = n_ex, n_em
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = x.view(-1, 1, self.n_ex, self.n_em)
        return self.head(self.feat(x)).squeeze(-1)


def _train_torch(model: nn.Module, X: np.ndarray, y: np.ndarray,
                 epochs: int = 80, batch: int = 32, lr: float = 1e-3
                 ) -> Tuple[nn.Module, List[float]]:
    model.to(DEVICE)
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                       torch.from_numpy(y.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss()
    hist: List[float] = []
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        hist.append(tot / len(ds))
        if (ep + 1) % 20 == 0:
            print(f"    epoch {ep+1:3d}/{epochs}  MSE={hist[-1]:.4f}")
    return model, hist


def _predict_torch(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        yp = model(torch.from_numpy(X.astype(np.float32)).to(DEVICE)).cpu().numpy()
    return yp.astype(np.float32)


class TorchRegressor:
    def __init__(self, arch: str, in_dim: int,
                 n_ex: int | None = None, n_em: int | None = None, **kw):
        self.arch = arch; self.in_dim = in_dim
        self.n_ex = n_ex; self.n_em = n_em
        self.kw = kw
        self.model: nn.Module | None = None
        self.history: List[float] = []

    def fit(self, X, y):
        torch.manual_seed(SEED)
        if self.arch == "MLP":
            self.model = MLPReg(self.in_dim)
        elif self.arch == "CNN2D":
            self.model = CNN2DReg(self.n_ex, self.n_em)
        else:
            raise ValueError(self.arch)
        self.model, self.history = _train_torch(self.model, X, y, **self.kw)
        return self

    def predict(self, X):
        return _predict_torch(self.model, X)


def train_dl_models_for_target(X_train: np.ndarray, y_train: np.ndarray,
                               n_ex: int, n_em: int, target_name: str
                               ) -> Tuple[Dict[str, TorchRegressor],
                                          Dict[str, float]]:
    print(f"[Task 3b] 训练深度学习模型 ({target_name}) ...")
    in_dim = X_train.shape[1]
    y_fit = np.log1p(y_train) if LOG_TARGET else y_train
    trained: Dict[str, TorchRegressor] = {}
    times: Dict[str, float] = {}
    for name, arch in [("MLP", "MLP"), ("CNN2D", "CNN2D")]:
        print(f"  - {name} ...")
        t0 = time.time()
        reg = TorchRegressor(arch, in_dim, n_ex=n_ex, n_em=n_em,
                             epochs=80, batch=32, lr=1e-3)
        reg.fit(X_train, y_fit)
        dt = time.time() - t0
        trained[name] = reg
        times[name] = dt
        torch.save(reg.model.state_dict(),
                   os.path.join(MODEL_DIR, f"{target_name}__{name}.pt"))
        print(f"      time = {dt:.1f}s")
    return trained, times


# ============================================================
# 任务 4: 评估
# ============================================================
def _inv(y_fit: np.ndarray) -> np.ndarray:
    return np.expm1(y_fit) if LOG_TARGET else y_fit


def evaluate_for_target(models: Dict[str, Any],
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        train_times: Dict[str, float],
                        target_name: str) -> pd.DataFrame:
    print(f"[Task 4] 评估 ({target_name}) ...")
    rows = []
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    y_train_fit = np.log1p(y_train) if LOG_TARGET else y_train

    for name, mdl in models.items():
        t0 = time.time()
        yp = _inv(mdl.predict(X_test))
        t_pred = time.time() - t0

        rmse = float(np.sqrt(mean_squared_error(y_test, yp)))
        mse  = float(mean_squared_error(y_test, yp))
        mae  = float(mean_absolute_error(y_test, yp))
        r2   = float(r2_score(y_test, yp))
        # MAPE: 排除 0 真值
        mask = np.abs(y_test) > 1e-9
        mape = (float(mean_absolute_percentage_error(y_test[mask], yp[mask]))
                if mask.any() else np.nan)

        cv_rmse_std = np.nan
        cv_r2_std   = np.nan
        if not isinstance(mdl, TorchRegressor):
            try:
                rmses, r2s = [], []
                for tr_idx, va_idx in cv.split(X_train):
                    m2 = type(mdl)(**mdl.get_params())
                    m2.fit(X_train[tr_idx], y_train_fit[tr_idx])
                    yv = _inv(m2.predict(X_train[va_idx]))
                    rmses.append(np.sqrt(mean_squared_error(y_train[va_idx], yv)))
                    r2s.append(r2_score(y_train[va_idx], yv))
                cv_rmse_std = float(np.std(rmses))
                cv_r2_std   = float(np.std(r2s))
            except Exception as e:
                print(f"    [CV warn] {name}: {e}")

        rows.append({
            "target": target_name,
            "model":  name,
            "RMSE": rmse, "MSE": mse, "MAE": mae,
            "MAPE": mape, "R2": r2,
            "cv_rmse_std": cv_rmse_std, "cv_r2_std": cv_r2_std,
            "train_time_s":   train_times.get(name, np.nan),
            "predict_time_s": t_pred,
            "y_pred": yp.tolist(),
        })
        print(f"  - {name:18s} RMSE={rmse:.4g}  R²={r2:.4f}  "
              f"MAE={mae:.4g}  MAPE={mape:.3f}")
    return pd.DataFrame(rows)


# ============================================================
# 任务 5: 可视化
# ============================================================
def plot_performance_bar(df: pd.DataFrame, target_name: str) -> None:
    """RMSE / MSE / R² 三联柱图, CV std 误差棒。"""
    d = df.copy().sort_values("R2", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.32))
    x = np.arange(len(d))
    colors = sns.color_palette("viridis", len(d))
    for ax, metric, label, err_col in zip(
            axes,
            ["RMSE", "MSE", "R2"],
            ["RMSE (µg/L)", "MSE (µg/L)$^{2}$", "R$^{2}$"],
            ["cv_rmse_std", None, "cv_r2_std"]):
        err = d[err_col].fillna(0) if err_col else None
        ax.bar(x, d[metric], color=colors, edgecolor="black",
               linewidth=0.6, yerr=err, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(d["model"], rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
    fig.suptitle(f"{target_name} — model performance "
                 f"(error bars: 5-fold CV std)", fontsize=12, y=1.03)
    fig.tight_layout()
    save_fig(fig, f"fig1_{target_name}_performance_bar")


def plot_pred_vs_true(df: pd.DataFrame, y_test: np.ndarray,
                      target_name: str) -> None:
    d = df.sort_values("R2", ascending=False).reset_index(drop=True)
    n = len(d); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(DOUBLE_COL, 3.4 * nrows * CM * 2.54))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.5, wspace=0.4)
    lo, hi = float(min(y_test.min(), 0)), float(y_test.max())
    pad = 0.05 * (hi - lo + 1e-9)

    for i, row in d.iterrows():
        r, c = divmod(i, ncols)
        sub = gs[r, c].subgridspec(2, 2, width_ratios=[4, 1],
                                   height_ratios=[1, 4],
                                   hspace=0.05, wspace=0.05)
        ax_m = fig.add_subplot(sub[1, 0])
        ax_t = fig.add_subplot(sub[0, 0], sharex=ax_m)
        ax_r = fig.add_subplot(sub[1, 1], sharey=ax_m)

        yp = np.asarray(row["y_pred"])
        ax_m.scatter(y_test, yp, s=14, alpha=0.7,
                     color=sns.color_palette("viridis", n)[i],
                     edgecolor="white", linewidth=0.3,
                     label="Test")
        ax_m.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                  "k--", lw=1, label="y = x")
        ax_m.set_xlim(lo - pad, hi + pad)
        ax_m.set_ylim(lo - pad, hi + pad)
        ax_m.set_xlabel("True (µg/L)")
        ax_m.set_ylabel("Predicted (µg/L)")
        ax_m.text(0.04, 0.96,
                  f"{row['model']}\nR²={row['R2']:.3f}\n"
                  f"RMSE={row['RMSE']:.3g}",
                  transform=ax_m.transAxes, va="top", ha="left",
                  fontsize=9,
                  bbox=dict(facecolor="white", alpha=0.85, lw=0))
        ax_m.legend(loc="lower right", fontsize=7, frameon=False)

        ax_t.hist(y_test, bins=30, color="#888", alpha=0.7)
        ax_r.hist(yp, bins=30, color="#888", alpha=0.7,
                  orientation="horizontal")
        ax_t.axis("off"); ax_r.axis("off")

    fig.suptitle(f"{target_name} — predicted vs true",
                 fontsize=12, y=1.01)
    save_fig(fig, f"fig2_{target_name}_pred_vs_true")


def plot_residuals(df: pd.DataFrame, y_test: np.ndarray,
                   target_name: str) -> None:
    best = df.sort_values("R2", ascending=False).iloc[0]
    yp = np.asarray(best["y_pred"])
    res = y_test - yp
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.33))
    col = sns.color_palette("viridis", 6)[1]

    axes[0].scatter(yp, res, s=14, alpha=0.7, color=col,
                    edgecolor="white", linewidth=0.3)
    axes[0].axhline(0, color="k", lw=1, ls="--")
    axes[0].set_xlabel("Predicted (µg/L)")
    axes[0].set_ylabel("Residual (µg/L)")
    axes[0].set_title(f"(a) Residual vs Predicted — {best['model']}")

    sns.histplot(res, bins=30, kde=True, ax=axes[1], color=col,
                 edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Residual (µg/L)")
    axes[1].set_title("(b) Residual distribution")

    stats.probplot(res, dist="norm", plot=axes[2])
    axes[2].get_lines()[0].set_markerfacecolor(col)
    axes[2].get_lines()[0].set_markeredgecolor("white")
    axes[2].get_lines()[0].set_markersize(5)
    axes[2].get_lines()[1].set_color("k")
    axes[2].set_title("(c) Normal Q-Q plot")

    fig.suptitle(f"{target_name} — residual analysis "
                 f"(best model: {best['model']})", fontsize=12, y=1.03)
    fig.tight_layout()
    save_fig(fig, f"fig3_{target_name}_residual_analysis")


def plot_feature_importance(models: Dict[str, Any],
                            ex_axis: np.ndarray, em_axis: np.ndarray,
                            target_name: str, top_n: int = 20) -> None:
    for name in ("RandomForest", "GradientBoosting", "XGBoost"):
        if name not in models:
            continue
        imp = getattr(models[name], "feature_importances_", None)
        if imp is None:
            continue
        n_em = em_axis.size
        idx = np.argsort(imp)[::-1][:top_n]
        labels = [f"Ex {ex_axis[i // n_em]:.0f} / Em {em_axis[i % n_em]:.0f}"
                  for i in idx]
        vals = imp[idx]
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6,
                                        0.32 * top_n + 1.0))
        colors = sns.color_palette("viridis", top_n)
        ax.barh(range(top_n)[::-1], vals, color=colors,
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(top_n)[::-1])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Feature importance")
        ax.set_title(f"{target_name} — Top-{top_n} wavelength pairs "
                     f"({name})")
        sns.despine()
        save_fig(fig, f"fig4_{target_name}_feature_importance_{name}")


def plot_importance_heatmap(models: Dict[str, Any],
                            ex_axis: np.ndarray, em_axis: np.ndarray,
                            target_name: str) -> None:
    """把 RandomForest 特征重要性还原为 EEM 坐标热图。"""
    if "RandomForest" not in models:
        return
    imp = getattr(models["RandomForest"], "feature_importances_", None)
    if imp is None:
        return
    n_ex, n_em = ex_axis.size, em_axis.size
    if imp.size != n_ex * n_em:
        return
    grid = imp.reshape(n_ex, n_em)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.4))
    pcm = ax.pcolormesh(em_axis, ex_axis, grid, cmap="viridis",
                        shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Feature importance")
    ax.set_xlabel("Emission wavelength (nm)")
    ax.set_ylabel("Excitation wavelength (nm)")
    ax.set_title(f"{target_name} — RF importance on EEM grid")
    save_fig(fig, f"fig4b_{target_name}_importance_heatmap")


def plot_pca_colored(X: np.ndarray, y: np.ndarray,
                     target_name: str) -> None:
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.3))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=18,
                    alpha=0.85, edgecolor="white", linewidth=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(f"{target_name} (µg/L)")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title(f"PCA — colored by {target_name}")
    sns.despine()
    fig.tight_layout()
    save_fig(fig, f"fig5_{target_name}_pca")


def plot_error_by_bin(df: pd.DataFrame, y_test: np.ndarray,
                      target_name: str, n_bins: int = 5) -> None:
    """按真值分箱, 箱线图展示不同浓度区间的预测误差。"""
    best = df.sort_values("R2", ascending=False).iloc[0]
    yp = np.asarray(best["y_pred"])
    err = yp - y_test
    try:
        bins = pd.qcut(y_test, q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(y_test, bins=n_bins)
    tbl = pd.DataFrame({"bin": bins.astype(str), "err": err})

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, SINGLE_COL * 1.2))
    order = sorted(tbl["bin"].unique(),
                   key=lambda s: float(s.split(",")[0].strip("([")))
    sns.boxplot(data=tbl, x="bin", y="err", ax=ax, order=order,
                palette="viridis", linewidth=0.8, fliersize=2)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel(f"True {target_name} bin (µg/L)")
    ax.set_ylabel("Prediction error (Pred − True)")
    ax.set_title(f"{target_name} — error by concentration bin "
                 f"({best['model']})")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    sns.despine()
    fig.tight_layout()
    save_fig(fig, f"fig6_{target_name}_error_by_bin")


def plot_learning_curves(models: Dict[str, Any],
                         X_train: np.ndarray, y_train: np.ndarray,
                         target_name: str) -> None:
    sk = {k: v for k, v in models.items()
          if not isinstance(v, TorchRegressor)}
    if not sk:
        return
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    y_fit = np.log1p(y_train) if LOG_TARGET else y_train
    sizes = np.linspace(0.2, 1.0, 5)
    n = len(sk); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(DOUBLE_COL,
                                      3.2 * nrows * CM * 2.54))
    axes = np.atleast_2d(axes).ravel()
    pal = sns.color_palette("viridis", 2)
    for i, (name, mdl) in enumerate(sk.items()):
        ax = axes[i]
        try:
            ts, tr_s, va_s = learning_curve(
                mdl, X_train, y_fit, cv=cv, train_sizes=sizes,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1, random_state=SEED)
            trm, trs = -tr_s.mean(1), tr_s.std(1)
            vam, vas = -va_s.mean(1), va_s.std(1)
            ax.plot(ts, trm, "o-", color=pal[0], label="Train")
            ax.fill_between(ts, trm - trs, trm + trs,
                            color=pal[0], alpha=0.18)
            ax.plot(ts, vam, "s-", color=pal[1], label="CV")
            ax.fill_between(ts, vam - vas, vam + vas,
                            color=pal[1], alpha=0.18)
            ax.set_xlabel("Training samples")
            ax.set_ylabel("RMSE (log)" if LOG_TARGET else "RMSE")
            ax.set_title(name)
            ax.legend(frameon=False, fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"failed:\n{e}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{target_name} — learning curves", fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, f"fig7_{target_name}_learning_curves")


def plot_dl_loss(dl_models: Dict[str, TorchRegressor],
                 target_name: str) -> None:
    if not dl_models:
        return
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.2))
    colors = sns.color_palette("viridis", len(dl_models) + 1)
    for i, (name, m) in enumerate(dl_models.items()):
        hist = getattr(m, "history", [])
        if hist:
            ax.plot(range(1, len(hist) + 1), hist, lw=1.8,
                    color=colors[i], label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_yscale("log")
    ax.set_title(f"{target_name} — DL training curves")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    save_fig(fig, f"fig8_{target_name}_dl_training_loss")


def plot_targets_overview(all_results: pd.DataFrame) -> None:
    """同一画板对比两个目标的 R² / RMSE。"""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))
    palette = sns.color_palette("Set2", len(TARGETS))
    for ax, metric, label in zip(axes, ["R2", "RMSE"],
                                 ["R$^{2}$", "RMSE (µg/L)"]):
        pivot = all_results.pivot(index="model", columns="target",
                                  values=metric)
        pivot = pivot.sort_values(TARGETS[0], ascending=False)
        x = np.arange(len(pivot))
        w = 0.4
        for j, t in enumerate(TARGETS):
            ax.bar(x + (j - 0.5) * w, pivot[t], w, color=palette[j],
                   edgecolor="black", linewidth=0.6, label=t)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(frameon=False)
    fig.suptitle("Overview — both targets", fontsize=12, y=1.04)
    fig.tight_layout()
    save_fig(fig, "fig0_overview_both_targets")


# ============================================================
# 主流程
# ============================================================
def run_target(j: int, target_name: str,
               X_tr: np.ndarray, X_te: np.ndarray,
               Y_tr: np.ndarray, Y_te: np.ndarray,
               n_ex: int, n_em: int,
               ex_axis: np.ndarray, em_axis: np.ndarray
               ) -> pd.DataFrame:
    print("\n" + "=" * 64)
    print(f">>> Target {j+1}/2 : {target_name}")
    print("=" * 64)
    y_tr = Y_tr[:, j]; y_te = Y_te[:, j]

    ml_models, ml_t = train_ml_models_for_target(X_tr, y_tr, target_name)
    dl_models, dl_t = train_dl_models_for_target(X_tr, y_tr,
                                                 n_ex, n_em, target_name)
    all_models = {**ml_models, **dl_models}
    all_times = {**ml_t, **dl_t}

    res_df = evaluate_for_target(all_models, X_tr, y_tr, X_te, y_te,
                                 all_times, target_name)

    set_nature_style()
    plot_performance_bar(res_df, target_name)
    plot_pred_vs_true(res_df, y_te, target_name)
    plot_residuals(res_df, y_te, target_name)
    plot_feature_importance(all_models, ex_axis, em_axis, target_name)
    plot_importance_heatmap(all_models, ex_axis, em_axis, target_name)
    plot_pca_colored(np.vstack([X_tr, X_te]),
                     np.concatenate([y_tr, y_te]), target_name)
    plot_error_by_bin(res_df, y_te, target_name)
    plot_learning_curves(all_models, X_tr, y_tr, target_name)
    plot_dl_loss(dl_models, target_name)

    return res_df


def main() -> None:
    try:
        print("=" * 64)
        print("EEM Mixture Antibiotic Concentration Regression Pipeline")
        print("=" * 64)
        print(f"Device           : {DEVICE}")
        print(f"Data directory   : {DATA_DIR}")
        print(f"Figure directory : {FIGURE_DIR}")
        print(f"Model directory  : {MODEL_DIR}")
        print(f"Target transform : {'log1p' if LOG_TARGET else 'identity'}")
        print("-" * 64)

        X, Y, meta_df, ex_axis, em_axis = load_and_parse_data(DATA_DIR)
        meta_df.to_csv(os.path.join(DATA_DIR, "metadata.csv"),
                       index=False, encoding="utf-8-sig")
        n_ex, n_em = X.shape[1], X.shape[2]

        X_tr, X_te, Y_tr, Y_te, scaler = preprocess_data(X, Y)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

        all_res: List[pd.DataFrame] = []
        for j, t in enumerate(TARGETS):
            all_res.append(run_target(j, t, X_tr, X_te, Y_tr, Y_te,
                                      n_ex, n_em, ex_axis, em_axis))

        full_df = pd.concat(all_res, ignore_index=True)
        full_df.drop(columns=["y_pred"]).to_csv(
            RESULT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n[Saved] {RESULT_CSV}")

        set_nature_style()
        plot_targets_overview(full_df)

        print("\n========== 性能汇总 (按 R² 降序, 每个目标内) ==========")
        for t in TARGETS:
            print(f"\n--- {t} ---")
            sub = full_df[full_df["target"] == t]
            print(sub[["model", "RMSE", "MSE", "MAE", "MAPE", "R2",
                       "train_time_s", "predict_time_s"]]
                  .sort_values("R2", ascending=False)
                  .to_string(index=False))

        print("\n全部完成 ✓")

    except Exception:
        print("[ERROR] 主流程异常:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
