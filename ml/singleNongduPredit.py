# -*- coding: utf-8 -*-
"""
EEM (Excitation-Emission Matrix) Fluorescence Spectroscopy
Antibiotic Concentration Regression Pipeline
==================================================================

任务: 从 Excel 文件读取三维荧光光谱 (EEM) 数据,
      回归预测样本的抗生素浓度 (µg/L)。

文件命名: <concentration>-<id>-<other>.xlsx, 第 1 段为浓度。

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
    pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn joblib openpyxl torch
----------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import sys
import glob
import time
import warnings
import traceback
from typing import Dict, List, Tuple, Any, Callable

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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
# ==============================================
# 一键屏蔽 PyCharm 调试器的烦人海报警告
# ==============================================
import warnings

# 忽略所有 UserWarning，直接干掉那个 pkg_resources 警告
warnings.filterwarnings("ignore", category=UserWarning)
# 额外忽略掉 sklearn、joblib 等训练时的无用警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 再强力清理一遍所有警告
import logging

logging.getLogger("setuptools").setLevel(logging.ERROR)
logging.getLogger("pkg_resources").setLevel(logging.ERROR)

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
DATA_DIR = r"/Users/houlinlin/master/data/EEM_data/lixiang/EEM-huan/strength/all"
FIGURE_DIR = os.path.join(DATA_DIR, "figure")
MODEL_DIR = os.path.join(DATA_DIR, "models")
RESULT_CSV = os.path.join(DATA_DIR, "results.csv")

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_SEARCH = 20  # 随机搜索迭代次数
LOG_TARGET = True  # 浓度跨多个数量级 -> 在 log1p 空间训练
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# Nature 风格 matplotlib 配置
# ============================================================
# 单栏 8.5 cm, 双栏 17.5 cm
CM = 1.0 / 2.54
SINGLE_COL = 8.5 * CM
DOUBLE_COL = 17.5 * CM

MODEL_MAP = {
    "RandomForest": "RF", "GradientBoosting": "GBRT", "SVR": "SVR",
    "KernelRidge": "KRR", "XGBoost": "XGB",
     "CNN2D": "CNN"
}
def set_nature_style() -> None:
    """深度定制 Nature 风格：Times New Roman + 极简轴"""
    plt.style.use("seaborn-v0_8-white")
    # 强制设置新罗马字体
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "mathtext.fontset": "stix", # 让公式也接近 Times 风格
    })

# 推荐使用的颜色循环 (Scientific color maps)
COLOR_CYCLE = ['#440154', '#31688e', '#35b779', '#fde725']  # Viridis 采样


def plot_relative_error(y_test, y_pred, model_name):
    rel_error = np.abs(y_test - y_pred) / (y_test + 1e-9) * 100
    fig, ax = plt.subplots(figsize=(8.5 * CM, 7 * CM))
    ax.scatter(y_test, rel_error, s=10, c='#31688e', alpha=0.6, edgecolors='none')
    ax.set_xscale('log')  # 浓度通常跨度大，用对数轴
    ax.axhline(10, color='red', ls='--', lw=0.8, label='10% Error')  # 设定参考线
    ax.set_xlabel('True Concentration (µg/L)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title(f'Prediction Stability - {model_name}')
    save_fig(fig, f"relative_error_{model_name}")


def save_fig(fig: plt.Figure, name: str) -> None:
    """保存 PDF (矢量) + PNG (600 dpi)."""
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.png"), dpi=600)
    plt.close(fig)


# ============================================================
# 任务 1: 数据读取
# ============================================================
FNAME_RE = re.compile(r"^([\-+0-9.eE]+)\s*-", )


def parse_concentration(fname: str) -> float | None:
    """从文件名第 1 段提取浓度 (float)。"""
    stem = os.path.splitext(fname)[0]
    parts = stem.split("-")
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def load_and_parse_data(folder_path: str
                        ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame,
np.ndarray, np.ndarray]:
    """
    读取目录内所有 EEM Excel:
        第 1 行为发射波长表头 (Em), 第 1 列为激发波长 (Ex)。
    返回:
        X (N, n_ex, n_em), y (N,), meta_df, ex_axis, em_axis
    """
    files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")))
    print(f"[Task 1] 发现 {len(files)} 个 Excel 文件, 开始加载 ...")
    if not files:
        raise FileNotFoundError(f"目录 {folder_path} 内未找到 .xlsx 文件。")

    X_list, y_list, meta_list = [], [], []
    ex_axis = em_axis = None
    expected_shape = None
    skipped = 0

    for i, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        conc = parse_concentration(fname)
        if conc is None:
            print(f"  [WARN] 文件名解析失败 (跳过): {fname}")
            skipped += 1
            continue
        try:
            df = pd.read_excel(fp, header=None)
            if ex_axis is None:
                em_axis = df.iloc[0, 1:].to_numpy(dtype=np.float32)
                ex_axis = df.iloc[1:, 0].to_numpy(dtype=np.float32)
            mat = df.iloc[1:, 1:].to_numpy(dtype=np.float32)
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            if expected_shape is None:
                expected_shape = mat.shape
            elif mat.shape != expected_shape:
                print(f"  [WARN] 形状 {mat.shape} ≠ {expected_shape} (跳过): {fname}")
                skipped += 1
                continue
        except Exception as e:
            print(f"  [WARN] 读取失败 {fname}: {e}")
            skipped += 1
            continue

        X_list.append(mat)
        y_list.append(conc)
        meta_list.append({"file": fname, "concentration": conc})

        if i % 200 == 0 or i == len(files):
            print(f"  - 已处理 {i}/{len(files)}")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta_list)
    print(f"[Task 1] 完成。有效样本: {len(y)} | 跳过: {skipped} "
          f"| EEM 形状: {expected_shape}")
    print(f"          浓度范围: [{y.min():.4g}, {y.max():.4g}] µg/L")
    return X, y, meta_df, ex_axis, em_axis


# ============================================================
# 任务 2-3: 划分 + 标准化 (用浓度分箱做"近似分层"以保持分布一致)
# ============================================================
def stratified_regression_split(y: np.ndarray, n_bins: int = 10
                                ) -> np.ndarray:
    """对连续目标按分位数分箱, 用作 train_test_split 的 stratify。"""
    n_bins = min(n_bins, max(2, len(np.unique(y))))
    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(y, bins=n_bins, labels=False)
    return np.asarray(bins)


def preprocess_data(X: np.ndarray, y: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray,
np.ndarray, np.ndarray,
StandardScaler]:
    X_flat = X.reshape(X.shape[0], -1)  # (N, n_ex * n_em)
    strat = stratified_regression_split(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_flat, y, test_size=TEST_SIZE,
        random_state=SEED, stratify=strat,
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    print(f"[Task 2-3] Train: {X_tr_s.shape} | Test: {X_te_s.shape}")
    return X_tr_s, X_te_s, y_tr, y_te, scaler


# ============================================================
# 任务 4a: 经典回归模型 + 超参数搜索
# ============================================================
def _model_search_space() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    space: Dict[str, Tuple[Any, Dict[str, Any]]] = {
        "RandomForest": (
            RandomForestRegressor(random_state=SEED, n_jobs=-1),
            {"n_estimators": [200, 400, 600],
             "max_depth": [None, 10, 20, 40],
             "min_samples_split": [2, 4, 8],
             "max_features": ["sqrt", 0.3, 0.5]},
        ),
        "SVR": (
            SVR(),
            {"C": [0.1, 1, 10, 100],
             "gamma": ["scale", 1e-3, 1e-4],
             "kernel": ["rbf"],
             "epsilon": [0.01, 0.05, 0.1]},
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=SEED),
            {"n_estimators": [100, 200, 400],
             "max_depth": [2, 3, 4],
             "learning_rate": [0.03, 0.05, 0.1],
             "subsample": [0.7, 1.0]},
        ),
        "KernelRidge": (
            KernelRidge(kernel="rbf"),
            {"alpha": [1e-3, 1e-2, 1e-1, 1],
             "gamma": [1e-4, 1e-3, 1e-2, "scale"]},
        ),
        # "ElasticNet": (
        #     ElasticNet(random_state=SEED, max_iter=10000),
        #     {"alpha": [1e-3, 1e-2, 1e-1, 1, 10],
        #      "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        # ),
    }
    if HAVE_XGB:
        space["XGBoost"] = (
            XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0,
                         tree_method="hist"),
            {"n_estimators": [200, 400, 600],
             "max_depth": [3, 5, 7],
             "learning_rate": [0.03, 0.05, 0.1],
             "subsample": [0.7, 0.9, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0]},
        )
    else:
        print("[Note] 未检测到 xgboost, 跳过 XGBoost。pip install xgboost 即可启用。")
    return space


def _fix_gamma_for_kr(grid: Dict[str, Any], n_features: int) -> Dict[str, Any]:
    """KernelRidge 不支持 gamma='scale', 替换为 1/n_features。"""
    g = grid.get("gamma", [])
    new_g = [(1.0 / n_features) if v == "scale" else v for v in g]
    grid = dict(grid)
    grid["gamma"] = new_g
    return grid


def train_ml_models(X_train: np.ndarray, y_train: np.ndarray,
                    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    print("[Task 4a] 训练经典回归模型 + 随机搜索超参 ...")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    space = _model_search_space()
    trained: Dict[str, Any] = {}
    times: Dict[str, Dict[str, float]] = {}

    y_fit = np.log1p(y_train) if LOG_TARGET else y_train

    for name, (est, grid) in space.items():
        if name == "KernelRidge":
            grid = _fix_gamma_for_kr(grid, X_train.shape[1])
        print(f"  - {name}: RandomizedSearchCV ...", flush=True)
        t0 = time.time()
        search = RandomizedSearchCV(
            est, grid, n_iter=N_ITER_SEARCH, cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1, random_state=SEED, refit=True, verbose=0,
        )
        search.fit(X_train, y_fit)
        t_fit = time.time() - t0
        best = search.best_estimator_
        trained[name] = best
        times[name] = {"train_time_s": t_fit}
        joblib.dump(best, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"      best params: {search.best_params_}")
        print(f"      cv RMSE (log-space) = {-search.best_score_:.4f} "
              f"| fit time = {t_fit:.1f}s")
    return trained, times


# ============================================================
# 任务 4b: 深度学习模型 (MLP + 2D-CNN)
# ============================================================
# class MLPReg(nn.Module):
#     def __init__(self, in_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(128, 1),
#         )
#
#     def forward(self, x):
#         return self.net(x).squeeze(-1)


class CNN2DReg(nn.Module):
    """把 EEM 当成 1 通道二维图像处理。"""

    def __init__(self, n_ex: int, n_em: int):
        super().__init__()
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
        self.n_ex, self.n_em = n_ex, n_em

    def forward(self, x):
        # x: (B, n_ex * n_em) -> (B, 1, n_ex, n_em)
        x = x.view(-1, 1, self.n_ex, self.n_em)
        return self.head(self.feat(x)).squeeze(-1)


def _train_torch_reg(model: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray,
                     epochs: int = 80, batch: int = 32, lr: float = 1e-3
                     ) -> Tuple[nn.Module, List[float]]:
    model.to(DEVICE)
    ds = TensorDataset(torch.from_numpy(X_tr).float(),
                       torch.from_numpy(y_tr).float())
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
            print(f"    epoch {ep + 1:3d}/{epochs}  MSE={hist[-1]:.4f}")
    return model, hist


def _predict_torch_reg(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        yh = model(torch.from_numpy(X).float().to(DEVICE)).cpu().numpy()
    return yh.astype(np.float32)


class TorchRegressor:
    """sklearn 风格包装。"""

    def __init__(self, arch: str, in_dim: int,
                 n_ex: int | None = None, n_em: int | None = None, **kw):
        self.arch = arch
        self.in_dim = in_dim
        self.n_ex, self.n_em = n_ex, n_em
        self.kw = kw
        self.model: nn.Module | None = None
        self.history: List[float] = []

    def fit(self, X, y):
        torch.manual_seed(SEED)
        if self.arch == "CNN2D":
            self.model = CNN2DReg(self.n_ex, self.n_em)
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")
        self.model, self.history = _train_torch_reg(
            self.model, X.astype(np.float32),
            y.astype(np.float32), **self.kw)
        return self

    def predict(self, X):
        return _predict_torch_reg(self.model, X.astype(np.float32))


def train_dl_models(X_train: np.ndarray, y_train: np.ndarray,
                    n_ex: int, n_em: int
                    ) -> Tuple[Dict[str, TorchRegressor], Dict[str, Dict[str, float]]]:
    print("[Task 4b] 训练深度学习回归模型 ...")
    in_dim = X_train.shape[1]
    y_fit = np.log1p(y_train) if LOG_TARGET else y_train
    trained: Dict[str, TorchRegressor] = {}
    times: Dict[str, Dict[str, float]] = {}

    for name, arch in [("CNN2D", "CNN2D")]:
        print(f"  - {name} ...")
        t0 = time.time()
        reg = TorchRegressor(arch, in_dim, n_ex=n_ex, n_em=n_em,
                             epochs=80, batch=32, lr=1e-3)
        reg.fit(X_train, y_fit)
        t_fit = time.time() - t0
        trained[name] = reg
        times[name] = {"train_time_s": t_fit}
        torch.save(reg.model.state_dict(),
                   os.path.join(MODEL_DIR, f"{name}.pt"))
        print(f"      fit time = {t_fit:.1f}s")
    return trained, times


# ============================================================
# 任务 4c: 评估
# ============================================================
def _inv_target(y_log_or_raw: np.ndarray) -> np.ndarray:
    return np.expm1(y_log_or_raw) if LOG_TARGET else y_log_or_raw


def evaluate_models(models: Dict[str, Any],
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    times: Dict[str, Dict[str, float]],
                    ) -> pd.DataFrame:
    print("[Task 4c] 在训练集和测试集上评估 ...")
    rows = []
    y_train_fit = np.log1p(y_train) if LOG_TARGET else y_train

    for name, mdl in models.items():
        # ---- 测试集预测 ----
        t_start = time.time()
        y_pred_fit = mdl.predict(X_test)
        t_pred = time.time() - t_start
        y_pred = _inv_target(y_pred_fit)

        # ---- 训练集预测 (用于绘图对比) ----
        y_train_pred = _inv_target(mdl.predict(X_train))

        # 计算指标
        test_r2 = float(r2_score(y_test, y_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        train_r2 = float(r2_score(y_train, y_train_pred))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))

        row = {
            "model": name,
            "R2": test_r2,
            "RMSE": test_rmse,
            "MSE": float(mean_squared_error(y_test, y_pred)),
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "train_R2": train_r2,
            "train_RMSE": train_rmse,
            "train_time_s": times.get(name, {}).get("train_time_s", np.nan),
            "predict_time_s": t_pred,
            "y_pred": y_pred.tolist(),
            "y_train_pred": y_train_pred.tolist()
        }
        rows.append(row)
        print(f"  - {name:18s} | Test R²: {test_r2:.4f} | Train R²: {train_r2:.4f} | Time: {row['train_time_s']:.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    return df


# ============================================================
# 任务 5: 可视化
# ============================================================
def plot_performance_bar(results_df: pd.DataFrame) -> None:
    """
    深度优化配色与细节的性能对比柱状图
    采用 Nature 风格：深青/暗红对比色
    """
    df = results_df.copy()
    df["abbr"] = df["model"].map(lambda x: MODEL_MAP.get(x, x))
    df = df.sort_values("R2", ascending=False)

    # 尺寸微调，确保精致
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 6 * CM))
    metrics = [("R2", "train_R2", "$R^2$"),
               ("RMSE", "train_RMSE", "RMSE"),
               ("MSE", "MSE", "MSE")]

    x = np.arange(len(df))
    width = 0.35

    # --- 顶级期刊配色方案 ---
    # C_TRAIN: 莫兰迪深青蓝 (Midnight Blue)
    # C_TEST:  莫兰迪干枯玫瑰红 (Muted Rose)
    c_train = "#1F4E79"
    c_test = "#C0504D"

    for i, (m_test, m_train, title) in enumerate(metrics):
        ax = axes[i]
        if m_train in df.columns:
            # 增加白色极细边框，增强立体感
            ax.bar(x - width / 2, df[m_train], width, label='Train',
                   color=c_train, edgecolor='white', lw=0.6, alpha=0.9)
            ax.bar(x + width / 2, df[m_test], width, label='Test',
                   color=c_test, edgecolor='white', lw=0.6, alpha=0.9)
        else:
            ax.bar(x, df[m_test], width * 1.5, color=c_test, edgecolor='white', lw=0.6)

        # 细节优化
        ax.set_xticks(x)
        ax.set_xticklabels(df["abbr"], rotation=45, ha='right', fontsize=7)

        # 仅保留左侧和底侧坐标轴
        sns.despine(ax=ax)

        # 增加极其微弱的水平网格线，辅助阅读但不抢戏
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.15)

        # 标题加粗并左对齐，这是顶刊风格
        ax.set_title(title, fontweight='bold', fontsize=9, loc='left', pad=10)

        if i == 0:
            ax.legend(frameon=False, loc='upper right', fontsize=7)

    fig.tight_layout()
    save_fig(fig, "fig1_nature_bar")


def plot_pred_vs_true(results_df: pd.DataFrame, y_test: np.ndarray) -> None:
    """提升高级感的预测值对比散点图"""
    df = results_df.sort_values("R2", ascending=False).reset_index(drop=True)
    n = len(df)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, 6 * nrows * CM))
    axes = np.atleast_2d(axes).ravel()

    # --- 高级感散点配色 ---
    # 训练集：浅灰（背景感）；测试集：深藏蓝/宝石红
    color_test = "#0D47A1"
    color_train = "#B0BEC5"

    for i, row in df.iterrows():
        ax = axes[i]
        abbr = MODEL_MAP.get(row["model"], row["model"])
        yp_test = np.asarray(row["y_pred"])
        yp_train = np.asarray(row["y_train_pred"])

        # 1. 先画对角线 (置于最底层)
        all_max = max(y_test.max(), yp_test.max(), y_tr_global.max())
        ax.plot([0, all_max], [0, all_max], color='#212121', ls='--', lw=0.8, alpha=0.6, zorder=1)

        # 2. 绘制训练集：极低透明度，无边框
        ax.scatter(y_tr_global, yp_train, s=8, alpha=0.15, c=color_train,
                   label='Train', edgecolors='none', zorder=2)

        # 3. 绘制测试集：较高透明度，白色细边框，增加 Z-index 确保在上方
        ax.scatter(y_test, yp_test, s=18, alpha=0.8, c=color_test,
                   label='Test', edgecolors='white', linewidths=0.4, zorder=3)

        # 细节美化
        ax.set_title(f"{abbr} ($R^2$: {row['R2']:.3f})", fontsize=9, loc='left')
        ax.set_xlabel("Measured (µg/L)")
        ax.set_ylabel("Predicted (µg/L)")

        # 使用科学计数法（如果数值较大）
        ax.ticklabel_format(style='plain', axis='both')

        if i == 0:
            ax.legend(frameon=False, loc='upper left', markerscale=1.2)

        sns.despine(ax=ax)

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    fig.tight_layout()
    save_fig(fig, "fig2_improved_scatter")


def plot_residuals(results_df: pd.DataFrame, y_test: np.ndarray) -> None:
    """最佳模型的残差三联图: 残差-预测 / 残差直方 / Q-Q。"""
    best = results_df.sort_values("R2", ascending=False).iloc[0]
    yp = np.asarray(best["y_pred"])
    res = y_test - yp

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.33))
    color = sns.color_palette("viridis", 6)[1]

    # (a) residual vs predicted
    axes[0].scatter(yp, res, s=14, alpha=0.7,
                    color=color, edgecolor="white", linewidth=0.3)
    axes[0].axhline(0, color="k", lw=1, ls="--")
    axes[0].set_xlabel("Predicted (µg/L)")
    axes[0].set_ylabel("Residual = True - Pred (µg/L)")
    axes[0].set_title(f"(a) Residual vs Predicted — {best['model']}")

    # (b) residual histogram + KDE
    sns.histplot(res, bins=30, kde=True, ax=axes[1], color=color,
                 edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Residual (µg/L)")
    axes[1].set_title("(b) Residual distribution")

    # (c) Q-Q plot
    stats.probplot(res, dist="norm", plot=axes[2])
    axes[2].get_lines()[0].set_markerfacecolor(color)
    axes[2].get_lines()[0].set_markeredgecolor("white")
    axes[2].get_lines()[0].set_markersize(5)
    axes[2].get_lines()[1].set_color("k")
    axes[2].set_title("(c) Normal Q-Q plot")

    fig.tight_layout()
    save_fig(fig, "fig3_residual_analysis")


def plot_feature_importance(models: Dict[str, Any],
                            ex_axis: np.ndarray, em_axis: np.ndarray,
                            top_n: int = 20) -> None:
    """Tree-ensemble 的 Top-N 重要 (Ex, Em) 波长对条形图。"""
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

        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4,
                                        0.32 * top_n + 1.0))
        colors = sns.color_palette("viridis", top_n)
        ax.barh(range(top_n)[::-1], vals, color=colors,
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(top_n)[::-1])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Feature importance")
        ax.set_title(f"Top-{top_n} wavelength pairs — {name}")
        sns.despine()
        save_fig(fig, f"fig4a_feature_importance_{name}")


def plot_pca(X: np.ndarray, y: np.ndarray) -> None:
    """PCA 2D 投影, 按浓度连续上色 (viridis)。"""
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.2))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=18,
                    alpha=0.85, edgecolor="white", linewidth=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Concentration (µg/L)")
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    ax.set_title("PCA projection of EEM features")
    sns.despine()
    fig.tight_layout()
    save_fig(fig, "fig4b_pca")


def plot_learning_curves(models: Dict[str, Any],
                         X_train: np.ndarray, y_train: np.ndarray) -> None:
    """对 sklearn 回归器绘制学习曲线 (train vs CV)。"""
    sk_models = {k: v for k, v in models.items()
                 if not isinstance(v, TorchRegressor)}
    if not sk_models:
        return
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    y_fit = np.log1p(y_train) if LOG_TARGET else y_train
    sizes = np.linspace(0.2, 1.0, 5)

    n = len(sk_models)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(DOUBLE_COL, 3.0 * nrows * CM * 2.54))
    axes = np.atleast_2d(axes).ravel()

    palette = sns.color_palette("viridis", 2)
    for i, (name, mdl) in enumerate(sk_models.items()):
        ax = axes[i]
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                mdl, X_train, y_fit, cv=cv, train_sizes=sizes,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1, random_state=SEED)
            tr_m = -train_scores.mean(1);
            tr_s = train_scores.std(1)
            va_m = -val_scores.mean(1);
            va_s = val_scores.std(1)
            ax.plot(train_sizes, tr_m, "o-", color=palette[0], label="Train RMSE")
            ax.fill_between(train_sizes, tr_m - tr_s, tr_m + tr_s,
                            color=palette[0], alpha=0.18)
            ax.plot(train_sizes, va_m, "s-", color=palette[1], label="CV RMSE")
            ax.fill_between(train_sizes, va_m - va_s, va_m + va_s,
                            color=palette[1], alpha=0.18)
            ax.set_xlabel("Training samples")
            ax.set_ylabel("RMSE (log-space)" if LOG_TARGET else "RMSE")
            ax.set_title(name)
            ax.legend(frameon=False, fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"learning_curve failed:\n{e}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Learning curves", fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig5_learning_curves")


def plot_dl_loss(dl_models: Dict[str, TorchRegressor]) -> None:
    if not dl_models:
        return
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.1))
    colors = sns.color_palette("viridis", len(dl_models) + 1)
    for i, (name, m) in enumerate(dl_models.items()):
        hist = getattr(m, "history", [])
        if not hist:
            continue
        ax.plot(range(1, len(hist) + 1), hist, lw=1.8,
                color=colors[i], label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_yscale("log")
    ax.set_title("Deep-learning training curves")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    save_fig(fig, "fig6_dl_training_loss")


def plot_nature_style(results_df: pd.DataFrame, models: Dict[str, Any],
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      ex_axis: np.ndarray, em_axis: np.ndarray,
                      dl_models: Dict[str, Any]) -> None:
    print("[Task 5] 生成 Nature 风格图表 ...")
    set_nature_style()
    plot_performance_bar(results_df)
    plot_pred_vs_true(results_df, y_test)
    plot_residuals(results_df, y_test)
    plot_feature_importance(models, ex_axis, em_axis, top_n=20)
    plot_pca(np.vstack([X_train, X_test]),
             np.concatenate([y_train, y_test]))
    plot_learning_curves(models, X_train, y_train)
    plot_dl_loss(dl_models)
    print(f"[Task 5] 图表保存至: {FIGURE_DIR}")


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    try:
        print("=" * 64)
        print("EEM Antibiotic Concentration Regression Pipeline")
        print("=" * 64)
        print(f"Device           : {DEVICE}")
        print(f"Data directory   : {DATA_DIR}")
        print(f"Figure directory : {FIGURE_DIR}")
        print(f"Model directory  : {MODEL_DIR}")
        print(f"Target transform : {'log1p' if LOG_TARGET else 'identity'}")
        print("-" * 64)

        # 1. 加载数据
        X, y, meta_df, ex_axis, em_axis = load_and_parse_data(DATA_DIR)

        # 获取 EEM 尺寸供 CNN 模型使用
        n_samples, n_ex, n_em = X.shape

        # 2-3. 划分 + 标准化 (只调用一次，确保顺序一致)
        global y_tr_global
        X_tr, X_te, y_tr, y_te, scaler = preprocess_data(X, y)
        y_tr_global = y_tr  # 赋值给全局变量供 plot_pred_vs_true 使用

        # 保存标准化器
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

        # 4a. 经典模型 + 超参搜索
        ml_models, ml_times = train_ml_models(X_tr, y_tr)

        # 4b. 深度模型
        dl_models, dl_times = train_dl_models(X_tr, y_tr, n_ex, n_em)

        all_models: Dict[str, Any] = {**ml_models, **dl_models}
        all_times = {**ml_times, **dl_times}

        # 4c. 评估
        results_df = evaluate_models(all_models, X_tr, y_tr, X_te, y_te,
                                     all_times)

        # 5. 可视化
        plot_nature_style(results_df, all_models,
                          X_tr, y_tr, X_te, y_te,
                          ex_axis, em_axis, dl_models)

        # 排行榜
        print("\n" + "=" * 80)
        print(f"{'Model':<10} | {'Test R2':<8} | {'Train R2':<8} | {'RMSE':<8} | {'Train(s)':<8} | {'Pred(s)':<8}")
        print("-" * 80)
        for _, r in results_df.sort_values("R2", ascending=False).iterrows():
            abbr = MODEL_MAP.get(r['model'], r['model'])
            print(
                f"{abbr:<10} | {r['R2']:<8.4f} | {r['train_R2']:<8.4f} | {r['RMSE']:<8.4f} | {r['train_time_s']:<8.2f} | {r['predict_time_s']:<8.4f}")
        print("=" * 80)
        print("\n========== 模型排行 (按 R² 降序) ==========")
        print(results_df[["model", "RMSE", "MSE", "MAE", "R2",
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
