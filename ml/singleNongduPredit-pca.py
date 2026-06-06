# -*- coding: utf-8 -*-
"""
EEM (Excitation-Emission Matrix) Fluorescence Spectroscopy
Antibiotic Concentration Regression Pipeline
   ==================  PCA 降维特征提取版  ==================

与 singleNongduPredit-features.py 的唯一区别:
    特征提取方法由"手工统计 + 边际谱 + Top-K 峰"换成
    PCA(主成分分析) 无监督降维:
      flatten(EEM) -> StandardScaler(像素级) -> PCA(n_components)
    所有经典/深度模型在 PCA 主成分得分上训练与预测。
其他配置、模型集合、可视化、保存路径策略全部保持一致
(输出目录改为 `figure_pca / models_pca / results_pca.csv`)。

参考 nongdu-GB.py 的结构与命名风格。
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

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
DATA_DIR = r"C:\Users\yafex\Desktop\huan"
FIGURE_DIR = os.path.join(DATA_DIR, "figure_pca")
MODEL_DIR = os.path.join(DATA_DIR, "models_pca")
RESULT_CSV = os.path.join(DATA_DIR, "results_pca.csv")

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_SEARCH = 20
LOG_TARGET = True
DEVICE = torch.device("cpu")

# -------- PCA 降维超参 --------
# 若为 float (0,1]) -> 累计方差比阈值 (例 0.99 = 保留 99% 方差);
# 若为 int        -> 直接指定主成分个数;
# 推荐先用 0.99 自动定 K, 跑完再看 n_components_ 决定要不要锁定。
PCA_N_COMPONENTS: float | int = 0.99
PCA_WHITEN = False

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

# MODEL_MAP = {
#     "RandomForest": "RF", "GradientBoosting": "GBRT", "SVR": "SVR",
#     "KernelRidge": "KRR", "XGBoost": "XGB",
#     "CNN2D": "CNN"
# }

MODEL_MAP = {
    "RandomForest": "RF", "GradientBoosting": "GBRT", "SVR": "SVR",
    "XGBoost": "XGB",
    "CNN2D": "CNN"
}



def set_nature_style() -> None:
    plt.style.use("seaborn-v0_8-white")
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
        "mathtext.fontset": "stix",
    })


COLOR_CYCLE = ['#440154', '#31688e', '#35b779', '#fde725']


def plot_relative_error(y_test, y_pred, model_name):
    rel_error = np.abs(y_test - y_pred) / (y_test + 1e-9) * 100
    fig, ax = plt.subplots(figsize=(8.5 * CM, 7 * CM))
    ax.scatter(y_test, rel_error, s=10, c='#31688e', alpha=0.6, edgecolors='none')
    ax.set_xscale('log')
    ax.axhline(10, color='red', ls='--', lw=0.8, label='10% Error')
    ax.set_xlabel('True Concentration (µg/L)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title(f'Prediction Stability - {model_name}')
    save_fig(fig, f"relative_error_{model_name}")


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.png"), dpi=600)
    plt.close(fig)


# ============================================================
# 任务 1: 数据读取
# ============================================================
FNAME_RE = re.compile(r"^([\-+0-9.eE]+)\s*-", )


def parse_concentration(fname: str) -> float | None:
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
# 任务 2-3: 划分 + 像素标准化 + PCA 降维 (核心改动)
# ============================================================
def stratified_regression_split(y: np.ndarray, n_bins: int = 10
                                ) -> np.ndarray:
    n_bins = min(n_bins, max(2, len(np.unique(y))))
    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(y, bins=n_bins, labels=False)
    return np.asarray(bins)


def preprocess_data_pca(X: np.ndarray, y: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray,
                                   StandardScaler, PCA, List[str]]:
    """
    1) flatten 每个 EEM 到 (N, n_ex*n_em)
    2) train_test_split (用浓度分箱分层)
    3) 在训练集上 fit StandardScaler (像素级) → 同步 transform 测试集
    4) 在标准化后的训练集上 fit PCA(n_components=PCA_N_COMPONENTS)
       → 同步 transform 测试集
    返回训练/测试主成分得分、scaler、pca、特征名(PC1..PCk)。
    """
    X_flat = X.reshape(X.shape[0], -1)
    strat = stratified_regression_split(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_flat, y, test_size=TEST_SIZE,
        random_state=SEED, stratify=strat,
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    pca = PCA(n_components=PCA_N_COMPONENTS,
              whiten=PCA_WHITEN,
              random_state=SEED).fit(X_tr_s)
    X_tr_p = pca.transform(X_tr_s).astype(np.float32)
    X_te_p = pca.transform(X_te_s).astype(np.float32)

    k = pca.n_components_
    cum_var = float(pca.explained_variance_ratio_.sum())
    feature_names = [f"PC{i+1}" for i in range(k)]
    print(f"[Task 2-3] PCA: ({X_flat.shape[1]}) -> ({k})  "
          f"cum_var={cum_var:.4f}")
    print(f"           Train PC: {X_tr_p.shape} | Test PC: {X_te_p.shape}")
    return X_tr_p, X_te_p, y_tr, y_te, scaler, pca, feature_names


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
        # "KernelRidge": (
        #     KernelRidge(kernel="rbf"),
        #     {"alpha": [1e-3, 1e-2, 1e-1, 1],
        #      "gamma": [1e-4, 1e-3, 1e-2, "scale"]},
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
    g = grid.get("gamma", [])
    new_g = [(1.0 / n_features) if v == "scale" else v for v in g]
    grid = dict(grid)
    grid["gamma"] = new_g
    return grid


def train_ml_models(X_train: np.ndarray, y_train: np.ndarray,
                    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    print("[Task 4a] 训练经典回归模型 + 随机搜索超参 (PCA 主成分输入) ...")
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
# 任务 4b: 深度学习模型 (2D-CNN, PCA 主成分填充为方形伪 2D)
# ============================================================
class CNN2DReg(nn.Module):
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
        x = x.contiguous().view(-1, 1, self.n_ex, self.n_em)
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
        if (ep + 1) % 5 == 0:
            print(f"    epoch {ep + 1:3d}/{epochs}  MSE={hist[-1]:.4f}")
    return model, hist


def _predict_torch_reg(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        yh = model(torch.from_numpy(X).float().to(DEVICE)).cpu().numpy()
    return yh.astype(np.float32)


class TorchRegressor:
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


def _pad_to_square(X_feat: np.ndarray) -> Tuple[np.ndarray, int]:
    n, F = X_feat.shape
    side = int(np.ceil(np.sqrt(F)))
    pad = side * side - F
    if pad == 0:
        return X_feat, side
    return np.pad(X_feat, ((0, 0), (0, pad)),
                  mode="constant", constant_values=0.0), side


def train_dl_models(X_train: np.ndarray, y_train: np.ndarray
                    ) -> Tuple[Dict[str, TorchRegressor],
                               Dict[str, Dict[str, float]],
                               int, np.ndarray]:
    print("[Task 4b] 训练深度学习回归模型 (基于 PCA 主成分) ...")
    X_pad, side = _pad_to_square(X_train)
    print(f"      主成分 {X_train.shape} -> 填充至 ({X_pad.shape[1]},) "
          f"= ({side} x {side}) 供 CNN 使用")
    in_dim = X_pad.shape[1]
    y_fit = np.log1p(y_train) if LOG_TARGET else y_train
    trained: Dict[str, TorchRegressor] = {}
    times: Dict[str, Dict[str, float]] = {}

    for name, arch in [("CNN2D", "CNN2D")]:
        print(f"  - {name} ...")
        t0 = time.time()
        reg = TorchRegressor(arch, in_dim, n_ex=side, n_em=side,
                             epochs=80, batch=32, lr=1e-3)
        reg.fit(X_pad, y_fit)
        t_fit = time.time() - t0
        trained[name] = reg
        times[name] = {"train_time_s": t_fit}
        torch.save(reg.model.state_dict(),
                   os.path.join(MODEL_DIR, f"{name}.pt"))
        print(f"      fit time = {t_fit:.1f}s")
    return trained, times, side, X_pad


# ============================================================
# 任务 4c: 评估
# ============================================================
def _inv_target(y_log_or_raw: np.ndarray) -> np.ndarray:
    return np.expm1(y_log_or_raw) if LOG_TARGET else y_log_or_raw


def evaluate_models(models: Dict[str, Any],
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    times: Dict[str, Dict[str, float]],
                    dl_inputs: Dict[str, Tuple[np.ndarray, np.ndarray]]
                                | None = None,
                    ) -> pd.DataFrame:
    print("[Task 4c] 在训练集和测试集上评估 ...")
    rows = []
    y_train_fit = np.log1p(y_train) if LOG_TARGET else y_train
    dl_inputs = dl_inputs or {}

    for name, mdl in models.items():
        X_tr_use, X_te_use = dl_inputs.get(name, (X_train, X_test))

        t_start = time.time()
        y_pred_fit = mdl.predict(X_te_use)
        t_pred = time.time() - t_start
        y_pred = _inv_target(y_pred_fit)

        y_train_pred = _inv_target(mdl.predict(X_tr_use))

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
            "train_MSE": float(mean_squared_error(y_train, y_train_pred)),
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
    df = results_df.copy()
    df["abbr"] = df["model"].map(lambda x: MODEL_MAP.get(x, x))
    df = df.sort_values("R2", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 6.5 * CM))
    metrics = [("R2", "train_R2", "$R^2$"),
               ("RMSE", "train_RMSE", "RMSE"),
               ("MSE", "train_MSE", "MSE")]

    x = np.arange(len(df))
    width = 0.38

    c_train = "#1F4E79"
    c_test = "#C0504D"

    for i, (m_test, m_train, title) in enumerate(metrics):
        ax = axes[i]
        # 不再在柱顶标注数值
        ax.bar(x - width / 2, df[m_train], width, label='Train', color=c_train, edgecolor='white', lw=0.4)
        ax.bar(x + width / 2, df[m_test], width, label='Test', color=c_test, edgecolor='white', lw=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(df["abbr"], rotation=45, ha='right', fontsize=7)
        ax.set_title(title, fontweight='bold', fontsize=9, pad=15)
        ax.legend(frameon=False, loc='best', fontsize=6)
        sns.despine(ax=ax)
        ax.yaxis.grid(True, linestyle='--', alpha=0.1)

    fig.tight_layout()
    save_fig(fig, "fig1_bar_with_labels")


def plot_pred_vs_true(results_df: pd.DataFrame, y_test: np.ndarray) -> None:
    df = results_df.sort_values("R2", ascending=False).reset_index(drop=True)
    n = len(df)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, 7 * nrows * CM))
    axes = np.atleast_2d(axes).ravel()

    color_train = "#1F4E79"
    color_test = "#C0504D"

    for i, row in df.iterrows():
        ax = axes[i]
        abbr = MODEL_MAP.get(row["model"], row["model"])
        yp_test = np.asarray(row["y_pred"])
        yp_train = np.asarray(row["y_train_pred"])

        all_vals = np.concatenate([y_test, yp_test, y_tr_global, yp_train])
        low, high = all_vals.min(), all_vals.max()
        ax.plot([low, high], [low, high], color='#333333', ls='--', lw=1, alpha=0.6, zorder=1)

        ax.scatter(y_tr_global, yp_train, s=12, alpha=0.5, c=color_train,
                   label='Train', edgecolors='none', zorder=2)

        ax.scatter(y_test, yp_test, s=28, alpha=0.9, c=color_test,
                   label='Test', edgecolors='white', linewidths=0.7, zorder=3)

        ax.set_title(f"Model: {abbr}", fontsize=10, loc='left', fontweight='bold')
        ax.set_xlabel("Measured (µg/L)", fontsize=8)
        ax.set_ylabel("Predicted (µg/L)", fontsize=8)

        ax.legend(frameon=True, facecolor='white', framealpha=0.8,
                  loc='upper left', fontsize=7, markerscale=1.2)

        sns.despine(ax=ax)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()

    output_path = os.path.join(FIGURE_DIR, "fig2_4k_clear_scatter")
    fig.savefig(f"{output_path}.png", dpi=600, bbox_inches='tight')
    fig.savefig(f"{output_path}.pdf", bbox_inches='tight')
    print(f"高清散点图已生成：{output_path}.png")
    plt.close(fig)


def plot_residuals(results_df: pd.DataFrame, y_test: np.ndarray) -> None:
    best = results_df.sort_values("R2", ascending=False).iloc[0]
    yp = np.asarray(best["y_pred"])
    res = y_test - yp

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.33))
    color = sns.color_palette("viridis", 6)[1]

    axes[0].scatter(yp, res, s=14, alpha=0.7,
                    color=color, edgecolor="white", linewidth=0.3)
    axes[0].axhline(0, color="k", lw=1, ls="--")
    axes[0].set_xlabel("Predicted (µg/L)")
    axes[0].set_ylabel("Residual = True - Pred (µg/L)")
    axes[0].set_title(f"(a) Residual vs Predicted — {best['model']}")

    sns.histplot(res, bins=30, kde=True, ax=axes[1], color=color,
                 edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("Residual (µg/L)")
    axes[1].set_title("(b) Residual distribution")

    stats.probplot(res, dist="norm", plot=axes[2])
    axes[2].get_lines()[0].set_markerfacecolor(color)
    axes[2].get_lines()[0].set_markeredgecolor("white")
    axes[2].get_lines()[0].set_markersize(5)
    axes[2].get_lines()[1].set_color("k")
    axes[2].set_title("(c) Normal Q-Q plot")

    fig.tight_layout()
    save_fig(fig, "fig3_residual_analysis")


def plot_feature_importance(models: Dict[str, Any],
                            feature_names: List[str],
                            top_n: int = 20) -> None:
    """Tree-ensemble 的 Top-N 重要主成分条形图。"""
    top_n = min(top_n, len(feature_names))
    for name in ("RandomForest", "GradientBoosting", "XGBoost"):
        if name not in models:
            continue
        imp = getattr(models[name], "feature_importances_", None)
        if imp is None:
            continue

        idx = np.argsort(imp)[::-1][:top_n]
        labels = [feature_names[i] for i in idx]
        vals = imp[idx]

        fig, ax = plt.subplots(
            figsize=(SINGLE_COL * 2.0, 0.42 * top_n + 1.4)
        )

        colors = sns.color_palette("viridis", top_n)
        ax.barh(
            range(top_n)[::-1],
            vals,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_yticks(range(top_n)[::-1])
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Feature importance")
        ax.set_title(f"Top-{top_n} principal components — {name}")
        ax.margins(y=0.01)

        sns.despine(ax=ax)

        fig.tight_layout()
        fig.subplots_adjust(left=0.28)

        save_fig(fig, f"fig4a_feature_importance_{name}")


def plot_pca_scatter(X_pc: np.ndarray, y: np.ndarray) -> None:
    """PCA 投影 (PC1 vs PC2), 按浓度连续上色。"""
    if X_pc.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.2))
    sc = ax.scatter(X_pc[:, 0], X_pc[:, 1], c=y, cmap="viridis", s=18,
                    alpha=0.85, edgecolor="white", linewidth=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Concentration (µg/L)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection (training features used by models)")
    sns.despine()
    fig.tight_layout()
    save_fig(fig, "fig4b_pca")


def plot_pca_variance(pca: PCA) -> None:
    """PCA 解释方差比 + 累计曲线 (PCA 降维专用诊断图)。"""
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    k = len(ratios)
    x = np.arange(1, k + 1)

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.1))
    ax.bar(x, ratios, color="#31688e",
           edgecolor="white", linewidth=0.3, label="Individual")
    ax2 = ax.twinx()
    ax2.plot(x, cum, "o-", color="#C0504D", lw=1.2,
             markersize=3, label="Cumulative")
    ax2.axhline(0.99, color="grey", ls="--", lw=0.7)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax2.set_ylabel("Cumulative")
    ax.set_title(f"PCA explained variance (k = {k}, "
                 f"cum = {cum[-1]:.3f})")
    sns.despine(ax=ax, right=False)
    fig.tight_layout()
    save_fig(fig, "fig4c_pca_explained_variance")


def plot_learning_curves(models: Dict[str, Any],
                         X_train: np.ndarray, y_train: np.ndarray) -> None:
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
            tr_m = -train_scores.mean(1)
            tr_s = train_scores.std(1)
            va_m = -val_scores.mean(1)
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
                      feature_names: List[str],
                      pca: PCA,
                      dl_models: Dict[str, Any]) -> None:
    print("[Task 5] 生成 Nature 风格图表 ...")
    set_nature_style()
    plot_performance_bar(results_df)
    plot_pred_vs_true(results_df, y_test)
    plot_residuals(results_df, y_test)
    plot_feature_importance(models, feature_names, top_n=20)
    plot_pca_scatter(np.vstack([X_train, X_test]),
                     np.concatenate([y_train, y_test]))
    plot_pca_variance(pca)
    plot_learning_curves(models, X_train, y_train)
    plot_dl_loss(dl_models)
    print(f"[Task 5] 图表保存至: {FIGURE_DIR}")


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    try:
        print("=" * 64)
        print("EEM Antibiotic Concentration Regression Pipeline (PCA)")
        print("=" * 64)
        print(f"Device           : {DEVICE}")
        print(f"Data directory   : {DATA_DIR}")
        print(f"Figure directory : {FIGURE_DIR}")
        print(f"Model directory  : {MODEL_DIR}")
        print(f"Target transform : {'log1p' if LOG_TARGET else 'identity'}")
        print(f"PCA n_components : {PCA_N_COMPONENTS}  whiten={PCA_WHITEN}")
        print("-" * 64)

        # 1. 加载原始 EEM
        X_raw, y, meta_df, ex_axis, em_axis = load_and_parse_data(DATA_DIR)

        # 2-3. PCA 降维 (核心改动) + 训练/测试集划分
        global y_tr_global
        X_tr, X_te, y_tr, y_te, scaler, pca, feature_names = \
            preprocess_data_pca(X_raw, y)
        y_tr_global = y_tr

        # 持久化预处理器
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
        joblib.dump(pca,    os.path.join(MODEL_DIR, "pca.joblib"))
        joblib.dump({"feature_names": feature_names,
                     "n_components": int(pca.n_components_),
                     "explained_variance_ratio":
                         pca.explained_variance_ratio_.tolist()},
                    os.path.join(MODEL_DIR, "pca_info.joblib"))

        # 4a. 经典模型 + 超参搜索
        ml_models, ml_times = train_ml_models(X_tr, y_tr)

        # 4b. 深度模型 (PCA 主成分填充到方形供 2D-CNN)
        dl_models, dl_times, side, X_tr_pad = train_dl_models(X_tr, y_tr)
        X_te_pad, _ = _pad_to_square(X_te)

        all_models: Dict[str, Any] = {**ml_models, **dl_models}
        all_times = {**ml_times, **dl_times}
        dl_inputs = {name: (X_tr_pad, X_te_pad) for name in dl_models}

        # 4c. 评估
        results_df = evaluate_models(all_models, X_tr, y_tr, X_te, y_te,
                                     all_times, dl_inputs=dl_inputs)

        # 5. 可视化
        plot_nature_style(results_df, all_models,
                          X_tr, y_tr, X_te, y_te,
                          feature_names, pca, dl_models)

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
