# -*- coding: utf-8 -*-
"""
EEM Mixture Antibiotic — MULTI-OUTPUT Regression Pipeline
   ============  手工特征提取版 (基于 v2)  ============

相对 v2 的唯一改动:
    在 load_and_parse_data 与 preprocess 之间插入 extract_eem_features(),
    把每个 (n_ex, n_em) 的 EEM 矩阵展平为一组手工特征向量
      (全局统计 + Ex/Em 边际谱 + Top-K 峰强度),
    所有多输出模型基于提取后的特征训练与预测。
其余逻辑 (划分/训练/评估/全部绘图) 全部与 v2 一致;
输出目录改名为 `figure_multi_feat / models_multi_feat / results_multi_feat.csv`
以避免覆盖 v2 结果。

只使用 SVR / RandomForest / GradientBoosting / XGBoost 四个多输出模型。
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

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

warnings.filterwarnings("ignore")

# ============================================================
# 全局配置
# ============================================================
DATA_DIR   = r"C:\Users\yafex\Desktop\hunhe"
FIGURE_DIR = os.path.join(DATA_DIR, "figure_multi_feat")
MODEL_DIR  = os.path.join(DATA_DIR, "models_multi_feat")
RESULT_CSV = os.path.join(DATA_DIR, "results_multi_feat.csv")

TARGETS = ["Ofloxacin", "Ciprofloxacin"]
N_TARGETS = len(TARGETS)

SEED          = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5
N_ITER_SEARCH = 15
LOG_TARGET    = True

# 手工特征提取超参 (与 singleNongduPredit-features.py 一致)
TOPK_PEAKS = 5

np.random.seed(SEED)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 配色 ---
COLOR_TRAIN = "#1F4E79"   # 训练集 / 环丙沙星
COLOR_TEST  = "#C0504D"   # 测试集 / 氧氟沙星
# 按抗生素配色 (按 fig4/fig5 需求: 环丙=#1F4E79, 氧氟=#C0504D)
COLOR_BY_TARGET = {
    "Ciprofloxacin": "#1F4E79",
    "Ofloxacin":     "#C0504D",
}

MODEL_MAP = {
    "RandomForest": "RF", "GradientBoosting": "GBRT",
    "SVR": "SVR", "XGBoost": "XGB",
}

CM = 1.0 / 2.54
SINGLE_COL = 8.5 * CM
DOUBLE_COL = 17.5 * CM


def set_nature_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-white")
    except OSError:
        plt.style.use("seaborn-white")
    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif"],
        "font.size":        8,
        "axes.titlesize":   9,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "legend.fontsize":  7,
        "axes.linewidth":   0.8,
        "savefig.dpi":      600,
        "pdf.fonttype":     42,
        "mathtext.fontset": "stix",
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.png"),
                dpi=600, bbox_inches="tight")
    plt.close(fig)


def _abbr(name: str) -> str:
    return MODEL_MAP.get(name, name)


# ============================================================
# 任务 1: 数据读取
# ============================================================
def parse_concentrations(fname: str) -> Tuple[float, float] | None:
    stem = os.path.splitext(fname)[0]
    parts = stem.split("-")
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def load_and_parse_data(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "*.xlsx")))
    print(f"[Task 1] 发现 {len(files)} 个 Excel 文件 ...")
    if not files:
        raise FileNotFoundError(f"{folder} 内未找到 .xlsx")

    X_list, Y_list, meta = [], [], []
    ex_axis = em_axis = None
    expected = None
    skipped = 0

    for i, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        c = parse_concentrations(fname)
        if c is None:
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
                skipped += 1
                continue
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")
            skipped += 1
            continue
        X_list.append(mat)
        Y_list.append(c)
        meta.append({"file": fname,
                     "Ofloxacin": c[0], "Ciprofloxacin": c[1]})
        if i % 200 == 0 or i == len(files):
            print(f"  - 已处理 {i}/{len(files)}")

    X = np.stack(X_list, 0)
    Y = np.asarray(Y_list, dtype=np.float32)
    print(f"[Task 1] 有效 {len(Y)} | 跳过 {skipped} | EEM {expected}")
    for j, t in enumerate(TARGETS):
        print(f"          {t}: [{Y[:, j].min():.4g}, {Y[:, j].max():.4g}]")
    return X, Y, pd.DataFrame(meta), ex_axis, em_axis


# ============================================================
# 任务 1.5: 特征提取 (新增) — 把 (N,n_ex,n_em) -> (N, F) 手工特征
# ============================================================
def extract_eem_features(X: np.ndarray,
                         ex_axis: np.ndarray,
                         em_axis: np.ndarray,
                         topk_peaks: int = TOPK_PEAKS
                         ) -> Tuple[np.ndarray, List[str]]:
    """
    对每个样本的 EEM 矩阵提取手工特征:
      1) 9 个全局统计量: mean / std / max / min / sum / p25 / p50 / p75 / p90
      2) 沿 Em 方向的平均/最大 → 每个 Ex 波长一个值 (2 * n_ex)
      3) 沿 Ex 方向的平均/最大 → 每个 Em 波长一个值 (2 * n_em)
      4) 全图 Top-K 峰强度 (topk_peaks 个)
    返回:
      features : (N, F) float32
      names    : 长度 F 的特征名 (用于可解释性绘图)
    """
    n, n_ex, n_em = X.shape
    feats: List[np.ndarray] = []
    names: List[str] = []

    flat = X.reshape(n, -1).astype(np.float64)
    feats.append(flat.mean(axis=1, keepdims=True));    names.append("g_mean")
    feats.append(flat.std(axis=1, keepdims=True));     names.append("g_std")
    feats.append(flat.max(axis=1, keepdims=True));     names.append("g_max")
    feats.append(flat.min(axis=1, keepdims=True));     names.append("g_min")
    feats.append(flat.sum(axis=1, keepdims=True));     names.append("g_sum")
    for q, tag in [(25, "p25"), (50, "p50"), (75, "p75"), (90, "p90")]:
        feats.append(np.percentile(flat, q, axis=1, keepdims=True))
        names.append(f"g_{tag}")

    ex_mean = X.mean(axis=2)
    ex_max  = X.max(axis=2)
    feats.append(ex_mean)
    names += [f"ExMean@Ex{ex_axis[i]:.0f}" for i in range(n_ex)]
    feats.append(ex_max)
    names += [f"ExMax@Ex{ex_axis[i]:.0f}"  for i in range(n_ex)]

    em_mean = X.mean(axis=1)
    em_max  = X.max(axis=1)
    feats.append(em_mean)
    names += [f"EmMean@Em{em_axis[j]:.0f}" for j in range(n_em)]
    feats.append(em_max)
    names += [f"EmMax@Em{em_axis[j]:.0f}"  for j in range(n_em)]

    sorted_desc = -np.sort(-flat, axis=1)[:, :topk_peaks]
    feats.append(sorted_desc)
    names += [f"Peak#{k+1}" for k in range(topk_peaks)]

    F = np.concatenate(feats, axis=1).astype(np.float32)
    print(f"[Task 1.5] 特征提取: {X.shape} -> {F.shape} "
          f"(全局 9 + 2*Ex({n_ex}) + 2*Em({n_em}) + TopK({topk_peaks}))")
    return F, names


# ============================================================
# 任务 2: 划分 + 标准化 (输入改为特征矩阵)
# ============================================================
def stratified_2d(Y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    labs = []
    for j in range(Y.shape[1]):
        nb = min(n_bins, max(2, len(np.unique(Y[:, j]))))
        try:
            b = pd.qcut(Y[:, j], q=nb, labels=False, duplicates="drop")
        except ValueError:
            b = pd.cut(Y[:, j], bins=nb, labels=False)
        labs.append(np.asarray(b))
    return labs[0] * (labs[1].max() + 1) + labs[1]


def preprocess(X_feat: np.ndarray, Y: np.ndarray):
    """X_feat 已为 (N, F) 提取特征。"""
    X_flat = X_feat if X_feat.ndim == 2 else X_feat.reshape(X_feat.shape[0], -1)
    strat = stratified_2d(Y)
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
    print(f"[Task 2] Train {X_tr_s.shape} | Test {X_te_s.shape}")
    return X_tr_s, X_te_s, Y_tr, Y_te, scaler


# ============================================================
# 任务 3: 多输出模型 + 随机搜索 (仅 SVR / RF / GBRT / XGB)
# ============================================================
def _build_search_spaces() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    space: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
    space["RandomForest"] = (
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        {"n_estimators":      [200, 400, 600],
         "max_depth":         [None, 10, 20, 40],
         "min_samples_split": [2, 4, 8],
         "max_features":      ["sqrt", 0.3, 0.5]},
    )
    space["SVR"] = (
        MultiOutputRegressor(SVR()),
        {"estimator__C":       [0.1, 1, 10, 100],
         "estimator__gamma":   ["scale", 1e-3, 1e-4],
         "estimator__epsilon": [0.01, 0.05, 0.1]},
    )
    space["GradientBoosting"] = (
        MultiOutputRegressor(GradientBoostingRegressor(random_state=SEED)),
        {"estimator__n_estimators":  [100, 200, 400],
         "estimator__max_depth":     [2, 3, 4],
         "estimator__learning_rate": [0.03, 0.05, 0.1],
         "estimator__subsample":     [0.7, 1.0]},
    )
    if HAVE_XGB:
        space["XGBoost"] = (
            MultiOutputRegressor(
                XGBRegressor(random_state=SEED, n_jobs=-1,
                             verbosity=0, tree_method="hist")),
            {"estimator__n_estimators":     [200, 400, 600],
             "estimator__max_depth":        [3, 5, 7],
             "estimator__learning_rate":    [0.03, 0.05, 0.1],
             "estimator__subsample":        [0.7, 0.9, 1.0],
             "estimator__colsample_bytree": [0.6, 0.8, 1.0]},
        )
    else:
        print("[Note] 未检测到 xgboost, 跳过 XGBoost。")
    return space


def train_ml_models(X_tr: np.ndarray, Y_tr: np.ndarray
                    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
    print("[Task 3] 训练多输出经典模型 ...")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    Y_fit = np.log1p(Y_tr) if LOG_TARGET else Y_tr
    trained: Dict[str, Any] = {}
    times: Dict[str, float] = {}

    for name, (est, grid) in _build_search_spaces().items():
        print(f"  - {name}: RandomizedSearchCV ...", flush=True)
        t0 = time.time()
        search = RandomizedSearchCV(
            est, grid, n_iter=N_ITER_SEARCH, cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1, random_state=SEED, refit=True)
        search.fit(X_tr, Y_fit)
        dt = time.time() - t0
        trained[name] = search.best_estimator_
        times[name] = dt
        joblib.dump(search.best_estimator_,
                    os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"      best={search.best_params_}")
        print(f"      cv RMSE(log)={-search.best_score_:.4f}  time={dt:.1f}s")
    return trained, times


# ============================================================
# 任务 4: 评估
# ============================================================
def _inv(Y_fit: np.ndarray) -> np.ndarray:
    return np.expm1(Y_fit) if LOG_TARGET else Y_fit


def evaluate_models(models: Dict[str, Any],
                    X_tr: np.ndarray, Y_tr: np.ndarray,
                    X_te: np.ndarray, Y_te: np.ndarray,
                    times: Dict[str, float]) -> pd.DataFrame:
    print("[Task 4] 评估 (训练 + 测试) ...")
    rows = []
    for name, mdl in models.items():
        t0 = time.time()
        Yp_te = _inv(np.asarray(mdl.predict(X_te)))
        t_pred = time.time() - t0
        Yp_tr = _inv(np.asarray(mdl.predict(X_tr)))

        per_target = {}
        for j, t in enumerate(TARGETS):
            per_target[t] = {
                "train_R2":   float(r2_score(Y_tr[:, j], Yp_tr[:, j])),
                "test_R2":    float(r2_score(Y_te[:, j], Yp_te[:, j])),
                "train_RMSE": float(np.sqrt(mean_squared_error(Y_tr[:, j], Yp_tr[:, j]))),
                "test_RMSE":  float(np.sqrt(mean_squared_error(Y_te[:, j], Yp_te[:, j]))),
                "train_MSE":  float(mean_squared_error(Y_tr[:, j], Yp_tr[:, j])),
                "test_MSE":   float(mean_squared_error(Y_te[:, j], Yp_te[:, j])),
                "train_MAE":  float(mean_absolute_error(Y_tr[:, j], Yp_tr[:, j])),
                "test_MAE":   float(mean_absolute_error(Y_te[:, j], Yp_te[:, j])),
            }
        mean_test_r2  = float(np.mean([per_target[t]["test_R2"]  for t in TARGETS]))
        mean_train_r2 = float(np.mean([per_target[t]["train_R2"] for t in TARGETS]))
        overall_test_r2 = float(r2_score(Y_te, Yp_te,
                                         multioutput="uniform_average"))
        total_test_rmse = float(np.sqrt(mean_squared_error(Y_te, Yp_te)))
        total_train_rmse = float(np.sqrt(mean_squared_error(Y_tr, Yp_tr)))

        row = {
            "model": name,
            "mean_test_R2": mean_test_r2,
            "mean_train_R2": mean_train_r2,
            "overall_test_R2": overall_test_r2,
            "total_test_RMSE": total_test_rmse,
            "total_train_RMSE": total_train_rmse,
            "train_time_s": times.get(name, np.nan),
            "predict_time_s": t_pred,
            "_Yp_te": Yp_te, "_Yp_tr": Yp_tr,
        }
        for t in TARGETS:
            for k, v in per_target[t].items():
                row[f"{t}__{k}"] = v
        rows.append(row)
        print(f"  - {name:18s} | meanTestR² {mean_test_r2:.4f} "
              f"| totalTestRMSE {total_test_rmse:.4g}")
        for t in TARGETS:
            p = per_target[t]
            print(f"       {t:15s} R²={p['test_R2']:.4f} "
                  f"RMSE={p['test_RMSE']:.4g} MAE={p['test_MAE']:.4g}")
    return pd.DataFrame(rows)


# ============================================================
# 任务 5: 可视化
# ============================================================
def _add_bar_labels(ax, rects, fmt="{:.3f}", size=5.5):
    for r in rects:
        h = r.get_height()
        pad = abs(h) * 0.01 + 1e-6
        ax.text(r.get_x() + r.get_width() / 2.0, h + pad,
                fmt.format(h), ha="center", va="bottom",
                fontsize=size, fontweight="bold")


# ---------- 图 1: 每种抗生素一张散点图, 模型按 3 列网格排列 ----------
def plot_pred_vs_true_per_target_model(results: pd.DataFrame,
                                       Y_tr: np.ndarray,
                                       Y_te: np.ndarray) -> None:
    """
    4K 高清散点图: 每种抗生素一张, 子图按模型 3 列网格,
    样式严格对齐 singleNongduPredit-new.py 的 plot_pred_vs_true。
      - 训练集 #1F4E79 (Train), 测试集 #C0504D (Test);
      - 子图标题 "Model: {abbr}";
      - 每张子图独立图例; 单张图按 meanTestR² 降序排模型;
      - dpi=600 高清 png + pdf 保存。
    """
    df = results.sort_values("mean_test_R2", ascending=False).reset_index(drop=True)
    n = len(df)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    color_train = COLOR_TRAIN   # #1F4E79
    color_test  = COLOR_TEST    # #C0504D

    for t in TARGETS:
        j = TARGETS.index(t)
        y_tr = Y_tr[:, j]
        y_te = Y_te[:, j]

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(DOUBLE_COL, 7 * nrows * CM))
        axes = np.atleast_2d(axes).ravel()

        last_i = -1
        for i, (_, row) in enumerate(df.iterrows()):
            ax = axes[i]
            last_i = i
            abbr = _abbr(row["model"])
            yp_test  = np.asarray(row["_Yp_te"])[:, j]
            yp_train = np.asarray(row["_Yp_tr"])[:, j]

            # 1. 绘制对角线 (置于最底层)
            all_vals = np.concatenate([y_te, yp_test, y_tr, yp_train])
            low, high = all_vals.min(), all_vals.max()
            ax.plot([low, high], [low, high],
                    color="#333333", ls="--", lw=1, alpha=0.6, zorder=1)

            # 2. 训练集: alpha 0.5
            ax.scatter(y_tr, yp_train, s=12, alpha=0.5, c=color_train,
                       label="Train", edgecolors="none", zorder=2)

            # 3. 测试集: 深红色 + 白色描边
            ax.scatter(y_te, yp_test, s=28, alpha=0.9, c=color_test,
                       label="Test", edgecolors="white",
                       linewidths=0.7, zorder=3)

            ax.set_title(f"Model: {abbr}", fontsize=10,
                         loc="left", fontweight="bold")
            ax.set_xlabel("Measured (µg/L)", fontsize=8)
            ax.set_ylabel("Predicted (µg/L)", fontsize=8)

            ax.legend(frameon=True, facecolor="white", framealpha=0.8,
                      loc="upper left", fontsize=7, markerscale=1.2)

            sns.despine(ax=ax)
            ax.tick_params(labelsize=7)

        # 隐藏多余格子
        for jj in range(last_i + 1, len(axes)):
            axes[jj].axis("off")

        fig.suptitle(f"Measured vs Predicted — {t}",
                     fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()

        output_path = os.path.join(FIGURE_DIR, f"fig1_pred_vs_true_{t}")
        fig.savefig(f"{output_path}.png", dpi=600, bbox_inches="tight")
        fig.savefig(f"{output_path}.pdf", bbox_inches="tight")
        print(f"高清散点图已生成: {output_path}.png")
        plt.close(fig)


# ---------- 图 3: 每种抗生素的 R²/RMSE/MSE — 全模型 Train vs Test ----------
def plot_metrics_train_vs_test_per_target(results: pd.DataFrame) -> None:
    """
    对每种抗生素生成一张图: 3 子图 (R²/RMSE/MSE),
    X 轴=模型, 每模型两根柱 (Train #1F4E79, Test #C0504D),
    包含全部模型, 柱顶标数值。
    """
    df = results.sort_values("mean_test_R2", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    width = 0.38

    for t in TARGETS:
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 6.5 * CM))
        metrics = [("R2", "$R^2$", "{:.3f}"),
                   ("RMSE", "RMSE (µg/L)", "{:.3g}"),
                   ("MSE", "MSE (µg/L$^2$)", "{:.3g}")]
        for i, (metric, label, fmt) in enumerate(metrics):
            ax = axes[i]
            tr = df[f"{t}__train_{metric}"].values
            te = df[f"{t}__test_{metric}"].values
            ax.bar(x - width / 2, tr, width, color=COLOR_TRAIN,
                   edgecolor="white", lw=0.5, label="Train")
            ax.bar(x + width / 2, te, width, color=COLOR_TEST,
                   edgecolor="white", lw=0.5, label="Test")
            ax.set_xticks(x)
            ax.set_xticklabels([_abbr(m) for m in df["model"]],
                               rotation=30, ha="right", fontsize=7)
            ax.set_title(label, fontweight="bold", fontsize=9, pad=12)
            ax.legend(frameon=False, fontsize=6, loc="best")
            sns.despine(ax=ax)
            ax.yaxis.grid(True, linestyle="--", alpha=0.15)
        fig.suptitle(f"{t} — Train vs Test (all models)",
                     fontsize=10, fontweight="bold", y=1.04)
        fig.tight_layout()
        save_fig(fig, f"fig3a_{t}_train_vs_test")


# ---------- 图 4 / 图 5: 训练集 / 测试集 — 各模型按抗生素分色 ----------
def _plot_metrics_by_target(results: pd.DataFrame,
                            split: str,    # "train" or "test"
                            outname: str) -> None:
    df = results.sort_values("mean_test_R2", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    width = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 6.5 * CM))
    metrics = [("R2", "$R^2$", "{:.3f}"),
               ("RMSE", "RMSE (µg/L)", "{:.3g}"),
               ("MSE", "MSE (µg/L$^2$)", "{:.3g}")]
    for i, (metric, label, fmt) in enumerate(metrics):
        ax = axes[i]
        # 第 1 个目标 = Ciprofloxacin (#1F4E79), 第 2 个 = Ofloxacin (#C0504D)
        order = ["Ciprofloxacin", "Ofloxacin"]
        for k, t in enumerate(order):
            vals = df[f"{t}__{split}_{metric}"].values
            offset = (k - 0.5) * width
            ax.bar(x + offset, vals, width,
                   color=COLOR_BY_TARGET[t],
                   edgecolor="white", lw=0.5, label=t)
        ax.set_xticks(x)
        ax.set_xticklabels([_abbr(m) for m in df["model"]],
                           rotation=30, ha="right", fontsize=7)
        ax.set_title(label, fontweight="bold", fontsize=9, pad=12)
        ax.legend(frameon=False, fontsize=6, loc="best")
        sns.despine(ax=ax)
        ax.yaxis.grid(True, linestyle="--", alpha=0.15)
    split_label = "Training set" if split == "train" else "Test set"
    fig.suptitle(f"{split_label} — metrics per model (by target)",
                 fontsize=10, fontweight="bold", y=1.04)
    fig.tight_layout()
    save_fig(fig, outname)


def plot_metrics_train_by_target(results: pd.DataFrame) -> None:
    _plot_metrics_by_target(results, "train", "fig4_train_set_by_target")


def plot_metrics_test_by_target(results: pd.DataFrame) -> None:
    _plot_metrics_by_target(results, "test", "fig5_test_set_by_target")


# ---------- 残差分布 (最佳模型, 两抗生素 × train/test) ----------
def plot_residual_distribution(results: pd.DataFrame,
                               Y_tr: np.ndarray, Y_te: np.ndarray) -> None:
    best = results.sort_values("mean_test_R2", ascending=False).iloc[0]
    Yp_tr = np.asarray(best["_Yp_tr"])
    Yp_te = np.asarray(best["_Yp_te"])
    name = _abbr(best["model"])

    fig, axes = plt.subplots(1, N_TARGETS,
                             figsize=(DOUBLE_COL, 6.5 * CM), sharey=False)
    for j, t in enumerate(TARGETS):
        ax = axes[j]
        res_tr = Y_tr[:, j] - Yp_tr[:, j]
        res_te = Y_te[:, j] - Yp_te[:, j]
        sns.histplot(res_tr, kde=True, ax=ax, color=COLOR_TRAIN,
                     stat="density", alpha=0.45,
                     edgecolor="white", linewidth=0.4,
                     label=f"Train (n={len(res_tr)})")
        sns.histplot(res_te, kde=True, ax=ax, color=COLOR_TEST,
                     stat="density", alpha=0.55,
                     edgecolor="white", linewidth=0.4,
                     label=f"Test (n={len(res_te)})")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("Residual = True - Pred (µg/L)")
        ax.set_ylabel("Density")
        ax.set_title(f"{t}", fontweight="bold")
        ax.legend(frameon=False, fontsize=7)
        sns.despine(ax=ax)
    fig.suptitle(f"Residual distribution — best model: {name}",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig2_residual_distribution")


# ---------- 误差曲线 ----------
def plot_error_curves(results: pd.DataFrame,
                      Y_tr: np.ndarray, Y_te: np.ndarray) -> None:
    best = results.sort_values("mean_test_R2", ascending=False).iloc[0]
    name = _abbr(best["model"])
    Yp_tr = np.asarray(best["_Yp_tr"])
    Yp_te = np.asarray(best["_Yp_te"])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 7 * CM))
    n_tr = Y_tr.shape[0]; n_te = Y_te.shape[0]
    idx_tr = np.arange(n_tr)
    idx_te = np.arange(n_tr, n_tr + n_te)

    for t in TARGETS:
        j = TARGETS.index(t)
        c = COLOR_BY_TARGET[t]
        err_tr = Yp_tr[:, j] - Y_tr[:, j]
        err_te = Yp_te[:, j] - Y_te[:, j]
        ax.plot(idx_tr, err_tr, "-", color=c, lw=0.9, alpha=0.85,
                label=f"{t} (Train)")
        ax.plot(idx_te, err_te, "--", color=c, lw=1.4, alpha=0.95,
                label=f"{t} (Test)")
    ax.axvline(n_tr - 0.5, color="grey", ls=":", lw=0.8)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Sample index  (Train | Test)")
    ax.set_ylabel("Prediction error (Pred - True, µg/L)")
    ax.set_title(f"Prediction error curve — {name}", fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_fig(fig, "fig6_error_curves")


# ---------- PCA 二维投影 (按抗生素浓度上色) ----------
def plot_pca(X: np.ndarray, Y: np.ndarray) -> None:
    """
    对展平后的 EEM 特征做 PCA 二维投影,
    针对每种抗生素生成一张图, 颜色按该抗生素浓度连续上色 (viridis)。
    """
    X_flat = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X_flat)
    var = pca.explained_variance_ratio_

    for j, t in enumerate(TARGETS):
        fig, ax = plt.subplots(
            figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.2))
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=Y[:, j], cmap="viridis",
                        s=18, alpha=0.85,
                        edgecolor="white", linewidth=0.3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(f"{t} concentration (µg/L)")
        ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
        ax.set_title(f"PCA projection of EEM features — {t}")
        sns.despine(ax=ax)
        fig.tight_layout()
        save_fig(fig, f"fig4b_pca_{t}")


# ---------- 特征重要度 (Top-20, 修复左侧标签溢出) ----------
def _multi_feature_importance(mdl: Any) -> np.ndarray | None:
    imp = getattr(mdl, "feature_importances_", None)
    if imp is not None:
        return np.asarray(imp)
    if hasattr(mdl, "estimators_"):
        sub_imps = []
        for sub in mdl.estimators_:
            sub_imp = getattr(sub, "feature_importances_", None)
            if sub_imp is None:
                return None
            sub_imps.append(np.asarray(sub_imp))
        return np.mean(np.stack(sub_imps, 0), axis=0)
    return None


def plot_feature_importance(models: Dict[str, Any],
                            feature_names: List[str],
                            top_n: int = 20) -> None:
    top_n = min(top_n, len(feature_names))
    for name in ("RandomForest", "GradientBoosting", "XGBoost"):
        if name not in models:
            continue
        imp = _multi_feature_importance(models[name])
        if imp is None:
            continue
        idx = np.argsort(imp)[::-1][:top_n]
        labels = [feature_names[i] for i in idx]
        vals = imp[idx]
        # 提取特征标签较长 → 加宽画布 + 大左边距
        fig, ax = plt.subplots(
            figsize=(SINGLE_COL * 2.4, 0.38 * top_n + 1.2))
        colors = sns.color_palette("viridis", top_n)
        y_pos = np.arange(top_n)[::-1]
        ax.barh(y_pos, vals, color=colors,
                edgecolor="black", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Feature importance (mean over targets)")
        ax.set_title(f"Top-{top_n} extracted features — {_abbr(name)} "
                     f"(multi-output)", fontweight="bold")
        ax.margins(y=0.01)
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.subplots_adjust(left=0.38)
        save_fig(fig, f"fig7_feature_importance_{name}")


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    try:
        print("=" * 64)
        print("EEM Multi-Output Antibiotic Regression Pipeline (FEATURE)")
        print("=" * 64)
        print(f"Data directory   : {DATA_DIR}")
        print(f"Figure directory : {FIGURE_DIR}")
        print(f"Model directory  : {MODEL_DIR}")
        print(f"Target transform : {'log1p' if LOG_TARGET else 'identity'}")
        print(f"Targets          : {TARGETS}")
        print("-" * 64)

        X_raw, Y, meta_df, ex_axis, em_axis = load_and_parse_data(DATA_DIR)
        meta_df.to_csv(os.path.join(DATA_DIR, "metadata_multi_feat.csv"),
                       index=False, encoding="utf-8-sig")

        # 特征提取 (核心新增)
        X_feat, feature_names = extract_eem_features(
            X_raw, ex_axis, em_axis, topk_peaks=TOPK_PEAKS)
        joblib.dump({"feature_names": feature_names},
                    os.path.join(MODEL_DIR, "feature_names.joblib"))

        X_tr, X_te, Y_tr, Y_te, scaler = preprocess(X_feat, Y)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

        ml_models, ml_t = train_ml_models(X_tr, Y_tr)
        results = evaluate_models(ml_models, X_tr, Y_tr, X_te, Y_te, ml_t)

        results_csv = results.drop(columns=["_Yp_te", "_Yp_tr"])
        results_csv.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
        print(f"[Saved] {RESULT_CSV}")

        set_nature_style()
        plot_pred_vs_true_per_target_model(results, Y_tr, Y_te)
        plot_residual_distribution(results, Y_tr, Y_te)
        plot_metrics_train_vs_test_per_target(results)   # fig3a × 2
        plot_metrics_train_by_target(results)            # fig4
        plot_metrics_test_by_target(results)             # fig5
        plot_pca(X_raw, Y)                               # fig4b × N_TARGETS
        plot_error_curves(results, Y_tr, Y_te)           # fig6
        plot_feature_importance(ml_models, feature_names, top_n=20)

        print("\n========== 模型排行 (按 meanTestR² 降序) ==========")
        cols = ["model", "mean_test_R2", "overall_test_R2",
                "total_test_RMSE"]
        for t in TARGETS:
            cols += [f"{t}__test_R2",
                     f"{t}__test_RMSE",
                     f"{t}__test_MAE"]
        cols += ["train_time_s", "predict_time_s"]
        print(results_csv[cols]
              .sort_values("mean_test_R2", ascending=False)
              .to_string(index=False))

        print(f"\n图表保存目录: {FIGURE_DIR}")
        print("全部完成 ✓")

    except Exception:
        print("[ERROR] 主流程异常:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
