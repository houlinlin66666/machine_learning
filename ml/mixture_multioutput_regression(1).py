# -*- coding: utf-8 -*-
"""
EEM (Excitation-Emission Matrix) Fluorescence Spectroscopy
Mixture Antibiotic Concentration — MULTI-OUTPUT Regression Pipeline
==================================================================

任务: 一次性预测两种抗生素 (Ofloxacin / Ciprofloxacin) 的浓度,
      使用单一多输出模型而不是两个独立模型。

文件命名: <ofloxacin>-<ciprofloxacin>-<other>.xlsx

----------------------------------------------------------------
评价指标:
    每个目标分别报告 R² / RMSE / MAE,
    并提供综合 (mean / total / overall) 指标用于横向比较。

可视化 (训练集 + 测试集同框):
    1. 真值 vs 预测散点图 (含 y=x 与线性拟合, 两种抗生素 + 训练/测试 区分)
    2. 残差分布 (KDE / 直方图, 两子图 × 训练/测试 对比)
    3. 性能柱状图 (R² / RMSE / MSE, 训练 vs 测试)
    4. 预测误差曲线图 (样本索引 vs 误差, 两抗生素 × 训练/测试)
    5. Top-20 特征重要度 (基于 RandomForest 多输出模型)

颜色: 训练 #1F4E79 (蓝), 测试 #C0504D (红)
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

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

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
FIGURE_DIR = os.path.join(DATA_DIR, "figure_multi")
MODEL_DIR  = os.path.join(DATA_DIR, "models_multi")
RESULT_CSV = os.path.join(DATA_DIR, "results_multi.csv")

TARGETS = ["Ofloxacin", "Ciprofloxacin"]
N_TARGETS = len(TARGETS)

SEED          = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5
N_ITER_SEARCH = 15
LOG_TARGET    = True

np.random.seed(SEED)

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 颜色 (训练 / 测试)
COLOR_TRAIN = "#1F4E79"
COLOR_TEST  = "#C0504D"
# 两种抗生素 (用于按目标着色)
COLOR_TARGETS = {"Ofloxacin": "#2A6F97", "Ciprofloxacin": "#E07A5F"}
# 训练/测试 marker
MARKER_TRAIN = "o"
MARKER_TEST  = "^"

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


def load_and_parse_data(folder: str
                        ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame,
                                   np.ndarray, np.ndarray]:
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
# 任务 2: 划分 + 标准化 (二维分箱近似分层)
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


def preprocess(X: np.ndarray, Y: np.ndarray):
    X_flat = X.reshape(X.shape[0], -1)
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
# 任务 3a: 多输出经典模型 + 随机搜索
# ============================================================
def _build_search_spaces(n_features: int
                         ) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    所有模型一次输出 N_TARGETS 维。
    - RF                 原生支持多输出
    - SVR / GBRT / XGB   用 MultiOutputRegressor 包裹
    超参网格的键名带前缀 'estimator__' 以适配 MultiOutputRegressor。
    """
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
    print("[Task 3a] 训练多输出经典模型 ...")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    Y_fit = np.log1p(Y_tr) if LOG_TARGET else Y_tr
    trained: Dict[str, Any] = {}
    times: Dict[str, float] = {}

    for name, (est, grid) in _build_search_spaces(X_tr.shape[1]).items():
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
        # 综合指标
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
            "_Yp_te": Yp_te, "_Yp_tr": Yp_tr,  # 内部保留供绘图
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
def _abbr(name: str) -> str:
    return MODEL_MAP.get(name, name)


# ---------- 图 1: 真值 vs 预测散点 ----------
def plot_pred_vs_true(results: pd.DataFrame,
                      Y_tr: np.ndarray, Y_te: np.ndarray) -> None:
    """每个模型一张子图: 训练(圆)+测试(三角), 两种抗生素用颜色区分。"""
    df = results.sort_values("mean_test_R2", ascending=False).reset_index(drop=True)
    n = len(df); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(DOUBLE_COL, 6.5 * nrows * CM))
    axes = np.atleast_2d(axes).ravel()

    for i, row in df.iterrows():
        ax = axes[i]
        Yp_tr = np.asarray(row["_Yp_tr"])
        Yp_te = np.asarray(row["_Yp_te"])
        all_vals = np.concatenate([Y_tr.ravel(), Y_te.ravel(),
                                   Yp_tr.ravel(), Yp_te.ravel()])
        lo, hi = float(all_vals.min()), float(all_vals.max())
        pad = 0.05 * (hi - lo + 1e-9)

        # y=x
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "--", color="#333", lw=1.0, alpha=0.7, zorder=1,
                label="y = x")

        for j, t in enumerate(TARGETS):
            c = COLOR_TARGETS[t]
            # train (圆形, 描边淡蓝)
            ax.scatter(Y_tr[:, j], Yp_tr[:, j], s=14, alpha=0.55,
                       c=c, marker=MARKER_TRAIN,
                       edgecolors=COLOR_TRAIN, linewidths=0.4,
                       label=f"{t} (Train)", zorder=2)
            # test (三角, 描边深红)
            ax.scatter(Y_te[:, j], Yp_te[:, j], s=24, alpha=0.9,
                       c=c, marker=MARKER_TEST,
                       edgecolors=COLOR_TEST, linewidths=0.6,
                       label=f"{t} (Test)", zorder=3)
            # 线性拟合 (合并 train+test 同一目标)
            xs = np.concatenate([Y_tr[:, j], Y_te[:, j]])
            ys = np.concatenate([Yp_tr[:, j], Yp_te[:, j]])
            if len(xs) >= 2 and np.ptp(xs) > 0:
                slope, intercept = np.polyfit(xs, ys, 1)
                xline = np.array([lo - pad, hi + pad])
                ax.plot(xline, slope * xline + intercept,
                        "-", color=c, lw=1.2, alpha=0.9, zorder=4)

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Measured (µg/L)")
        ax.set_ylabel("Predicted (µg/L)")
        ax.set_title(f"{_abbr(row['model'])}  "
                     f"meanR²={row['mean_test_R2']:.3f}",
                     fontweight="bold", loc="left")
        sns.despine(ax=ax)
        if i == 0:
            ax.legend(frameon=True, facecolor="white", framealpha=0.85,
                      fontsize=6, loc="upper left", markerscale=1.1)

    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    save_fig(fig, "fig1_pred_vs_true_scatter")


# ---------- 图 2: 残差分布 ----------
def plot_residual_distribution(results: pd.DataFrame,
                               Y_tr: np.ndarray, Y_te: np.ndarray) -> None:
    """最佳模型: 两子图 (每种抗生素) × 训练/测试 KDE+直方。"""
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
        ax.set_xlabel(f"Residual = True - Pred (µg/L)")
        ax.set_ylabel("Density")
        ax.set_title(f"{t}", fontweight="bold")
        ax.legend(frameon=False, fontsize=7)
        sns.despine(ax=ax)
    fig.suptitle(f"Residual distribution — best model: {name}",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig2_residual_distribution")


# ---------- 图 3: 性能柱状图 (R² / RMSE / MSE) ----------
def plot_performance_bars(results: pd.DataFrame) -> None:
    """
    每个指标一张子图; X 轴=两种抗生素; 每个抗生素两根柱 (train, test);
    每个模型一张图组? -> 改为: 选最佳模型展示; 同时给出全模型对比附图。
    这里按需求"X轴为两种抗生素的名称, 每组包含训练+测试" => 选最佳模型。
    """
    best = results.sort_values("mean_test_R2", ascending=False).iloc[0]
    name = _abbr(best["model"])

    metrics = [("R2", "$R^2$"), ("RMSE", "RMSE (µg/L)"),
               ("MSE", "MSE (µg/L$^2$)")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 6.5 * CM))
    x = np.arange(N_TARGETS)
    w = 0.36

    for i, (metric, label) in enumerate(metrics):
        ax = axes[i]
        train_vals = [best[f"{t}__train_{metric}"] for t in TARGETS]
        test_vals  = [best[f"{t}__test_{metric}"]  for t in TARGETS]
        b1 = ax.bar(x - w / 2, train_vals, w, color=COLOR_TRAIN,
                    edgecolor="white", lw=0.5, label="Train")
        b2 = ax.bar(x + w / 2, test_vals, w, color=COLOR_TEST,
                    edgecolor="white", lw=0.5, label="Test")
        for rects in (b1, b2):
            for r in rects:
                h = r.get_height()
                ax.text(r.get_x() + r.get_width() / 2,
                        h + (abs(h) * 0.01 + 1e-6),
                        f"{h:.3f}" if metric == "R2" else f"{h:.3g}",
                        ha="center", va="bottom",
                        fontsize=6, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS, rotation=10, ha="right")
        ax.set_title(label, fontweight="bold")
        ax.legend(frameon=False, fontsize=6, loc="best")
        sns.despine(ax=ax)
        ax.yaxis.grid(True, linestyle="--", alpha=0.2)

    fig.suptitle(f"Performance by target — {name}", fontsize=10, y=1.04)
    fig.tight_layout()
    save_fig(fig, "fig3_performance_bars_best")

    # 附加: 所有模型在两个目标上的 test R² 对比
    fig2, ax = plt.subplots(figsize=(DOUBLE_COL, 6 * CM))
    df = results.sort_values("mean_test_R2", ascending=False)
    models_abbr = [_abbr(m) for m in df["model"]]
    xm = np.arange(len(df))
    for j, t in enumerate(TARGETS):
        ax.bar(xm + (j - 0.5) * 0.4,
               df[f"{t}__test_R2"].values, 0.4,
               color=COLOR_TARGETS[t], edgecolor="white", lw=0.5, label=t)
    ax.set_xticks(xm); ax.set_xticklabels(models_abbr, rotation=20, ha="right")
    ax.set_ylabel("Test $R^2$")
    ax.set_title("Per-target test $R^2$ across models", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig2.tight_layout()
    save_fig(fig2, "fig3b_all_models_test_r2")


# ---------- 图 4: 预测误差曲线 ----------
def plot_error_curves(results: pd.DataFrame,
                      Y_tr: np.ndarray, Y_te: np.ndarray) -> None:
    """最佳模型: 样本索引 vs (pred - true), 两抗生素 + train/test。"""
    best = results.sort_values("mean_test_R2", ascending=False).iloc[0]
    name = _abbr(best["model"])
    Yp_tr = np.asarray(best["_Yp_tr"])
    Yp_te = np.asarray(best["_Yp_te"])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 7 * CM))
    n_tr = Y_tr.shape[0]; n_te = Y_te.shape[0]
    idx_tr = np.arange(n_tr)
    idx_te = np.arange(n_tr, n_tr + n_te)

    for j, t in enumerate(TARGETS):
        c = COLOR_TARGETS[t]
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
    save_fig(fig, "fig4_error_curves")


# ---------- 图 5: Top-20 特征重要度 ----------
def _multi_feature_importance(mdl: Any) -> np.ndarray | None:
    """从多输出模型提取特征重要度向量 (length = n_features)。"""
    # 原生支持
    imp = getattr(mdl, "feature_importances_", None)
    if imp is not None:
        return np.asarray(imp)
    # MultiOutputRegressor: 取子估计器平均
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
                            ex_axis: np.ndarray, em_axis: np.ndarray,
                            top_n: int = 20) -> None:
    n_em = em_axis.size
    for name in ("RandomForest", "GradientBoosting", "XGBoost"):
        if name not in models:
            continue
        imp = _multi_feature_importance(models[name])
        if imp is None:
            continue
        idx = np.argsort(imp)[::-1][:top_n]
        labels = [f"Ex {ex_axis[i // n_em]:.0f} / Em {em_axis[i % n_em]:.0f}"
                  for i in idx]
        vals = imp[idx]
        # 加宽画布 + 拉大左边距, 避免长标签 (Ex xxx / Em xxx) 溢出
        fig, ax = plt.subplots(
            figsize=(SINGLE_COL * 2.2, 0.38 * top_n + 1.2))
        colors = sns.color_palette("viridis", top_n)
        y_pos = np.arange(top_n)[::-1]
        ax.barh(y_pos, vals, color=colors,
                edgecolor="black", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Feature importance (mean over targets)")
        ax.set_title(f"Top-{top_n} wavelength pairs — {_abbr(name)} "
                     f"(multi-output)", fontweight="bold")
        ax.margins(y=0.01)
        sns.despine(ax=ax)
        # 先 tight_layout, 再强制留出左侧 32% 给 y 轴标签
        fig.tight_layout()
        fig.subplots_adjust(left=0.32)
        save_fig(fig, f"fig5_feature_importance_{name}")


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    try:
        print("=" * 64)
        print("EEM Multi-Output Antibiotic Regression Pipeline")
        print("=" * 64)
        print(f"Data directory   : {DATA_DIR}")
        print(f"Figure directory : {FIGURE_DIR}")
        print(f"Model directory  : {MODEL_DIR}")
        print(f"Target transform : {'log1p' if LOG_TARGET else 'identity'}")
        print(f"Targets          : {TARGETS}")
        print("-" * 64)

        X, Y, meta_df, ex_axis, em_axis = load_and_parse_data(DATA_DIR)
        meta_df.to_csv(os.path.join(DATA_DIR, "metadata_multi.csv"),
                       index=False, encoding="utf-8-sig")
        n_ex, n_em = X.shape[1], X.shape[2]

        X_tr, X_te, Y_tr, Y_te, scaler = preprocess(X, Y)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

        ml_models, ml_t = train_ml_models(X_tr, Y_tr)
        all_models = ml_models
        all_times = ml_t

        results = evaluate_models(all_models, X_tr, Y_tr,
                                  X_te, Y_te, all_times)

        # CSV (剔除内部预测数组)
        results_csv = results.drop(columns=["_Yp_te", "_Yp_tr"])
        results_csv.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
        print(f"[Saved] {RESULT_CSV}")

        set_nature_style()
        plot_pred_vs_true(results, Y_tr, Y_te)
        plot_residual_distribution(results, Y_tr, Y_te)
        plot_performance_bars(results)
        plot_error_curves(results, Y_tr, Y_te)
        plot_feature_importance(all_models, ex_axis, em_axis, top_n=20)

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
