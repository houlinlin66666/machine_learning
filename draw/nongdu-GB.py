# -*- coding: utf-8 -*-
"""
EEM (Excitation-Emission Matrix) Fluorescence Spectroscopy
GradientBoosting-Only Regression Pipeline (Nature Style)
==================================================================
优化版：仅保留 GradientBoosting，并修复特征重要性标签显示不全的问题
"""

from __future__ import annotations
import os
import re
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

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# 屏蔽无关警告
warnings.filterwarnings("ignore")

# ============================================================
# 全局配置
# ============================================================
DATA_DIR = r"C:\Users\666\Desktop\yang"
FIGURE_DIR = os.path.join(DATA_DIR, "GB")
MODEL_DIR = os.path.join(DATA_DIR, "models")
RESULT_CSV = os.path.join(DATA_DIR, "results_gradient_boosting.csv")

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_SEARCH = 30
LOG_TARGET = True

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Nature 尺寸规范 (cm to inches)
CM = 1.0 / 2.54
SINGLE_COL = 10 * CM  # 适当加宽以容纳纵坐标标签
DOUBLE_COL = 18 * CM


def set_nature_style() -> None:
    """深度定制 Nature 风格绘图参数"""
    plt.style.use("seaborn-v0_8-white")
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "axes.labelpad": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    """强制使用 bbox_inches='tight' 确保标签不被截断"""
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 任务 1-3: 数据处理 (解析浓度并转换 EEM)
# ============================================================
def parse_concentration(fname: str) -> float | None:
    stem = os.path.splitext(fname)[0]
    parts = stem.split("-")
    try:
        return float(parts[0])
    except:
        return None


def load_and_parse_data(folder_path: str):
    files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")))
    if not files: raise FileNotFoundError("未找到数据文件")
    X_list, y_list, meta_list = [], [], []
    ex_axis = em_axis = None
    for fp in files:
        conc = parse_concentration(os.path.basename(fp))
        if conc is None: continue
        df = pd.read_excel(fp, header=None)
        if ex_axis is None:
            em_axis = df.iloc[0, 1:].to_numpy(dtype=np.float32)
            ex_axis = df.iloc[1:, 0].to_numpy(dtype=np.float32)
        mat = df.iloc[1:, 1:].to_numpy(dtype=np.float32)
        X_list.append(np.nan_to_num(mat))
        y_list.append(conc)
        meta_list.append({"file": os.path.basename(fp), "concentration": conc})
    return np.stack(X_list), np.asarray(y_list), pd.DataFrame(meta_list), ex_axis, em_axis


def preprocess_data(X: np.ndarray, y: np.ndarray):
    X_flat = X.reshape(X.shape[0], -1)
    # 按浓度分箱进行层化划分，保证训练/测试集浓度分布一致
    bins = np.digitize(y, np.histogram_bin_edges(y, bins=min(10, len(np.unique(y)))))
    X_tr, X_te, y_tr, y_te = train_test_split(X_flat, y, test_size=TEST_SIZE, random_state=SEED, stratify=bins)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te, scaler


# ============================================================
# 任务 4: GradientBoosting 训练
# ============================================================
def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray):
    print("[Task 4] 正在对 GradientBoosting 模型进行随机搜索调参...")
    y_fit = np.log1p(y_train) if LOG_TARGET else y_train

    gb = GradientBoostingRegressor(random_state=SEED)
    param_grid = {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "max_features": ["sqrt", 0.3, 0.5]
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        gb, param_grid, n_iter=N_ITER_SEARCH, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=SEED
    )
    search.fit(X_train, y_fit)

    best_model = search.best_estimator_
    joblib.dump(best_model, os.path.join(MODEL_DIR, "GradientBoosting_model.joblib"))
    print(f"      最佳参数: {search.best_params_}")
    return best_model


# ============================================================
# 任务 5: 优化后的特征重要性绘图 (解决纵坐标显示不全问题)
# ============================================================
def plot_optimized_importance(model, ex_axis, em_axis, top_n=20):
    """
    通过动态高度调整与 tight_layout 解决标签截断问题
    """
    imp = model.feature_importances_
    n_em = len(em_axis)
    idx = np.argsort(imp)[::-1][:top_n]

    # 构造标签: Ex 激发 / Em 发射
    labels = [f"Ex {ex_axis[i // n_em]:.0f} / Em {em_axis[i % n_em]:.0f}" for i in idx]
    vals = imp[idx]

    # 1. 动态设置高度：top_n=20 时，高度应足够拉伸，避免标签重叠
    fig_height = top_n * 0.45 * CM * 2.54  # 约每行 0.45cm 空间
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, fig_height))

    colors = sns.color_palette("viridis", top_n)
    # 绘图（使用翻转确保最重要的在最上方）
    ax.barh(range(top_n), vals[::-1], color=colors[::-1], edgecolor="black", linewidth=0.5)

    # 2. 设置刻度位置与标签
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=8)

    ax.set_xlabel("Relative Importance (Gini Score)")
    ax.set_title(f"Top {top_n} Wavelength Pairs (GradientBoosting)")

    sns.despine()
    # 3. 核心：使用 tight_layout 自动调整轴间距
    plt.tight_layout()
    save_fig(fig, "fig4_feature_importance_gb")


def evaluate_and_plot(model, X_te, y_te, ex_axis, em_axis):
    set_nature_style()
    y_pred_fit = model.predict(X_te)
    y_pred = np.expm1(y_pred_fit) if LOG_TARGET else y_pred_fit

    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    print(f"\n评估完成: R2 = {r2:.4f}, RMSE = {rmse:.4g}")

    # 预测散点图
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    ax.scatter(y_te, y_pred, alpha=0.7, color='#35b779', s=15, edgecolors='white', lw=0.3)
    ax.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'k--', lw=0.8)
    ax.set_xlabel("True Concentration (µg/L)")
    ax.set_ylabel("Predicted Concentration (µg/L)")
    ax.set_title(f"GradientBoosting Regression (R²={r2:.3f})")
    plt.tight_layout()
    save_fig(fig, "fig2_pred_vs_true_gb")

    # 特征重要性
    plot_optimized_importance(model, ex_axis, em_axis)


# ============================================================
# 主流程序
# ============================================================
def main():
    try:
        X, y, meta, ex, em = load_and_parse_data(DATA_DIR)
        X_tr_s, X_te_s, y_tr, y_te, scaler = preprocess_data(X, y)

        # 训练 GradientBoosting
        best_gb = train_gradient_boosting(X_tr_s, y_tr)

        # 评估与绘图
        evaluate_and_plot(best_gb, X_te_s, y_te, ex, em)

        # 保存详细结果
        res_df = pd.DataFrame(
            {"True": y_te, "Pred": np.expm1(best_gb.predict(X_te_s)) if LOG_TARGET else best_gb.predict(X_te_s)})
        res_df.to_csv(RESULT_CSV, index=False)

        print(f"\n全部任务已完成！结果 CSV 及图片已保存至：{DATA_DIR}")

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()