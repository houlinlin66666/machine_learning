# -*- coding: utf-8 -*-
"""
EEM (Excitation-Emission Matrix) Fluorescence Spectroscopy
GradientBoosting-Only Regression Pipeline (Nature Style)
==================================================================
已更新：分别输出训练时长、预测时长及总运行耗时
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
DATA_DIR = r"C:\Users\yafex\Desktop\yang"
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

# Nature 尺寸规范
CM = 1.0 / 2.54
SINGLE_COL = 10 * CM


def set_nature_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "savefig.dpi": 600, "pdf.fonttype": 42,
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 核心处理函数
# ============================================================
def load_and_parse_data(folder_path: str):
    # (解析逻辑保持不变)
    files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")))
    if not files: raise FileNotFoundError("未找到数据文件")
    X_list, y_list, meta_list = [], [], []
    ex_axis = em_axis = None
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        try:
            conc = float(stem.split("-")[0])
        except:
            continue
        df = pd.read_excel(fp, header=None)
        if ex_axis is None:
            em_axis = df.iloc[0, 1:].to_numpy(dtype=np.float32)
            ex_axis = df.iloc[1:, 0].to_numpy(dtype=np.float32)
        mat = df.iloc[1:, 1:].to_numpy(dtype=np.float32)
        X_list.append(np.nan_to_num(mat))
        y_list.append(conc)
    return np.stack(X_list), np.asarray(y_list), ex_axis, em_axis


def preprocess_data(X: np.ndarray, y: np.ndarray):
    X_flat = X.reshape(X.shape[0], -1)
    bins = np.digitize(y, np.histogram_bin_edges(y, bins=min(10, len(np.unique(y)))))
    X_tr, X_te, y_tr, y_te = train_test_split(X_flat, y, test_size=TEST_SIZE, random_state=SEED, stratify=bins)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te


# ============================================================
# 主程序逻辑
# ============================================================
def main():
    overall_start = time.time()
    set_nature_style()

    print("=" * 50)
    print(f"程序启动时间: {time.strftime('%H:%M:%S')}")
    print("=" * 50)

    try:
        # 1. 数据加载与预处理
        X, y, ex, em = load_and_parse_data(DATA_DIR)
        X_tr_s, X_te_s, y_tr, y_te = preprocess_data(X, y)

        # 2. 训练计时
        print("\n[Step 1] 开始模型训练 (GradientBoosting + 随机搜索)...")
        train_start = time.time()

        y_fit = np.log1p(y_tr) if LOG_TARGET else y_tr
        gb = GradientBoostingRegressor(random_state=SEED)
        param_grid = {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0]
        }
        search = RandomizedSearchCV(gb, param_grid, n_iter=N_ITER_SEARCH, cv=CV_FOLDS, n_jobs=-1, random_state=SEED)
        search.fit(X_tr_s, y_fit)
        best_model = search.best_estimator_

        train_end = time.time()
        train_duration = train_end - train_start

        # 3. 预测计时
        print("[Step 2] 开始模型预测...")
        predict_start = time.time()

        y_pred_raw = best_model.predict(X_te_s)
        y_pred = np.expm1(y_pred_raw) if LOG_TARGET else y_pred_raw

        predict_end = time.time()
        predict_duration = predict_end - predict_start

        # 4. 评估与绘图 (略过计时，保持 Nature 风格)
        r2 = r2_score(y_te, y_pred)
        # (此处省略绘图函数细节，逻辑同前)
        # evaluate_and_plot(best_model, X_te_s, y_te, ex, em)

        # 5. 输出最终报告
        overall_end = time.time()

        print("\n" + " > " * 15)
        print(f"【性能报告】")
        print(f"训练耗时 (包含调参): {train_duration:.4f} 秒")
        print(f"预测耗时 (测试集):   {predict_duration:.4f} 秒")
        print(f"单样本平均预测耗时:  {predict_duration / len(X_te_s):.6f} 秒")
        print(f"程序总运行时间:      {overall_end - overall_start:.2f} 秒")
        print(f"模型评估 R²:         {r2:.4f}")
        print(" > " * 15)

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()