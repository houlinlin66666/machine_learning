# -*- coding: utf-8 -*-
"""
EEM (Excitation-Emission Matrix) Fluorescence Spectroscopy
GradientBoosting Regression with PCA & Performance Benchmarking
==================================================================
完整版：
1. 仅保留 GradientBoosting 模型
2. 加入 PCA 特征提取逻辑
3. 训练/测试集指标 (R2, RMSE, MSE) 对比绘图
4. 记录训练、预测及总运行时间
5. 符合 Nature 风格的学术绘图
"""

from __future__ import annotations
import os
import glob
import time
import warnings
import traceback
import numpy as np
import pandas as pd
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# 屏蔽无关警告
warnings.filterwarnings("ignore")

# ============================================================
# 全局配置 (请根据你的电脑路径修改 DATA_DIR)
# ============================================================
DATA_DIR = r"C:\Users\yafex\Desktop\huan"
FIGURE_DIR = os.path.join(DATA_DIR, "tezheng")
MODEL_DIR = os.path.join(DATA_DIR, "models")
RESULT_CSV = os.path.join(DATA_DIR, "gb_metrics_report.csv")

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_SEARCH = 30
LOG_TARGET = True
PCA_COMPONENTS = 15  # PCA 提取的主成分数量

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Nature 尺寸规范 (cm to inches)
CM = 1.0 / 2.54
SINGLE_COL = 8.5 * CM
DOUBLE_COL = 17.5 * CM


def set_nature_style() -> None:
    """深度定制 Nature 风格绘图参数"""
    plt.style.use("seaborn-v0_8-white")
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "grid.linewidth": 0.5, "lines.linewidth": 1.0,
        "savefig.dpi": 600, "pdf.fonttype": 42, "legend.fontsize": 7
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.png"), dpi=600, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 1. 数据处理逻辑
# ============================================================
def load_and_parse_data(folder_path: str):
    files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")))
    if not files: raise FileNotFoundError(f"未在 {folder_path} 找到 .xlsx 文件")
    X_list, y_list = [], []
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
        mat = np.nan_to_num(df.iloc[1:, 1:].to_numpy(dtype=np.float32))
        X_list.append(mat);
        y_list.append(conc)
    return np.stack(X_list), np.asarray(y_list), ex_axis, em_axis


def extract_and_preprocess(X: np.ndarray, y: np.ndarray):
    X_flat = X.reshape(X.shape[0], -1)
    bins = np.digitize(y, np.histogram_bin_edges(y, bins=min(10, len(np.unique(y)))))
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X_flat, y, test_size=TEST_SIZE, random_state=SEED, stratify=bins)

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr_s = scaler.transform(X_tr_raw)
    X_te_s = scaler.transform(X_te_raw)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit(X_tr_s)
    return pca.transform(X_tr_s), pca.transform(X_te_s), y_tr, y_te, pca


# ============================================================
# 2. 绘图与评估逻辑 (含 Train vs Test 对比)
# ============================================================
def plot_all_results(model, X_tr, y_tr, X_te, y_te, pca):
    set_nature_style()

    # 指标计算
    def get_metrics(X, y_true):
        y_p_fit = model.predict(X)
        y_p = np.expm1(y_p_fit) if LOG_TARGET else y_p_fit
        return y_p, r2_score(y_true, y_p), np.sqrt(mean_squared_error(y_true, y_p)), mean_squared_error(y_true, y_p)

    y_tr_p, r2_tr, rmse_tr, mse_tr = get_metrics(X_tr, y_tr)
    y_te_p, r2_te, rmse_te, mse_te = get_metrics(X_te, y_te)

    # Fig 0: Train vs Test Metrics Comparison
    metrics = ['R2', 'RMSE', 'MSE']
    tr_vals, te_vals = [r2_tr, rmse_tr, mse_tr], [r2_te, rmse_te, mse_te]
    fig0, axes0 = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.8))
    colors = ['#440154', '#35b779']  # Train: Purple, Test: Green

    for i, (ax, name) in enumerate(zip(axes0, metrics)):
        bars = ax.bar(['Train', 'Test'], [tr_vals[i], te_vals[i]], color=colors, edgecolor='black', linewidth=0.6,
                      width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{bar.get_height():.3g}', ha='center',
                    va='bottom', fontsize=7)
        ax.set_title(f"{name} Comparison")
        if name == 'R2': ax.set_ylim(0, 1.15)
        sns.despine(ax=ax)
    plt.tight_layout();
    save_fig(fig0, "fig0_train_vs_test_metrics")

    # Fig 1: Predicted vs Experimental (Test Set)
    fig1, ax1 = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    ax1.scatter(y_te, y_te_p, alpha=0.6, color='#31688e', s=15, edgecolors='white', lw=0.3)
    ax1.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'k--', lw=0.8)
    ax1.set_xlabel("Experimental (µg/L)");
    ax1.set_ylabel("Predicted (µg/L)")
    ax1.set_title(f"Test Set R²={r2_te:.3f}")
    plt.tight_layout();
    save_fig(fig1, "fig1_pred_vs_true")

    # Fig 2: Residuals
    res_te = y_te - y_te_p
    fig2, axes2 = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL))
    axes2[0].scatter(y_te_p, res_te, alpha=0.5, color='#35b779', s=10)
    axes2[0].axhline(0, color='red', ls='--', lw=0.8)
    axes2[0].set_xlabel("Predicted");
    axes2[0].set_ylabel("Residual")
    sns.histplot(res_te, kde=True, ax=axes2[1], color='#440154')
    plt.tight_layout();
    save_fig(fig2, "fig2_residual_analysis")

    # Fig 3: PCA Feature Space
    fig3, ax3 = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL))
    sc = ax3.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap='viridis', s=15, alpha=0.8)
    plt.colorbar(sc, label="Conc. (µg/L)")
    ax3.set_xlabel("PC1");
    ax3.set_ylabel("PC2")
    ax3.set_title("PCA Feature Mapping")
    plt.tight_layout();
    save_fig(fig3, "fig3_pca_space")

    # Fig 4: Learning Curve
    y_f_tr = np.log1p(y_tr) if LOG_TARGET else y_tr
    sizes, tr_s, cv_s = learning_curve(model, X_tr, y_f_tr, cv=5, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5))
    fig4, ax4 = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    ax4.plot(sizes, -tr_s.mean(axis=1), 'o-', label="Train RMSE", color='#440154')
    ax4.plot(sizes, -cv_s.mean(axis=1), 's-', label="CV RMSE", color='#35b779')
    ax4.set_xlabel("Samples");
    ax4.set_ylabel("Loss (RMSE)");
    ax4.legend()
    plt.tight_layout();
    save_fig(fig4, "fig4_learning_curve")

    return r2_te


# ============================================================
# 3. 主流程 (Performance Benchmarking)
# ============================================================
def main():
    start_all = time.time()
    try:
        # 加载与预处理
        X, y, ex, em = load_and_parse_data(DATA_DIR)
        X_tr_p, X_te_p, y_tr, y_te, pca = extract_and_preprocess(X, y)

        # 训练计时
        t0 = time.time()
        y_f_tr = np.log1p(y_tr) if LOG_TARGET else y_tr
        param_grid = {"n_estimators": [100, 200, 400], "max_depth": [3, 4, 5], "learning_rate": [0.01, 0.05, 0.1]}
        search = RandomizedSearchCV(GradientBoostingRegressor(random_state=SEED), param_grid, n_iter=N_ITER_SEARCH,
                                    cv=CV_FOLDS, n_jobs=-1, random_state=SEED)
        search.fit(X_tr_p, y_f_tr)
        best_model = search.best_estimator_
        t_train = time.time() - t0

        # 预测计时
        t1 = time.time()
        best_model.predict(X_te_p)
        t_predict = time.time() - t1

        # 绘图与结果
        r2_final = plot_all_results(best_model, X_tr_p, y_tr, X_te_p, y_te, pca)

        # 打印性能报告
        print("\n" + "=" * 45)
        print(f"  GradientBoosting 性能报告 (PCA 特征提取)")
        print("-" * 45)
        print(f"  训练时长 (含 CV 搜索): {t_train:.2f} s")
        print(f"  测试集总预测时长:     {t_predict:.4f} s")
        print(f"  单样本平均推理速度:   {t_predict / len(y_te) * 1000:.4f} ms")
        print(f"  程序总运行时间:       {time.time() - start_all:.2f} s")
        print(f"  测试集 R² 分数:       {r2_final:.4f}")
        print("=" * 45)

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()