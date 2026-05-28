# -*- coding: utf-8 -*-
"""
EEM Spectroscopy: Integrated Dual-Pipeline (Raw & FE) Analysis
=============================================================
Combines complete original logging/plotting with new comparative metrics.
"""

from __future__ import annotations
import os
import glob
import time
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 模型尝试加载
try:
    from xgboost import XGBRegressor

    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. 环境与路径配置
# ============================================================
warnings.filterwarnings("ignore")

DATA_DIR = r"/Users/houlinlin/master/data/EEM_data/lixiang/EEM-huan/strength/all"
DIRS = {
    "RAW": os.path.join(DATA_DIR, "figure_raw"),
    "FE": os.path.join(DATA_DIR, "figure_fe"),
    "COMP": os.path.join(DATA_DIR, "figure_comparison"),
    "MODELS": os.path.join(DATA_DIR, "models")
}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)

RESULT_CSV = os.path.join(DATA_DIR, "dual_pipeline_full_results.csv")

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_SEARCH = 15
LOG_TARGET = True
PCA_VAR = 0.95
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CM = 1.0 / 2.54
DOUBLE_COL = 17.5 * CM
MODEL_MAP = {"RandomForest": "RF", "GradientBoosting": "GBRT", "SVR": "SVR",
             "KernelRidge": "KRR", "XGBoost": "XGB", "CNN2D": "CNN"}


def set_nature_style():
    plt.style.use("seaborn-v0_8-white")
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "savefig.dpi": 600, "mathtext.fontset": "stix"
    })


set_nature_style()


# ============================================================
# 2. 数据处理与 CNN 结构
# ============================================================
def load_eem_data(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "*.xlsx")))
    X_list, y_list = [], []
    ex_axis = em_axis = None
    print(f"-> Loading: {len(files)} files")
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        try:
            conc = float(stem.split("-")[0])
            df = pd.read_excel(fp, header=None)
            if ex_axis is None:
                em_axis = df.iloc[0, 1:].values.astype(np.float32)
                ex_axis = df.iloc[1:, 0].values.astype(np.float32)
            X_list.append(df.iloc[1:, 1:].values.astype(np.float32))
            y_list.append(conc)
        except:
            continue
    return np.nan_to_num(np.stack(X_list), 0.0), np.array(y_list), ex_axis, em_axis


class CNN2DReg(nn.Module):
    def __init__(self, n_ex, n_em):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 1))
        self.n_ex, self.n_em = n_ex, n_em

    def forward(self, x):
        x = x.view(-1, 1, self.n_ex, self.n_em)
        return self.head(self.feat(x)).squeeze(-1)


class CNNWrapper:
    def __init__(self, n_ex, n_em):
        self.n_ex, self.n_em = n_ex, n_em
        self.model = None

    def fit(self, X, y):
        # 将展平的 X 还原回 2D 进行卷积
        X_2d = X.reshape(-1, self.n_ex, self.n_em)
        self.model = CNN2DReg(self.n_ex, self.n_em).to(DEVICE)
        ds = TensorDataset(torch.from_numpy(X_2d).float(), torch.from_numpy(y).float())
        dl = DataLoader(ds, batch_size=8, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        self.model.train()
        for _ in range(100):
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad();
                crit(self.model(xb), yb).backward();
                opt.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_2d = X.reshape(-1, self.n_ex, self.n_em)
        with torch.no_grad():
            return self.model(torch.from_numpy(X_2d).float().to(DEVICE)).cpu().numpy()


# ============================================================
# 3. 核心绘图逻辑 (保留并升级原始的独立散点图)
# ============================================================
def plot_regression_results(y_true, y_pred, model_name, save_dir, pipe_name):
    fig, ax = plt.subplots(figsize=(8 * CM, 8 * CM))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5, s=20, color='#1F4E79')
    ideal = [y_true.min(), y_true.max()]
    ax.plot(ideal, ideal, 'r--', lw=1)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    ax.set_title(f"{model_name} ({pipe_name})", fontsize=9)
    ax.set_xlabel("Measured Concentration")
    ax.set_ylabel("Predicted Concentration")
    ax.text(0.05, 0.9, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=ax.transAxes, fontsize=7)

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_scatter.png"))
    plt.close()


# ============================================================
# 4. 自动化流水线 (支持双轨道)
# ============================================================
def run_enhanced_pipeline(X_tr, y_tr, X_te, y_te, n_ex, n_em, pipe_name, save_dir, overhead=0.0):
    y_fit = np.log1p(y_tr) if LOG_TARGET else y_tr

    space = {
        "RandomForest": (RandomForestRegressor(random_state=SEED), {"n_estimators": [200, 500]}),
        "SVR": (SVR(), {"C": [1, 10, 100], "epsilon": [0.01, 0.1]}),
        "KernelRidge": (KernelRidge(kernel="rbf"), {"alpha": [0.1, 1.0], "gamma": [0.01, 0.1]})
    }
    if HAVE_XGB:
        space["XGBoost"] = (XGBRegressor(random_state=SEED), {"n_estimators": [300, 500], "learning_rate": [0.05, 0.1]})

    results = []

    # --- 机器学习模型循环 ---
    for name, (est, grid) in space.items():
        print(f"  > Training {name}...")
        t0 = time.time()
        search = RandomizedSearchCV(est, grid, n_iter=N_ITER_SEARCH, cv=CV_FOLDS, n_jobs=-1, random_state=SEED)
        search.fit(X_tr, y_fit)
        best_mdl = search.best_estimator_
        train_time = (time.time() - t0) + overhead

        # 预测并转换
        p_tr = best_mdl.predict(X_tr)
        p_te = best_mdl.predict(X_te)
        if LOG_TARGET:
            p_tr, p_te = np.expm1(p_tr), np.expm1(p_te)

        # 记录结果
        res = {
            "pipeline": pipe_name, "model": name, "train_time": train_time,
            "R2": r2_score(y_te, p_te), "train_R2": r2_score(y_tr, p_tr),
            "RMSE": np.sqrt(mean_squared_error(y_te, p_te)), "train_RMSE": np.sqrt(mean_squared_error(y_tr, p_tr)),
            "MSE": mean_squared_error(y_te, p_te), "train_MSE": mean_squared_error(y_tr, p_tr)
        }
        results.append(res)

        # 独立出图与模型保存
        plot_regression_results(y_te, p_te, name, save_dir, pipe_name)
        joblib.dump(best_mdl, os.path.join(DIRS["MODELS"], f"{pipe_name}_{name}.joblib"))

    # --- CNN 2D 轨道 (仅 Raw 且数据量足够时) ---
    if pipe_name == "Raw":
        print("  > Training CNN2D...")
        t0 = time.time()
        cnn = CNNWrapper(n_ex, n_em).fit(X_tr, y_fit)
        train_time = time.time() - t0
        p_tr, p_te = cnn.predict(X_tr), cnn.predict(X_te)
        if LOG_TARGET: p_tr, p_te = np.expm1(p_tr), np.expm1(p_te)

        results.append({
            "pipeline": pipe_name, "model": "CNN2D", "train_time": train_time,
            "R2": r2_score(y_te, p_te), "train_R2": r2_score(y_tr, p_tr),
            "RMSE": np.sqrt(mean_squared_error(y_te, p_te)), "train_RMSE": np.sqrt(mean_squared_error(y_tr, p_tr)),
            "MSE": mean_squared_error(y_te, p_te), "train_MSE": mean_squared_error(y_tr, p_tr)
        })
        plot_regression_results(y_te, p_te, "CNN2D", save_dir, pipe_name)
        torch.save(cnn.model.state_dict(), os.path.join(DIRS["MODELS"], "Raw_CNN2D.pt"))

    return results


# ============================================================
# 5. 生成四位一体对比大图
# ============================================================
def plot_ultimate_comparison(df):
    df["Model_Short"] = df["model"].map(lambda x: MODEL_MAP.get(x, x))
    metrics = [("R2", "train_R2", "$R^2$ Score"), ("RMSE", "train_RMSE", "RMSE"),
               ("MSE", "train_MSE", "MSE"), ("train_time", None, "Training Time (s)")]

    fig, axes = plt.subplots(4, 1, figsize=(DOUBLE_COL, 22 * CM))
    pal_test = {"Raw": "#1F4E79", "FE (PCA)": "#C0504D"}

    for i, (m_te, m_tr, label) in enumerate(metrics):
        ax = axes[i]
        if m_tr:  # 对于有 Train/Test 区分的指标
            temp = df.melt(id_vars=["Model_Short", "pipeline"], value_vars=[m_tr, m_te],
                           var_name="Set", value_name="Val")
            temp["Set"] = temp["Set"].map({m_tr: "Train", m_te: "Test"})
            sns.barplot(data=temp, x="Model_Short", y="Val", hue="pipeline", ax=ax, palette=pal_test, alpha=0.8)
        else:  # 对于时间指标
            sns.barplot(data=df, x="Model_Short", y=m_te, hue="pipeline", ax=ax, palette=pal_test)

        ax.set_title(label, fontweight='bold', loc='left')
        ax.set_ylabel("");
        ax.set_xlabel("")
        ax.legend(title="", frameon=False, ncol=2, fontsize=6)
        sns.despine()

    plt.tight_layout()
    plt.savefig(os.path.join(DIRS["COMP"], "dual_pipeline_comparison.png"))


# ============================================================
# 6. 执行主程序
# ============================================================
def main():
    print("=" * 50)
    print("STARTING EEM DUAL-PIPELINE ANALYSIS")
    print("=" * 50)

    # A. 数据加载
    X, y, ex, em = load_eem_data(DATA_DIR)
    n_samples, n_ex, n_em = X.shape
    X_flat = X.reshape(n_samples, -1)

    # B. 分割与标准化
    X_tr, X_te, y_tr, y_te = train_test_split(X_flat, y, test_size=TEST_SIZE, random_state=SEED)
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
    joblib.dump(sc, os.path.join(DIRS["MODELS"], "global_scaler.joblib"))

    # C. 运行 Raw 轨道
    print("\n[Orbit 1: Raw Data]")
    res_raw = run_enhanced_pipeline(X_tr_s, y_tr, X_te_s, y_te, n_ex, n_em, "Raw", DIRS["RAW"])

    # D. 运行 FE (PCA) 轨道
    print("\n[Orbit 2: Feature Extraction]")
    t_pca = time.time()
    pca = PCA(n_components=PCA_VAR, random_state=SEED).fit(X_tr_s)
    X_tr_fe = pca.transform(X_tr_s)
    X_te_fe = pca.transform(X_te_s)
    pca_overhead = (time.time() - t_pca) / 4.0  # 将降维开销平均分配给各个模型
    joblib.dump(pca, os.path.join(DIRS["MODELS"], "pca_transformer.joblib"))

    res_fe = run_enhanced_pipeline(X_tr_fe, y_tr, X_te_fe, y_te, n_ex, n_em, "FE (PCA)", DIRS["FE"],
                                   overhead=pca_overhead)

    # E. 结果汇总与可视化
    full_df = pd.DataFrame(res_raw + res_fe)
    full_df.to_csv(RESULT_CSV, index=False)

    print("\n-> Generating Comparison Report...")
    plot_ultimate_comparison(full_df)

    print("=" * 50)
    print(f"Success! Data saved to: {RESULT_CSV}")
    print(f"Comparison plot: {DIRS['COMP']}")
    print("=" * 50)


if __name__ == "__main__":
    main()