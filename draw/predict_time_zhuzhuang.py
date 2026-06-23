# -*- coding: utf-8 -*-
"""
EEM (Excitation-Emission Matrix) Fluorescence Spectroscopy
Computational Efficiency Optimization Plotter (Mac PPT Version)
================================================================
功能：原生读取桌面上的 time.xlsx 文件，绘制精美红蓝分组柱状图。
优化：专为 Mac 环境与 PPT 比例进行了比例重构，确保柱体饱满、尺寸完美。
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. 全局路径配置（Mac 本地桌面路径）
# ============================================================
DATA_DIR = r"/Users/houlinlin/Desktop"
EXCEL_PATH = os.path.join(DATA_DIR, "time.xlsx")

OUTPUT_PNG = os.path.join(DATA_DIR, "fig_computational_time_comparison_yang_pr.png")
OUTPUT_PDF = os.path.join(DATA_DIR, "fig_computational_time_comparison_yang_pr.pdf")

# 模型缩写映射
MODEL_MAP = {
    "RandomForest": "RF",
    "GradientBoosting": "GBRT",
    "SVR": "SVR",
    "KernelRidge": "KRR",
    "XGBoost": "XGB",
    "CNN2D": "CNN"
}


# ============================================================
# 2. PPT 级学术风格 Matplotlib 参数重构
# ============================================================
def set_ppt_academic_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],  # 增加后备字体防止Mac不认
        "font.size": 10,  # 提高整体字号，确保PPT后排老师看得清
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,  # 略微加粗边框线，增强视觉反差
        "savefig.dpi": 300,  # PPT 插入无需 600dpi（会导致图像体积过大且缩放变形），300dpi 刚刚好
        "pdf.fonttype": 42,
        "mathtext.fontset": "stix",
        "figure.autolayout": False  # 关闭全自动，改用手工精确边缘控制
    })


# ============================================================
# 3. 主绘图业务逻辑
# ============================================================
def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"[错误] 未在路径找到输入文件: {EXCEL_PATH}")
        return

    print(f"[Task 1] 开始从桌面读取 Excel 数据: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH, header=None, names=["model", "before_feat", "after_feat"])

    # 转换缩写并按照耗时降序排列
    df["abbr"] = df["model"].map(lambda x: MODEL_MAP.get(x, x))
    df = df.sort_values("before_feat", ascending=False).reset_index(drop=True)
    df["speedup"] = df["before_feat"] / df["after_feat"]

    print("[Task 2] 正在加载 PPT 优化版绘图引擎...")
    set_ppt_academic_style()

    # 【重大调整】：直接采用标准的 4:3 黄金比例画布 (长 7 英寸，高 4.5 英寸)
    # 这样导出的图片放进 PPT 里不需要任何拉伸，柱子比例最饱满
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    x = np.arange(len(df))

    # 【重大调整】：将单柱宽度从 0.36 缩窄至 0.30，增加组间距
    # 这样可以留出足够的空白，让柱状图看起来极其舒展、不拥挤
    width = 0.30

    # 颜色完全按照你的要求：哑光红 vs 深邃蓝
    color_before = "#C0504D"
    color_after = "#1F4E79"

    # 绘制柱状图 (添加 zorder 确保网格线在柱子背后)
    rects1 = ax.bar(x - width / 2, df["before_feat"], width, label='Before Feature Extraction',
                    color=color_before, edgecolor='black', lw=0.6, zorder=3)
    rects2 = ax.bar(x + width / 2, df["after_feat"], width, label='After Feature Extraction',
                    color=color_after, edgecolor='black', lw=0.6, zorder=3)

    # ============================================================
    # 4. 智能化柱顶提速倍数标注 (位置与字号微调)
    # ============================================================
    for rect, speedup in zip(rects2, df["speedup"]):
        height = rect.get_height()
        if speedup > 1.05:
            # 针对耗时极短的模型（如KRR/SVR/CNN），如果柱子太矮，把文字往上提，避免和横坐标重叠
            offset = 3 if height > 5 else 8
            ax.annotate(f'{speedup:.1f}×',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold', color=color_after)

    # ============================================================
    # 5. 图面高级美化
    # ============================================================
    ax.set_ylabel('Total Time (seconds)', fontsize=10, fontweight='bold')
    # ax.set_title('Computational Efficiency Optimization per Model', fontsize=11, fontweight='bold', pad=15)

    ax.set_xticks(x)
    ax.set_xticklabels(df["abbr"], rotation=0, ha='center', fontweight='bold')

    # 移除上方和右侧的边框
    sns.despine(ax=ax)

    # 轻量化水平网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.2, zorder=1)

    # 图例位置（去掉边框，改到最上方合适的位置）
    ax.legend(frameon=False, loc='upper right')

    # 精确的手工边缘控制，防止由于 tight_layout 导致的柱子宽度变形
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.15)

    # 同步输出
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_PDF, bbox_inches='tight')

    print("-" * 60)
    print(f"[成功] 专为 PPT 优化的耗时对比图已生成！")
    print(f"👉 赶紧打开桌面这个图片看看，比例绝对变漂亮了: {OUTPUT_PNG}")
    print("-" * 60)
    plt.close(fig)


if __name__ == "__main__":
    main()