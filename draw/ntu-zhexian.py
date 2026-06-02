import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter

# 命名（）-（）-  浓度-浊度-   经过平滑后， 取固定ex绘制出的荧光强度随em变化的曲线图
def set_nature_style():
    plt.rcParams.update(plt.rcParamsDefault)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.linewidth": 1.5,
        "font.weight": "bold",
        "savefig.dpi": 600,
        "mathtext.fontset": "stix"
    })


def plot_eem_final_v3(folder_path, target_ex=280):
    set_nature_style()

    # --- 数据处理逻辑 (略，保持不变) ---
    files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")))
    all_data = []
    for fp in files:
        fname = os.path.basename(fp)
        parts = fname.replace(".xlsx", "").split("-")
        try:
            conc, turb = float(parts[0]), float(parts[1])
        except:
            continue
        df = pd.read_excel(fp, index_col=0)
        df.columns = pd.to_numeric(df.columns, errors='coerce')
        ex_val = target_ex if target_ex in df.columns else df.columns[np.abs(df.columns - target_ex).argmin()]
        col_data = df[ex_val]
        mask = (col_data.index >= 400) & (col_data.index <= 500)
        sub_data = col_data.loc[mask]
        all_data.append({"conc": conc, "turb": turb, "em": sub_data.index.values,
                         "intensity": savgol_filter(sub_data.values, 7, 2)})

    df_meta = pd.DataFrame(all_data)
    u_concs = sorted(df_meta['conc'].unique())[::-1]
    u_turbs = sorted(df_meta['turb'].unique())

    colors = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD']
    turb_map = {t: colors[i % len(colors)] for i, t in enumerate(u_turbs)}
    ls_list = ['-', (0, (4, 1.5)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    conc_map = {c: ls_list[i % len(ls_list)] for i, c in enumerate(u_concs[::-1])}
    if 10 in conc_map and 1 in conc_map:
        conc_map[10], conc_map[1] = conc_map[1], conc_map[10]

    fig, ax = plt.subplots(figsize=(10 / 2.54, 8 / 2.54))

    for item in all_data:
        ax.plot(item["em"], item["intensity"], color=turb_map[item['turb']],
                linestyle=conc_map[item['conc']], lw=1.5, zorder=1)

    # =============================================================
    # 【参数调节区】
    # =============================================================

    # 1. 表格位置与大小 [左, 底, 宽, 高]
    # 如果觉得表格挡住了线，就把前两个数调小（往左下移）或者调大（往右上移）
    TABLE_POS = [0.12, 0.70, 0.28, 0.18]

    # 2. 字体大小
    FS_TITLE = 7.0  # "Turbidity" 标题字号
    FS_NUM = 6.0  # 10, 20, 30 这些数字的字号

    # 3. 文字相对于表格的“距离” (单位是百分比，相对于表格宽度/高度)
    # 调大这些数值，字就离表格越远
    TURB_Y_OFFSET = 1.05  # 浊度数字(10,20..)离表格顶部的距离
    TURB_TITLE_Y = 1.35  # "Turbidity (NTU)" 标题离顶部的距离

    CONC_X_OFFSET = -0.02  # 浓度数字(1,10..)离表格左侧的距离
    CONC_TITLE_X = -0.22  # "Conc. (μg/L)" 标题离左侧的距离

    # --- 绘制图例表格 ---
    legend_ax = ax.inset_axes(TABLE_POS, facecolor='white', zorder=10)
    legend_ax.patch.set_alpha(1.0)
    legend_ax.set_xlim(0, 1);
    legend_ax.set_ylim(0, 1)
    legend_ax.set_xticks([]);
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_linewidth(0.3)
    # ========================================

    n_rows, n_cols = len(u_concs), len(u_turbs)

    # 绘制浊度文字
    for j, t in enumerate(u_turbs):
        legend_ax.text(j / n_cols + 0.5 / n_cols, TURB_Y_OFFSET, f"{int(t)}",
                       ha='center', va='bottom', fontweight='bold', fontsize=FS_NUM)
    legend_ax.text(0.5, TURB_TITLE_Y, "Turbidity (NTU)", ha='center', fontweight='bold', fontsize=FS_TITLE)

    # 绘制浓度文字
    for i, c in enumerate(u_concs):
        legend_ax.text(CONC_X_OFFSET, 1 - (i / n_rows + 0.5 / n_rows), f"{int(c)}",
                       ha='right', va='center', fontweight='bold', fontsize=FS_NUM)
    legend_ax.text(CONC_TITLE_X, 0.5, "Conc. (μg/L)", ha='center', va='center',
                   fontweight='bold', fontsize=FS_TITLE, rotation=90)

    # 绘制格子
    for i, c in enumerate(u_concs):
        for j, t in enumerate(u_turbs):
            x_m, y_m = j / n_cols + 0.5 / n_cols, 1 - (i / n_rows + 0.5 / n_rows)
            legend_ax.plot([x_m - 0.35 / n_cols, x_m + 0.35 / n_cols], [y_m, y_m],
                           color=turb_map[t], linestyle=conc_map[c], lw=1.0)
            rect = plt.Rectangle((j / n_cols, 1 - (i + 1) / n_rows), 1 / n_cols, 1 / n_rows,
                                 fill=False, edgecolor='black', lw=0.4)
            legend_ax.add_patch(rect)

    # --- 修饰与保存 ---
    # 将 pad 从 15 改小（例如改为 5 或者 3）
    # pad 代表标题文字与坐标轴顶框线之间的距离，数值越小越近
    ax.set_title(f"Ex = {target_ex} nm", loc='right', fontweight='bold', fontsize=12, pad=10)
    ax.set_xlabel("Emission Wavelength (nm)", fontweight='bold', fontsize=11)
    ax.set_ylabel("Fluorescence Intensity (A.U.)", fontweight='bold', fontsize=11)
    ax.tick_params(width=1.5, length=5, labelsize=9, direction='in', top=True, right=True)

    # 控制整个大图的边距，防止字被切掉
    plt.subplots_adjust(bottom=0.15, left=0.22, top=0.82, right=0.95)

    # 【这里是保存代码】
    save_name = os.path.join(folder_path, "Result_Final.png")
    plt.savefig(save_name, dpi=600, bbox_inches='tight')
    print(f"图片已成功保存到: {save_name}")

    plt.show()


# 执行
DATA_FOLDER = r"/Users/houlinlin/master/data/EEM_data/ntu/huan-ntu/excel/adjust"
plot_eem_final_v3(DATA_FOLDER)