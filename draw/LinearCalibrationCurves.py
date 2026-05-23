import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

# --- 1. 环境与风格配置 (Nature Style) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# 强制所有 MathText (公式) 也使用新罗马风格
plt.rcParams['mathtext.fontset'] = 'stix'


def get_averaged_peak_intensity(directory, range_filter=(100, 1000)):
    """
    提取数据，并对相同浓度的最大荧光强度取平均值
    """
    data_dict = {}  # 用于存储 {浓度: [强度列表]}

    if not os.path.exists(directory):
        return np.array([]), np.array([])

    files = [f for f in os.listdir(directory) if f.endswith(('.xlsx', '.xls'))]
    for file in files:
        try:
            conc_val = float(file.split('-')[0])
            if range_filter[0] <= conc_val <= range_filter[1]:
                file_path = os.path.join(directory, file)
                df = pd.read_excel(file_path)
                # 提取矩阵最大强度
                max_int = df.iloc[:, 1:].values.max()

                if conc_val not in data_dict:
                    data_dict[conc_val] = []
                data_dict[conc_val].append(max_int)
        except:
            continue

    # 计算平均值
    sorted_concs = sorted(data_dict.keys())
    avg_intensities = [np.mean(data_dict[c]) for c in sorted_concs]

    return np.array(sorted_concs), np.array(avg_intensities)


def plot_linear_calibration(orig_dir, inter_dir):
    # 2. 提取并平均化数据
    o_conc, o_int = get_averaged_peak_intensity(orig_dir)
    i_conc, i_int = get_averaged_peak_intensity(inter_dir)

    if len(o_conc) == 0:
        print("未找到原始实验数据点，请检查路径和 100-1000 的过滤区间。")
        return

    # 3. 线性拟合 (基于平均值点进行拟合)
    all_c = np.concatenate([o_conc, i_conc])
    all_i = np.concatenate([o_int, i_int])

    # 剔除重复浓度（如果原始和插值有重叠浓度，统一取均值进行拟合线计算）
    unique_concs = np.unique(all_c)
    unique_ints = []
    for c in unique_concs:
        mask = (all_c == c)
        unique_ints.append(np.mean(all_i[mask]))

    slope, intercept, r_value, p_value, std_err = stats.linregress(unique_concs, unique_ints)

    # 4. 开始绘图
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # 绘制拟合线
    line_x = np.array([min(unique_concs), max(unique_concs)])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Linear Fit')

    # 绘制插值生成的平均值 (空心圆点)
    ax.scatter(i_conc, i_int, facecolors='none', edgecolors='#E64B35', s=50,
               linewidth=1.0, label='Interpolated (Avg.)', alpha=0.9)

    # 绘制原始实验的平均值 (实心圆点)
    ax.scatter(o_conc, o_int, color='#4DBBD5', s=70,
               edgecolors='black', linewidth=0.8, label='Original (Avg.)', zorder=5)

    # 5. 标注 R² 和 公式 (全 Times New Roman 风格)
    # 使用 \mathrm 确保单位和字母不倾斜（根据学术规范）
    eq_text = f"$y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {r_value ** 2:.4f}$"
    ax.text(0.05, 0.92, eq_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='none'))

    # 6. 细节美化
    # 修改横坐标单位为微克每升，使用 LaTeX 渲染 mu
    ax.set_xlabel(r'Antibiotic Concentration ($\mu$g/L)', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('Fluorescence Intensity (a.u.)', fontsize=13, fontweight='bold', labelpad=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 调整布局间距
    plt.subplots_adjust(bottom=0.20, left=0.15)

    # 设置刻度字体
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(11)

    ax.legend(frameon=False, loc='lower right', fontsize=11)

    plt.show()


# --- 执行 ---
if __name__ == "__main__":
    ORIG_DIR = r'D:\data\EEM_data\lixiang\EEM-yang\strength\origin'
    INTER_DIR = r'D:\data\EEM_data\lixiang\EEM-yang\strength\interpolated'

    # 注意：range_filter 已根据你的代码设置为 100-1000
    plot_linear_calibration(ORIG_DIR, INTER_DIR)