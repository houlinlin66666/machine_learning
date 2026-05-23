import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. 学术风格配置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def get_filtered_data(directory, fix_value, fix_index):
    """提取固定某一浓度时，另一浓度变化的数据"""
    x_list, y_list = [], []
    if not os.path.exists(directory):
        return np.array([]), np.array([])

    files = [f for f in os.listdir(directory) if f.endswith(('.xlsx', '.xls'))]
    for f in files:
        try:
            parts = f.split('-')
            if float(parts[fix_index]) == float(fix_value):
                change_index = 1 if fix_index == 0 else 0
                x_val = float(parts[change_index])
                df = pd.read_excel(os.path.join(directory, f))
                # 选取最大值作为特征强度（更符合特征峰观察）
                intensity = df.iloc[:, 1:].values.mean()
                x_list.append(x_val)
                y_list.append(intensity)
        except:
            continue
    indices = np.argsort(x_list)
    return np.array(x_list)[indices], np.array(y_list)[indices]


def plot_combined_calibration(orig_dir, inter_dir):
    plt.figure(figsize=(8, 7), dpi=300)

    # --- 数据准备 ---
    # 实验1: 固定 氧氟沙星=50, 观察 环丙沙星
    x_o1, y_o1 = get_filtered_data(orig_dir, 50, 0)
    x_i1, y_i1 = get_filtered_data(inter_dir, 50, 0)

    # 实验2: 固定 环丙沙星=50, 观察 氧氟沙星
    x_o2, y_o2 = get_filtered_data(orig_dir, 50, 1)
    x_i2, y_i2 = get_filtered_data(inter_dir, 50, 1)

    # --- 绘制第一组：环丙沙星响应 (红色系) ---
    if len(x_o1) > 1:
        # 合并原始与合成数据进行统一拟合
        all_x1 = np.concatenate([x_o1, x_i1])
        all_y1 = np.concatenate([y_o1, y_i1])
        slope1, intercept1, r1, _, _ = stats.linregress(all_x1, all_y1)

        plt.scatter(x_o1, y_o1, color='#D9042B', marker='o', s=60, label='Fixed Oflo. (Exp.)', edgecolors='k')
        plt.scatter(x_i1, y_i1, color='#D9042B', marker='x', s=40, label='Fixed Oflo. (Syn.)')

        line_x1 = np.array([0, max(all_x1)])
        plt.plot(line_x1, slope1 * line_x1 + intercept1, color='#D9042B', linestyle='-',
                 label=f'Cipro. Response ($R^2$={r1 ** 2:.4f})', alpha=0.8)

    # --- 绘制第二组：氧氟沙星响应 (蓝色系) ---
    if len(x_o2) > 1:
        all_x2 = np.concatenate([x_o2, x_i2])
        all_y2 = np.concatenate([y_o2, y_i2])
        slope2, intercept2, r2, _, _ = stats.linregress(all_x2, all_y2)

        plt.scatter(x_o2, y_o2, color='#2B2D42', marker='o', s=60, label='Fixed Cipro. (Exp.)', edgecolors='k')
        plt.scatter(x_i2, y_i2, color='#2B2D42', marker='x', s=40, label='Fixed Cipro. (Syn.)')

        line_x2 = np.array([0, max(all_x2)])
        plt.plot(line_x2, slope2 * line_x2 + intercept2, color='#2B2D42', linestyle='--',
                 label=f'Oflo. Response ($R^2$={r2 ** 2:.4f})', alpha=0.8)

    # --- 细节修饰 ---
    plt.xlabel('Concentration of Variable Component ($\mu$g/L)', fontsize=12, fontweight='bold')
    plt.ylabel('Peak Fluorescence Intensity (a.u.)', fontsize=12, fontweight='bold')
    plt.title('Linear Additivity Calibration Curves\n(Cross-Validation at 50 $\mu$g/L Base)', fontsize=14, pad=15,
              fontweight='bold')

    plt.legend(loc='best', frameon=False, fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xlim(0, None)
    plt.ylim(0, None)

    plt.tight_layout()
    plt.show()


# --- 执行 ---
if __name__ == "__main__":
    plot_combined_calibration(
        orig_dir=r'D:\data\EEM_data\hunhe-lixiang\excel\strength\origin',
        inter_dir=r'D:\data\EEM_data\hunhe-lixiang\excel\strength\interpolated'
    )