import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# --- 1. 学术风格配置 (Nature Style) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def plot_fluorescence_spectra(directory):
    data_groups = {}
    all_emission_waves = None

    if not os.path.exists(directory):
        print(f"路径不存在: {directory}")
        return

    # --- 2. 数据读取与平均逻辑 ---
    files = [f for f in os.listdir(directory) if f.endswith(('.xlsx', '.xls', '.csv'))]
    for f in files:
        try:
            # 假设文件名格式: 氧氟沙星浓度-浊度-Index.xlsx
            parts = f.replace('.csv', '').replace('.xlsx', '').replace('.xls', '').split('-')
            ofl_conc = float(parts[0]) # 氧氟沙星浓度
            turbidity = float(parts[1]) # 浊度
            key = (ofl_conc, turbidity)

            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(directory, f), index_col=0)
            else:
                df = pd.read_excel(os.path.join(directory, f), index_col=0)

            target_col = '270' if '270' in df.columns else 270
            if target_col in df.columns:
                # 约束条件：取发射波长 <= 500
                mask = df.index <= 500
                valid_df = df.loc[mask]
                if all_emission_waves is None:
                    all_emission_waves = valid_df.index.values
                intensity = valid_df[target_col].values
                if key not in data_groups:
                    data_groups[key] = []
                data_groups[key].append(intensity)
        except Exception as e:
            print(f"解析 {f} 出错: {e}")

    averaged_records = []
    for (ofl, turb), val_list in data_groups.items():
        averaged_records.append({'ofl': ofl, 'turb': turb, 'y': np.mean(val_list, axis=0)})

    # --- 3. 绘图与颜色管理 ---
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    unique_ofls = sorted(list(set(r['ofl'] for r in averaged_records)))
    unique_turbs = sorted(list(set(r['turb'] for r in averaged_records)))

    # 使用 Spectral 调色盘，色彩更具高级感
    cmap = plt.cm.get_cmap('Spectral', len(unique_ofls))
    legend_colors = {}

    for i, ofl in enumerate(unique_ofls):
        ofl_data = [r for r in averaged_records if r['ofl'] == ofl]
        ofl_data.sort(key=lambda x: x['turb'])

        base_rgb = cmap(i)

        for r in ofl_data:
            # 颜色深度逻辑：浊度越高，alpha越高
            t_idx = unique_turbs.index(r['turb'])
            alpha_val = 0.4 + (0.6 * (t_idx + 1) / len(unique_turbs))
            current_color = (*base_rgb[:3], alpha_val)
            legend_colors[(ofl, r['turb'])] = current_color

            # 三次样条平滑
            x_new = np.linspace(all_emission_waves.min(), all_emission_waves.max(), 300)
            spline = make_interp_spline(all_emission_waves, r['y'], k=3)
            y_smooth = np.maximum(spline(x_new), 0)

            ax.plot(x_new, y_smooth, color=current_color, linewidth=1.3)

    # --- 4. 细节修饰 (Nature 风格) ---
    ax.set_xlabel('Emission Wavelength (nm)', fontsize=12, labelpad=10)
    ax.set_ylabel('Fluorescence Intensity (a.u.)', fontsize=12, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(all_emission_waves.min(), 500)
    ax.set_ylim(0, None)

    # --- 5. 矩阵图例 (OFL vs Turbidity) ---
    cell_text = [["" for _ in range(len(unique_turbs) + 1)] for _ in range(len(unique_ofls) + 1)]
    cell_text[0][0] = "OFL / SO$_4^{2-}$"
    for j, t in enumerate(unique_turbs): cell_text[0][j + 1] = f"{int(t)}"
    for i, c in enumerate(unique_ofls): cell_text[i + 1][0] = f"{int(c)}"
    for i in range(len(unique_ofls)):
        for j in range(len(unique_turbs)): cell_text[i + 1][j + 1] = "●"

    # 将图例放置在左上方空白处
    table = ax.table(cellText=cell_text, loc='upper left', cellLoc='center',
                     bbox=[0.05, 0.55, 0.35, 0.38], edges='closed')

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # 动态着色单元格内的圆点
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('none')
        if i > 0 and j > 0:
            ofl, turb = unique_ofls[i - 1], unique_turbs[j - 1]
            if (ofl, turb) in legend_colors:
                cell.get_text().set_color(legend_colors[(ofl, turb)])
                cell.get_text().set_fontsize(12)
        if i == 0 or j == 0:
            cell.get_text().set_weight('bold')
            cell.set_facecolor('#f0f0f0')

    plt.title('Fluorescence Response of Ofloxacin (Ex = 270 nm)', fontsize=14, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 路径已更新为您的数据目录
    data_dir = r'C:\Users\666\Desktop\jia\yang\SO4'
    plot_fluorescence_spectra(data_dir)