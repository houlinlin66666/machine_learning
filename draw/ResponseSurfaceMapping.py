import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm

# --- 1. 学术风格配置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def extract_data(directory):
    """从文件名解析浓度并提取特征强度"""
    c1_list, c2_list, intensity_list = [], [], []
    if not os.path.exists(directory):
        print(f"路径不存在: {directory}")
        return np.array([]), np.array([]), np.array([])

    files = [f for f in os.listdir(directory) if f.endswith(('.xlsx', '.xls'))]
    for f in files:
        try:
            parts = f.split('-')
            c1, c2 = float(parts[0]), float(parts[1])
            df = pd.read_excel(os.path.join(directory, f))
            # 提取矩阵均值
            intensity = df.iloc[:, 1:].values.mean()
            intensity = max(0, intensity)
            c1_list.append(c1)
            c2_list.append(c2)
            intensity_list.append(intensity)
        except:
            continue
    return np.array(c1_list), np.array(c2_list), np.array(intensity_list)


def plot_3d_response_surface(orig_dir, inter_dir):
    # 加载数据
    c1_o, c2_o, z_o = extract_data(orig_dir)
    c1_i, c2_i, z_i = extract_data(inter_dir)

    if len(c1_o) == 0:
        print("未发现原始数据点，请检查路径。")
        return

    # --- 2. 创建画布 ---
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # 构建平滑响应面
    xi = np.linspace(c1_o.min(), c1_o.max(), 100)
    yi = np.linspace(c2_o.min(), c2_o.max(), 100)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata((c1_o, c2_o), z_o, (grid_x, grid_y), method='cubic')
    grid_z = np.where(grid_z < 0, 0, grid_z)

    # 绘制表面
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.magma, alpha=0.4,
                           linewidth=0, antialiased=True, zorder=1)

    # --- 3. 绘制散点 ---
    # 原始实验点
    ax.scatter(c1_o, c2_o, z_o, color='#D9042B', s=55, alpha=1, edgecolors='k',
               linewidth=0.8, label='Experimental (Original)', zorder=10)

    # 合成插值点
    if len(c1_i) > 0:
        ax.scatter(c1_i, c2_i, z_i, facecolors='none', edgecolors='#2B2D42', s=25,
                   alpha=0.6, linewidth=0.5, label='Synthesized (Interpolated)', zorder=5)

    # --- 4. 核心修改：标签贴近轴线 + 字号优化 ---
    label_font_size = 9.5
    tick_font_size = 8

    # labelpad 设为较小值 (4-6)，让名字离轴线更近
    ax.set_xlabel('Ofloxacin\n($\mu$g/L)',
                  fontsize=label_font_size, fontweight='bold', labelpad=4)
    ax.set_ylabel('Ciprofloxacin\n($\mu$g/L)',
                  fontsize=label_font_size, fontweight='bold', labelpad=4)
    ax.set_zlabel('Intensity\n(a.u.)',
                  fontsize=label_font_size, fontweight='bold', labelpad=6)

    # 刻度数字缩小并贴近轴线
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size, pad=2)

    # Z轴物理约束
    ax.set_zlim(0, None)

    # --- 5. 相机与视角优化 ---
    # 稍微拉远相机距离（默认10），给外围文字留出呼吸空间
    ax.dist = 11.5
    # 调整仰角和方位角
    ax.view_init(elev=26, azim=-125)

    # 优化背景平面
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # 颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.35, aspect=18, pad=0.08)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Gradient', fontsize=9, fontweight='bold')

    # 图例
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=8)

    plt.title('3D Concentration-Response Surface', fontsize=12, pad=20, fontweight='bold')

    # 自动调整布局，避免边缘切断
    plt.tight_layout()
    plt.show()


# --- 调用执行 ---
if __name__ == "__main__":
    # 请确保文件夹路径在你的电脑上是正确的
    plot_3d_response_surface(
        orig_dir=r'D:\data\EEM_data\hunhe-lixiang\excel\strength\origin',
        inter_dir=r'D:\data\EEM_data\hunhe-lixiang\excel\strength\interpolated'
    )