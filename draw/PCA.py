import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# --- 1. 学术风格设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def load_eem_with_conc(directory, data_type_label):
    """
    读取数据并从文件名提取浓度，仅保留 100-1000 范围内的样本
    """
    vectors = []
    concs = []
    types = []

    if not os.path.exists(directory):
        return np.array([]), [], []

    files = [f for f in os.listdir(directory) if f.endswith(('.xlsx', '.xls'))]
    for file in files:
        try:
            # 提取第一个横杠前的浓度值
            conc_val = float(file.split('-')[0])

            # --- 修改部分：增加浓度区间过滤 ---
            if 100 <= conc_val <= 1000:
                file_path = os.path.join(directory, file)
                df = pd.read_excel(file_path)
                # 展平矩阵
                vector = df.iloc[:, 1:].values.flatten()

                vectors.append(vector)
                concs.append(conc_val)
                types.append(data_type_label)
            # --- 过滤结束 ---

        except Exception as e:
            print(f"解析文件 {file} 出错: {e}")

    return np.array(vectors), concs, types


def plot_concentration_pca(orig_dir, aug_dir, inter_dir):
    # 1. 加载所有数据（函数内部已实现过滤）
    o_vec, o_conc, o_type = load_eem_with_conc(orig_dir, "Original")
    a_vec, a_conc, a_type = load_eem_with_conc(aug_dir, "Augmented")
    i_vec, i_conc, i_type = load_eem_with_conc(inter_dir, "Interpolated")

    # 检查是否有数据，防止 vstack 报错
    valid_data = [v for v in [o_vec, a_vec, i_vec] if v.size > 0]
    if not valid_data:
        print("错误：指定浓度范围内（100-1000）未找到任何数据点。")
        return

    # 合并
    all_vec = np.vstack(valid_data)
    all_conc = o_conc + a_conc + i_conc
    all_type = o_type + a_type + i_type

    # 2. PCA 处理
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(all_vec)

    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled_matrix)
    var_exp = pca.explained_variance_ratio_

    df = pd.DataFrame({
        'PC1': pca_res[:, 0],
        'PC2': pca_res[:, 1],
        'Concentration': all_conc,
        'Type': all_type
    })

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 浓度连续色板 (保持原有的 magma)
    cmap = sns.color_palette("magma", as_cmap=True)

    # 分层绘制
    # Augmented
    if not df[df['Type'] == 'Augmented'].empty:
        ax.scatter(df[df['Type'] == 'Augmented']['PC1'],
                   df[df['Type'] == 'Augmented']['PC2'],
                   c=df[df['Type'] == 'Augmented']['Concentration'],
                   cmap=cmap, s=25, alpha=0.2, marker='o', edgecolors='none')

    # Interpolated
    if not df[df['Type'] == 'Interpolated'].empty:
        ax.scatter(df[df['Type'] == 'Interpolated']['PC1'],
                   df[df['Type'] == 'Interpolated']['PC2'],
                   c=df[df['Type'] == 'Interpolated']['Concentration'],
                   cmap=cmap, s=45, alpha=0.6, marker='s', edgecolors='white', linewidth=0.5)

    # Original
    if not df[df['Type'] == 'Original'].empty:
        scatter_o = ax.scatter(df[df['Type'] == 'Original']['PC1'],
                               df[df['Type'] == 'Original']['PC2'],
                               c=df[df['Type'] == 'Original']['Concentration'],
                               cmap=cmap, s=80, alpha=1.0, marker='o', edgecolors='black', linewidth=1.2)

        # 添加浓度颜色条 (单位修改为微克每升)
        cbar = plt.colorbar(scatter_o)
        cbar.set_label(r'Concentration ($\mu$g/L)', fontsize=10, fontweight='bold')

    # 细节修饰
    ax.set_xlabel(f'PC1 ({var_exp[0]:.2%})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.2%})', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Original', markerfacecolor='gray', markersize=10,
               markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='Interpolated', markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Augmented', markerfacecolor='gray', markersize=5, alpha=0.4)
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='upper right')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


# --- 调用 ---
if __name__ == "__main__":
    plot_concentration_pca(
        orig_dir=r'D:\data\EEM_data\lixiang\EEM-yang\strength\origin',
        aug_dir=r'D:\data\EEM_data\lixiang\EEM-yang\strength\augmented',
        inter_dir=r'D:\data\EEM_data\lixiang\EEM-yang\strength\interpolated'
    )