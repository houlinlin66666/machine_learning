import pandas as pd
import numpy as np
import os


def generate_ion_impact_data(input_path, output_root):
    # 1. 兼容性读取：根据后缀名自动选择读取方式
    try:
        if input_path.endswith('.xlsx') or input_path.endswith('.xls'):
            df = pd.read_excel(input_path, index_col=0)
        else:
            df = pd.read_csv(input_path, index_col=0)
    except Exception as e:
        print(f"读取文件失败，请检查路径或文件是否被占用: {e}")
        return

    # 获取文件名第一个数字 X
    file_basename = os.path.basename(input_path)
    prefix_x = file_basename.split('-')[0] if '-' in file_basename else "Sample"

    # 定义浓度梯度
    concentrations = [10, 100, 200]

    # --- A. 处理氯离子 (Chloride) ---
    # 模拟物理机制：显著猝灭 (Stern-Volmer)
    cl_dir = os.path.join(output_root, "Cl")
    os.makedirs(cl_dir, exist_ok=True)
    ksv_cl = 0.005

    for conc in concentrations:
        factor = 1 / (1 + ksv_cl * conc)
        noise = np.random.normal(1, 0.001, df.shape)
        new_df = df * factor * noise

        save_path = os.path.join(cl_dir, f"{prefix_x}-{conc}.xlsx")
        new_df.to_excel(save_path)
        print(f"✅ 氯离子文件已生成: {save_path} (强度 x{factor:.3f})")

    # --- B. 处理硫酸根 (Sulfate) ---
    # 模拟物理机制：微弱离子强度影响 (微降)
    so4_dir = os.path.join(output_root, "SO4")
    os.makedirs(so4_dir, exist_ok=True)

    for conc in concentrations:
        # 硫酸根影响极小，假设 200mg/L 时仅下降 1%
        factor = 1 - (0.00005 * conc)
        noise = np.random.normal(1, 0.001, df.shape)
        new_df = df * factor * noise

        save_path = os.path.join(so4_dir, f"{prefix_x}-{conc}.xlsx")
        new_df.to_excel(save_path)
        print(f"✅ 硫酸根文件已生成: {save_path} (强度 x{factor:.3f})")


# --- 执行区 ---
if __name__ == "__main__":
    # 请确保安装了 openpyxl 库: pip install openpyxl

    # 1. 原始文件路径 (确保文件名包含 '-')
    input_file = r"D:\data\EEM_data\lixiang\EEM-yang\strength\origin\1.0-1-2.xlsx"

    # 2. 目标根目录
    target_root = r'D:\data\EEM_data\yan-jia-yang'

    generate_ion_impact_data(input_file, target_root)