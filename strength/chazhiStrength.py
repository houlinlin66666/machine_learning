import pandas as pd
import numpy as np
import os


def augment_eem_dataset(label_file, data_dir, output_dir, aug_per_sample=2, missing_concs=None, aug_per_missing=2):
    """
    所有生成的文件统一命名为：浓度-编号.xlsx
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取标签文件 (请确保路径正确)
    try:
        labels = pd.read_excel(label_file, header=None)
        labels.columns = ['file_name', 'conc']
    except Exception as e:
        print(f"读取标签文件失败，请检查路径: {label_file}")
        print(f"错误信息: {e}")
        return

    unique_concs = sorted(labels['conc'].unique())

    # 核心计数器：记录每个浓度生成到第几个样本了
    conc_counts = {}
    new_label_data = []

    def save_matrix_with_name(matrix, em_axis, col_names, conc):
        """统一命名保存函数"""
        # 更新该浓度的计数
        conc_counts[conc] = conc_counts.get(conc, 0) + 1
        # 构造文件名：例如 100-1.xlsx
        new_name = f"{conc}-{conc_counts[conc]}"

        df = pd.DataFrame(matrix, columns=col_names[1:])
        df.insert(0, col_names[0], em_axis)

        save_path = os.path.join(output_dir, f"{new_name}.xlsx")
        df.to_excel(save_path, index=False)
        new_label_data.append([new_name, conc])

    # --- 任务 1: 已有样本处理 (原始重命名 + 扰动增强) ---
    print("正在处理已有样本及增强...")
    for _, row in labels.iterrows():
        orig_name = row['file_name']
        conc = row['conc']
        source_path = os.path.join(data_dir, f"{orig_name}.xlsx")

        if not os.path.exists(source_path):
            print(f"跳过不存在的文件: {source_path}")
            continue

        df = pd.read_excel(source_path)
        em_axis = df.iloc[:, 0]
        matrix = df.iloc[:, 1:].values
        col_names = df.columns

        # 1.1 保存原始数据的“浓度-编号”版本
        save_matrix_with_name(matrix, em_axis, col_names, conc)

        # 1.2 生成指定数量的扰动样本
        for i in range(aug_per_sample):
            aug_matrix = matrix * np.random.uniform(0.98, 1.02)
            noise = np.random.normal(0, 0.002 * np.mean(matrix), matrix.shape)
            aug_matrix = np.maximum(0, aug_matrix + noise)
            save_matrix_with_name(aug_matrix, em_axis, col_names, conc)

    # --- 任务 2: 缺失浓度处理 (插值 + 扰动增强) ---
    if missing_concs:
        print("正在生成缺失浓度的插值数据...")
        for t_conc in missing_concs:
            lower_list = [c for c in unique_concs if c < t_conc]
            upper_list = [c for c in unique_concs if c > t_conc]

            if lower_list and upper_list:
                c_low, c_high = lower_list[-1], upper_list[0]
                name_low = labels[labels['conc'] == c_low].iloc[0]['file_name']
                name_high = labels[labels['conc'] == c_high].iloc[0]['file_name']

                df_low = pd.read_excel(os.path.join(data_dir, f"{name_low}.xlsx"))
                df_high = pd.read_excel(os.path.join(data_dir, f"{name_high}.xlsx"))

                weight = (t_conc - c_low) / (c_high - c_low)
                interp_matrix = df_low.iloc[:, 1:].values + weight * (
                            df_high.iloc[:, 1:].values - df_low.iloc[:, 1:].values)

                # 2.1 保存插值基准
                save_matrix_with_name(interp_matrix, df_low.iloc[:, 0], df_low.columns, t_conc)

                # 2.2 生成指定数量的插值增强样本
                for i in range(aug_per_missing):
                    aug_matrix = interp_matrix * np.random.uniform(0.98, 1.02)
                    noise = np.random.normal(0, 0.002 * np.mean(interp_matrix), interp_matrix.shape)
                    aug_matrix = np.maximum(0, aug_matrix + noise)
                    save_matrix_with_name(aug_matrix, df_low.iloc[:, 0], df_low.columns, t_conc)

    # 3. 保存更新后的 label 文件
    new_label_df = pd.DataFrame(new_label_data)
    new_label_df.to_excel(os.path.join(output_dir, "label_augmented.xlsx"), index=False, header=False)
    print(f"所有任务已完成！")
    print(f"数据保存路径: {output_dir}")
    print(f"新标签文件: {os.path.join(output_dir, 'label_augmented.xlsx')}")


# --- 主程序运行入口 ---
if __name__ == '__main__':
    # 建议统一使用绝对路径
    BASE_DIR = r'D:\data\EEM_data\lixiang\EEM-yang\over_noNull'

    augment_eem_dataset(
        label_file=os.path.join(BASE_DIR, 'label.xlsx'),  # 自动拼接路径，解决 FileNotFoundError
        data_dir=BASE_DIR,
        output_dir=r'D:\data\EEM_data\lixiang\EEM-yang\strength',
        aug_per_sample=3,  # 已有浓度：每个原样本生成 3 个增强，共 4 个文件
        missing_concs=[0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 2, 4, 6, 7, 8, 9, 200, 400, 600, 800, 900],
        aug_per_missing=11  # 缺失浓度：每个浓度生成 1 个插值基准 + 6 个增强，共 7 个文件
    )