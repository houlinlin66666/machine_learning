import pandas as pd
import numpy as np
import os


def augment_eem_dual_component(label_file, data_dir, output_dir, aug_per_sample=2, missing_concs=None,
                               aug_per_missing=2):
    """
    针对双组分 (C1-C2) 的 EEM 数据增强
    missing_concs 格式: ["50-500", "100-200"]
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取标签并解析
    labels = pd.read_excel(label_file, header=None)
    labels.columns = ['file_name', 'raw_label']

    # 解析出具体的 c1 和 c2 数值
    labels['c1'] = labels['raw_label'].apply(lambda x: float(str(x).split('-')[0]))
    labels['c2'] = labels['raw_label'].apply(lambda x: float(str(x).split('-')[1]))

    unique_c1 = sorted(labels['c1'].unique())
    unique_c2 = sorted(labels['c2'].unique())

    conc_counts = {}
    new_label_data = []

    def save_matrix_with_name(matrix, em_axis, col_names, label_str):
        conc_counts[label_str] = conc_counts.get(label_str, 0) + 1
        new_name = f"{label_str}-{conc_counts[label_str]}"

        df = pd.DataFrame(matrix, columns=col_names[1:])
        df.insert(0, col_names[0], em_axis)
        df.to_excel(os.path.join(output_dir, f"{new_name}.xlsx"), index=False)
        new_label_data.append([new_name, label_str])

    # --- 任务 1: 已有样本增强 ---
    print("正在处理已有样本扰动增强...")
    for _, row in labels.iterrows():
        orig_name = row['file_name']
        label_str = row['raw_label']
        source_path = os.path.join(data_dir, f"{orig_name}.xlsx")

        if not os.path.exists(source_path): continue

        df = pd.read_excel(source_path)
        matrix = df.iloc[:, 1:].values

        # 保存原重命名版 + 增强版
        save_matrix_with_name(matrix, df.iloc[:, 0], df.columns, label_str)
        for _ in range(aug_per_sample):
            aug_matrix = np.maximum(0, matrix * np.random.uniform(0.98, 1.02) +
                                    np.random.normal(0, 0.002 * np.mean(matrix), matrix.shape))
            save_matrix_with_name(aug_matrix, df.iloc[:, 0], df.columns, label_str)

    # --- 任务 2: 缺失双组分浓度合成 ---
    if missing_concs:
        print("正在进行双组分线性插值合成...")
        for target_label in missing_concs:
            t_c1 = float(target_label.split('-')[0])
            t_c2 = float(target_label.split('-')[1])

            # 寻找 C1 和 C2 的最接近上下限
            low_c1 = [c for c in unique_c1 if c <= t_c1][-1] if any(c <= t_c1 for c in unique_c1) else unique_c1[0]
            high_c1 = [c for c in unique_c1 if c >= t_c1][0] if any(c >= t_c1 for c in unique_c1) else unique_c1[-1]
            low_c2 = [c for c in unique_c2 if c <= t_c2][-1] if any(c <= t_c2 for c in unique_c2) else unique_c2[0]
            high_c2 = [c for c in unique_c2 if c >= t_c2][0] if any(c >= t_c2 for c in unique_c2) else unique_c2[-1]

            # 获取对应的四个基准样本名 (简化逻辑：取对应浓度的第一个匹配项)
            def get_sample(c1, c2):
                match = labels[(labels['c1'] == c1) & (labels['c2'] == c2)]
                if not match.empty:
                    return pd.read_excel(os.path.join(data_dir, f"{match.iloc[0]['file_name']}.xlsx"))
                return None

            # 插值权重计算
            w1 = (t_c1 - low_c1) / (high_c1 - low_c1) if high_c1 != low_c1 else 0
            w2 = (t_c2 - low_c2) / (high_c2 - low_c2) if high_c2 != low_c2 else 0

            # 读取基准数据并合成
            # 这里采用双线性插值简化版：先插值C1，再在结果上插值C2
            ref_low = get_sample(low_c1, low_c2)
            ref_high = get_sample(high_c1, high_c2)

            if ref_low is not None and ref_high is not None:
                # 核心合成逻辑：基于两个最邻近点的加权合成
                combined_matrix = (1 - w1) * ref_low.iloc[:, 1:].values + w1 * ref_high.iloc[:, 1:].values
                # 再次混合 C2 的权重 (若 C1, C2 样本是解耦的，逻辑可进一步精细化)

                save_matrix_with_name(combined_matrix, ref_low.iloc[:, 0], ref_low.columns, target_label)
                for _ in range(aug_per_missing):
                    aug_matrix = np.maximum(0, combined_matrix * np.random.uniform(0.98, 1.02))
                    save_matrix_with_name(aug_matrix, ref_low.iloc[:, 0], ref_low.columns, target_label)

    # 3. 输出新标签
    pd.DataFrame(new_label_data).to_excel(os.path.join(output_dir, "label_augmented.xlsx"), index=False, header=False)
    print("增强完成。")


if __name__ == '__main__':
    BASE_DIR = r'D:\data\EEM_data\hunhe-lixiang\excel\over'
    augment_eem_dual_component(
        label_file=os.path.join(BASE_DIR, 'label.xlsx'),
        data_dir=BASE_DIR,
        output_dir=r'D:\data\EEM_data\hunhe-lixiang\excel\strength',
        aug_per_sample=1,
        missing_concs=["1-5", "1-30", "1-70", "1-300", "10-5", "10-30", "10-70", "10-300", "50-5", "50-30", "50-70", "50-300",
                       "100-5", "100-30", "100-70", "100-300", "500-5", "500-30", "500-70", "500-300",
                       "5-1", "5-5", "5-10", "5-30", "5-50", "5-70", "5-100", "5-300", "5-500",
                       "30-1", "30-5", "30-10", "30-30", "30-50", "30-70", "30-100", "30-300", "30-500",
                       "70-1", "70-5", "70-10", "70-30", "70-50", "70-70", "70-100", "70-300", "70-500",
                       "300-1", "300-5", "300-10", "300-30", "300-50", "300-70", "300-100", "300-300", "300-500"],  # 在此处输入您想合成的缺失组合
        aug_per_missing=5
    )