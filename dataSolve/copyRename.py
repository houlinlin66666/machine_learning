import os
import shutil
import pandas as pd


def batch_rename_and_copy(label_file, source_dir, output_dir):
    """
    label_file: label.xlsx 文件路径
    source_dir: 存放原始“样品XXX.xlsx”文件的文件夹
    output_dir: 重命名后文件存放的目标文件夹
    """
    # 1. 读取 Excel 标签文件
    # 默认第一列是原名，第二列是新名
    df = pd.read_excel(label_file)

    # 创建目标文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 获取源文件夹中所有以“样品”开头且是 Excel 的文件
    files = [f for f in os.listdir(source_dir) if f.startswith('样品') and f.endswith(('.xlsx', '.xls'))]

    success_count = 0
    for file_name in files:
        # 获取不带扩展名的文件名，用于在 Excel 中匹配
        base_name = os.path.splitext(file_name)[0]

        # 3. 匹配 label 文件第一列
        # iloc[:, 0] 是第一列，iloc[:, 1] 是第二列
        match = df[df.iloc[:, 0] == base_name]

        if not match.empty:
            new_name_val = str(match.iloc[0, 1])  # 获取对应的 1-10-1 这种名字
            extension = os.path.splitext(file_name)[1]
            new_file_name = new_name_val + extension

            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(output_dir, new_file_name)

            # 4. 执行复制并重命名
            shutil.copy2(source_path, target_path)
            print(f"成功: {file_name} -> {new_file_name}")
            success_count += 1
        else:
            print(f"跳过 (未匹配): {file_name}")

    print(f"\n任务结束：成功处理 {success_count} 个文件。")


# --- 使用前配置 ---
if __name__ == '__main__':
    batch_rename_and_copy(
        label_file=r'/Users/houlinlin/master/data/EEM_data/youji/noadd/HUAN/excel/over/label.xlsx',  # 你的 label 文件路径
        source_dir=r'/Users/houlinlin/master/data/EEM_data/youji/noadd/HUAN/excel/over',  # 原始样品文件所在的文件夹
        output_dir=r'/Users/houlinlin/master/data/EEM_data/youji/noadd/HUAN/excel/rename'  # 处理后存放的文件夹
    )