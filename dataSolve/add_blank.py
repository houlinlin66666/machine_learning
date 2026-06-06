import os
import shutil
import re
from pathlib import Path


def copy_excel_with_suffix(source_path, target_folder, index):
    """
    复制单个Excel文件并按规则重命名

    :param source_path: 源文件完整路径
    :param target_folder: 目标文件夹路径
    :param index: 编号（1, 2, 3）
    :return: 是否复制成功
    """
    filename = os.path.basename(source_path)

    # 使用正则表达式匹配文件名模式 X-Y(FD3).xlsx
    match = re.match(r'^(\d+-\d+)\((FD3)\)\.xlsx$', filename)

    if not match:
        print(f"警告：{filename} 不符合命名格式，跳过该文件")
        return False

    # 获取基础部分和FD3部分
    base_part = match.group(1)  # 如 "1-1", "10-1", "500-100"
    fd3_part = match.group(2)  # "FD3"

    # 构建新文件名: X-Y-1(FD3).xlsx
    new_filename = f"{base_part}-{index}({fd3_part}).xlsx"
    new_filepath = os.path.join(target_folder, new_filename)

    try:
        # 复制文件
        shutil.copy2(source_path, new_filepath)
        print(f"已复制：{filename} -> {new_filename}")
        return True
    except Exception as e:
        print(f"复制文件 {filename} 到 {new_filename} 时出错: {e}")
        return False


def batch_copy_excel_with_suffixes(source_dir, target_dir, suffixes=[1, 2, 3]):
    """
    批量复制Excel文件并添加后缀

    :param source_dir: 源文件夹路径
    :param target_dir: 目标文件夹路径
    :param suffixes: 后缀列表，默认[1, 2, 3]
    """
    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"错误：源文件夹不存在 - {source_dir}")
        return

    # 创建目标文件夹
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    print(f"目标文件夹: {target_dir}")

    # 获取源文件夹中所有.xlsx文件
    excel_files = [f for f in os.listdir(source_dir)
                   if f.endswith('.xlsx') and re.match(r'^\d+-\d+\(FD3\)\.xlsx$', f)]

    if not excel_files:
        print(f"警告：源文件夹 {source_dir} 中未找到符合格式的Excel文件")
        print("符合格式的文件名应为：X-Y(FD3).xlsx，其中X和Y为数字")
        return

    print(f"找到 {len(excel_files)} 个符合条件的Excel文件")
    print("-" * 50)

    # 处理每个文件
    total_success = 0
    for filename in excel_files:
        source_path = os.path.join(source_dir, filename)

        # 为每个文件创建指定数量的副本
        file_success = 0
        for suffix in suffixes:
            if copy_excel_with_suffix(source_path, target_dir, suffix):
                file_success += 1

        if file_success == len(suffixes):
            total_success += 1

    # 输出统计信息
    print("-" * 50)
    print(f"批量处理完成！")
    print(f"源文件夹: {source_dir}")
    print(f"目标文件夹: {target_dir}")
    print(f"成功处理: {total_success}/{len(excel_files)} 个文件")
    print(f"生成文件总数: {total_success * len(suffixes)} 个")


# -------------------------- 配置参数（请根据实际情况修改）--------------------------
SOURCE_FOLDER = r"D:\data\EEM_data\hunhe-lixiang\excel\hunhe-blank"  # 例如：r"D:\Excel_Files\Original"
TARGET_FOLDER = r"D:\data\EEM_data\hunhe-lixiang\excel\hunhe-y-h\blank"  # 例如：r"D:\Excel_Files\Copies"
SUFFIXES = [1, 2, 3]  # 后缀编号列表
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    # 运行批量处理
    batch_copy_excel_with_suffixes(SOURCE_FOLDER, TARGET_FOLDER, SUFFIXES)