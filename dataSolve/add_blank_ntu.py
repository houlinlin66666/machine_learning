import os
import shutil
import re
from pathlib import Path


def copy_excel_with_custom_suffix(source_path, target_folder, suffix_part):
    """
    复制单个Excel文件并按规则重命名（如：原名称-10-1）

    :param source_path: 源文件完整路径
    :param target_folder: 目标文件夹路径
    :param suffix_part: 后缀部分（如 "10-1"、"30-2"）
    :return: 是否复制成功
    """
    filename = os.path.basename(source_path)

    # 匹配文件名模式：X-Y(FD3).xlsx
    match = re.match(r'^(\d+-\d+)\((FD3)\)\.xlsx$', filename)
    if not match:
        print(f"警告：{filename} 不符合命名格式，跳过该文件")
        return False

    base_part = match.group(1)  # 提取 "X-Y" 部分（如10-10）
    fd3_part = match.group(2)  # 提取 "FD3" 部分

    # 构建新文件名：X-Y-10-1(FD3).xlsx
    new_filename = f"{base_part}-{suffix_part}({fd3_part}).xlsx"
    new_filepath = os.path.join(target_folder, new_filename)

    try:
        shutil.copy2(source_path, new_filepath)
        print(f"已复制：{filename} -> {new_filename}")
        return True
    except Exception as e:
        print(f"复制文件 {filename} 到 {new_filename} 时出错: {e}")
        return False


def batch_copy_excel_custom_suffixes(source_dir, target_dir):
    """
    批量复制Excel文件，生成指定后缀的副本（10-1、10-2、30-1、30-2、50-1、50-2）
    """
    # 检查源文件夹
    if not os.path.exists(source_dir):
        print(f"错误：源文件夹不存在 - {source_dir}")
        return

    # 创建目标文件夹
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    print(f"目标文件夹: {target_dir}")

    # 筛选符合格式的Excel文件
    excel_files = [
        f for f in os.listdir(source_dir)
        # ^(\d+)\((FD3)\)\.xlsx$
        # ^\d+-\d+\(FD3\)\.xlsx$
        if f.endswith('.xlsx') and re.match(r'^\d+-\d+\(FD3\)\.xlsx$', f)
    ]

    if not excel_files:
        print(f"警告：源文件夹 {source_dir} 中未找到符合格式的Excel文件")
        print("符合格式的文件名应为：X-Y(FD3).xlsx（X、Y为数字）")
        return

    print(f"找到 {len(excel_files)} 个符合条件的Excel文件")
    print("-" * 50)

    # 定义需要生成的后缀列表
    suffix_list = ["1-1", "1-2", "5-1", "5-2", "10-1", "10-2"]
    total_success = 0

    # 处理每个文件
    for filename in excel_files:
        source_path = os.path.join(source_dir, filename)
        file_success = 0

        # 为当前文件生成所有指定后缀的副本
        for suffix in suffix_list:
            if copy_excel_with_custom_suffix(source_path, target_dir, suffix):
                file_success += 1

        if file_success == len(suffix_list):
            total_success += 1

    # 输出统计
    print("-" * 50)
    print(f"批量处理完成！")
    print(f"源文件夹: {source_dir}")
    print(f"目标文件夹: {target_dir}")
    print(f"成功处理: {total_success}/{len(excel_files)} 个文件")
    print(f"生成文件总数: {total_success * len(suffix_list)} 个")


# -------------------------- 配置参数（请修改为你的实际路径）--------------------------
SOURCE_FOLDER = r"D:\data\EEM_data\youji\HA\HUAN\excel\blank-short"
TARGET_FOLDER = r"D:\data\EEM_data\youji\HA\HUAN\excel\blank"
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    batch_copy_excel_custom_suffixes(SOURCE_FOLDER, TARGET_FOLDER)