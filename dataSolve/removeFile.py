import os
import shutil
#挪动文件
#这个是单组分的挪动方式（源数据）
def organize_eem_files(source_dir, target_dir):
    # 1. 定义第一个数字的排除列表
    exclude_first = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 2, 4, 6, 7, 8, 9, 200, 400, 600, 800, 900]

    # 2. 定义第二个数字的匹配列表
    match_second = [1, 5, 9]

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    count = 0
    # 遍历源目录下的所有 Excel 文件
    for file_name in os.listdir(source_dir):
        if not file_name.endswith(('.xlsx', '.xls')):
            continue

        # 分割文件名，假设格式为 7-2-1.xlsx
        # 去掉扩展名后按 '-' 分割
        name_without_ext = os.path.splitext(file_name)[0]
        parts = name_without_ext.split('-')

        if len(parts) >= 2:
            try:
                # 转换前两个数字进行判断
                val_1 = float(parts[0])
                val_2 = float(parts[1])

                # 条件判断：
                # 第一个数不在排除列表中 AND 第二个数在匹配列表中
                if val_1 not in exclude_first and val_2 in match_second:
                    src_path = os.path.join(source_dir, file_name)
                    dst_path = os.path.join(target_dir, file_name)

                    shutil.move(src_path, dst_path)
                    count += 1
                    print(f"Moving: {file_name}")
            except ValueError:
                # 如果文件名分割后不是数字则跳过
                continue

    print(f"\n--- 任务完成 ---")
    print(f"共计移动文件: {count} 个")
    print(f"目标位置: {target_dir}")
#这个是单组分的挪动方式（增强）
import os
import shutil


def extract_non_standard_samples(source_dir, target_dir):
    # 1. 定义标准浓度列表（这些是要留在原处的，不挪动）
    standard_list = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 2, 4, 6, 7, 8, 9, 200, 400, 600, 800, 900]

    # 确保目标文件夹存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    moved_count = 0
    print("开始整理文件...")

    for file_name in os.listdir(source_dir):
        # 仅处理 Excel 文件
        if not file_name.endswith(('.xlsx', '.xls')):
            continue

        # 假设文件名格式为 7-2-1.xlsx
        name_without_ext = os.path.splitext(file_name)[0]
        parts = name_without_ext.split('-')

        if len(parts) >= 1:
            try:
                # 提取第一个括号内的数字（浓度）
                first_val = float(parts[0])

                # 判断条件：如果第一个数字【不在】标准列表中
                # 使用 np.isclose 或者简单的 round 避免浮点数存储精度导致的匹配失败
                is_standard = any(abs(first_val - s) < 1e-7 for s in standard_list)

                if not is_standard:
                    src_path = os.path.join(source_dir, file_name)
                    dst_path = os.path.join(target_dir, file_name)

                    # 移动文件
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                    print(f"已移动非标样本: {file_name}")

            except ValueError:
                # 如果文件名开头不是数字，跳过
                continue

    print("-" * 30)
    print(f"任务完成！")
    print(f"共移动文件: {moved_count} 个")
    print(f"目标文件夹: {target_dir}")


import os
import shutil

#挪到插值生成文件
def move_all_excels(source_dir, target_dir):
    # 确保目标文件夹存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    count = 0
    # 遍历源目录
    for file_name in os.listdir(source_dir):
        # 匹配所有 xlsx 和 xls 文件
        if file_name.lower().endswith(('.xlsx', '.xls')):
            src_path = os.path.join(source_dir, file_name)
            dst_path = os.path.join(target_dir, file_name)

            try:
                # 移动文件
                shutil.move(src_path, dst_path)
                count += 1
                print(f"[{count}] 已移动: {file_name}")
            except Exception as e:
                print(f"移动 {file_name} 失败: {e}")

    print("-" * 30)
    print(f"任务完成！总共挪动了 {count} 个 Excel 文件。")

# --- 填入你的路径 ---
if __name__ == "__main__":
    # 源文件夹路径（你现在的 strength 文件夹）
    SOURCE = r'D:\data\EEM_data\lixiang\EEM-yang\strength'
    # 目标文件夹路径（你想挪到的固定位置）
    TARGET = r'D:\data\EEM_data\lixiang\EEM-yang\strength\interpolated'

    # organize_eem_files(SOURCE, TARGET)
    # extract_non_standard_samples(SOURCE, TARGET)
    move_all_excels(SOURCE, TARGET)