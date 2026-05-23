import os
import shutil

# --- 配置路径 ---
source_dir = r'D:\data\EEM_data\hunhe-lixiang\excel\strength'  # 原始文件路径
target_dir = r'D:\data\EEM_data\hunhe-lixiang\excel\strength\interpolated'  # 目标存放路径

# --- 过滤逻辑配置 ---
# 排除的浓度组合 (C1-C2)
missing_concs = [
    "1-5", "1-30", "1-70", "1-300", "10-5", "10-30", "10-70", "10-300", "50-5", "50-30", "50-70", "50-300",
    "100-5", "100-30", "100-70", "100-300", "500-5", "500-30", "500-70", "500-300",
    "5-1", "5-5", "5-10", "5-30", "5-50", "5-70", "5-100", "5-300", "5-500",
    "30-1", "30-5", "30-10", "30-30", "30-50", "30-70", "30-100", "30-300", "30-500",
    "70-1", "70-5", "70-10", "70-30", "70-50", "70-70", "70-100", "70-300", "70-500",
    "300-1", "300-5", "300-10", "300-30", "300-50", "300-70", "300-100", "300-300", "300-500"
]
# 需要保留的第三个数字 (重复次数/索引)
valid_indices = ["1", "3", "5"]


# 原始文件
def batch_move_files():
    # 如果目标文件夹不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标文件夹: {target_dir}")

    count = 0
    # 遍历文件夹
    files = [f for f in os.listdir(source_dir) if f.endswith(('.xlsx', '.xls'))]

    for file_name in files:
        try:
            # 假设命名规则是: C1-C2-Index-Other.xlsx
            parts = file_name.split('-')

            if len(parts) >= 3:
                c1 = parts[0]
                c2 = parts[1]
                idx = parts[2]

                # 拼接浓度组合用于匹配
                current_conc = f"{c1}-{c2}"

                # 条件判断：
                # 1. 浓度组合不在排除列表中
                # 2. 第三个数字在 [1, 3, 5] 中
                if current_conc not in missing_concs and idx in valid_indices:
                    src_path = os.path.join(source_dir, file_name)
                    dst_path = os.path.join(target_dir, file_name)

                    # 执行复制（推荐用 copy，防止原数据丢失；若确定要搬走可用 move）
                    shutil.move(src_path, dst_path)
                    count += 1
                    print(f"已同步: {file_name}")

        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")

    print(f"\n任务完成！共计转移文件数: {count}")


import shutil
import os


def batch_move():

    # 确保目标文件夹存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取源文件夹内所有项目
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(target_dir, item)

        # 执行移动操作（剪切）
        # 如果只想复制，请将 move 改为 copy2 (文件) 或 copytree (文件夹)
        shutil.move(s, d)

print(f"所有数据已从 {source_dir} 转移到 {target_dir}")

if __name__ == "__main__":
    batch_move()
