import os


def rename_excel_files(target_dir):
    """
    将文件夹中的 excel 文件重命名为 '原名-1.xlsx'
    """
    # 检查路径是否存在
    if not os.path.exists(target_dir):
        print(f"错误：路径不存在 -> {target_dir}")
        return

    # 获取文件夹中所有文件
    files = os.listdir(target_dir)

    count = 0
    for file_name in files:
        # 1. 筛选 Excel 文件 (支持 .xlsx 和 .xls)
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):

            # 2. 分离文件名和扩展名 (例如: '样品1', '.xlsx')
            name_part, ext_part = os.path.splitext(file_name)

            # 3. 构造新文件名
            new_name = f"{name_part}-3{ext_part}"

            # 4. 获取完整路径进行操作
            old_path = os.path.join(target_dir, file_name)
            new_path = os.path.join(target_dir, new_name)

            # 5. 执行重命名
            try:
                os.rename(old_path, new_path)
                print(f"成功: {file_name} -> {new_name}")
                count += 1
            except Exception as e:
                print(f"跳过: {file_name} (原因: {e})")

    print(f"\n处理完成！共重命名了 {count} 个 Excel 文件。")


# --- 使用说明 ---
if __name__ == "__main__":
    # 请在此处填入你的实际文件夹路径
    # 注意：前面的 r 表示原始字符串，防止转义字符出错
    my_folder = r'D:\data\EEM_data\hunhe-lixiang\excel\strength'

    rename_excel_files(my_folder)