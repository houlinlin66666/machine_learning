import os
import pandas as pd
from scipy.interpolate import interp1d


def fill_nan_in_excel_files(input_folder, output_folder):
    """
    对输入文件夹中的所有 Excel 文件进行处理，使用线性插值填充空缺值，并保存到输出文件夹。

    参数:
    input_folder (str): 输入 Excel 文件所在的文件夹路径。
    output_folder (str): 处理后文件的输出文件夹路径。
    """
    # 检查并创建输出文件夹
    if not os.path.exists(output_folder):
        print(f"输出文件夹 {output_folder} 不存在，正在创建...")
        os.makedirs(output_folder)
        print(f"输出文件夹 {output_folder} 创建成功。")

    # 遍历输入文件夹中的所有文件
    file_list = os.listdir(input_folder)
    file_count = len([name for name in file_list if name.endswith('.xlsx')])
    current_file = 0
    for file_name in file_list:
        if file_name.endswith('.xlsx'):
            current_file += 1
            file_path = os.path.join(input_folder, file_name)
            print(f"正在处理第 {current_file} 个文件，共 {file_count} 个文件: {file_name}")

            # 读取 Excel 文件
            try:
                excel_file = pd.ExcelFile(file_path)
            except Exception as e:
                print(f"读取文件 {file_name} 时出现错误: {e}，跳过该文件。")
                continue

            # 获取所有表名
            sheet_names = excel_file.sheet_names

            # 创建一个新的 Excel 文件写入对象
            output_file_path = os.path.join(output_folder, file_name.replace('.xlsx', '_nonull.xlsx'))
            with pd.ExcelWriter(output_file_path) as writer:
                for sheet_name in sheet_names:
                    print(f"正在处理文件 {file_name} 的工作表 {sheet_name}...")
                    # 获取指定工作表中的数据
                    try:
                        df = excel_file.parse(sheet_name)
                        df = df.head(53).copy()
                        print(f"工作表 {sheet_name} 保留前 {len(df)} 行数据进行处理")
                    except Exception as e:
                        print(f"读取工作表 {sheet_name} 时出现错误: {e}，跳过该工作表。")
                        continue

                    # 对每一列进行线性插值
                    df_filled_interp = df.copy()
                    for col in df.columns[1:]:  # 假设第一列是发射波长，不参与插值，从第二列开始
                        x = df.index[~df[col].isnull()]
                        y = df[col].dropna()
                        if len(x) > 1:
                            try:
                                f = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
                                interp_values = f(df.index)
                                # 检查插值结果是否为负，若为负则置为 0
                                interp_values[interp_values < 0] = 0
                                df_filled_interp[col] = interp_values
                            except Exception as e:
                                print(f"在对工作表 {sheet_name} 的列 {col} 进行插值时出现错误: {e}")

                    # 将填充后的数据写入到新 Excel 文件的相应工作表中
                    try:
                        df_filled_interp.to_excel(writer, sheet_name=sheet_name, index=False)
                    except Exception as e:
                        print(f"将工作表 {sheet_name} 的处理后数据写入新文件时出现错误: {e}")

if __name__ == "__main__":
    # 输入文件夹路径，需要根据实际情况修改
    input_folder = r"D:\data\EEM_data\youji\niu-rou\YANG\excel\over"
    # 输出文件夹路径，需要根据实际情况修改
    output_folder = r"D:\data\EEM_data\youji\niu-rou\YANG\excel\notNull"
    fill_nan_in_excel_files(input_folder, output_folder)