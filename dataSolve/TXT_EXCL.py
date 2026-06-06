import os
import pandas as pd
from pathlib import Path


def process_fluorescence_txt(txt_input_path, excel_output_dir):
    """
    处理单个荧光光谱txt文件，提取EX、EM和光强数据并保存为Excel
    :param txt_input_path: 单个txt文件路径
    :param excel_output_dir: Excel输出文件夹路径
    """
    # 读取txt文件内容
    with open(txt_input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().strip().split('\n')

    # 找到Data points开始的位置（忽略前面所有无关信息）
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Data points'):
            data_start_idx = i + 1  # 下一行是EX波长表头
            break

    if data_start_idx is None:
        print(f"警告：{os.path.basename(txt_input_path)} 中未找到Data points部分，跳过该文件")
        return

    # 提取EX波长（表头行）
    header_line = lines[data_start_idx].strip()
    if not header_line:
        print(f"警告：{os.path.basename(txt_input_path)} 表头为空，跳过该文件")
        return

    ex_wavelengths = list(map(float, header_line.split()))

    # 提取EM波长和对应光强数据
    em_wavelengths = []
    intensity_data = []

    for line in lines[data_start_idx + 1:]:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        parts = line.split()
        if len(parts) < 2:
            continue  # 跳过无效数据行

        # 第一列为EM波长，后面为光强值
        try:
            em_wave = float(parts[0])
            intensities = list(map(float, parts[1:]))
            # 确保光强数据长度与EX波长一致
            if len(intensities) == len(ex_wavelengths):
                em_wavelengths.append(em_wave)
                intensity_data.append(intensities)
        except ValueError:
            continue  # 跳过无法转换为数字的行

    # 创建DataFrame
    df = pd.DataFrame(intensity_data, columns=ex_wavelengths)
    df.insert(0, 'EM_Wavelength(nm)', em_wavelengths)  # 第一列设为EM波长

    # 生成输出Excel文件名（与txt同名）
    txt_filename = os.path.basename(txt_input_path)
    excel_filename = os.path.splitext(txt_filename)[0] + '.xlsx'
    excel_output_path = os.path.join(excel_output_dir, excel_filename)

    # 保存Excel文件
    df.to_excel(excel_output_path, index=False, engine='openpyxl')
    print(f"已处理：{txt_filename} -> {excel_filename}")


def batch_process_txt_to_excel(input_dir, output_dir):
    """
    批量处理文件夹中所有txt文件
    :param input_dir: 输入txt文件夹路径
    :param output_dir: 输出Excel文件夹路径
    """
    # 创建输出文件夹（不存在则创建）
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 遍历输入文件夹中所有txt文件
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.TXT')]

    if not txt_files:
        print(f"警告：输入文件夹 {input_dir} 中未找到txt文件")
        return

    print(f"找到 {len(txt_files)} 个txt文件，开始批量处理...")
    for txt_file in txt_files:
        txt_path = os.path.join(input_dir, txt_file)
        process_fluorescence_txt(txt_path, output_dir)

    print(f"\n批量处理完成！所有Excel文件已保存至：{output_dir}")


# -------------------------- 配置参数（请根据实际情况修改）--------------------------
INPUT_FOLDER = r"/Users/houlinlin/master/data/EEM_data/yan/Cl/huan"  # 例如：r"C:\Users\XXX\Fluorescence_Data\TxtFiles"
OUTPUT_FOLDER = r"/Users/houlinlin/master/data/EEM_data/yan/Cl/huan/excel"  # 例如：r"C:\Users\XXX\Fluorescence_Data\ExcelFiles"
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    # 运行批量处理
    batch_process_txt_to_excel(INPUT_FOLDER, OUTPUT_FOLDER)