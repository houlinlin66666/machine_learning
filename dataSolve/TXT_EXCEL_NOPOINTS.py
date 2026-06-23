import os
import pandas as pd
from pathlib import Path


def process_fluorescence_txt(txt_input_path, excel_output_dir):
    """
    处理三维荧光光谱TXT文件（Hitachi F-7100 格式）
    """
    with open(txt_input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.rstrip() for line in f]

    # 找到 "Wavelength data" 起始行
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Wavelength data"):
            data_start_idx = i
            break

    if data_start_idx is None:
        print(f"⚠ 未找到 Wavelength data：{os.path.basename(txt_input_path)}")
        return

    # 找到 EX 波长所在行（Wavelength data 后第 3 行）
    ex_line_idx = data_start_idx + 2
    ex_wavelengths = list(map(float, lines[ex_line_idx].split()))

    intensity_data = []
    em_wavelengths = []

    # 从第 4 行开始是 (EM, 各 EX 下强度)
    for line in lines[ex_line_idx + 1:]:
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            em = float(parts[0])
            values = list(map(float, parts[1:]))

            if len(values) == len(ex_wavelengths):
                em_wavelengths.append(em)
                intensity_data.append(values)
        except ValueError:
            continue

    # 构造 DataFrame
    df = pd.DataFrame(intensity_data, columns=ex_wavelengths)
    df.insert(0, "EM_Wavelength(nm)", em_wavelengths)

    # 保存 Excel
    txt_name = os.path.basename(txt_input_path)
    excel_name = Path(txt_name).stem + ".xlsx"
    excel_path = os.path.join(excel_output_dir, excel_name)

    df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"✅ 已转换：{txt_name} -> {excel_name}")


def batch_process_txt_to_excel(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    txt_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".txt")
    ]

    if not txt_files:
        print("⚠ 未找到 TXT 文件")
        return

    print(f"🔍 共发现 {len(txt_files)} 个 TXT 文件")
    for txt_file in txt_files:
        process_fluorescence_txt(
            os.path.join(input_dir, txt_file),
            output_dir
        )

    print(f"\n🎉 全部完成，Excel 已保存至：\n{output_dir}")


# ================== 参数配置 ==================
INPUT_FOLDER = r"/Users/houlinlin/master/data/EEM_data/yan/blank/sds"
OUTPUT_FOLDER = r"/Users/houlinlin/master/data/EEM_data/yan/blank/sds/excel"
# ============================================

if __name__ == "__main__":
    batch_process_txt_to_excel(INPUT_FOLDER, OUTPUT_FOLDER)