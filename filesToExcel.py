import re
import pandas as pd
import os


def extract_energy_from_file(file_path):
    # 初始化变量
    last_energy = None
    energy_pattern = r"SCF Done:  E\(RPM6\) =\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"

    # 读取文件并查找最后一个匹配的能量值
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(energy_pattern, line)
            if match:
                last_energy = float(match.group(1))

    return last_energy


def process_files(input_directory):
    records = []

    # 遍历指定文件夹中的所有文件
    for file in os.listdir(input_directory):
        if file.endswith(".out"):
            full_path = os.path.join(input_directory, file)
            coordinates = extract_coordinates_from_filename(file)
            energy = extract_energy_from_file(full_path)

            if coordinates and energy is not None:
                records.append(coordinates + (energy,))

    return records


def extract_coordinates_from_filename(filename):
    # 使用正则表达式提取两个一位小数（包括负数）
    matches = re.findall(r"-?\d\.\d", filename)
    if len(matches) == 2:
        return float(matches[0]), float(matches[1])
    else:
        return None


# 示例用法
input_directory = "/Users/zyd/Downloads/Gaussian/output_Energy"  # 替换为您的文件夹路径
records = process_files(input_directory)

# 将结果保存到DataFrame中
df = pd.DataFrame(records, columns=["X", "Y", "Energy"])

# 将DataFrame保存到Excel文件中
output_file = "/Users/zyd/Downloads/Gaussian/outputExcel_Energy.xlsx"  # 替换为您的输出文件路径
df.to_excel(output_file, index=False)
