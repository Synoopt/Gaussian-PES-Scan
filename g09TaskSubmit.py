import subprocess
import os


def run_g09_in_batch(input_directory, output_directory):
    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历指定文件夹中的所有文件
    for input_file in os.listdir(input_directory):
        full_input_path = os.path.join(input_directory, input_file)

        # 确保它是一个文件
        if os.path.isfile(full_input_path):
            # 获取不带扩展名的文件名
            file_name_without_ext = os.path.splitext(input_file)[0]

            # 将文件名中非扩展名的点转为\.
            file_name_without_ext = file_name_without_ext.replace(".", r"\.", 2)

            # 构造输入和输出文件的路径
            input_file = os.path.join(input_directory, file_name_without_ext + ".gjf")
            output_file = os.path.join(output_directory, file_name_without_ext + ".out")

            # 构建g09命令
            command = f"g09 < {input_file} > {output_file}"

            # 执行命令
            subprocess.run(command, shell=True)

            print(f"处理完成: {input_file}")

    print("所有文件处理完毕")


if __name__ == "__main__":
    input_dir = ""
    output_dir = ""
    run_g09_in_batch(input_dir, output_dir)
