import os


def calculate_points_rounded():
    x_min = float(input("请输入x的最小值: "))
    x_max = float(input("请输入x的最大值: "))
    x_step = float(input("请输入x的步长: "))

    y_min = float(input("请输入y的最小值: "))
    y_max = float(input("请输入y的最大值: "))
    y_step = float(input("请输入y的步长: "))

    points = []

    x = x_min
    while x <= x_max:
        y = y_min
        while y <= y_max:
            rounded_x = round(x, 1)
            rounded_y = round(y, 1)
            points.append((rounded_x, rounded_y))
            y += y_step
        x += x_step

    return points


def replace_coordinates_in_file(template_file_path, coordinates, output_directory):
    """
    这个函数接受一个模板文件路径、一个坐标列表和一个输出目录。
    它会用列表中的每个坐标替换模板中的占位符，并在指定目录中为每对坐标创建新文件。
    """
    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for x, y in coordinates:
        with open(template_file_path, "r") as file:
            content = file.read()

        # 用x和y的值替换占位符
        formatted_x = "{:.8f}".format(x)
        formatted_y = "{:.8f}".format(y)
        content = content.replace("[$1]", formatted_x).replace("[$2]", formatted_y)

        # 在指定的输出目录中为每对坐标创建一个新文件
        new_file_name = f"{output_directory}/{x}{y}.gjf"
        with open(new_file_name, "w") as new_file:
            new_file.write(content)


if __name__ == "__main__":
    coordinates = calculate_points_rounded()
    template_path = "/Users/zyd/Downloads/Gaussian/template.gjf"
    output_path = "/Users/zyd/Downloads/Gaussian/scan"
    replace_coordinates_in_file(template_path, coordinates, output_path)
