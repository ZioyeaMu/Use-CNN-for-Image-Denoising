import os
import re


def create_sequential_folder(base_path, prefix):
    """
    在指定路径下创建按编号顺序排列的新文件夹，并允许指定前缀。

    参数:
        base_path (str): 要创建新文件夹的基路径。
        prefix (str): 文件夹名称的前缀。

    返回:
        str: 新创建的文件夹的路径。
    """
    # 获取基路径下所有子文件夹
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    # 使用正则表达式匹配文件夹名中的前缀和数字编号
    pattern = re.compile(rf'^{prefix}(\d+)$')
    max_num = -1
    for folder in subfolders:
        match = pattern.match(folder)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num

    # 确定新文件夹的编号
    new_num = max_num + 1 if max_num != -1 else 1

    # 创建新文件夹
    new_folder_name = f"{prefix}{new_num:03d}"
    new_folder_path = os.path.join(base_path, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    return new_folder_path


# 示例用法
if __name__ == "__main__":
    base_path = "./runs"  # 指定基路径
    prefix = "train"  # 指定前缀
    new_folder = create_sequential_folder(base_path, prefix)
    print(f"创建的新文件夹路径: {new_folder}")