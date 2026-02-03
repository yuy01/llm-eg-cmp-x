import os

def print_tree_with_counts(path):
    """
    以树形结构打印目录，并显示每个目录下的文件数量。
    """
    if not os.path.exists(path):
        print(f"错误：路径 '{path}' 不存在。")
        return

    # 获取根目录名称
    root_name = os.path.basename(os.path.abspath(path))
    
    # 统计根目录下的文件数
    try:
        root_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        root_count = len(root_files)
    except PermissionError:
        print(f"没有权限访问: {path}")
        return

    print(f"📂 {root_name} (当前层文件数: {root_count})")
    
    # 全局计数器（列表是可变的，可以在递归中修改）
    total_files = [root_count]
    
    # 开始递归打印
    _print_tree_recursive(path, "", total_files)
    
    print("-" * 50)
    print(f"总计文件数量 (包含所有子目录): {total_files[0]}")

def _print_tree_recursive(current_path, prefix, total_counter):
    """
    递归打印子目录
    """
    try:
        # 获取当前路径下的所有内容
        items = os.listdir(current_path)
    except PermissionError:
        print(f"{prefix}└── [权限被拒绝]")
        return

    # 筛选出所有子文件夹并排序，保证显示顺序一致
    subdirs = [d for d in items if os.path.isdir(os.path.join(current_path, d))]
    subdirs.sort()

    count = len(subdirs)
    
    for index, dirname in enumerate(subdirs):
        # 判断是否是该层级的最后一个文件夹（决定图标是 ├── 还是 └──）
        is_last = (index == count - 1)
        
        # 构建完整路径
        full_path = os.path.join(current_path, dirname)
        
        # 统计该子文件夹内的文件数量
        try:
            files_in_subdir = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]
            file_count = len(files_in_subdir)
        except PermissionError:
            file_count = "?"
        
        # 累加总数
        if isinstance(file_count, int):
            total_counter[0] += file_count

        # 设置连接符
        connector = "└── " if is_last else "├── "
        
        # 打印当前行
        # 输出格式：└── 文件夹名 (文件数: 10)
        print(f"{prefix}{connector}{dirname} (文件数: {file_count})")
        
        # 准备下一级的缩进前缀
        # 如果当前是最后一个，下一级就不需要竖线 │ 了，只需要空格
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        # 递归调用下一级
        _print_tree_recursive(full_path, new_prefix, total_counter)

if __name__ == "__main__":
    target_path = "/home/hadoop/data/cldfeed/IC_datas/IC_2class_round2_pillow_deduplicate"
    print_tree_with_counts(target_path)
