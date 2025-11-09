import os

def get_models_cache_dir() -> str:
    """
    自动从当前文件开始，向上查找项目根目录下的 models 文件夹。
    返回其绝对路径。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 一直向上查找，直到找到 models 目录为止
    while True:
        models_dir = os.path.join(current_dir, "models")
        if os.path.isdir(models_dir):
            return models_dir

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("未找到 models 目录，请检查项目结构。")
        current_dir = parent_dir