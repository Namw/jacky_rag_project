import os

# 查看permanent目录下的内容
permanent_path = "data/vectorstore/permanent"

for item in os.listdir(permanent_path):
    item_path = os.path.join(permanent_path, item)
    if os.path.isdir(item_path):
        print(f"\n文件夹: {item}")
        # 查看文件夹里的文件
        try:
            files = os.listdir(item_path)
            print(f"包含 {len(files)} 个文件:")
            for f in files[:5]:  # 只显示前5个文件
                print(f"  - {f}")
            if len(files) > 5:
                print(f"  ... 还有 {len(files)-5} 个文件")
        except Exception as e:
            print(f"  无法读取: {e}")