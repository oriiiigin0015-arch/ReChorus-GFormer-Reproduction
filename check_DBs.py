import os

# 设定 data 目录的相对路径
# 根据你之前的报错，data 应该在当前脚本的上级目录的 data 文件夹中
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

print(f"📂 正在检查目录: {os.path.abspath(data_dir)}")

if os.path.exists(data_dir):
    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    print(f"\n✅ 发现 {len(subfolders)} 个数据集文件夹:")
    for name in subfolders:
        # 检查里面是否有 train.csv
        has_train = os.path.exists(os.path.join(data_dir, name, 'train.csv'))
        status = "🟢 (可用)" if has_train else "🔴 (缺少 train.csv)"
        print(f"  - {name} \t{status}")
    
    if not subfolders:
        print("⚠️ data 目录下没有文件夹！")
else:
    print(f"❌ 找不到 data 目录！请检查路径。")