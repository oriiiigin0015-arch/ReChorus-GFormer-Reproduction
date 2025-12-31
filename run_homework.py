import os
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import re
import sys
import time

# ================= 配置区域 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

PYTHON_EXEC = sys.executable 
SRC_DIR = "./src" 
DATASET = "MovieLens_1M" 
SEP = "\t" 

MODELS = ["BPRMF", "LightGCN", "DirectAU"]

COMMON_ARGS = [
    "--dataset", DATASET,
    "--test_all", "1",
    "--epoch", "30",         # 保持30轮，这次LR对了会很强
    "--batch_size", "2048",
    "--lr", "0.001",         # 【关键】恢复到 0.001
    "--num_workers", "0",    # 【关键】修复 Windows 崩溃问题
    "--path", "../data",
    "--cuda", "0"
]
# ===========================================

def patch_code():
    """ 确保代码已修复 """
    print("🔧 [V15] 正在应用防崩溃补丁...")
    # 再次执行修复逻辑
    files_to_fix = {
        os.path.join(SRC_DIR, "models", "BaseModel.py"): [("np.object", "object")],
        os.path.join(SRC_DIR, "models", "general", "LightGCN.py"): [(".cuda()", "")],
        os.path.join(SRC_DIR, "utils", "utils.py"): [
            (r'np\.float(?![0-9])', 'float'),
            (r'np\.int(?![0-9])', 'int'),
            (r'np\.bool(?![0-9])', 'bool')
        ]
    }
    
    for path, rules in files_to_fix.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: content = f.read()
            new_content = content
            for old, new in rules:
                if old.startswith("np."):
                    new_content = new_content.replace(old, new)
                else:
                    new_content = re.sub(old, new, new_content)
            
            if new_content != content:
                with open(path, 'w', encoding='utf-8') as f: f.write(new_content)
    print("🔧 环境检查完毕。\n")

results = []

def parse_metrics(output):
    try:
        lines = output.split('\n')
        last_hr = 0.0
        last_ndcg = 0.0
        for line in reversed(lines):
            if "HR@10" in line and "NDCG@10" in line and "Test" in line:
                hr_part = line.split("HR@10")[1].split(",")[0].replace(":", "").strip()
                ndcg_part = line.split("NDCG@10")[1].split(",")[0].replace(":", "").strip()
                hr_match = re.search(r"0\.\d+", hr_part)
                ndcg_match = re.search(r"0\.\d+", ndcg_part)
                if hr_match and ndcg_match:
                    last_hr = float(hr_match.group())
                    last_ndcg = float(ndcg_match.group())
                    return last_hr, last_ndcg
        return 0.0, 0.0
    except Exception:
        return 0.0, 0.0

def run_experiment():
    patch_code()
    
    train_path = os.path.join("data", DATASET, "train.csv")
    if os.path.exists(train_path):
        try:
            with open(train_path, 'r') as f:
                if ',' in f.readline(): 
                    global SEP
                    SEP = ","
        except: pass

    print(f"🚀 开始运行实验 (Epochs: 30, LR: 0.001, Workers: 0)")
    print(f"📂 数据集: {DATASET}")
    
    for model in MODELS:
        print(f"\n{'='*40}")
        print(f"▶️  正在训练模型: {model}")
        print(f"{'='*40}")
        
        cmd = [PYTHON_EXEC, "main.py", "--model_name", model, "--sep", SEP] + COMMON_ARGS
        
        full_log = ""
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=SRC_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='ignore',
                bufsize=1
            )
            
            print("⏳ 正在计算中 (已过滤刷屏进度条)...")
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    full_log += line
                    keywords = ["Epoch", "Test", "HR@", "Loss", "Device", "Load"]
                    is_progress_bar = "%" in line or "it/s" in line or "Predict" in line
                    
                    if any(k in line for k in keywords) and not is_progress_bar:
                        print(f"   {line.strip()}")
            
            duration = time.time() - start_time
            
            if process.returncode != 0:
                print(f"❌ {model} 运行异常 (Code: {process.returncode})")
                print("最后10行日志:")
                print("\n".join(full_log.split('\n')[-10:]))
            else:
                hr, ndcg = parse_metrics(full_log)
                print(f"✅ {model} 完成! 耗时: {duration:.1f}s")
                print(f"   🏆 最终成绩: HR@10 = {hr}, NDCG@10 = {ndcg}")
                results.append({"Model": model, "HR@10": hr, "NDCG@10": ndcg})
                
        except Exception as e:
            print(f"❌ 系统错误: {e}")

def plot_results():
    if not results:
        print("\n⚠️ 没有数据可绘图")
        return

    df = pd.DataFrame(results)
    print("\n" + "="*20 + " 最终汇总 " + "="*20)
    print(df)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#d62728' if m == 'DirectAU' else '#1f77b4' for m in df['Model']]
    
    ax[0].bar(df['Model'], df['HR@10'], color=colors)
    ax[0].set_title(f'Hit Ratio @ 10\n({DATASET})')
    ax[0].set_ylabel('HR@10')
    for i, v in enumerate(df['HR@10']):
        ax[0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    ax[1].bar(df['Model'], df['NDCG@10'], color=colors)
    ax[1].set_title(f'NDCG @ 10\n({DATASET})')
    ax[1].set_ylabel('NDCG@10')
    for i, v in enumerate(df['NDCG@10']):
        ax[1].text(i, v, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    save_path = os.path.join(os.getcwd(), f'homework_result_{DATASET}.png')
    plt.savefig(save_path)
    print(f"\n📊 成功! 结果图已保存为: {save_path}")

if __name__ == "__main__":
    run_experiment()
    plot_results()