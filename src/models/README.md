# 机器学习大作业 - ReChorus复现与实验

本项目是机器学习课程的大作业，基于 [ReChorus](https://github.com/THUwangcy/ReChorus) 框架进行了推荐系统模型的复现与实验对比。

## 项目结构

- **`src/`**: 核心代码目录，包含模型定义 (`models/`) 和工具函数 (`utils/`)。
- **`data/`**: 数据集目录 (如 `Grocery_and_Gourmet_Food`, `MovieLens` 等)。
- **`log/`**: 实验运行日志及生成的推荐结果 CSV 文件。
- **`run_homework.py`**: **作业主运行脚本**，用于执行实验流程。
- **`check_DBs.py`**: 数据集检查工具。
- **`homework_result_*.png`**: 实验结果的可视化对比图。

## 包含模型

本项目主要复现并对比了以下推荐算法：
1. **LightGCN**: 简化的图卷积网络推荐模型。
2. **BPRMF**: 基于贝叶斯个性化排序的矩阵分解。
3. **DirectAU**: (以及其他你实验中用到的模型)。

## 环境依赖

请确保安装了 Python 3.x 及以下依赖库：
- PyTorch
- NumPy
- Pandas
- Scipy
- (其他 ReChorus 需要的依赖)

## 如何运行

### 1. 运行主实验脚本
直接运行根目录下的脚本即可开始训练和测试：

```bash
python run_homework.py
```

### 2. 运行指定模型 (ReChorus 原生方式)
也可以使用 src 下的入口文件运行特定参数的模型，例如：

```bash
python src/main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food --lr 1e-3 --batch_size 2048
```

## 实验结果

实验运行结束后，日志文件保存在 `log/` 目录下。
可视化结果图表将生成在项目根目录，例如：`homework_result_Grocery_and_Gourmet_Food.png`。

## 引用与致谢

本项目基于清华大学 [ReChorus](https://github.com/THUwangcy/ReChorus) 框架开发。
