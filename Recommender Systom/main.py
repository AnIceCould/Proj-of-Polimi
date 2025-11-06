# -*- coding: utf-8 -*-
"""
主执行脚本 - Kaggle电影推荐竞赛

本脚本完成了以下流程:
1. 设置项目常量和配置.
2. 加载并预处理训练数据，构建用户-物品交互矩阵 (URM).
3. 将数据分割为本地训练集和验证集.
4. 初始化、训练并评估一个基线推荐模型 (ItemKNNCF).
5. 以标准化的格式打印评估结果.

"""

# -------------------------------------------------------------------------------------
# 1. 导入必要的库
# -------------------------------------------------------------------------------------
import pandas as pd
import scipy.sparse as sps
import numpy as np

# 从官方代码库导入必要的模块
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout

# -------------------------------------------------------------------------------------
# 2. 项目配置与常量
# -------------------------------------------------------------------------------------
# 数据文件路径
# 使用 os.path.join 在未来可以更好地处理跨平台路径问题，但目前这样也足够清晰
DATA_TRAIN_PATH = 'dataset/data_train.csv'

# 随机种子，用于确保实验结果的可复现性
RANDOM_SEED = 1234

# 本地验证时，训练集所占的百分比
TRAIN_PERCENTAGE = 0.80

# 评估时使用的推荐列表长度 (cutoff)
EVALUATION_CUTOFF = 20


# -------------------------------------------------------------------------------------
# 3. 辅助函数
# -------------------------------------------------------------------------------------
def load_and_preprocess_data(file_path: str) -> sps.csr_matrix:
    """
    加载CSV数据文件，并将其转换为CSR格式的稀疏矩阵.

    Args:
        file_path (str): 训练数据CSV文件的路径.

    Returns:
        sps.csr_matrix: 用户-物品交互矩阵 (URM).
    """
    print("--- 正在加载和预处理数据... ---")
    df_train = pd.read_csv(file_path, dtype={'row': int, 'col': int})

    n_users = df_train['row'].max() + 1
    n_items = df_train['col'].max() + 1

    # 使用COO格式高效构建稀疏矩阵
    urm_all = sps.coo_matrix(
        ([1.0] * len(df_train['row']), (df_train['row'], df_train['col'])),
        shape=(n_users, n_items),
        dtype=float
    )
    # 转换为CSR格式，便于后续的行切片和矩阵运算
    urm_all = urm_all.tocsr()
    print(f"数据加载完成. URM 维度: {urm_all.shape}\n")
    return urm_all


def print_results_formatted(results_dict: dict, model_name: str = "Model"):
    """
    接收评估结果字典和模型名称，并以清晰的格式打印关键指标.

    Args:
        results_dict (dict): 来自 Evaluator.evaluateRecommender() 的结果字典.
        model_name (str): 要打印的模型名称.
    """
    res = results_dict.get(EVALUATION_CUTOFF, {})

    print(f"--- 模型评估结果: {model_name} ---")
    print(f"--------------------------------------------------")
    print(f"{f'RECALL@{EVALUATION_CUTOFF}':<25}: {res.get('RECALL', -1):.4f}   <-- 竞赛官方指标")
    print(f"{f'PRECISION@{EVALUATION_CUTOFF}':<25}: {res.get('PRECISION', -1):.4f}")
    print(f"{f'MAP@{EVALUATION_CUTOFF}':<25}: {res.get('MAP', -1):.4f}")
    print(f"{f'HIT_RATE@{EVALUATION_CUTOFF}':<25}: {res.get('HIT_RATE', -1):.4f}")
    print(f"{f'ITEM_COVERAGE@{EVALUATION_CUTOFF}':<25}: {res.get('COVERAGE_ITEM', -1):.4f}")
    print(f"{f'AVG_POPULARITY@{EVALUATION_CUTOFF}':<25}: {res.get('AVERAGE_POPULARITY', -1):.4f}")
    print(f"--------------------------------------------------\n")


# -------------------------------------------------------------------------------------
# 4. 主执行流程
# -------------------------------------------------------------------------------------
def main():
    """
    脚本的主执行函数.
    """
    # 设置全局随机种子以保证可复现性
    np.random.seed(RANDOM_SEED)

    # 加载数据
    urm_all = load_and_preprocess_data(DATA_TRAIN_PATH)

    # 分割数据用于本地验证
    print("--- 正在分割数据用于本地验证... ---")
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(
        urm_all,
        train_percentage=TRAIN_PERCENTAGE
    )
    print("数据分割完成.\n")

    # 初始化评估器
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[EVALUATION_CUTOFF])

    # -------------------------------------------
    # ----- 基线模型: Item-based KNN CF -----
    # -------------------------------------------
    print("--- 正在训练基线模型 (ItemKNNCF)... ---")
    # 初始化模型实例
    recommender_itemknn = ItemKNNCFRecommender(URM_train)

    # 训练模型 (拟合相似度矩阵)
    # 这些是模型的超参数，后续优化的重点
    recommender_itemknn.fit(topK=100, shrink=10, similarity='cosine')
    print("模型训练完成.\n")

    # 评估模型
    results_dict, _ = evaluator_validation.evaluateRecommender(recommender_itemknn)

    # 打印格式化的结果
    print_results_formatted(results_dict, "ItemKNNCF Baseline")

    # -------------------------------------------
    # ----- 在这里添加和测试你的新模型... -----
    # -------------------------------------------
    # 示例:
    # print("--- 正在训练新模型 (P3alpha)... ---")
    # recommender_p3alpha = P3alphaRecommender(URM_train)
    # recommender_p3alpha.fit(topK=50, alpha=0.8)
    # results_dict_p3, _ = evaluator_validation.evaluateRecommender(recommender_p3alpha)
    # print_results_formatted(results_dict_p3, "P3alpha")


if __name__ == '__main__':
    # 当直接运行此 .py 文件时，执行 main() 函数
    main()