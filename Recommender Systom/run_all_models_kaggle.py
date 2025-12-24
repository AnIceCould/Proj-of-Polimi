# =============================================================================
# 推荐模型自动化基准测试脚本 (Kaggle竞赛定制版)
#
# 目的: 自动运行一系列推荐模型，使用默认参数进行训练和评估，
#       并将它们的 Recall@20 分数记录到日志文件中，以便快速比较和筛选。
# =============================================================================
import traceback
import os
import numpy as np
import pandas as pd
import scipy.sparse as sps

# 导入所有需要的推荐器
# Recommender_import_list 中包含了所有模型的导入语句
from RecSys_Course_AT_PoliMi.Recommenders.Recommender_import_list import *
# 导入评估器
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from RecSys_Course_AT_PoliMi.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

# =============================================================================
# 1. 项目配置
# =============================================================================
DATA_TRAIN_PATH = 'dataset/data_train.csv'
OUTPUT_FOLDER = "result_experiments/"  # 统一输出目录
RANDOM_SEED = 1234
TRAIN_PERCENTAGE = 0.80
EVALUATION_CUTOFF = 20


# =============================================================================
# 2. 辅助函数 (将它们直接放在脚本中，避免从main导入)
# =============================================================================
def load_and_preprocess_data(file_path: str) -> sps.csr_matrix:
    """加载CSV数据并转换为CSR稀疏矩阵。"""
    print("--- 正在加载和预处理数据... ---")
    df_train = pd.read_csv(file_path, dtype={'row': int, 'col': int})
    n_users = df_train['row'].max() + 1
    n_items = df_train['col'].max() + 1
    urm_all = sps.coo_matrix(
        ([1.0] * len(df_train['row']), (df_train['row'], df_train['col'])),
        shape=(n_users, n_items),
        dtype=float
    ).tocsr()
    print(f"数据加载完成. URM 维度: {urm_all.shape}\n")
    return urm_all


# =============================================================================
# 3. 主执行流程
# =============================================================================
if __name__ == '__main__':

    # 设置随机种子
    np.random.seed(RANDOM_SEED)

    # --- 数据加载和分割 ---
    urm_all = load_and_preprocess_data(DATA_TRAIN_PATH)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(
        urm_all,
        train_percentage=TRAIN_PERCENTAGE
    )

    # --- 兼容模型的列表 ---
    # 已经移除了所有需要额外特征(ICM/UCM)的模型，并修正了类名
    recommender_class_list = [
        SLIM_BPR_Cython,
    ]

    # --- 评估器 ---
    evaluator = EvaluatorHoldout(URM_validation, cutoff_list=[EVALUATION_CUTOFF])

    # --- 日志文件 ---
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    logFile = open(os.path.join(OUTPUT_FOLDER, "benchmark_results.txt"), "w")  # "w"模式会覆盖旧文件

    # --- 循环评估 ---
    for recommender_class in recommender_class_list:
        try:
            model_name = recommender_class.RECOMMENDER_NAME
            print(f"\n--- 正在评估模型: {model_name} ---")

            # 初始化模型
            recommender_object = recommender_class(URM_train)

            # 训练模型 (使用默认参数)
            recommender_object.fit()

            # 评估模型
            results_df, _ = evaluator.evaluateRecommender(recommender_object)

            # 获取并记录 Recall@20 的分数
            recall_score = results_df.loc[EVALUATION_CUTOFF].get('RECALL', 0.0)
            result_string = f"Recall@{EVALUATION_CUTOFF}: {recall_score:.5f}"
            print(f"模型: {model_name}, 结果: {result_string}")
            logFile.write(f"{model_name:<40} | {result_string}\n")
            logFile.flush()

        except Exception as e:
            error_message = f"模型: {recommender_class.RECOMMENDER_NAME} - 发生异常: {str(e)}"
            print(error_message)
            traceback.print_exc()
            logFile.write(f"{recommender_class.RECOMMENDER_NAME:<40} | FAILED: {str(e)}\n")
            logFile.flush()

    logFile.close()
    print(f"\n基准测试完成！结果已保存在: {os.path.join(OUTPUT_FOLDER, 'benchmark_results.txt')}")