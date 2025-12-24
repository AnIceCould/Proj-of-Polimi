#%%
# -*- coding: utf-8 -*-

import pandas as pd
import scipy.sparse as sps
import numpy as np
import os
import json
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer
from RecSys_Course_AT_PoliMi.Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from RecSys_Course_AT_PoliMi.Recommenders.Recommender_import_list import *
from RecSys_Course_AT_PoliMi.Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.BaseRecommender import BaseRecommender

# ================= 配置与数据加载 =================
DATA_FOLDER = 'dataset'
DATA_TRAIN_PATH = os.path.join(DATA_FOLDER, 'data_train.csv')
OUTPUT_FOLDER = 'temp_output'
MODEL_FOLDER = 'model_output'
RANDOM_SEED = 1234
TRAIN_PERCENTAGE = 0.80
EVALUATION_CUTOFF = 20
#%%
def load_and_preprocess_data(file_path: str) -> sps.csr_matrix:
    """
    加载CSV数据文件，并将其转换为CSR格式的稀疏矩阵.
    """
    print("--- 正在加载和预处理数据... ---")
    df_train = pd.read_csv(file_path, dtype={'row': int, 'col': int})

    n_users = df_train['row'].max() + 1
    n_items = df_train['col'].max() + 1

    urm_all = sps.coo_matrix(
        ([1.0] * len(df_train['row']), (df_train['row'], df_train['col'])),
        shape=(n_users, n_items),
        dtype=float
    ).tocsr()
    print(f"数据加载完成. URM 维度: {urm_all.shape}")
    return urm_all


urm_all = load_and_preprocess_data(DATA_TRAIN_PATH)
#%%


# ================= 全局配置 =================
N_ROUNDS = 5        # 进行多少轮“对决”
EVALUATION_CUTOFF = 20  # 评估指标的 cutoff，比如 Recall@20

# 候选参数池
ials_params_list = [
    {"num_factors": 160, "epochs": 35, "confidence_scaling": "linear", "alpha": 6.833425043, "epsilon": 0.002595667, "reg": 1.45E-05},
    {"num_factors": 189, "epochs": 60, "confidence_scaling": "linear", "alpha": 12.28096809, "epsilon": 0.002133677, "reg": 1.00E-05},
    {"num_factors": 58,  "epochs": 20, "confidence_scaling": "log",    "alpha": 50.0,        "epsilon": 5.585081218, "reg": 0.000775936},
    {"num_factors": 131, "epochs": 45, "confidence_scaling": "log",    "alpha": 11.89248214, "epsilon": 6.416178267, "reg": 0.000456289}
]

slim_params_list = [
    {"topK": 821,  "l1_ratio": 0.019917914, "alpha": 0.001},
    {"topK": 1000, "l1_ratio": 0.005891513, "alpha": 0.001},
    {"topK": 1000, "l1_ratio": 0.029739176, "alpha": 0.001},
    {"topK": 1000, "l1_ratio": 2.37E-05,    "alpha": 0.001}
]

def run_round_based_tuning_recall(recommender_class, params_list, urm_all, model_name="Model", n_rounds=5):
    """
    执行多轮对决参数选择策略 (基于 Recall@20)
    """
    print(f"\n==================================================")
    print(f"开始 {model_name} 的多轮对决 (Rounds: {n_rounds})")
    print(f"指标: Recall@{EVALUATION_CUTOFF}")
    print(f"==================================================")

    # 初始化记分板：winning_counts[i] 表示第 i 组参数赢了多少轮
    winning_counts = [0] * len(params_list)

    for round_idx in range(n_rounds):
        print(f"\n>>> ROUND : {round_idx + 1} / {n_rounds}")

        # 1. 每一轮都重新随机划分数据
        # 第一次划分：分离出用于本轮“决胜”的 Test 集 (20%)
        URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(urm_all, train_percentage=0.8)

        # 第二次划分：分离出用于训练的 Train (80% of 80% = 64%)
        # 实际上我们用 URM_train 来 fit 模型，在 URM_test 上验证
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.8)

        # 初始化本轮的评估器 (针对本轮的随机 Test 集)
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[EVALUATION_CUTOFF])

        round_winner_index = None
        max_recall = -1.0

        # 2. 遍历所有候选参数
        for i, params in enumerate(params_list):
            try:
                # 初始化模型
                recommender = recommender_class(URM_train)

                # 训练模型
                recommender.fit(**params)

                # 3. 在 Test 集上评估
                results_df, _ = evaluator_test.evaluateRecommender(recommender)

                # === 关键修改：抓取 RECALL 而不是 MAP ===
                # results_df 的索引是 cutoff 值，列是指标名称
                current_recall = results_df.loc[EVALUATION_CUTOFF]["RECALL"]

                print(f"   Config {i}: Recall@{EVALUATION_CUTOFF} = {current_recall:.6f}")

                # 记录本轮最高 Recall
                if current_recall > max_recall:
                    max_recall = current_recall
                    round_winner_index = i

            except Exception as e:
                print(f"   Config {i} Failed: {str(e)}")
                continue

        # 4. 记录本轮胜者
        if round_winner_index is not None:
            winning_counts[round_winner_index] += 1
            print(f"   [Round {round_idx + 1} Winner] Config {round_winner_index} (Recall: {max_recall:.6f})")

    # ================= 决出最终赢家 =================
    print(f"\n--------------------------------------------------")
    print(f"{model_name} 对决结果 (基于 Recall@{EVALUATION_CUTOFF}):")
    for i, count in enumerate(winning_counts):
        print(f"Config {i}: 获胜 {count} 轮")

    # 找到获胜次数最多的索引
    final_winner_index = winning_counts.index(max(winning_counts))
    print(f"\n>>> 最终冠军配置 (Index {final_winner_index}):")
    print(f"{params_list[final_winner_index]}")
    print(f"==================================================\n")

    return params_list[final_winner_index]


# ==========================================================
# 执行代码
# ==========================================================

# 假设 urm_all 已经加载完毕
# urm_all = load_and_preprocess_data(...)

# 1. 运行 IALS 的多轮对决
best_ials_params = run_round_based_tuning_recall(
    IALSRecommender,
    ials_params_list,
    urm_all,
    model_name="IALS",
    n_rounds=N_ROUNDS
)

# 2. 运行 SLIM 的多轮对决
best_slim_params = run_round_based_tuning_recall(
    SLIMElasticNetRecommender,
    slim_params_list,
    urm_all,
    model_name="SLIMElasticNet",
    n_rounds=N_ROUNDS
)

# 3. 全量训练
print("\n正在使用冠军参数进行全量训练 (Full Retrain)...")

final_ials = IALSRecommender(urm_all)
final_ials.fit(**best_ials_params)
# final_ials.save_model(...)

final_slim = SLIMElasticNetRecommender(urm_all)
final_slim.fit(**best_slim_params)
# final_slim.save_model(...)

print("全量训练完成。")