import pandas as pd
import scipy.sparse as sps
import numpy as np
import os
import torch
import itertools
import time
import traceback

from RecSys_Course_AT_PoliMi.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch_OptimizerMask


# ==========================================
# 0. 配置与路径
# ==========================================
DATA_FOLDER = 'dataset'
DATA_TRAIN_PATH = os.path.join(DATA_FOLDER, 'data_train.csv')

# 结果保存路径
OUTPUT_FOLDER = 'result_experiments/MultVAE_GridSearch_CPU/'
MODEL_SAVE_NAME = "MultVAE_Best_Model_Sub"
LOG_FILE_NAME = 'search_results_realtime.csv'
LOG_FILE_PATH = os.path.join(OUTPUT_FOLDER, LOG_FILE_NAME)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
RANDOM_SEED = 1234
TRAIN_PERCENTAGE = 0.80
EVALUATION_CUTOFF = 20

print("--- 使用全量数据进行最终训练 ---")



# ==========================================
# 1. 核心：随机种子 (CPU版)
# ==========================================
def set_global_seed(seed):
    """
    设置全局随机种子，确保CPU结果可复现
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_global_seed(RANDOM_SEED)

# 设置随机种子
set_global_seed(RANDOM_SEED)

# ==========================================
# 2. 数据加载
# ==========================================
print("--- 正在加载和预处理数据... ---")
df_train = pd.read_csv(DATA_TRAIN_PATH, dtype={'row': int, 'col': int})
n_users = df_train['row'].max() + 1
n_items = df_train['col'].max() + 1

urm_all = sps.coo_matrix(
    ([1.0] * len(df_train['row']), (df_train['row'], df_train['col'])),
    shape=(n_users, n_items),
    dtype=float
).tocsr()

print(f"URM Shape: {urm_all.shape}")

# 全部用户-物品矩阵作为训练数据
URM_train = urm_all

best_params = {
    'learning_rate': 1e-3,
    'dropout': 0.5,
    'encoding_size': 64,
    'epochs': 100,
    'batch_size': 128,
    'l2_reg': 0.01,
    'total_anneal_steps': 200000,
    'anneal_cap': 0.2,
    'next_layer_size_multiplier': 2,
    'max_n_hidden_layers': 3
}

# 初始化模型（CPU）
recommender = MultVAERecommender_PyTorch_OptimizerMask(
    URM_train,
    use_gpu=False
)

# 使用你已经搜索到的最佳参数（示例）
recommender.fit(
    epochs=best_params['epochs'],
    learning_rate=best_params['learning_rate'],
    batch_size=best_params['batch_size'],
    dropout=best_params['dropout'],
    l2_reg=best_params['l2_reg'],
    anneal_cap=best_params['anneal_cap'],
    total_anneal_steps=best_params['total_anneal_steps'],
    sgd_mode='adam',

    encoding_size=best_params['encoding_size'],
    next_layer_size_multiplier=best_params['next_layer_size_multiplier'],
    max_n_hidden_layers=best_params['max_n_hidden_layers'],

    stop_on_validation=False  # ❗关闭验证（提交比赛必须如此）
)

# 保存模型
recommender.save_model(OUTPUT_FOLDER, MODEL_SAVE_NAME)

print("训练完成，模型已保存.")
