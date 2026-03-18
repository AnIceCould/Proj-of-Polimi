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

# ==========================================
# 🔥 修正：严格使用你提供的 Import 路径和类名
# ==========================================
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch_OptimizerMask

# ==========================================
# 0. 配置与路径
# ==========================================
DATA_FOLDER = 'dataset'
DATA_TRAIN_PATH = os.path.join(DATA_FOLDER, 'data_train.csv')

# 结果保存路径
OUTPUT_FOLDER = 'result_experiments/MultVAE_GridSearch_CPU/'
MODEL_SAVE_NAME = "MultVAE_Best_Model"
LOG_FILE_NAME = 'search_results_realtime.csv'
LOG_FILE_PATH = os.path.join(OUTPUT_FOLDER, LOG_FILE_NAME)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

RANDOM_SEED = 1234
TRAIN_PERCENTAGE = 0.80
EVALUATION_CUTOFF = 20


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

# 切分验证集
URM_train, URM_validation = split_train_in_two_percentage_global_sample(
    urm_all,
    train_percentage=TRAIN_PERCENTAGE
)

# 初始化评估器
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[EVALUATION_CUTOFF])

# ==========================================
# 3. 定义搜索空间
# ==========================================
# 注意：MultVAERecommender_PyTorch_OptimizerMask 通常会自动处理网络层
# 这里我们主要调整 dropout, learning_rate 和 encoding_size
param_grid = {
    'learning_rate': [1e-3],
    'dropout': [0.5],  # 你的类可能用的是 'dropout' 而不是 'dropout_p'，保持你代码里的写法
    'encoding_size': [64],
    'epochs': [50],
    'batch_size': [128],
    'l2_reg': [0.01],
    'total_anneal_steps': [200000],
    'anneal_cap': [0.2],

    # 你的类支持这些参数用于构建网络
    'next_layer_size_multiplier': [2],
    'max_n_hidden_layers': [3]
}

keys, values = zip(*param_grid.items())
search_space = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"--- 总共需要测试 {len(search_space)} 组参数 ---")
print(f"--- 结果将实时写入: {LOG_FILE_PATH} ---")

# ==========================================
# 4. 搜索循环 (含实时日志)
# ==========================================

best_recall = 0.0
best_params = None

if os.path.exists(LOG_FILE_PATH):
    print("检测到已有日志文件，尝试读取历史最佳成绩...")
    try:
        existing_df = pd.read_csv(LOG_FILE_PATH)
        if not existing_df.empty and 'RECALL' in existing_df.columns:
            best_recall = existing_df['RECALL'].max()
            print(f"历史最佳 RECALL: {best_recall:.6f}")
    except:
        pass

for idx, params in enumerate(search_space):
    start_time = time.time()
    print(f"\n[{idx + 1}/{len(search_space)}] Testing params: {params}")

    set_global_seed(RANDOM_SEED)

    current_recall = 0.0
    status = "SUCCESS"
    error_msg = ""

    try:
        # 🔥 修正：使用你的类，并强制 CPU
        recommender = MultVAERecommender_PyTorch_OptimizerMask(
            URM_train,
            use_gpu=False  # 强制 CPU
        )

        recommender.fit(
            epochs=params['epochs'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            dropout=params['dropout'],  # 使用你代码中的参数名
            l2_reg=params['l2_reg'],
            anneal_cap=params['anneal_cap'],
            total_anneal_steps=params['total_anneal_steps'],
            sgd_mode='adam',

            # 网络结构参数
            encoding_size=params['encoding_size'],
            next_layer_size_multiplier=params['next_layer_size_multiplier'],
            max_n_hidden_layers=params['max_n_hidden_layers'],

            # Early Stopping
            validation_every_n=5,
            stop_on_validation=True,
            evaluator_object=evaluator_validation,
            lower_validations_allowed=3,
            validation_metric="RECALL"
        )

        # 评估
        result_df, _ = evaluator_validation.evaluateRecommender(recommender)
        current_recall = result_df.loc[EVALUATION_CUTOFF]['RECALL']
        print(f"   -> Recall@20: {current_recall:.6f}")

        if current_recall > best_recall:
            print(f"   >>> 新纪录 (Old: {best_recall:.6f} -> New: {current_recall:.6f})! 保存模型...")
            best_recall = current_recall
            best_params = params
            recommender.save_model(OUTPUT_FOLDER, file_name=MODEL_SAVE_NAME)

    except Exception as e:
        status = "FAILED"
        error_msg = str(e)
        print(f"   !!! 训练出错: {error_msg}")
        traceback.print_exc()

    # 实时日志写入
    elapsed_time = time.time() - start_time
    log_entry = params.copy()
    log_entry['RECALL'] = current_recall
    log_entry['STATUS'] = status
    log_entry['TIME_SEC'] = round(elapsed_time, 2)
    log_entry['ERROR'] = error_msg

    log_df = pd.DataFrame([log_entry])
    file_exists = os.path.isfile(LOG_FILE_PATH)
    try:
        log_df.to_csv(LOG_FILE_PATH, mode='a', header=not file_exists, index=False)
    except Exception as e:
        print(f"   !!! 写入日志失败: {e}")

print("\n" + "=" * 50)
print("搜索结束")
print(f"最优 RECALL: {best_recall:.6f}")
print("=" * 50)