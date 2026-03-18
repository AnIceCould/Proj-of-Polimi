import pandas as pd
import scipy.sparse as sps
import numpy as np
import os
import torch
import itertools
import time
import sys
import traceback

# 强制隐藏 GPU，确保 PyTorch 只看得到 CPU
# 这必须在 import torch 之前或者程序最开始执行
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from RecSys_Course_AT_PoliMi.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout

# ==========================================
# 1. 导入 LightGCN
# ==========================================
from Recommenders.GraphBased.LightGCNRecommender import LightGCNRecommender

# ==========================================
# 0. 配置与路径
# ==========================================
DATA_FOLDER = 'dataset'
DATA_TRAIN_PATH = os.path.join(DATA_FOLDER, 'data_train.csv')

# 结果保存路径
OUTPUT_FOLDER = 'result_experiments/LightGCN_GridSearch_CPU/'
MODEL_SAVE_NAME = "LightGCN_Best_Model"
LOG_FILE_NAME = 'search_results_realtime.csv'
LOG_FILE_PATH = os.path.join(OUTPUT_FOLDER, LOG_FILE_NAME)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

RANDOM_SEED = 1234
TRAIN_PERCENTAGE = 0.80
EVALUATION_CUTOFF = 20


# ==========================================
# 2. 核心：随机种子 (CPU版)
# ==========================================
def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


set_global_seed(RANDOM_SEED)

# ==========================================
# 3. 数据加载
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
# 4. 定义搜索空间 (CPU 极简版)
# ==========================================
param_grid = {
    # 核心超参
    'learning_rate': [1e-3],  # 1e-3 是最稳的，先固定
    'n_layers': [2, 3],  # 核心差异点：试试 2层 和 3层
    'embedding_size': [64],  # 64 足够，128 在 CPU 上会太慢

    # 训练控制
    'epochs': [50],  # 50 轮配合 Early Stopping
    'batch_size': [1024],  # 调大 Batch Size 可以显著提升 CPU 训练速度
    'l2_reg': [1e-4],  # LightGCN 标准正则化系数
    'dropout': [0.0],  # 图卷积通常不需要太高的 dropout，0.0 或 0.1
    'sgd_mode': ['adam']
}

keys, values = zip(*param_grid.items())
search_space = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"--- 总共需要测试 {len(search_space)} 组参数 ---")
print(f"--- 警告：LightGCN 在 CPU 上运行较慢，请耐心等待 ---")
print(f"--- 结果将实时写入: {LOG_FILE_PATH} ---")

# ==========================================
# 5. 搜索循环 (含实时日志)
# ==========================================

best_recall = 0.0
best_params = None

# 读取断点（如果有）
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
        # 实例化 LightGCN
        # 注意：PoliMi 的 LightGCN 内部会自动检测 device，
        # 但我们已经在开头通过 os.environ 屏蔽了 CUDA，所以它会强制用 CPU
        recommender = LightGCNRecommender(URM_train)

        recommender.fit(
            epochs=params['epochs'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            n_layers=params['n_layers'],
            embedding_size=params['embedding_size'],
            l2_reg=params['l2_reg'],
            dropout=params['dropout'],
            sgd_mode=params['sgd_mode'],

            # Early Stopping
            validation_every_n=5,
            stop_on_validation=True,
            evaluator_object=evaluator_validation,
            lower_validations_allowed=3,  # 3次不提升就停
            validation_metric="RECALL"
        )

        # 评估
        result_df, _ = evaluator_validation.evaluateRecommender(recommender)
        current_recall = result_df.loc[EVALUATION_CUTOFF]['RECALL']
        print(f"   -> Recall@20: {current_recall:.6f}")

        # 保存最佳模型
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