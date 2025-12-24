#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna 贝叶斯优化 MultVAE（中文注释版本）
- 试验次数：100
- 使用模型内置的 early stopping（validation_every_n...）
- 每个 trial 训练结束后，评估 Recall@20，并实时保存历史最优模型
- 将每次 trial 的结果写入 CSV，便于崩溃后检查/恢复
"""

import os
import time
import json
import random
import traceback
import numpy as np
import pandas as pd
import scipy.sparse as sps
import optuna

# 请确认下面的模块路径在你的项目中可用
from RecSys_Course_AT_PoliMi.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch_OptimizerMask

# ------------------------------
# 配置（你可以根据机器资源调整）
# ------------------------------
RANDOM_SEED = 1234
TRAIN_PERCENTAGE = 0.8
EVALUATION_CUTOFF = 20

N_TRIALS = 50                     # 你要求的 100 次试验
N_STARTUP_TRIALS = 10              # 前几个随机试验用于采样器初始化
SEARCH_EPOCHS = 40                  # 贝叶斯搜索阶段每次训练的 epoch（短）
FINAL_TRAIN_EPOCHS = 150           # 若要用最佳参数做最终全量训练，可用此值
VALIDATION_EVERY_N = 5             # 模型内部每 N 个 epoch 做一次 validation（触发早停）
LOWER_VALIDATIONS_ALLOWED = 3      # internal early stopping 参数
SAMPLE_VALIDATION_USERS = None     # 若设为整数（如 3000）则在评估时仅采样这么多有交互的验证用户（可加速），默认 None -> 全量验证

OUTPUT_FOLDER = "bo_optuna_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
TRIALS_LOG_PATH = os.path.join(OUTPUT_FOLDER, "trials_log.csv")
BEST_PARAMS_PATH = os.path.join(OUTPUT_FOLDER, "best_params.json")
BEST_MODEL_PATH = os.path.join(OUTPUT_FOLDER, "best_model")  # DataIO 会在内部生成文件

# 若你希望在搜索阶段评估使用验证子集以加速，可启用下面的开关：
USE_VALIDATION_SUBSAMPLE = True if SAMPLE_VALIDATION_USERS is not None else False

# ------------------------------
# 随机种子
# ------------------------------
def set_global_seed(seed):
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_global_seed(RANDOM_SEED)

# ------------------------------
# 数据加载（若你已经在运行环境中有 urm_all，则注释掉下面加载）
# ------------------------------
# 下面是一个示例，若你已经在外部构造了 `urm_all`，可以注释掉这段并保证 urm_all 在全局可见。
DATA_TRAIN_PATH = 'dataset/data_train.csv'
df_train = pd.read_csv(DATA_TRAIN_PATH, dtype={'row': int, 'col': int})
n_users = df_train['row'].max() + 1
n_items = df_train['col'].max() + 1
urm_all = sps.coo_matrix((np.ones(len(df_train), dtype=np.float32), (df_train['row'], df_train['col'])),
                         shape=(n_users, n_items), dtype=np.float32).tocsr()

# ------------------------------
# 将全部数据拆分为训练 / 验证（和你当前本地测试流程一致）
# ------------------------------
print("正在根据 TRAIN_PERCENTAGE 拆分 URM_train / URM_validation ...")
URM_train, URM_validation = split_train_in_two_percentage_global_sample(urm_all, train_percentage=TRAIN_PERCENTAGE)
n_users, n_items = URM_train.shape
print(f"URM_train: {URM_train.shape}, URM_validation: {URM_validation.shape}")

# ------------------------------
# 若需要，可构建验证子集（只评估部分用户以加速）
# ------------------------------
def build_validation_subset(URM_validation, n_sample_users, random_seed=RANDOM_SEED):
    """
    返回与 URM_validation 形状相同，但只有 n_sample_users 行保留原始交互的稀疏矩阵。
    这样 EvaluatorHoldout 只会对这些用户计算指标，可显著加速评估。
    """
    from scipy.sparse import csr_matrix
    rng = np.random.RandomState(random_seed)
    users_with_interactions = np.unique(URM_validation.nonzero()[0])
    if len(users_with_interactions) == 0:
        raise ValueError("URM_validation 没有交互，无法构建验证集！")
    n_sample = min(n_sample_users, len(users_with_interactions))
    sampled = rng.choice(users_with_interactions, size=n_sample, replace=False)

    # 构造输出矩阵（先全0，然后赋值相应行）
    VR = csr_matrix(URM_validation.shape, dtype=URM_validation.dtype).tolil()
    for u in sampled:
        VR[u, :] = URM_validation[u]
    return VR.tocsr(), sampled

# ------------------------------
# 记录/恢复历史最优与日志功能
# ------------------------------
def load_existing_best():
    if os.path.exists(BEST_PARAMS_PATH):
        try:
            with open(BEST_PARAMS_PATH, "r") as f:
                data = json.load(f)
                return data
        except Exception:
            return None
    return None

def append_trial_log(row_dict, csv_path=TRIALS_LOG_PATH):
    df = pd.DataFrame([row_dict])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=header, index=False)

# 读取已有最优
existing = load_existing_best()
global_best_recall = -1.0
if existing is not None and "best_value" in existing:
    # best_value 存储为正 recall（非负值）
    global_best_recall = existing.get("best_value", -1.0)
    print(f"检测到已有最优 recall: {global_best_recall:.6f}")

# ------------------------------
# Objective 函数（每个 trial 调用）
# ------------------------------
def objective(trial):
    """
    Optuna objective，每个 trial 会：
    1) 从 trial 中抽取超参
    2) 用 URM_train 训练模型（短 epoch + 模型内 early stopping）
    3) 训练结束后在 URM_validation（或子集）上计算 Recall@20
    4) 记录日志 & 若优于历史最优则保存模型
    返回值：负的 Recall（因为 Optuna 默认做最小化）
    """

    # 随机化 seed（使每个 trial 有小差异）
    trial_seed = RANDOM_SEED + trial.number
    set_global_seed(trial_seed)

    # -------------- 搜索空间 --------------
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 5e-3)
    dropout = trial.suggest_uniform("dropout", 0.2, 0.6)
    encoding_size = trial.suggest_int("encoding_size", 64, 512)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    l2_reg = trial.suggest_loguniform("l2_reg", 1e-6, 1e-2)
    anneal_cap = trial.suggest_uniform("anneal_cap", 0.05, 0.4)
    total_anneal_steps = trial.suggest_int("total_anneal_steps", 50000, 300000, step=25000)
    next_layer_size_multiplier = trial.suggest_int("next_layer_size_multiplier", 2, 4)
    max_n_hidden_layers = trial.suggest_int("max_n_hidden_layers", 1, 3)

    # 固定训练策略（搜索阶段用较小 epoch）
    epochs = SEARCH_EPOCHS

    # 打印当前超参（便于日志）
    print(f"[Trial {trial.number}] params: lr={learning_rate:.5g}, dropout={dropout}, enc={encoding_size}, "
          f"batch={batch_size}, l2={l2_reg:.2e}, anneal_cap={anneal_cap:.3f}")

    # 构建评估用的 validation 矩阵（可以是全量或子采样）
    if USE_VALIDATION_SUBSAMPLE:
        URM_val_for_eval, sampled_users = build_validation_subset(URM_validation, SAMPLE_VALIDATION_USERS, random_seed=trial_seed)
    else:
        URM_val_for_eval = URM_validation
        sampled_users = None

    evaluator = EvaluatorHoldout(URM_val_for_eval, cutoff_list=[EVALUATION_CUTOFF])

    # -------------- 训练 --------------
    trial_start_time = time.time()
    try:
        # 初始化 recommender（根据是否有 cuda 自动选择）
        recommender = MultVAERecommender_PyTorch_OptimizerMask(URM_train, use_gpu=False)

        recommender.fit(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            dropout=dropout,
            l2_reg=l2_reg,
            anneal_cap=anneal_cap,
            total_anneal_steps=total_anneal_steps,
            sgd_mode='adam',
            encoding_size=encoding_size,
            next_layer_size_multiplier=next_layer_size_multiplier,
            max_n_hidden_layers=max_n_hidden_layers,

            # 重要：启用内部 Early Stopping（模型会在 validation_every_n 时评估）
            stop_on_validation=True,
            validation_every_n=VALIDATION_EVERY_N,
            evaluator_object=evaluator,
            lower_validations_allowed=LOWER_VALIDATIONS_ALLOWED,
            validation_metric="RECALL"
        )

        train_time = time.time() - trial_start_time

        # -------------- 评估（在 full/subsample 的 validation 上）--------------
        # 即便模型内部已做 validation，我们仍在外部完整评估一次以确保公平比较
        result_df, _ = EvaluatorHoldout(URM_val_for_eval, cutoff_list=[EVALUATION_CUTOFF]).evaluateRecommender(recommender)
        recall_at_20 = float(result_df.loc[EVALUATION_CUTOFF]["RECALL"])

        # 记录 trial 的日志行
        log_row = {
            "trial": trial.number,
            "status": "OK",
            "recall_at_20": recall_at_20,
            "train_time_sec": round(train_time, 2),
            "learning_rate": learning_rate,
            "dropout": dropout,
            "encoding_size": encoding_size,
            "batch_size": batch_size,
            "l2_reg": l2_reg,
            "anneal_cap": anneal_cap,
            "total_anneal_steps": total_anneal_steps,
            "next_layer_size_multiplier": next_layer_size_multiplier,
            "max_n_hidden_layers": max_n_hidden_layers,
            "sampled_validation_users": sampled_users is not None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

        append_trial_log(log_row, TRIALS_LOG_PATH)

        # -------------- 若优于历史最优，则保存模型 --------------
        global global_best_recall
        if recall_at_20 > global_best_recall:
            prev_best = global_best_recall
            global_best_recall = recall_at_20
            # 保存模型（DataIO 会存成 folder_path + file_name）
            model_file_name = f"multvae_best_trial_{trial.number}_recall_{recall_at_20:.6f}"
            try:
                recommender.save_model(OUTPUT_FOLDER + "/", file_name=model_file_name)
                # 保存当前最优参数与数值
                with open(BEST_PARAMS_PATH, "w") as f:
                    json.dump({
                        "best_trial": trial.number,
                        "best_value": recall_at_20,
                        "best_params": log_row
                    }, f, indent=2)
                # 复制为统一的 best model 名称（可选）
                # 注意：DataIO.save_data 会生成文件夹，若你想保持某个固定文件名可以在此实现复制操作
            except Exception as e_save:
                print(f"警告：保存最优模型失败: {e_save}")
            print(f"[Trial {trial.number}] 新历史最优 Recall@20: {prev_best:.6f} -> {recall_at_20:.6f}, 模型已保存为 {model_file_name}")

        # 返回负 recall（Optuna 最小化）
        return -recall_at_20

    except Exception as e:
        # 捕获异常，写入日志
        err = traceback.format_exc()
        log_row = {
            "trial": trial.number,
            "status": "FAILED",
            "error": str(e),
            "traceback": err,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        append_trial_log(log_row, TRIALS_LOG_PATH)
        print(f"[Trial {trial.number}] 发生异常，已记录日志。异常信息：{e}")
        # 返回一个糟糕的目标值（Optuna 会继续）
        return 0.0

# ------------------------------
# 创建 Optuna Study 并运行
# ------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # 设置采样器与 study
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="multvae_recall20_bo")

    print(f"开始运行 Optuna 贝叶斯优化，trial 数 = {N_TRIALS} ...")
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("检测到中断 (KeyboardInterrupt)，保存当前试验日志并退出。")
    except Exception as e:
        print("优化过程中发生未捕获异常，详细信息：")
        traceback.print_exc()

    # 保存 study 的 trials_dataframe，便于后续分析
    try:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(os.path.join(OUTPUT_FOLDER, "optuna_trials_dataframe.csv"), index=False)
    except Exception as e:
        print(f"保存 Optuna trials dataframe 失败: {e}")

    # 打印与保存最佳结果
    if study.best_trial is not None:
        best = study.best_trial
        print("=== OPTUNA 最佳试验 ===")
        print("Trial:", best.number)
        print("Value (negative recall):", best.value)
        print("Params:", best.params)
        # 若尚未在 BEST_PARAMS_PATH 保存过（上面已保存），再存一次
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump({
                "best_trial": best.number,
                "best_value": -best.value,
                "best_params": best.params
            }, f, indent=2)
        print("已将最优参数写入：", BEST_PARAMS_PATH)

    print("Optuna 优化完成。查看文件夹以获取详细日志与最优模型：", OUTPUT_FOLDER)
