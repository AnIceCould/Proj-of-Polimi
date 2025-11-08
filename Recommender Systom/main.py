# -*- coding: utf-8 -*-
"""
主控制脚本 - Kaggle电影推荐竞赛

本脚本是项目的中央控制中心，可以执行以下任务:
1. 手动运行单个模型的实验.
2. 自动化地评估一系列模型的基线性能.
3. 使用最佳模型生成最终的提交文件.

通过修改 `if __name__ == '__main__':` 代码块中的函数调用来选择要执行的任务。
"""

# -------------------------------------------------------------------------------------
# 1. 导入必要的库
# -------------------------------------------------------------------------------------
import pandas as pd
import scipy.sparse as sps
import numpy as np
import os
import traceback

# 从官方代码库导入所有推荐器，为自动化评估做准备
from RecSys_Course_AT_PoliMi.Recommenders.Recommender_import_list import *
from RecSys_Course_AT_PoliMi.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout

# -------------------------------------------------------------------------------------
# 2. 项目配置与常量
# -------------------------------------------------------------------------------------
DATA_FOLDER = 'dataset'
OUTPUT_FOLDER = 'temp_output'
DATA_TRAIN_PATH = os.path.join(DATA_FOLDER, 'data_train.csv')
DATA_TARGET_USERS_PATH = os.path.join(DATA_FOLDER, 'data_target_users_test.csv')
RANDOM_SEED = 1234
TRAIN_PERCENTAGE = 0.80
EVALUATION_CUTOFF = 20


# -------------------------------------------------------------------------------------
# 3. 辅助函数 (与之前相同)
# -------------------------------------------------------------------------------------
def load_and_preprocess_data(file_path: str) -> sps.csr_matrix:
    # ... (此函数内容保持不变) ...
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


def print_results_formatted(results_df, model_name: str = "Model"):
    # ... (此函数内容保持不变) ...
    if results_df.empty or EVALUATION_CUTOFF not in results_df.index:
        print(f"--- 在 cutoff={EVALUATION_CUTOFF} 处没有找到 '{model_name}' 的评估结果 ---")
        return
    res_series = results_df.loc[EVALUATION_CUTOFF]
    print(f"--- 模型评估结果: {model_name} ---")
    print(f"--------------------------------------------------")
    print(f"{f'RECALL@{EVALUATION_CUTOFF}':<25}: {res_series.get('RECALL', -1):.4f}   <-- 竞赛官方指标")
    print(f"{f'PRECISION@{EVALUATION_CUTOFF}':<25}: {res_series.get('PRECISION', -1):.4f}")
    print(f"{f'MAP@{EVALUATION_CUTOFF}':<25}: {res_series.get('MAP', -1):.4f}")
    print(f"--------------------------------------------------\n")


# -------------------------------------------------------------------------------------
# 4. 核心功能函数 (将不同任务封装)
# -------------------------------------------------------------------------------------
def run_manual_experiments(URM_train, URM_validation):
    """手动运行和测试单个模型。"""
    print("\n##### 任务: 手动运行实验 #####")
    evaluator = EvaluatorHoldout(URM_validation, cutoff_list=[EVALUATION_CUTOFF])

    # --- 实验 1: ItemKNNCF ---
    recommender = ItemKNNCFRecommender(URM_train)
    recommender.fit(topK=100, shrink=10, similarity='cosine')
    results_df, _ = evaluator.evaluateRecommender(recommender)
    print_results_formatted(results_df, "ItemKNNCF")

    # --- 实验 2: P3alpha ---
    recommender = P3alphaRecommender(URM_train)
    recommender.fit(topK=100, alpha=0.8, implicit=True)
    results_df, _ = evaluator.evaluateRecommender(recommender)
    print_results_formatted(results_df, "P3alpha")


def run_automated_evaluation(URM_train, URM_validation):
    """自动化地评估一系列模型的基线性能。"""
    print("\n##### 任务: 自动化模型评估 #####")

    recommender_class_list = [
        TopPop, ItemKNNCFRecommender, UserKNNCFRecommender,
        P3alphaRecommender, RP3betaRecommender,
        SLIM_BPR_Cython_Epoch,
        IALSRecommender,
        MatrixFactorization_BPR_Cython_Epoch,
        MatrixFactorization_FunkSVD_Cython_Epoch,
        EASE_R_Recommender,
        PureSVDRecommender,
    ]

    evaluator = EvaluatorHoldout(URM_validation, cutoff_list=[EVALUATION_CUTOFF])
    log_file_path = os.path.join(OUTPUT_FOLDER, "result_all_algorithms_kaggle.txt")

    with open(log_file_path, "w") as logFile:
        for recommender_class in recommender_class_list:
            try:
                model_name = recommender_class.RECOMMENDER_NAME
                print(f"\n--- 正在评估模型: {model_name} ---")

                recommender = recommender_class(URM_train)
                recommender.fit()

                results_df, _ = evaluator.evaluateRecommender(recommender)
                recall_score = results_df.loc[EVALUATION_CUTOFF].get('RECALL', 0.0)

                result_string = f"Recall@{EVALUATION_CUTOFF}: {recall_score:.4f}"
                print(f"模型: {model_name}, 结果: {result_string}")
                logFile.write(f"模型: {model_name}, 结果: {result_string}\n")
                logFile.flush()

            except Exception as e:
                error_message = f"模型: {recommender_class.RECOMMENDER_NAME} - 发生异常: {str(e)}"
                print(error_message)
                traceback.print_exc()
                logFile.write(error_message + "\n")
                logFile.flush()


def generate_submission_file(urm_all):
    """使用最佳模型生成提交文件。"""
    print("\n##### 任务: 生成提交文件 #####")

    model_config = {
        "class": ItemKNNCFRecommender,
        "params": {'topK': 100, 'shrink': 10, 'similarity': 'cosine'}
    }

    submission_filename = f"submission_{model_config['class'].RECOMMENDER_NAME}.csv"
    SUBMISSION_PATH = os.path.join(OUTPUT_FOLDER, submission_filename)

    final_model_class = model_config["class"]
    final_model_params = model_config["params"]

    print(f"--- 正在使用模型 '{final_model_class.RECOMMENDER_NAME}' 在全量数据上进行训练... ---")
    final_recommender = final_model_class(urm_all)
    final_recommender.fit(**final_model_params)
    print("最终模型训练完成。\n")

    df_target_users = pd.read_csv(DATA_TARGET_USERS_PATH)
    target_user_ids = df_target_users['user_id'].values

    submission = []
    for user_id in target_user_ids:
        recommended_items = final_recommender.recommend(user_id, cutoff=EVALUATION_CUTOFF, remove_seen_flag=True)
        submission.append((user_id, ' '.join(map(str, recommended_items))))

    df_submission = pd.DataFrame(submission, columns=['user_id', 'item_list'])
    df_submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"--- 提交文件已成功生成! ---")
    print(f"文件保存在: {SUBMISSION_PATH}")


# -------------------------------------------------------------------------------------
# 5. 主执行入口
# -------------------------------------------------------------------------------------
if __name__ == '__main__':
    # --- 通用设置 ---
    np.random.seed(RANDOM_SEED)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # --- 数据准备 ---
    urm_all = load_and_preprocess_data(DATA_TRAIN_PATH)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(
        urm_all,
        train_percentage=TRAIN_PERCENTAGE
    )

    # =========================================================================
    # =============== 在这里选择你要执行的任务 (取消注释对应的行) ===============
    # =========================================================================

    # 任务1: 运行手动配置的几个实验
    # run_manual_experiments(URM_train, URM_validation)

    # 任务2: 自动化评估所有主流模型
    run_automated_evaluation(URM_train, URM_validation)

    # 任务3: 生成提交文件
    # generate_submission_file(urm_all)

    print("\n脚本执行完毕。")