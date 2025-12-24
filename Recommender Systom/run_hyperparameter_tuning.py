# =============================================================================
# 推荐模型自动化超参数优化脚本 (Kaggle竞赛定制版)
#
# 目的: 使用贝叶斯搜索为一系列推荐模型自动寻找最佳超参数组合，
#       并保存最佳模型、参数和评估结果。
# =============================================================================
import os
import numpy as np
import traceback
from functools import partial
import multiprocessing

# 导入所有需要的推荐器
from RecSys_Course_AT_PoliMi.Recommenders.Recommender_import_list import *
# 导入数据加载和分割函数
from RecSys_Course_AT_PoliMi.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
# 导入评估器
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
# 导入核心的超参数搜索函数
from RecSys_Course_AT_PoliMi.HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.FactorizationMachines import LightFMRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorizationImpressions_Cython import \
    MatrixFactorization_FunkSVD_Cython
from Recommenders.Neural import MultVAE_PyTorch_Recommender
from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
from Recommenders.GraphBased.LightGCNRecommender import LightGCNRecommender

# 导入你的数据加载函数 (假设此脚本与main.py在同一目录)
from main import load_and_preprocess_data

# =============================================================================
# 1. 项目与搜索配置
# =============================================================================
DATA_TRAIN_PATH = 'dataset/data_train.csv'
OUTPUT_FOLDER_PATH = "result_experiments/"
RANDOM_SEED = 1234
TRAIN_PERCENTAGE = 0.80

# --- 核心搜索配置 ---
# 竞赛的优化目标
METRIC_TO_OPTIMIZE = "RECALL"
CUTOFF_TO_OPTIMIZE = 20

# 搜索强度: 每个模型尝试的参数组合数量
# 建议值: 测试时用 10-20，正式搜索用 50-100
N_CASES = 50
N_RANDOM_STARTS = int(N_CASES / 5)  # 初始的随机探索次数

# 是否并行执行 (如果你的CPU核心数多，可以设为True来加速)
PARALLELIZE_SEARCH = True

# =============================================================================
# 2. 主执行流程
# =============================================================================
if __name__ == '__main__':

    # --- 数据准备 ---
    np.random.seed(RANDOM_SEED)
    urm_all = load_and_preprocess_data(DATA_TRAIN_PATH)
    # 创建一个用于调优的 训练集/验证集 分割
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(
        urm_all,
        train_percentage=TRAIN_PERCENTAGE
    )

    # --- 待优化的模型列表 ---
    # 已经移除了不兼容的模型，并修正了类名
    collaborative_algorithm_list = [
        LightGCNRecommender,
        MultVAE_PyTorch_Recommender,
        LightFMRecommender,
        SLIM_BPR_Python
    ]

    # --- 初始化评估器 ---
    # 这个评估器将用于在搜索过程中评估每个参数组合的性能
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[CUTOFF_TO_OPTIMIZE])

    # --- 创建部分应用函数 (Partial Function) ---
    # 使用 partial 可以固定 runHyperparameterSearch_Collaborative 的部分参数，
    # 使得它更容易被 multiprocessing 的 map 函数调用。
    runParameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                       URM_train=URM_train,
                                                       metric_to_optimize=METRIC_TO_OPTIMIZE,
                                                       cutoff_to_optimize=CUTOFF_TO_OPTIMIZE,
                                                       n_cases=N_CASES,
                                                       n_random_starts=N_RANDOM_STARTS,
                                                       evaluator_validation_earlystopping=evaluator_validation,
                                                       evaluator_validation=evaluator_validation,
                                                       evaluator_test=None,  # 我们在调优时不需要测试集
                                                       output_folder_path=OUTPUT_FOLDER_PATH,
                                                       resume_from_saved=True,
                                                       similarity_type_list=None,  # None表示使用算法的默认相似度类型
                                                       parallelizeKNN=False)  # 在并行运行多个算法时，通常关闭内部的并行

    # --- 执行搜索 ---
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)

    if PARALLELIZE_SEARCH:
        # 并行执行：为列表中的每个算法启动一个独立的搜索进程
        # 这会占用大量的CPU资源，但速度快
        print(f"--- 开始并行超参数搜索，使用 {multiprocessing.cpu_count()} 个CPU核心 ---")
        pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)
        pool.close()
        pool.join()
    else:
        # 串行执行：一个接一个地为列表中的算法进行搜索
        # 速度慢，但易于调试，CPU占用低
        print("--- 开始串行超参数搜索 ---")
        for recommender_class in collaborative_algorithm_list:
            try:
                runParameterSearch_Collaborative_partial(recommender_class)
            except Exception as e:
                print(f"在优化模型 {recommender_class.RECOMMENDER_NAME} 时发生异常: {str(e)}")
                traceback.print_exc()

    print("\n--- 所有模型的超参数搜索已完成! ---")