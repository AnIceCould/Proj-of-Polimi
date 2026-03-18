#%% md
# ### 库
#%%
# -*- coding: utf-8 -*-

import pandas as pd
import scipy.sparse as sps
import numpy as np
import os
from tqdm import tqdm

from RecSys_Course_AT_PoliMi.Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from RecSys_Course_AT_PoliMi.Evaluation.Evaluator import EvaluatorHoldout
from RecSys_Course_AT_PoliMi.Recommenders.Recommender_import_list import *
import lightgbm as lgb
from sklearn.model_selection import KFold

# 数据文件路径
DATA_FOLDER = 'dataset'
DATA_TRAIN_PATH = os.path.join(DATA_FOLDER, 'data_train.csv')
DATA_TARGET_USERS_PATH = os.path.join(DATA_FOLDER, 'data_target_users_test.csv')
OUTPUT_FOLDER = 'temp_output'
MODEL_FOLDER = 'model_output'
SUBMISSION_FOLDER = 'temp_output' # 提交文件的保存目录

# 随机种子，用于确保实验结果的可复现性
RANDOM_SEED = 1234

# 本地验证时，训练集所占的百分比
TRAIN_PERCENTAGE = 0.80

# 评估时使用的推荐列表长度 (cutoff)
EVALUATION_CUTOFF = 20

# 设置全局随机种子
np.random.seed(RANDOM_SEED)

best_ials_params = {
    "num_factors": 58,
    "epochs": 30,
    "confidence_scaling": "log",
    "alpha": 49.99999999999999,
    "epsilon": 5.585081217734329,
    "reg": 0.0007759360926311159
}

best_slim_params = {
    "topK": 1000,
    "l1_ratio": 0.029739176029882,
    "alpha": 0.001
}

print("项目配置加载完成.")

#%% md
# ### 辅助函数
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

def load_best_model(recommender_class, urm_train, file_name, modelfile_path):
    """
    加载一个由超参数搜索脚本保存的最佳模型。
    """
    file_path = os.path.join(modelfile_path, file_name)

    print(f"--- 正在加载预训练模型: {file_name} ---")

    # 检查模型文件是否存在
    if not os.path.exists(file_path + ".zip"):
        print(f">>> 警告: 模型文件 '{file_path}.zip' 不存在!")
        print(">>> 请确保超参数优化已完成，并且文件名正确。")
        return None

    # 1. 初始化一个“空”的模型对象
    recommender_instance = recommender_class(urm_train)

    # 2. 调用 .load_model() 方法加载数据
    recommender_instance.load_model(folder_path=modelfile_path, file_name=file_name)

    print("模型加载成功！\n")
    return recommender_instance

def safe_min_max_scale(scores):
    """
    一个健壮的 Min-Max 归一化函数。
    如果所有分数都相同，则返回一个全零数组。
    这个函数应该只处理不含 -inf 的、干净的分数数组。
    """
    # 确保输入是有限数值
    if not np.all(np.isfinite(scores)):
        # 如果包含 inf 或 nan，这是一个上游错误信号，我们返回全零
        return np.zeros_like(scores, dtype=np.float32)

    min_val, max_val = scores.min(), scores.max()
    denominator = max_val - min_val

    if denominator == 0:
        return np.zeros_like(scores, dtype=np.float32)
    else:
        return (scores - min_val) / denominator
#%% md
# ### 第一次分割
#%%
# 加载数据
urm_all = load_and_preprocess_data(DATA_TRAIN_PATH)

# 分割数据用于本地验证
print("\n--- 正在分割数据用于本地验证... ---")
URM_train, URM_validation = split_train_in_two_percentage_global_sample(
    urm_all,
    train_percentage=TRAIN_PERCENTAGE
)
print("数据分割完成.")
print(f"训练集维度: {URM_train.shape}, 验证集维度: {URM_validation.shape}\n")


# 初始化评估器
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[EVALUATION_CUTOFF])
print("评估器初始化完成.")
#%% md
# ### 第二次分割 k-fold
#%%
# 定义用于存放 K-Fold 模型的文件夹名称
FOLD_MODEL_FOLDER = "k_fold_models"
# 创建文件夹，如果已存在则不报错
os.makedirs(FOLD_MODEL_FOLDER, exist_ok=True)

print(f"所有在 K-Fold 中训练的模型将被保存在 '{FOLD_MODEL_FOLDER}/' 文件夹中。")

print("\n--- 阶段一：使用 K-Fold 生成 OOF 训练数据 (带模型保存) ---")

# 我们将在完整的 urm_train (80%数据) 上进行操作
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

meta_features = []
meta_labels = []
urm_train_coo = URM_train.tocoo()

for fold_num, (train_index, val_index) in enumerate(kf.split(urm_train_coo.row)):
    # 使用 f-string 格式化，方便阅读
    print(f"\n{'='*20} FOLD {fold_num + 1}/5 {'='*20}")

    urm_train_fold = sps.csr_matrix((urm_train_coo.data[train_index],
                                     (urm_train_coo.row[train_index], urm_train_coo.col[train_index])),
                                    shape=URM_train.shape)

    # --- 训练并保存基模型 ---
    print(f"--- [Fold {fold_num + 1}] 正在训练基模型 ---")

    # 训练 SLIM
    recommender_slim_fold = SLIMElasticNetRecommender(urm_train_fold)
    recommender_slim_fold.fit(**best_slim_params)
    # 关键一步：保存模型
    slim_filename = f"SLIM_fold_{fold_num + 1}"
    recommender_slim_fold.save_model(folder_path=FOLD_MODEL_FOLDER, file_name=slim_filename)
    print(f"SLIM (Fold {fold_num + 1}) 训练完成并已保存至 '{slim_filename}.zip'")

    # 训练 IALS
    recommender_ials_fold = IALSRecommender(urm_train_fold)
    recommender_ials_fold.fit(**best_ials_params)
    # 关键一步：保存模型
    ials_filename = f"IALS_fold_{fold_num + 1}"
    recommender_ials_fold.save_model(folder_path=FOLD_MODEL_FOLDER, file_name=ials_filename)
    print(f"IALS (Fold {fold_num + 1}) 训练完成并已保存至 '{ials_filename}.zip'")

    # 3. 为当前 fold 的验证集交互生成特征
    print("正在为验证集交互生成 OOF 特征...")
    val_users = np.unique(urm_train_coo.row[val_index])
    for user_id in tqdm(val_users, desc="生成OOF特征"):

        # a) 获取该用户在当前 fold 验证集中的正样本
        user_mask_in_val = (urm_train_coo.row[val_index] == user_id)
        positive_items = urm_train_coo.col[val_index][user_mask_in_val]

        # b) 为正样本进行负采样
        num_positive = len(positive_items)
        if num_positive == 0: continue

        user_seen_items = URM_train[user_id].indices
        unseen_items = np.setdiff1d(np.arange(URM_train.shape[1]), user_seen_items)
        negative_items = np.random.choice(unseen_items, size=num_positive, replace=False)

        items_to_score = np.concatenate([positive_items, negative_items])

        # c) 获取分数并进行特征工程
        slim_scores_full = recommender_slim_fold._compute_item_score(np.array([user_id]))[0]
        ials_scores_full = recommender_ials_fold._compute_item_score(np.array([user_id]))[0]

        slim_scores_raw = slim_scores_full[items_to_score]
        ials_scores_raw = ials_scores_full[items_to_score]

        slim_scores_norm = safe_min_max_scale(slim_scores_raw)
        ials_scores_norm = safe_min_max_scale(ials_scores_raw)

        for i in range(len(items_to_score)):
            f_slim = slim_scores_norm[i]
            f_ials = ials_scores_norm[i]
            meta_features.append([f_slim, f_ials, f_slim - f_ials, f_slim * f_ials])
            meta_labels.append(1 if i < num_positive else 0)

# --- 组合最终的元模型训练数据 ---
X_train_meta = pd.DataFrame(meta_features, columns=['slim_norm', 'ials_norm', 'score_diff', 'score_prod'])
y_train_meta = np.array(meta_labels)

print(f"\nOOF 特征生成完毕！元模型训练数据维度: {X_train_meta.shape}")
#%% md
# ### 训练元模型
#%%
print("\n--- 训练元模型 (LightGBM) ---")

lgbm = lgb.LGBMClassifier(objective='binary',
                          n_estimators=1000,
                          random_state=RANDOM_SEED)

# 使用早停法来找到最佳迭代次数
lgbm.fit(X_train_meta, y_train_meta,
         eval_set=[(X_train_meta, y_train_meta)],
         eval_metric='auc',
         callbacks=[lgb.early_stopping(50, verbose=False)])

print("元模型训练完成！")
#%% md
# ### 本地验证
#%%
# 定义模型所在的文件夹
COMBINE_MODEL_FOLDER = "temp_output"

print("--- 正在加载本地验证的模型... ---")

# 加载 SLIMElasticNetRecommender
recommender_slim_local_full = load_best_model(
    recommender_class=SLIMElasticNetRecommender,
    urm_train=URM_train, # 使用 URM_train 初始化
    file_name="SLIMElasticNetRecommender_best_model",
    modelfile_path=COMBINE_MODEL_FOLDER
)

# 加载 IALSRecommender
recommender_ials_local_full = load_best_model(
    recommender_class=IALSRecommender,
    urm_train=URM_train, # 使用 URM_train 初始化
    file_name="IALSRecommender_best_model",
    modelfile_path=COMBINE_MODEL_FOLDER
)
#%%
print("\n--- 在本地验证集上评估 Stacking 模型 ---")
print("用于评估的基模型已准备就绪。")

# 开始评估
cumulative_recall = 0
num_eval_users = 0
users_to_evaluate = np.unique(URM_validation.tocoo().row)

for user_id in tqdm(users_to_evaluate, desc="正在本地评估"):

    true_items = URM_validation[user_id].indices
    if len(true_items) == 0:
        continue

    num_eval_users += 1

    # a. 生成候选物品 (不变)
    slim_recs = recommender_slim_local_full.recommend(user_id, cutoff=100)
    ials_recs = recommender_ials_local_full.recommend(user_id, cutoff=100)
    candidate_items = np.union1d(slim_recs, ials_recs)

    if len(candidate_items) == 0:
        continue

    # b. 获取原始分数并正确提取
    slim_scores_full = recommender_slim_local_full._compute_item_score(np.array([user_id]))[0]
    ials_scores_full = recommender_ials_local_full._compute_item_score(np.array([user_id]))[0]

    slim_scores_raw = slim_scores_full[candidate_items]
    ials_scores_raw = ials_scores_full[candidate_items]

    # c. 进行与训练时完全相同的特征工程
    slim_scores_norm = safe_min_max_scale(slim_scores_raw)
    ials_scores_norm = safe_min_max_scale(ials_scores_raw)

    # d. 创建与训练时完全相同的特征 DataFrame
    #    确保列名 ('slim_norm', 'ials_norm', 'score_diff', 'score_prod')
    X_test_meta = pd.DataFrame({
        'slim_norm': slim_scores_norm,
        'ials_norm': ials_scores_norm,
        'score_diff': slim_scores_norm - ials_scores_norm,
        'score_prod': slim_scores_norm * ials_scores_norm
    })

    # e. 使用元模型得到最终分数
    final_scores = lgbm.predict_proba(X_test_meta)[:, 1]

    # f. 排序并推荐
    top_local_indices = np.argsort(final_scores)[::-1][:EVALUATION_CUTOFF]
    recommended_items = candidate_items[top_local_indices]

    # g. 计算 Recall
    hits = len(set(recommended_items) & set(true_items))
    recall_at_20 = hits / len(true_items)
    cumulative_recall += recall_at_20

# 计算并报告最终的本地验证分数
final_avg_recall = cumulative_recall / num_eval_users if num_eval_users > 0 else 0

print("\n--- 本地评估完成 ---")
print(f"Stacking 模型在本地验证集上的 Recall@20: {final_avg_recall:.5f}")
#%% md
# ### 最终预测与提交
#%%
print("\n--- 最终预测与提交 ---")

# 1. 在完整的 `urm_all` 数据上重新训练基模型，以获得最强的预测性能
print("在全部 `urm_all` 数据上训练最终的基模型...")

# 加载 SLIMElasticNetRecommender
recommender_slim_full = load_best_model(
    recommender_class=SLIMElasticNetRecommender,
    urm_train=urm_all, # 使用 urm_all 初始化
    file_name="5-1final_model_SLIMElasticNetRecommender-better",
    modelfile_path=MODEL_FOLDER
)

# 加载 IALSRecommender
recommender_ials_full = load_best_model(
    recommender_class=IALSRecommender,
    urm_train=urm_all, # 使用 urm_all 初始化
    file_name="5-2final_model_IALSRecommender",
    modelfile_path=MODEL_FOLDER
)

print("最终的基模型已准备就绪。")

# 2. 读取目标用户
df_target_users = pd.read_csv(DATA_TARGET_USERS_PATH)
target_user_ids = df_target_users['user_id'].values

submission = []

# 3. 为每个目标用户生成推荐
for user_id in tqdm(target_user_ids, desc="生成最终推荐"):

    # a. 生成候选物品集
    slim_recs = recommender_slim_full.recommend(user_id, cutoff=100)
    ials_recs = recommender_ials_full.recommend(user_id, cutoff=100)
    candidate_items = np.union1d(slim_recs, ials_recs)

    if len(candidate_items) == 0:
        submission.append((user_id, ''))
        continue

    # b. 获取原始分数并正确提取
    slim_scores_full = recommender_slim_full._compute_item_score(np.array([user_id]))[0]
    ials_scores_full = recommender_ials_full._compute_item_score(np.array([user_id]))[0]

    slim_scores_raw = slim_scores_full[candidate_items]
    ials_scores_raw = ials_scores_full[candidate_items]

    # c. 进行与训练时完全相同的特征工程
    slim_scores_norm = safe_min_max_scale(slim_scores_raw)
    ials_scores_norm = safe_min_max_scale(ials_scores_raw)

    # d. 创建与训练时完全相同的特征 DataFrame
    #    确保列名与训练 lgbm 时使用的列名完全一致！
    X_test_meta = pd.DataFrame({
        'slim_norm': slim_scores_norm,
        'ials_norm': ials_scores_norm,
        'score_diff': slim_scores_norm - ials_scores_norm,
        'score_prod': slim_scores_norm * ials_scores_norm
    })

    # e. 使用元模型得到最终分数
    final_scores = lgbm.predict_proba(X_test_meta)[:, 1]

    # f. 排序并推荐
    top_local_indices = np.argsort(final_scores)[::-1][:EVALUATION_CUTOFF]
    recommended_items = candidate_items[top_local_indices]

    submission.append((user_id, ' '.join(map(str, recommended_items))))

# 4. 保存提交文件
df_submission = pd.DataFrame(submission, columns=['user_id', 'item_list'])
df_submission.to_csv("submission_stacking_final.csv", index=False)

print("\n提交文件 'submission_stacking_final.csv' 已生成！")