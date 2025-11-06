import torch

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings

from preprocess import load_and_preprocess_data, stratified_group_kfold_split
from utils import train_fold, train_epoch,  evaluate

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

warnings.filterwarnings('ignore')

# PyTorch GPU 显存优化
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 设备配置 - 优先使用 CUDA，GPU 不可用时降级到 CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"✓ CUDA 可用，使用 GPU 进行训练")
    print(f"  GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA 版本: {torch.version.cuda}")
else:
    DEVICE = torch.device('cpu')
    print(f"✗ CUDA 不可用，降级使用 CPU 进行训练")

print(f"\n最终使用设备: {DEVICE}")

def main():
    # 配置参数
    csv_path = './data/basket.csv'
    lookback = 7  # 使用前7天数据
    n_splits = 10  # 5折交叉验证
    
    print("="*80)
    print("篮球比赛预测模型 - 基于扩散模型")
    print("="*80)
    
    # 1. 加载和预处理数据
    print("\n[步骤1] 加载数据...")
    X, y, dates, weekdays = load_and_preprocess_data(csv_path, lookback=lookback)
    
    print(f"\n输入特征形状: {X.shape}")
    print(f"输出标签形状: {y.shape}")
    print(f"特征说明:")
    print(f"  - 维度0: 比赛结果 (0/1)")
    print(f"  - 维度1: 星期几 (归一化)")
    print(f"  - 序列长度: {lookback}")
    
    # 2. 按星期进行5折交叉验证
    print(f"\n[步骤2] 执行{n_splits}折分层群组交叉验证...")
    print(f"按星期进行分组，确保每折星期分布一致...")
    
    folds = stratified_group_kfold_split(y, weekdays, n_splits=n_splits, random_state=SEED)
    
    all_results = []
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_indices_map = {}  # 记录fold_idx与all_results中位置的映射
    
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- 第{fold_idx+1}折 ---")
        
        # 划分训练和测试集
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"训练集: {len(X_train)} 样本 (正样本: {(y_train==1).sum()}, 负样本: {(y_train==0).sum()})")
        print(f"测试集: {len(X_test)} 样本 (正样本: {(y_test==1).sum()}, 负样本: {(y_test==0).sum()})")
        
        # 训练模型
        if len(X_test) > 0:  # 只有当测试集非空时才训练
            fold_result = train_fold(X_train, y_train, X_test, y_test, fold_idx, DEVICE)
            
            fold_indices_map[fold_idx] = len(all_results)  # 记录映射关系
            all_results.append(fold_result)
            fold_accuracies.append(fold_result['accuracy'])
            fold_precisions.append(fold_result['precision'])
            fold_recalls.append(fold_result['recall'])
            fold_f1s.append(fold_result['f1'])
        else:
            print(f"  跳过此折：测试集为空")
            continue
    
    # 3. 汇总结果
    print(f"\n\n[步骤3] 交叉验证汇总结果")
    print("="*80)
    print(f"\n5折交叉验证结果:")
    print(f"  平均准确率: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"  平均精确率: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
    print(f"  平均召回率: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
    print(f"  平均F1分数: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    
    # 4. 输出每个样本的预测概率
    print(f"\n[步骤4] 生成预测概率报告")
    print("="*80)
    
    # 汇总所有预测
    all_indices = []
    all_probs = []
    all_preds = []
    all_labels = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        # 只处理未被跳过的折
        if fold_idx in fold_indices_map:
            result_idx = fold_indices_map[fold_idx]
            fold_result = all_results[result_idx]
            all_indices.extend(test_idx)
            all_probs.extend(fold_result['probs'])
            all_preds.extend(fold_result['preds'])
            all_labels.extend(fold_result['labels'])
    
    # 检查是否有有效结果
    if len(all_probs) > 0:
        sorted_idx = np.argsort(all_indices)
        all_indices = np.array(all_indices)[sorted_idx]
        all_probs = np.array(all_probs)[sorted_idx]
        all_preds = np.array(all_preds)[sorted_idx]
        all_labels = np.array(all_labels)[sorted_idx]
        
        # 创建结果数据框
        results_df = pd.DataFrame({
            'Date': dates[all_indices],
            'True_Label': all_labels,
            'Prediction': all_preds.astype(int),
            'Probability': all_probs,
            'WeekDay': weekdays[all_indices]
        })
        
        # 保存预测结果
        output_path = './results/predictions.csv'
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已保存到: {output_path}")
        
        # 显示样本预测
        print(f"\n预测样本统计 (前20个样本):")
        print(results_df.head(20).to_string(index=False))
        

    else:
        print(f"\n没有有效的预测结果（所有折的测试集都为空）")


if __name__ == '__main__':
    main()
