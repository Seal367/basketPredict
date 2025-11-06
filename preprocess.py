# ==================== 数据加载与预处理 ====================
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def load_and_preprocess_data(csv_path, lookback=7):
    """
    加载并预处理数据
    Args:
        csv_path: CSV文件路径
        lookback: 历史回溯天数
    """
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"\n数据统计:")
    print(f"总样本数: {len(df)}")
    print(f"中奖(1): {(df['result'] == 1).sum()}")
    print(f"未中奖(0): {(df['result'] == 0).sum()}")
    print(f"无比赛(2): {(df['result'] == 2).sum()}")
    print(f"日期范围: {df['Date'].min()} 到 {df['Date'].max()}")

    # 特征工程：创建时间序列数据
    X = []
    y = []
    dates = []
    weekdays = []

    for i in range(lookback, len(df)):
        # 获取历史lookback天的结果和星期信息
        hist_results = df['result'].iloc[i - lookback:i].values.astype(np.float32)
        hist_weekdays = df['WeekDay'].iloc[i - lookback:i].values.astype(np.float32) / 7.0  # 归一化

        # 组合成特征
        features = np.stack([hist_results, hist_weekdays], axis=0)  # shape: (2, lookback)
        X.append(features)

        # 目标：下一天是否中奖 (只看0和1，忽略2)
        next_result = df['result'].iloc[i]
        if next_result == 2:
            # 如果下一天无比赛，跳过这个样本
            continue
        y.append(1 if next_result == 1 else 0)

        dates.append(df['Date'].iloc[i])
        weekdays.append(df['WeekDay'].iloc[i])

    X = np.array(X)
    y = np.array(y)
    dates = np.array(dates)
    weekdays = np.array(weekdays)

    print(f"\n样本处理后: {len(X)}")
    print(f"正样本(中奖): {(y == 1).sum()}")
    print(f"负样本(未中): {(y == 0).sum()}")

    return X, y, dates, weekdays

def stratified_group_kfold_split(y, groups, n_splits=5, random_state=42):
    """
    按星期进行分组的分层K折交叉验证
    确保每折中星期分布一致
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    folds = list(sgkf.split(np.zeros_like(y), y, groups))
    return folds
