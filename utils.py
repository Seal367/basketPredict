# ==================== 训练与评估函数 ====================
import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

from model import DiffusionPredictor


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1).float()

        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend((outputs > 0.5).cpu().detach().numpy().flatten())
        all_labels.extend(y_batch.cpu().detach().numpy().flatten())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """
    评估模型
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1).float()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_preds.extend((outputs > 0.5).cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())

    # 处理空验证集的情况
    if len(val_loader) > 0:
        avg_loss = total_loss / len(val_loader)
    else:
        avg_loss = 0.0

    # 处理空标签的情况
    if len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    else:
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return avg_loss, accuracy, precision, recall, f1, all_probs, all_preds, all_labels


def train_fold(X_train, y_train, X_val, y_val, fold_idx, device):
    """
    训练单个折
    """
    # 转换为张量
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型 - 超大改進版本
    model = DiffusionPredictor(
        input_dim=2,
        seq_len=X_train.shape[2],
        hidden_dim=4096,  # 扩大到 4096
        num_timesteps=1000,  # 维持时间步
        dropout=0.15  # 改进 dropout 率以適应更大模型
    ).to(device)

    # 优化器和损失函数 - 適应超大模型
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-5)  # 最低化书不率以適应大模型
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-7  # 適应超大模型
    )

    # 训练
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    best_model_state = None

    print(f"\n  第{fold_idx + 1}折训练:")
    for epoch in range(100):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_recall, val_f1, val_probs, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                if 'best_model_state' in locals():
                    model.load_state_dict(best_model_state)
                break

    # 最终评估
    final_loss, final_acc, final_prec, final_recall, final_f1, final_probs, final_preds, final_labels = evaluate(
        model, val_loader, criterion, device
    )

    print(f"  第{fold_idx + 1}折最终结果:")
    print(f"    准确率: {final_acc:.4f}")
    print(f"    精确率: {final_prec:.4f}")
    print(f"    召回率: {final_recall:.4f}")
    print(f"    F1分数: {final_f1:.4f}")

    return {
        'model': model,
        'accuracy': final_acc,
        'precision': final_prec,
        'recall': final_recall,
        'f1': final_f1,
        'probs': final_probs,
        'preds': final_preds,
        'labels': final_labels
    }
