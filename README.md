# 篮球比赛预测模型

基于扩散模型的篮球比赛结果预测系统，使用 Transformer 编码器-解码器架构和深度扩散网络进行二分类预测。

## 📋 项目概述

该项目构建了一个高级的机器学习模型，用于预测篮球比赛的结果。模型采用以下创新技术：

- **扩散模型架构**：使用基于扩散的条件生成模型进行预测
- **Transformer 编码器-解码器**：用于特征提取和决策生成
- **多头自注意力机制**：捕捉复杂的时间序列依赖关系
- **8层残差网络**：增强模型的表达能力
- **余弦退火 Beta 调度**：改进的噪声衰减策略
- **10折交叉验证**：全面评估模型泛化能力

## 🏗️ 项目结构

```
basketPrediction/
├── model.py                 # 扩散模型定义和架构
├── preprocess.py            # 数据预处理和加载
├── utils.py                 # 训练、评估等工具函数
├── run.py                   # 主程序入口
├── requirements.txt         # Python 依赖配置
├── setup_and_run.sh         # Conda 环境自动化部署脚本 (Linux/Mac)
├── docker/
│   ├── Dockerfile           # Docker 镜像构建配置
│   ├── Dockerfile.prod      # 生产优化版本 (多阶段构建)
│   ├── docker-compose.yml   # Docker Compose 编排配置
│   ├── build_and_run.sh     # Docker 自动化脚本
│   └── DOCKER_README.md     # Docker 使用指南
├── data/                    # 数据文件目录
│   └── basket.csv           # 篮球比赛数据
├── results/                 # 输出结果目录
│   └── predictions.csv      # 模型预测结果
└── README.md                # 项目说明文档
```

## 🔧 核心模块说明

### model.py - 模型定义

**关键组件：**

| 类名 | 说明 | 功能 |
|------|------|------|
| `SinusoidalEmbedding` | 正弦位置编码 | 对时间步进行编码 |
| `AttentionBlock` | 多头自注意力 | 8头自注意力机制，捕捉特征间关系 |
| `ResidualBlock` | 残差块 | 结合注意力和 MLP 的残差连接 |
| `DiffusionPredictor` | 扩散预测器 | 完整的扩散模型，包含编码器、去噪网络、解码器 |

**模型架构参数：**
- 隐藏维度：4096
- 去噪层数：8
- 注意力头数：8
- MLP 中间层倍数：8
- Transformer 编码层：2
- Transformer 解码层：2
- 时间步数：1000
- Dropout 比例：0.15

### preprocess.py - 数据预处理

**核心函数：**
- `load_and_preprocess_data()`：加载 CSV 数据、特征标准化、生成时间序列
- `stratified_group_kfold_split()`：按星期进行分层群组 K 折交叉验证

### utils.py - 训练工具

**核心函数：**
- `train_epoch()`：单个 epoch 的训练循环
- `evaluate()`：模型评估（准确率、精确率、召回率、F1 分数）
- `train_fold()`：单个折的完整训练流程

### run.py - 主程序

**工作流程：**
1. 加载并预处理数据
2. 执行 10 折分层群组交叉验证
3. 训练模型（100 个 epoch，早期停止）
4. 汇总交叉验证结果
5. 生成预测概率报告
6. 保存结果到 CSV 文件

## 🚀 快速开始

### 方式1：Conda 虚拟环境（推荐）

**Linux/Mac 用户：**
```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

**Windows 用户：**
```powershell
.\setup_and_run.ps1
```

脚本会自动：
- 创建 conda 虚拟环境 (`basket_env`)
- 安装所有依赖库（使用清华 PyPI 镜像加速）
- 显示环境信息
- 运行模型训练

### 方式2：手动 Conda 环境

```bash
# 创建虚拟环境
conda create -n basket_env python=3.10 -y

# 激活环境
conda activate basket_env

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行程序
python run.py
```

### 方式3：Docker 容器化部署

```bash
# 构建镜像
docker-compose build

# 运行容器（后台）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止容器
docker-compose down
```

详见 `docker/DOCKER_README.md`

### 方式4：直接 pip 安装（非隔离环境）

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python run.py
```

## 📦 系统依赖

### Python 版本
- Python 3.9 或更高版本（推荐 3.10）

### 主要依赖库

| 库名 | 版本 | 用途 |
|------|------|------|
| torch | >=2.0.0 | 深度学习框架 |
| torchvision | >=0.15.0 | 计算机视觉工具库 |
| torchaudio | >=2.0.0 | 音频处理库 |
| pandas | >=1.5.0 | 数据处理和分析 |
| numpy | >=1.23.0 | 数值计算 |
| scikit-learn | >=1.2.0 | 机器学习工具 |
| matplotlib | >=3.7.0 | 数据可视化 |
| seaborn | >=0.12.0 | 高级可视化 |
| tqdm | >=4.65.0 | 进度条工具 |

### 可选：GPU 加速

- **NVIDIA GPU**：自动检测 CUDA，优先使用 GPU
- **CUDA**：11.8 或 12.1
- **cuDNN**：8.0 或以上

## 💾 数据格式

### 输入数据 (data/basket.csv)

CSV 格式，包含以下列：
- 日期
- 比赛结果（0 或 1）
- 其他相关特征

### 输出结果 (results/predictions.csv)

| 列名 | 说明 |
|------|------|
| Date | 比赛日期 |
| True_Label | 实际结果 |
| Prediction | 预测结果 |
| Probability | 预测概率 |
| WeekDay | 星期几 |

## 📊 模型训练

### 训练参数

```python
# 数据配置
lookback = 7              # 使用前 7 天数据
n_splits = 10             # 10 折交叉验证

# 模型配置
hidden_dim = 4096         # 隐藏层维度
num_timesteps = 1000      # 扩散时间步

# 训练配置
batch_size = 8            # 批大小
learning_rate = 0.0002    # 初始学习率
epochs = 100              # 每折最多 100 个 epoch
patience = 10             # 早期停止耐心值
```

### 优化器和调度

- **优化器**：AdamW（权重衰减：1e-5）
- **学习率调度**：余弦退火（CosineAnnealingLR）
- **最小学习率**：1e-7

### 显存优化

- Batch size：8（相比默认 32 减少 75% 显存占用）
- 定期清理：每 20 个 epoch 执行一次 `torch.cuda.empty_cache()`
- 折间释放：每个折完成后释放模型和优化器

## 🎯 性能指标

模型通过 10 折交叉验证评估，报告以下指标：

- **准确率 (Accuracy)**：正确预测的比例
- **精确率 (Precision)**：预测为正样本中实际为正的比例
- **召回率 (Recall)**：实际正样本中被正确预测的比例
- **F1 分数**：精确率和召回率的调和平均

## 🔍 故障排除

### 问题1：CUDA 显存不足

**解决方案：**
- 减少 `batch_size`
- 使用 CPU 模式（自动降级）
- 清理 GPU 显存：`torch.cuda.empty_cache()`

### 问题2：依赖安装失败

**解决方案：**
```bash
# 使用清华 PyPI 镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 问题3：环境激活失败

**确保 conda 已正确安装：**
```bash
conda --version
conda env list
```

### 问题4：GPU 未被识别

**检查 GPU 可用性：**
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 📈 训练流程详解

```
数据加载和预处理
    ↓
10 折分层群组交叉验证
    ├─ 第 1 折
    │   ├─ 数据划分 (训练/测试)
    │   ├─ 模型初始化
    │   ├─ 循环训练 (最多 100 个 epoch)
    │   │   ├─ 前向传播
    │   │   ├─ 反向传播
    │   │   ├─ 验证评估
    │   │   └─ 早期停止检查
    │   └─ 显存释放
    ├─ 第 2-10 折
    │   └─ 重复流程...
    ↓
结果汇总和统计
    ├─ 平均准确率、精确率、召回率、F1
    └─ 标准差统计
    ↓
生成预测报告
    └─ 保存为 CSV 格式
```

## 🔄 版本控制

项目使用 Git 管理版本，所有代码、配置和文档已上传到 GitHub：

```bash
git clone https://github.com/Seal367/basketPredict.git
cd basketPredict
```

## 📝 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交问题和改进建议！

## ✉️ 联系方式

如有问题，请通过 GitHub Issues 联系。

---

**最后更新**：2025-11-06  
**项目状态**：开发中 🚧
