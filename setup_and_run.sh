#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}篮球比赛预测 - Conda 环境一键部署${NC}"
echo -e "${BLUE}================================${NC}\n"

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ 错误：Conda 未安装或不在 PATH 中${NC}"
    echo -e "${YELLOW}请访问 https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html 安装 Conda${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Conda 已安装${NC}\n"

# 项目配置
PROJECT_NAME="basketPrediction"
ENV_NAME="basket_env"
PYTHON_VERSION="3.10"

echo -e "${BLUE}[步骤1] 检查虚拟环境...${NC}"

# 检查环境是否已存在
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}⚠ 环境 '$ENV_NAME' 已存在${NC}"
    read -p "是否删除并重建？(y/n): " rebuild
    if [ "$rebuild" = "y" ] || [ "$rebuild" = "Y" ]; then
        echo -e "${BLUE}删除现有环境...${NC}"
        conda remove -n $ENV_NAME -y --all
    else
        echo -e "${YELLOW}使用现有环境${NC}"
    fi
fi

# 创建虚拟环境
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${BLUE}创建 Conda 虚拟环境: $ENV_NAME (Python $PYTHON_VERSION)...${NC}"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 虚拟环境创建失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 虚拟环境创建成功${NC}\n"
else
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}\n"
fi

echo -e "${BLUE}[步骤2] 激活虚拟环境并安装依赖...${NC}"

# 获取 conda 初始化脚本
eval "$(conda shell.bash hook)"

# 激活虚拟环境
conda activate $ENV_NAME

# 检查是否成功激活
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo -e "${RED}✗ 虚拟环境激活失败${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 虚拟环境已激活: $CONDA_DEFAULT_ENV${NC}\n"

# 检查 requirements.txt 是否存在
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗ 错误：requirements.txt 文件不存在${NC}"
    exit 1
fi

echo -e "${BLUE}安装依赖库...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ 依赖安装失败${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 依赖安装成功${NC}\n"

# 显示环境信息
echo -e "${BLUE}[步骤3] 环境信息${NC}"
echo -e "Python 版本: $(python --version)"
echo -e "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
echo -e "CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo '无法检查')\n"

# 询问是否立即运行
read -p "是否现在运行 run.py? (y/n): " run_now
if [ "$run_now" = "y" ] || [ "$run_now" = "Y" ]; then
    echo -e "\n${BLUE}[步骤4] 运行 run.py...${NC}\n"
    python run.py
else
    echo -e "\n${YELLOW}您可以手动激活环境后运行:${NC}"
    echo -e "  conda activate $ENV_NAME"
    echo -e "  python run.py"
fi

echo -e "\n${BLUE}================================${NC}"
echo -e "${BLUE}完成！${NC}"
echo -e "${BLUE}================================${NC}"
