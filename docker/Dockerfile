# 使用官方 PyTorch 镜像作为基础镜像（包含 CUDA 支持）
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tsinghua.edu.cn/simple

# 复制项目文件
COPY .. .

# 创建数据和结果目录
RUN mkdir -p /app/data /app/results

# 暴露端口（如果需要 Jupyter 等）
EXPOSE 8888

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 默认命令：运行训练脚本
CMD ["python", "run.py"]
