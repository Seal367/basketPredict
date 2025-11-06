# Docker 部署指南

## 快速开始

### 前置条件
- Docker 已安装
- Docker GPU 支持 (nvidia-docker 或 Docker Desktop with GPU support)
- 项目文件已准备好

### Linux/Mac 用户

#### 方式1：使用自动化脚本（推荐）
```bash
chmod +x build_and_run.sh
./build_and_run.sh
```
然后选择选项 `3) 构建并运行`

#### 方式2：使用 docker-compose（推荐）
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

#### 方式3：使用 Docker 命令
```bash
# 构建镜像
docker build -t basketball-prediction:latest .

# 运行容器
docker run -it \
  --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results:rw \
  -e CUDA_VISIBLE_DEVICES=0 \
  basketball-prediction:latest
```


## 文件说明

| 文件 | 说明 |
|------|------|
| `Dockerfile` | Docker 镜像构建配置文件 |
| `docker-compose.yml` | Docker Compose 编排配置 |
| `.dockerignore` | Docker 构建时忽略的文件 |
| `build_and_run.sh` | Linux/Mac 自动化部署脚本 |
| `build_and_run.ps1` | Windows PowerShell 部署脚本 |

## 常用命令

### 构建镜像
```bash
docker build -t basketball-prediction:latest .
```

### 运行容器
```bash
# 后台运行
docker run -d \
  --name basketball-pred \
  --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results:rw \
  basketball-prediction:latest

# 交互式运行
docker run -it \
  --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results:rw \
  basketball-prediction:latest
```

### 查看日志
```bash
docker logs -f basketball-pred
```

### 进入容器
```bash
docker exec -it basketball-pred /bin/bash
```

### 停止容器
```bash
docker stop basketball-pred
docker rm basketball-pred
```

### 删除镜像
```bash
docker rmi basketball-prediction:latest
```

## GPU 支持

### Linux 用户
需要安装 nvidia-docker：
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 数据卷挂载

容器会自动挂载以下目录：

| 容器路径 | 宿主机路径 | 权限 | 说明 |
|---------|----------|------|------|
| `/app/data` | `./data` | 只读 | 输入数据 |
| `/app/results` | `./results` | 读写 | 训练结果输出 |
| `/app/models` | `./models` | 读写 | 模型保存目录 |

## 环境变量

- `CUDA_VISIBLE_DEVICES`: GPU 设备ID (默认: 0)
- `PYTHONUNBUFFERED`: Python 输出不缓冲 (默认: 1)

## 故障排除

### 问题1：GPU 无法识别
```bash
# 检查 GPU
docker run --rm --gpus all ubuntu nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

### 问题2：权限错误
```bash
# Linux 用户需要将用户加入 docker 组
sudo usermod -aG docker $USER
newgrp docker
```

### 问题3：磁盘空间不足
```bash
# 清理未使用的镜像和容器
docker system prune -a
```

### 问题4：内存不足
编辑 docker-compose.yml，在 `services` 中添加：
```yaml
deploy:
  resources:
    limits:
      memory: 16G
```

## 性能优化建议

1. **减少镜像大小**：使用 `--no-cache` 选项重建
2. **使用多阶段构建**：详见 Advanced Dockerfile
3. **启用 BuildKit**：`DOCKER_BUILDKIT=1 docker build .`
4. **限制日志大小**：详见 docker-compose.yml logging 配置

## 推送到仓库

### 推送到 Docker Hub
```bash
docker tag basketball-prediction:latest username/basketball-prediction:latest
docker login
docker push username/basketball-prediction:latest
```

### 推送到私有仓库
```bash
docker tag basketball-prediction:latest registry.example.com/basketball-prediction:latest
docker push registry.example.com/basketball-prediction:latest
```

## 生产环境建议

1. 使用 Kubernetes 或 Docker Swarm 进行编排
2. 配置日志收集 (ELK, Splunk 等)
3. 实现健康检查和自动重启
4. 使用容器镜像扫描工具检测漏洞
5. 配置资源限制和监控告警

## 获取帮助

遇到问题？尝试以下步骤：

1. 检查 Docker 日志：`docker logs -f container_name`
2. 进入容器调试：`docker exec -it container_name bash`
3. 查看 Docker 文档：https://docs.docker.com/
4. 查看项目 GitHub Issues
