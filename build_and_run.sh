#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}篮球比赛预测 Docker 部署脚本${NC}"
echo -e "${BLUE}================================${NC}\n"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误：Docker 未安装或不在 PATH 中${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker 已安装${NC}\n"

# 检查 docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠ docker-compose 未找到，使用 docker compose 命令${NC}"
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# 选择操作
echo -e "${YELLOW}请选择操作:${NC}"
echo "1) 构建镜像"
echo "2) 运行容器"
echo "3) 构建并运行 (推荐)"
echo "4) 停止容器"
echo "5) 查看日志"
echo "6) 删除镜像和容器"
read -p "输入选项 (1-6): " choice

case $choice in
    1)
        echo -e "\n${BLUE}[步骤1] 构建 Docker 镜像...${NC}"
        $COMPOSE_CMD build --no-cache
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ 镜像构建成功${NC}"
        else
            echo -e "${RED}✗ 镜像构建失败${NC}"
            exit 1
        fi
        ;;
    2)
        echo -e "\n${BLUE}[步骤2] 启动容器...${NC}"
        $COMPOSE_CMD up -d
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ 容器启动成功${NC}"
            echo -e "${BLUE}查看日志: $COMPOSE_CMD logs -f${NC}"
        else
            echo -e "${RED}✗ 容器启动失败${NC}"
            exit 1
        fi
        ;;
    3)
        echo -e "\n${BLUE}[步骤1] 构建 Docker 镜像...${NC}"
        $COMPOSE_CMD build --no-cache
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ 镜像构建失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ 镜像构建成功${NC}\n"
        
        echo -e "${BLUE}[步骤2] 启动容器...${NC}"
        $COMPOSE_CMD up
        ;;
    4)
        echo -e "\n${BLUE}停止容器...${NC}"
        $COMPOSE_CMD down
        echo -e "${GREEN}✓ 容器已停止${NC}"
        ;;
    5)
        echo -e "\n${BLUE}查看实时日志 (按 Ctrl+C 退出):${NC}"
        $COMPOSE_CMD logs -f
        ;;
    6)
        echo -e "\n${YELLOW}确认删除所有镜像和容器? (y/n):${NC}"
        read -p "" confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo -e "${BLUE}删除中...${NC}"
            $COMPOSE_CMD down -v
            docker rmi basketball-prediction:latest
            echo -e "${GREEN}✓ 删除完成${NC}"
        else
            echo -e "${YELLOW}已取消${NC}"
        fi
        ;;
    *)
        echo -e "${RED}无效的选项${NC}"
        exit 1
        ;;
esac

echo -e "\n${BLUE}================================${NC}"
echo -e "${BLUE}操作完成${NC}"
echo -e "${BLUE}================================${NC}"
