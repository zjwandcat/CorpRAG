# ============================================================================
# 金融研报智能体 API - Docker 镜像构建文件
# ============================================================================
# 基础镜像：使用 Python 3.11 slim 版本，体积小、安全性高
FROM python:3.11-slim

# 设置环境变量，优化 Python 运行行为
# PYTHONUNBUFFERED=1：禁用输出缓冲，确保日志实时输出（容器化必备）
# PYTHONDONTWRITEBYTECODE=1：不生成 .pyc 字节码文件，减少镜像体积
# PIP_NO_CACHE_DIR=1：pip 安装时不缓存，减少镜像层大小
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 安装系统依赖
# build-essential：C/C++ 编译工具链，部分 Python 包（如 chromadb）需要编译
# curl：用于容器内健康检查
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 先复制依赖文件并安装（利用 Docker 层缓存机制）
# 只要 requirements.txt 不变，这一层就会被缓存，大幅加速后续构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目全部代码到容器内
COPY . .

# 暴露 FastAPI 服务端口（本项目运行在 8001 端口）
EXPOSE 8001

# 启动命令：使用 uvicorn 运行 FastAPI 应用
# --host 0.0.0.0：监听所有网络接口（容器内必须，否则外部无法访问）
# --port 8001：指定服务端口
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]