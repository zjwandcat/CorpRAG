# ============================================================================
# 金融研报智能体 API - Makefile 一键操作
# ============================================================================
# 使用方式：make <目标名>
# 示例：make build    → 构建镜像
#       make up       → 启动所有服务
#       make health   → 检查 API 健康状态
# ============================================================================

# 构建镜像（根据 Dockerfile 构建 FastAPI 应用镜像）
build:
	docker-compose build

# 启动所有服务（后台运行，-d 表示 detach 模式）
up:
	docker-compose up -d

# 停止并移除所有容器
down:
	docker-compose down

# 查看 API 服务实时日志（-f 表示 follow，持续输出）
logs:
	docker-compose logs -f api

# 查看所有容器运行状态
ps:
	docker-compose ps

# 检查 API 健康状态（向宿主机 8001 端口发送请求）
health:
	curl http://localhost:8001/health

# 发送测试对话请求（验证 API 是否正常响应）
chat:
	curl -X POST http://localhost:8001/api/v1/chat -H "Content-Type: application/json" -d '{"query": "你好"}'

# 彻底清理：停止容器 + 删除数据卷 + 清除本地缓存
clean:
	docker-compose down -v
	rm -rf chroma_db/__pycache__

# 重新构建并启动（--build 强制重新构建镜像）
rebuild:
	docker-compose up -d --build

# 进入 API 容器的交互式终端（调试用）
shell:
	docker-compose exec api bash

# 声明所有目标为伪目标（不对应实际文件，每次都执行）
.PHONY: build up down logs ps health chat clean rebuild shell