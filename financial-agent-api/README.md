# 企业内部办公知识库智能体 API

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

企业内部办公知识库智能问答系统，基于 **FastAPI + LangChain + NVIDIA NIM + ChromaDB** 构建。支持多轮对话、混合检索增强生成、工具调用、文档管理，采用 Docker Compose 多容器编排部署。

## ✨ 核心特性

- 🤖 **智能对话**：基于 LangChain Agent 框架，支持多轮对话和上下文记忆
- � **混合检索引擎**：向量 + BM25 + RRF 融合排序，提升检索精准度
- � **多格式文档**：支持 PDF/TXT/DOCX/MD 文档上传、切片、向量化存储
- � **员工信息查询**：集成 OA 系统接口，快速查询员工联系方式
- 🌐 **联网搜索**：内库查无结果时自主联网搜索（DuckDuckGo）
- 📧 **数字员工动作**：模拟发送邮件通知，支持自动化任务执行
- 🏢 **部门级权限**：文档按部门分类，检索时支持部门过滤
- ⚡ **语义缓存**：高频问题直接返回缓存结果，提升响应速度
- 🛡️ **安全审计日志**：全量请求链路记录，满足企业合规要求
- 🐳 **容器化部署**：Docker Compose 一键部署，支持 GPU 加速
- 🔐 **类型安全**：mypy strict 模式，完整的类型标注
- 📝 **代码质量**：ruff linter + formatter，遵循 PEP8/PEP20 规范
- 📝 **PRD 自动撰写**：基于 LLM 自动生成结构化产品需求文档，支持导出为 Word
- 📊 **流程图生成**：将业务描述转化为 Mermaid 代码，无缝对接 Visio/Draw.io
- 🖥️ **低保真原型**：快速生成基于 HTML/Tailwind 的前端原型，提升设计效率

## 📋 目录

- [架构概览](#架构概览)
- [技术栈](#技术栈)
- [快速启动](#快速启动)
- [配置说明](#配置说明)
- [API 接口](#api-接口)
- [项目结构](#项目结构)
- [开发指南](#开发指南)
- [部署指南](#部署指南)
- [常见问题](#常见问题)

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Compose 编排                          │
│                     agent-network (bridge)                       │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │   api 服务        │  │   nim 服务        │  │ chromadb 服务   │ │
│  │   (FastAPI)      │  │   (NVIDIA NIM)   │  │  (ChromaDB)    │ │
│  │                  │  │                  │  │                │ │
│  │  宿主机:8001     │  │  宿主机:8002     │  │  宿主机:8003   │ │
│  │  容器:8001       │  │  容器:8000       │  │  容器:8000     │ │
│  │                  │  │                  │  │                │ │
│  │  ./data ──────►  │  │  nim-cache ────► │  │ chromadb-data► │ │
│  │  ./chroma_db ──► │  │  (命名卷)        │  │  (命名卷)      │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
│         │                      │                       │         │
│         └──── HTTP ────────────┘                       │         │
│         └──── HTTP ────────────────────────────────────┘         │
│                                                                  │
│  数据卷挂载：                                                    │
│  ├── ./data:/app/data           (知识库文档持久化)               │
│  ├── ./chroma_db:/app/chroma_db (向量数据库持久化)              │
│  ├── nim-cache:/opt/nim/.cache  (NIM 模型缓存)                 │
│  └── chromadb-data:/chroma/chroma (ChromaDB 数据)              │
└─────────────────────────────────────────────────────────────────┘
```

### 核心流程

```
用户提问 → FastAPI 路由 → Agent Chain → 工具选择 → 混合检索 → LLM 缓存 → LLM 推理 → 响应返回
                ↓              ↓           ↓           ↓           ↓
            限流保护      会话管理    工具调用    向量+BM25+RRF   审计日志
                                        ↓
                               ┌────────┼────────┬────────┬────────┐
                               ↓        ↓        ↓        ↓        ↓
                           内部文档  员工信息  联网搜索  邮件通知  产品经理效能
                                                                  ↓
                                                          ┌───────┼───────┐
                                                          ↓       ↓       ↓
                                                      PRD生成  流程图  HTML原型
```

## 🛠️ 技术栈

| 类别 | 技术 | 版本 | 说明 |
|------|------|------|------|
| **Web 框架** | FastAPI | 0.115+ | 高性能异步框架，自动生成 OpenAPI 文档 |
| **ASGI 服务器** | Uvicorn | 0.34+ | 生产级 ASGI 服务器，支持热重载 |
| **Agent 框架** | LangChain | 0.3+ | 工具调用、对话链、RAG 管道 |
| **LLM 推理** | NVIDIA NIM | - | 支持云端 API 和本地部署，GPU 加速 |
| **Embedding** | nv-embedqa-e5-v5 | - | 办公文档向量化模型 |
| **向量数据库** | ChromaDB | 0.6+ | 轻量级向量存储与相似度检索 |
| **混合检索** | BM25 + RRF | - | 关键字检索与向量检索融合排序 |
| **中文分词** | jieba | - | BM25 中文分词支持 |
| **联网搜索** | DuckDuckGo | - | 外部信息查询 |
| **文档解析** | python-docx + unstructured | - | DOCX/MD 格式支持 |
| **语义缓存** | InMemoryCache / SQLiteCache | - | LLM 响应缓存 |
| **限速中间件** | SlowAPI | 0.1.9+ | 基于 IP 的请求限速 |
| **文档加载** | PyPDF | 5.0+ | PDF 文件解析与文本提取 |
| **数据验证** | Pydantic | 2.10+ | 数据模型验证与序列化 |
| **容器编排** | Docker Compose | - | 多容器编排，一键部署 |
| **代码检查** | Ruff | 0.8+ | 快速 Python linter 和 formatter |
| **类型检查** | MyPy | 1.13+ | 静态类型检查，strict 模式 |

## 🚀 快速启动

### 方式一：Docker 部署（推荐）

#### 1. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 NGC API Key
# NGC_API_KEY=your_ngc_api_key_here
```

> **获取 NGC API Key**：访问 https://ngc.nvidia.com 注册并获取

#### 2. 构建并启动所有服务

```bash
make rebuild
```

#### 3. 检查服务状态

```bash
python scripts/check_services.py
```

预期输出：
```
[✅] FastAPI 应用 正常
[✅] NVIDIA NIM 正常
[✅] ChromaDB 正常
```

#### 4. 开始使用

```bash
# 方式 1：使用 Makefile 命令
make chat

# 方式 2：访问 Swagger UI
# 浏览器打开 http://localhost:8001/docs

# 方式 3：使用 curl
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "公司的报销流程是怎样的？"}'
```

### 方式二：本地开发部署

#### 1. 安装依赖

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 pyproject.toml
pip install -e .
```

#### 2. 配置 API Key

创建 `nim_config.txt` 文件，填入 NVIDIA API Key：

```bash
echo "your_nvidia_api_key" > nim_config.txt
```

#### 3. 启动服务

```bash
# 方式 1：使用 run.py（推荐，支持热重载）
python run.py

# 方式 2：使用 uvicorn 命令
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# 方式 3：使用批处理脚本（Windows）
start-local.bat
```

#### 4. 访问服务

浏览器打开 http://localhost:8001/docs 查看 Swagger UI

## ⚙️ 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM 服务地址（云端 API 或本地服务） |
| `NIM_MODEL_NAME` | `deepseek-ai/deepseek-v4-pro` | 大语言模型名称 |
| `NIM_EMBEDDING_MODEL` | `nvidia/nv-embedqa-e5-v5` | Embedding 模型名称 |
| `NVIDIA_API_KEY` | - | NVIDIA API Key（可从环境变量或 `nim_config.txt` 读取） |
| `CHROMA_DB_DIR` | `./chroma_db` | ChromaDB 数据目录 |
| `KNOWLEDGE_DIR` | `./data/knowledge_base` | 知识库文件目录 |
| `LLM_CACHE_TYPE` | `memory` | LLM 缓存类型（memory 或 sqlite） |
| `CHUNK_SIZE` | `500` | 文档切片大小（字符数） |
| `CHUNK_OVERLAP` | `50` | 切片重叠大小（字符数） |
| `TOP_K` | `3` | 检索返回的文档数量 |
| `SIMILARITY_THRESHOLD` | `1.3` | 相似度阈值（L2 距离） |
| `RATE_LIMIT_RPM` | `15` | 每分钟请求限制 |
| `RETRY_MAX_ATTEMPTS` | `5` | 最大重试次数 |

### 支持的模型

| 模型类型 | 模型名称 | 说明 |
|---------|---------|------|
| **LLM** | `deepseek-ai/deepseek-v4-pro` | DeepSeek V4 Pro（默认） |
| **LLM** | `moonshotai/kimi-k2.6` | Moonshot Kimi K2.6 |
| **Embedding** | `nvidia/nv-embedqa-e5-v5` | NVIDIA 办公 Embedding 模型（默认） |
| **Embedding** | `text-embedding-3-small` | OpenAI Embedding 模型 |

### 端口说明

| 宿主机端口 | 容器端口 | 服务 | 说明 |
|-----------|---------|------|------|
| 8001 | 8001 | FastAPI (api) | 应用 API 入口，Swagger UI |
| 8002 | 8000 | NVIDIA NIM (nim) | 大模型推理服务 |
| 8003 | 8000 | ChromaDB (chromadb) | 向量数据库 API |

> **提示**：如果本机已运行 NIM 服务，可在 `docker-compose.yml` 中注释掉 `nim` 服务，并将 `api` 服务的 `NIM_BASE_URL` 改为 `http://host.docker.internal:8000/v1`。

## 📡 API 接口

### 对话接口

#### POST `/api/v1/chat`

与 AI 助手对话（支持限流保护）

**请求体**：
```json
{
  "query": "公司的报销流程是怎样的？",
  "session_id": "user-123"
}
```

**响应体**：
```json
{
  "response": "根据内部文档检索结果...",
  "tools_used": ["search_internal_documents"],
  "session_id": "user-123"
}
```

#### DELETE `/api/v1/session/{session_id}`

清除指定会话的历史记录

**响应体**：
```json
{
  "message": "会话 user-123 已清除"
}
```

### 文档管理接口

#### POST `/api/v1/docs/upload`

上传 PDF/TXT/DOCX/MD 文档

**请求**：`multipart/form-data`
- `file`: 文件对象
- `department`: 所属部门（可选，用于部门级权限过滤）

**响应体**：
```json
{
  "message": "文档上传成功",
  "chunks": 42,
  "filename": "report.pdf"
}
```

#### GET `/api/v1/docs/count`

查询向量库中的文档数量

**响应体**：
```json
{
  "count": 128
}
```

#### DELETE `/api/v1/docs/clear`

清空向量库

**响应体**：
```json
{
  "message": "向量库已清空"
}
```


#### POST `/api/v1/docs/export/prd`

将 PRD 文档（Markdown 格式）导出为 Word (.docx) 格式

**请求体**：
```json
{
  "feature_name": "智能报销审批",
  "content": "# 需求背景\n..."
}
```

**响应**：返回 .docx 文件下载


#### GET `/health`

检查服务健康状态

**响应体**：
```json
{
  "status": "ok"
}
```

### 完整 API 文档

启动服务后访问：
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc
- OpenAPI JSON: http://localhost:8001/openapi.json

## 📁 项目结构

```
financial-agent-api/
├── app/                          # 应用核心代码
│   ├── __init__.py
│   ├── main.py                   # FastAPI 入口（CORS、限速、路由挂载）
│   ├── exceptions.py             # 自定义异常体系
│   ├── api/                      # 路由层
│   │   ├── __init__.py
│   │   ├── routes_chat.py        # 对话接口（含限速装饰器）
│   │   └── routes_docs.py        # 文档管理接口
│   ├── agent/                    # Agent 核心（员工查询、联网搜索、邮件通知）
│   │   ├── __init__.py
│   │   ├── chain.py              # 对话链与工具调度
│   │   └── tools.py              # @tool 工具定义
│   ├── core/                     # 核心配置
│   │   ├── __init__.py
│   │   ├── config.py             # 环境变量配置
│   │   ├── dependencies.py       # 依赖注入
│   │   ├── enums.py              # 枚举定义
│   │   ├── limiter.py            # 限流器实例
│   │   └── protocols.py          # 协议定义
│   ├── models/                   # 数据模型
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic 模型
│   └── rag/                      # RAG 检索（混合检索）
│       ├── __init__.py
│       ├── loader.py             # PDF/TXT/DOCX/MD 加载与切片
│       └── vectorstore.py        # ChromaDB 构建/检索
├── data/                         # 知识库文件目录
│   └── knowledge_base/           # PDF/TXT/DOCX/MD 知识库文档存放
├── chroma_db/                    # 向量数据库本地存储
├── scripts/                      # 工具脚本
│   └── check_services.py         # 服务健康检查
├── .dockerignore                 # Docker 构建排除
├── .env.example                  # 环境变量模板
├── .env                          # 环境变量（需自行创建）
├── docker-compose.yml            # 多容器编排配置
├── Dockerfile                    # 应用镜像构建
├── Makefile                      # 一键操作命令
├── pyproject.toml                # 项目配置（依赖、linter）
├── requirements.txt              # Python 依赖
├── run.py                        # 本地启动脚本
├── nim_config.txt                # NVIDIA API Key 配置文件
├── start-local.bat               # Windows 本地启动脚本
├── start-docker.bat              # Windows Docker 启动脚本
├── stop-docker.bat               # Windows Docker 停止脚本
└── README.md                     # 项目文档
```

### 核心模块说明

| 模块 | 职责 | 关键文件 |
|------|------|---------|
| **api** | HTTP 路由层，处理请求响应 | `routes_chat.py`, `routes_docs.py` |
| **agent** | Agent 逻辑，工具定义（文档检索/员工查询/联网搜索/邮件通知/PRD生成/流程图/原型）与调度 | `chain.py`, `tools.py` |
| **core** | 核心配置、依赖注入、枚举 | `config.py`, `dependencies.py` |
| **models** | Pydantic 数据模型 | `schemas.py` |
| **rag** | RAG 检索，文档加载、切片、向量化与混合检索 | `loader.py`, `vectorstore.py` |

## 👨‍💻 开发指南

### 代码规范

本项目遵循以下规范：
- **PEP 8**：代码风格
- **PEP 20**：Python 之禅
- **类型标注**：所有函数必须有完整类型标注（mypy strict 模式）
- **文档字符串**：所有公共函数必须有 docstring

### 代码质量工具

```bash
# 代码格式化
ruff format .

# 代码检查
ruff check .

# 类型检查
mypy app/

# 运行所有检查
ruff format . && ruff check . && mypy app/
```

### Makefile 命令速查

| 命令 | 说明 |
|------|------|
| `make build` | 构建 Docker 镜像 |
| `make up` | 启动所有服务（后台） |
| `make down` | 停止并移除所有容器 |
| `make logs` | 查看 API 服务实时日志 |
| `make ps` | 查看容器运行状态 |
| `make health` | 检查 API 健康状态 |
| `make chat` | 发送测试对话请求 |
| `make clean` | 彻底清理（删除容器+数据卷+缓存） |
| `make rebuild` | 重新构建并启动 |
| `make shell` | 进入 API 容器终端 |

### 添加新工具

1. 在 `app/agent/tools.py` 中定义新工具：

```python
@tool
def get_meeting_room(room_id: str) -> str:
    """查询会议室信息

    Args:
        room_id: 会议室编号

    Returns:
        会议室详细信息（位置、容量、设备等）
    """
    # 实现逻辑
    return result
```

2. 在 `app/agent/chain.py` 中注册工具：

```python
tools = [search_internal_documents, get_employee_info, get_meeting_room]
```

### 添加新异常

1. 在 `app/exceptions.py` 中定义新异常类：

```python
class MyNewError(AgentError):
    """新异常说明"""
    pass
```

2. 在路由中捕获并处理：

```python
except MyNewError as exc:
    raise HTTPException(status_code=400, detail=str(exc))
```

## 🚢 部署指南

### Docker 部署（生产环境推荐）

#### 1. 准备工作

```bash
# 克隆项目
git clone <repository-url>
cd financial-agent-api

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 NGC_API_KEY
```

#### 2. 构建和启动

```bash
# 构建镜像
make build

# 启动服务
make up

# 查看日志
make logs
```

#### 3. 验证部署

```bash
# 检查服务状态
make ps

# 健康检查
make health

# 测试对话
make chat
```

#### 4. 停止服务

```bash
# 停止服务
make down

# 完全清理（包括数据卷）
make clean
```

### 本地部署（开发环境）

#### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 2. 配置 API Key

```bash
# 方式 1：环境变量
export NVIDIA_API_KEY="your_api_key"

# 方式 2：配置文件
echo "your_api_key" > nim_config.txt
```

#### 3. 启动服务

```bash
python run.py
```

### GPU 支持

Docker Compose 配置已包含 GPU 支持：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

确保主机已安装：
- NVIDIA 驱动
- NVIDIA Container Toolkit

验证 GPU 可用：

```bash
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

## ❓ 常见问题

### Q1: 如何切换云端 API 和本地 NIM？

**A**: 修改环境变量 `NIM_BASE_URL`：

```bash
# 使用云端 API（默认）
NIM_BASE_URL=https://integrate.api.nvidia.com/v1

# 使用本地 NIM Docker
NIM_BASE_URL=http://nim:8000/v1

# 使用本机运行的 NIM
NIM_BASE_URL=http://localhost:8000/v1
```

### Q2: 如何更换大语言模型？

**A**: 修改环境变量 `NIM_MODEL_NAME`：

```bash
# 使用 DeepSeek V4 Pro
NIM_MODEL_NAME=deepseek-ai/deepseek-v4-pro

# 使用 Kimi K2.6
NIM_MODEL_NAME=moonshotai/kimi-k2.6
```

同时在 `docker-compose.yml` 中更新 NIM 镜像：

```yaml
nim:
  image: nvcr.io/nim/moonshotai/kimi-k2.6:latest
```

### Q3: 请求频率超限怎么办？

**A**: 系统会返回 HTTP 429 错误，响应头包含 `Retry-After`：

```bash
# 调整限速阈值
RATE_LIMIT_RPM=30  # 每分钟 30 次
```

### Q4: 如何添加新的知识库文档？

**A**: 三种方式：

1. **API 上传**：
```bash
curl -X POST http://localhost:8001/api/v1/docs/upload \
  -F "file=@report.pdf" \
  -F "department=财务部"
```

2. **文件系统**：直接放入 `data/knowledge_base/` 目录，重启服务自动加载

3. **Docker 挂载**：将宿主机目录挂载到容器：
```yaml
volumes:
  - ./my_docs:/app/data/knowledge_base
```

### Q5: ChromaDB 数据存储在哪里？

**A**: 
- **本地部署**：`./chroma_db/` 目录
- **Docker 部署**：
  - Bind Mount: `./chroma_db/` (宿主机)
  - Named Volume: `chromadb-data` (Docker 管理)

### Q6: 如何查看容器日志？

**A**:
```bash
# API 服务日志
make logs
# 或
docker-compose logs -f api

# NIM 服务日志
docker-compose logs -f nim

# ChromaDB 日志
docker-compose logs -f chromadb

# 所有服务日志
docker-compose logs -f
```

### Q7: 端口冲突怎么办？

**A**: 修改 `docker-compose.yml` 中的端口映射：

```yaml
services:
  api:
    ports:
      - "9001:8001"  # 改为其他端口
  nim:
    ports:
      - "9002:8000"
  chromadb:
    ports:
      - "9003:8000"
```

### Q8: 如何完全重置系统？

**A**:
```bash
# 停止并删除所有容器、数据卷、缓存
make clean

# 重新构建和启动
make rebuild
```

## 📊 性能优化建议

### 1. 混合检索优化

```bash
# 调整检索参数
TOP_K=5                    # 增加检索文档数
SIMILARITY_THRESHOLD=1.0   # 降低相似度阈值
```

### 2. 文档切片优化

```bash
# 调整切片参数
CHUNK_SIZE=1000      # 增大切片，减少碎片
CHUNK_OVERLAP=100    # 增加重叠，提高连续性
```

### 3. 限流配置优化

```bash
# 根据实际需求调整
RATE_LIMIT_RPM=30    # 提高限速阈值
RETRY_MAX_ATTEMPTS=3 # 减少重试次数
```

### 4. GPU 内存优化

在 `docker-compose.yml` 中调整：

```yaml
nim:
  shm_size: 16gb  # 减少共享内存（默认 32gb）
```

## 🔒 安全建议

1. **API Key 保护**：
   - 不要将 `nim_config.txt` 和 `.env` 提交到版本控制
   - 使用环境变量或密钥管理服务

2. **限流保护**：
   - 生产环境建议降低 `RATE_LIMIT_RPM`
   - 考虑添加用户认证和授权

3. **网络安全**：
   - 使用 HTTPS
   - 配置防火墙规则
   - 限制容器网络访问

4. **数据安全**：
   - 定期备份 `chroma_db/` 和 `data/`
   - 敏感文档考虑加密存储

## 📄 License

MIT License

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/my-feature`
3. 提交更改：`git commit -m 'Add my feature'`
4. 推送分支：`git push origin feature/my-feature`
5. 提交 Pull Request

## 📮 联系方式

- 项目主页：[GitHub Repository]
- 问题反馈：[GitHub Issues]
- 文档：[在线文档]

---

**Made with ❤️ by Enterprise Knowledge Agent Team**