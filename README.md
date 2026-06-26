# 企业内部办公知识库智能体

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://python.langchain.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11+-purple.svg)](https://www.llamaindex.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

企业内部办公知识库智能问答系统，提供 **LangChain** 和 **LlamaIndex** 双版本实现。两者共享知识库文档和配置文件，可同时运行在不同端口，满足不同技术栈的选型需求。

## ✨ 核心特性

- 🤖 **双版本实现**：LangChain（Tool Calling）与 LlamaIndex（AgentWorkflow）并行，共享知识库
- 🔀 **多 Provider 支持**：NVIDIA NIM / 讯飞星辰 / 智谱AI 三大 LLM Provider，可在线切换
- 🧠 **LangGraph 状态机**：LangChain 版本已升级为 LangGraph 状态机拓扑，支持多轮对话记忆（MemorySaver）、条件路由
- 📡 **SSE 流式推送**：LangChain 版本支持 SSE（Server-Sent Events）实时状态推送，事件类型包括 stream_start、agent_start、tool_call、tool_result、retrieval_result、stream_end 等
- 🎯 **Reranker 精排**：支持 NVIDIA Reranker（nvidia/llama-nemotron-rerank-1b-v2）和智谱AI Reranker，不可用时自动降级
- 🔀 **双 RAG 引擎路由**：LangChain 版本支持方案 A（自建 RAG + Reranker）和方案 B（NVIDIA RAG Blueprint）自动路由与降级
- 🔍 **混合检索引擎**：向量 + BM25 + RRF 融合排序，提升检索精准度
- 📄 **多格式文档**：支持 PDF/TXT/DOCX/MD/CSV 文档上传、切片、向量化存储
- 👤 **员工信息查询**：集成 OA 系统接口，快速查询员工联系方式
- 🌐 **联网搜索**：内库查无结果时自主联网搜索（DuckDuckGo）
- 📧 **数字员工动作**：模拟发送邮件通知，支持自动化任务执行
- 📝 **PRD 自动撰写**：基于 LLM 自动生成结构化产品需求文档，支持导出为 Word
- 📊 **流程图生成**：将业务描述转化为 Mermaid 代码，无缝对接 Visio/Draw.io
- 🖥️ **低保真原型**：快速生成基于 HTML/Tailwind 的前端原型，提升设计效率
- 📈 **CSV 数据统计**：自然语言查询 CSV 数据（LlamaIndex 独有，PandasQueryEngine）
- ⚡ **语义缓存**：高频问题直接返回缓存结果，提升响应速度
- 🛡️ **安全审计日志**：全量请求链路记录，满足企业合规要求
- 🐳 **容器化部署**：Docker Compose 一键部署，支持 GPU 加速（LangChain 版本）
- 🔐 **类型安全**：mypy strict 模式，完整的类型标注
- 📝 **代码质量**：ruff linter + formatter，遵循 PEP8/PEP20 规范
- 🔄 **Embedding 降级机制**：讯飞/智谱 Provider 下自动降级到 HuggingFace 本地 Embedding（BAAI/bge-small-zh-v1.5），使用 hf-mirror.com 镜像

## 📋 目录

- [双版本对比](#双版本对比)
- [架构概览](#架构概览)
- [技术栈](#技术栈)
- [快速启动](#快速启动)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [API 接口](#api-接口)
- [核心工具列表](#核心工具列表)
- [入门示例](#入门示例)
- [开发指南](#开发指南)
- [部署指南](#部署指南)
- [安全建议](#安全建议)
- [常见问题](#常见问题)
- [GitHub 上架清理清单](#github-上架清理清单)

## 🔄 双版本对比

本项目提供 **LangChain** 和 **LlamaIndex** 两个独立版本，两者共享知识库文档和 API Key 配置，可同时运行在不同端口：

| 对比维度 | LangChain 版本 | LlamaIndex 版本 |
|---------|---------------|----------------|
| **目录** | `financial-agent-api/` | `llama-index-app/` |
| **端口** | 8001 | 8002 |
| **Agent 框架** | LangGraph 状态机（Tool Calling + 条件路由） | LlamaIndex AgentWorkflow (ReActAgent) |
| **RAG 实现** | ChromaDB + BM25 + RRF 手动融合 + Reranker 精排 | VectorStoreIndex + BM25 + QueryFusionRetriever (RRF) |
| **向量库目录** | `chroma_db/` | `chroma_db_li/`（与 LangChain 版隔离） |
| **知识库目录** | `data/knowledge_base/` | 共享 `financial-agent-api/data/knowledge_base/` |
| **LLM** | 多 Provider（NVIDIA NIM / 讯飞星辰 / 智谱AI） | 多 Provider（NVIDIA NIM / 讯飞星辰 / 智谱AI） |
| **Embedding** | NVIDIAEmbeddings (nv-embedqa-e5-v5)，讯飞/智谱下降级 HuggingFace (BAAI/bge-small-zh-v1.5) | NVIDIAEmbedding (nv-embedqa-e5-v5)，回退 HuggingFace (BAAI/bge-small-zh-v1.5) |
| **Reranker 精排** | ✅ NVIDIA Reranker + 智谱AI Reranker，自动降级 | ❌ |
| **SSE 流式推送** | ✅ Server-Sent Events 实时状态推送 | ❌ |
| **LangGraph 状态机** | ✅ 多轮对话记忆 + 条件路由 | ❌ |
| **多 Provider 支持** | ✅ NIM / 讯飞 / 智谱 在线切换 | ✅ NIM / 讯飞 / 智谱 在线切换 |
| **双 RAG 引擎路由** | ✅ 方案 A（自建 RAG）+ 方案 B（Blueprint）自动路由与降级 | ❌ |
| **文档切片** | RecursiveCharacterTextSplitter | SentenceSplitter（中文分隔符优化） |
| **Docker 部署** | ✅ Docker Compose（3 容器） | ❌ 仅本地部署 |
| **CSV 数据查询** | ❌ | ✅ PandasQueryEngine |
| **NVIDIA Blueprints** | ✅ RAG Blueprint 适配层 | ✅ PRD 优化模板 + RRF 融合优化 |
| **PRD 导出 Word** | ✅ `/api/v1/docs/export/prd` | ✅ `/api/v1/docs/export/prd` |
| **Embedding 回退** | ✅ HuggingFace 本地回退（讯飞/智谱 Provider） | ✅ HuggingFace 本地回退 |
| **一键启动** | `start-langchain.bat` | `start-llamaindex.bat` |

### 如何选择？

- **选 LangChain 版本**：需要 Docker 生产部署、LangGraph 状态机、SSE 流式推送、Reranker 精排、双 RAG 引擎路由、审计日志、语义缓存
- **选 LlamaIndex 版本**：需要 CSV 数据统计查询、NVIDIA Blueprints 优化模板、HuggingFace 本地 Embedding 回退
- **两个都跑**：同时启动，分别监听 8001 和 8002 端口，共享同一份知识库

## 🏗️ 架构概览

### 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            项目根目录 (rag/)                              │
│                                                                          │
│  ┌──────────────────────────────┐    ┌──────────────────────────────┐   │
│  │    LangChain 版本 (:8001)     │    │    LlamaIndex 版本 (:8002)    │   │
│  │                              │    │                              │   │
│  │  ┌──────────────────────┐   │    │  ┌──────────────────────┐   │   │
│  │  │   FastAPI 路由层      │   │    │  │   FastAPI 路由层      │   │   │
│  │  │   (含 SSE 流式端点)   │   │    │  └──────────┬───────────┘   │   │
│  │  └────────┬─────────────┘   │    │             │               │   │
│  │           │                  │    │  ┌──────────▼───────────┐   │   │
│  │  ┌────────▼─────────────┐   │    │  │  AgentWorkflow        │   │   │
│  │  │  LangGraph 状态机     │   │    │  │  (ReActAgent)         │   │   │
│  │  │  (Tool Calling +     │   │    │  └──────────┬───────────┘   │   │
│  │  │   条件路由 + 记忆)    │   │    │             │               │   │
│  │  └────────┬─────────────┘   │    │  ┌──────────▼───────────┐   │   │
│  │           │                  │    │  │  VectorStoreIndex     │   │   │
│  │  ┌────────▼─────────────┐   │    │  │  + BM25 + RRF 融合    │   │   │
│  │  │  双 RAG 引擎路由      │   │    │  └──────────┬───────────┘   │   │
│  │  │  ┌────────┬────────┐ │   │    │             │               │   │
│  │  │  │方案A   │方案B   │ │   │    │  ┌──────────▼───────────┐   │   │
│  │  │  │自建RAG │Blueprint│ │   │    │  │  chroma_db_li/        │   │   │
│  │  │  │+Rerank │ RAG    │ │   │    │  │  (独立向量库)          │   │   │
│  │  │  └───┬────┴───┬────┘ │   │    │  └──────────────────────┘   │   │
│  │  └──────┼────────┼──────┘   │    │                              │   │
│  │         │        │          │    │                              │   │
│  │  ┌──────▼───┐    │          │    │                              │   │
│  │  │ Reranker │    │          │    │                              │   │
│  │  │ 精排     │    │          │    │                              │   │
│  │  └──────┬───┘    │          │    │                              │   │
│  │         │        │          │    │                              │   │
│  │  ┌──────▼────────▼──────────┐   │                              │   │
│  │  │  ChromaDB + BM25 + RRF   │   │                              │   │
│  │  │  + Reranker 精排          │   │                              │   │
│  │  └──────────┬───────────────┘   │                              │   │
│  │             │                    │                              │   │
│  │  ┌──────────▼───────────────┐   │                              │   │
│  │  │  chroma_db/              │   │                              │   │
│  │  │  (独立向量库)             │   │                              │   │
│  │  └──────────────────────────┘   │                              │   │
│  └──────────────┬───────────────────┘└──────────────┬───────────────┘   │
│                 │                                   │                   │
│                 └──────────┬────────────────────────┘                   │
│                            │                                            │
│                 ┌──────────▼──────────┐                                 │
│                 │  data/knowledge_base/ │  ← 共享知识库文档              │
│                 └──────────┬──────────┘                                 │
│                            │                                            │
│                 ┌──────────▼──────────┐                                 │
│                 │  nim_config.txt      │  ← 共享 API Key 配置           │
│                 └─────────────────────┘                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

### 多 Provider 路由

```
┌─────────────────────────────────────────────────────┐
│                  Provider 路由层                      │
│                                                       │
│   PROVIDER=nim          PROVIDER=xfyun    PROVIDER=zhipu  │
│   ┌─────────────┐      ┌──────────────┐  ┌──────────────┐ │
│   │ ChatNVIDIA  │      │ 讯飞星辰      │  │ 智谱AI       │ │
│   │ DeepSeek V4 │      │ xopqwen36v35b│  │ glm-4.7-flash│ │
│   └──────┬──────┘      └──────┬───────┘  └──────┬───────┘ │
│          │                    │                  │         │
│          │  NVIDIAEmbeddings  │  HuggingFace     │ HuggingFace│
│          │  (nv-embedqa-e5)  │  (bge-small-zh)  │ (bge-small-zh)│
│          │                    │  hf-mirror.com   │ hf-mirror.com│
│          └────────────────────┴──────────────────┘         │
│                         │                                   │
│                  统一 LLM 接口                               │
│                  统一 Embedding 接口                          │
└─────────────────────────────────────────────────────────────┘
```

### SSE 流式推送流程

```
┌────────┐     POST /api/v1/chat/stream     ┌──────────────┐
│  客户端  │ ──────────────────────────────► │  FastAPI      │
│        │                                   │              │
│        │  ◄── event: stream_start ─────── │  LangGraph   │
│        │  ◄── event: agent_start ──────── │  状态机执行   │
│        │  ◄── event: tool_call ────────── │              │
│        │  ◄── event: tool_result ──────── │  工具调度     │
│        │  ◄── event: retrieval_result ──  │  检索结果     │
│        │  ◄── event: stream_end ───────── │  完成         │
│        │                                   │              │
└────────┘                                   └──────────────┘

SSE 事件类型：
  stream_start     → 会话开始，包含 session_id
  agent_start      → Agent 开始推理
  tool_call        → 工具调用开始（含工具名、参数）
  tool_result      → 工具调用结果
  retrieval_result → 检索结果（含来源、分数）
  stream_end       → 流式结束，包含完整 answer
```

### Docker Compose 架构（仅 LangChain 版本）

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
用户提问 → FastAPI 路由 → LangGraph 状态机 → 条件路由 → 工具选择 → 混合检索 → Reranker 精排 → LLM 推理 → 响应返回
                ↓              ↓                ↓            ↓           ↓            ↓            ↓           ↓
            限流保护      会话管理/记忆     状态转移      工具调用    向量+BM25+RRF  相关性重排    审计日志    SSE 推送
                                               ↓
                                ┌────────┬──────┼──────┬────────┬────────┐
                                ↓        ↓      ↓      ↓        ↓        ↓
                            内部文档  员工信息  联网搜索  邮件通知  CSV查询  产品经理效能
                                                                               ↓
                                                                       ┌───────┼───────┐
                                                                       ↓       ↓       ↓
                                                                   PRD生成  流程图  HTML原型
```

## 🛠️ 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI 0.115+ | 高性能异步框架，自动生成 OpenAPI 文档 |
| **ASGI 服务器** | Uvicorn 0.34+ | 生产级 ASGI 服务器，支持热重载 |
| **Agent 框架** | LangChain 0.3+ / LlamaIndex 0.11+ | 双版本实现 |
| **状态机** | LangGraph | 状态机拓扑 + 多轮对话记忆（MemorySaver）+ 条件路由 |
| **LLM 推理** | NVIDIA NIM / 讯飞星辰 / 智谱AI | 多 Provider 支持，可在线切换 |
| **Embedding** | nv-embedqa-e5-v5 / BAAI/bge-small-zh-v1.5 | NVIDIA 云端 + HuggingFace 本地降级 |
| **Reranker** | NVIDIA Reranker / 智谱AI Reranker | 检索结果精排，不可用时自动降级 |
| **向量数据库** | ChromaDB 0.6+ | 轻量级向量存储与相似度检索 |
| **混合检索** | BM25 + RRF | 关键字检索与向量检索融合排序 |
| **流式推送** | SSE (Server-Sent Events) | 实时状态推送，支持多种事件类型 |
| **HTTP 客户端** | httpx | NVIDIA RAG Blueprint 通信 |
| **中文分词** | jieba | BM25 中文分词支持 |
| **联网搜索** | DuckDuckGo | 外部信息查询 |
| **文档解析** | PyPDF + python-docx + unstructured | 多格式支持 |
| **语义缓存** | InMemoryCache / SQLiteCache | LLM 响应缓存 |
| **限速中间件** | SlowAPI 0.1.9+ | 基于 IP 的请求限速 |
| **数据验证** | Pydantic 2.10+ | 数据模型验证与序列化 |
| **容器编排** | Docker Compose | 多容器编排，一键部署 |
| **代码检查** | Ruff 0.8+ | 快速 Python linter 和 formatter |
| **类型检查** | MyPy 1.13+ | 静态类型检查，strict 模式 |

## 🚀 快速启动

### 前置条件

- Python 3.10+
- NVIDIA API Key（[获取地址](https://build.nvidia.com)）或讯飞/智谱 API Key

### 方式一：一键启动（Windows，推荐新手）

```bash
# LangChain 版本（端口 8001）
start-langchain.bat

# LlamaIndex 版本（端口 8002）
start-llamaindex.bat
```

> 💡 脚本会自动安装依赖、配置环境并启动服务，首次启动后自动打开浏览器。

### 方式二：Docker 部署（LangChain 版本，推荐生产环境）

```bash
cd financial-agent-api
cp .env.example .env
# 编辑 .env 填入 NGC_API_KEY
make rebuild
```

### 方式三：本地开发

```bash
# LangChain 版本
cd financial-agent-api
pip install -r requirements.txt
python run.py

# LlamaIndex 版本
cd llama-index-app
pip install -r requirements.txt
python run.py
```

### 启动后访问

| 版本 | Swagger UI | 聊天界面 | 健康检查 |
|------|-----------|---------|---------|
| LangChain | http://localhost:8001/docs | http://localhost:8001/ | http://localhost:8001/health |
| LlamaIndex | http://localhost:8002/docs | http://localhost:8002/ | http://localhost:8002/health |

> ⚠️ 首次使用请在聊天界面配置 API Key（支持 NVIDIA / 讯飞 / 智谱），获取地址：https://build.nvidia.com

## 📁 项目结构

```
rag/
├── financial-agent-api/           # LangChain 版本（端口 8001）
│   ├── app/                       # 应用核心代码
│   │   ├── main.py                # FastAPI 入口（CORS、限速、路由挂载）
│   │   ├── exceptions.py          # 自定义异常体系
│   │   ├── api/                   # 路由层
│   │   │   ├── routes_chat.py     # 对话接口（含限速装饰器、SSE 流式端点）
│   │   │   └── routes_docs.py     # 文档管理接口
│   │   ├── agent/                 # Agent 核心
│   │   │   ├── graph.py           # LangGraph 状态机核心（条件路由 + 记忆）
│   │   │   ├── chain.py           # 对话链与工具调度
│   │   │   └── tools.py           # @tool 工具定义
│   │   ├── core/                  # 核心配置
│   │   │   ├── config.py          # 环境变量配置（多 Provider 支持）
│   │   │   ├── dependencies.py    # 依赖注入
│   │   │   ├── enums.py           # 枚举定义
│   │   │   ├── limiter.py         # 限流器实例
│   │   │   └── protocols.py       # 协议定义
│   │   ├── models/                # 数据模型
│   │   │   └── schemas.py         # Pydantic 模型
│   │   ├── rag/                   # RAG 检索
│   │   │   ├── loader.py          # PDF/TXT/DOCX/MD 加载与切片
│   │   │   ├── vectorstore.py     # ChromaDB 构建/检索
│   │   │   ├── engine_router.py   # 双 RAG 引擎路由（方案A/B 自动切换）
│   │   │   ├── reranker.py        # Reranker 精排封装（NVIDIA / 智谱AI）
│   │   │   └── nvidia_blueprint_client.py  # NVIDIA RAG Blueprint 适配层
│   │   └── static/                # 前端静态文件
│   ├── data/                      # 知识库文件目录
│   │   └── knowledge_base/        # PDF/TXT/DOCX/MD 知识库文档
│   ├── chroma_db/                 # 向量数据库本地存储
│   ├── scripts/                   # 工具脚本
│   │   └── check_services.py      # 服务健康检查
│   ├── docker-compose.yml         # 多容器编排配置
│   ├── Dockerfile                 # 应用镜像构建
│   ├── Makefile                   # 一键操作命令
│   ├── pyproject.toml             # 项目配置
│   ├── requirements.txt           # Python 依赖
│   └── run.py                     # 本地启动脚本
│
├── llama-index-app/               # LlamaIndex 版本（端口 8002）
│   ├── app/                       # 应用核心代码
│   │   ├── main.py                # FastAPI 入口
│   │   ├── exceptions.py          # 自定义异常体系
│   │   ├── api/                   # 路由层
│   │   │   ├── routes_chat.py     # 对话接口
│   │   │   └── routes_docs.py     # 文档管理接口
│   │   ├── agent/                 # Agent 核心
│   │   │   ├── workflow.py        # AgentWorkflow 工作流
│   │   │   └── tools.py           # FunctionTool 工具定义
│   │   ├── core/                  # 核心配置
│   │   │   ├── config.py          # 环境变量配置（共享 nim_config.txt）
│   │   │   ├── dependencies.py    # 依赖注入
│   │   │   ├── enums.py           # 枚举定义
│   │   │   ├── limiter.py         # 限流器实例
│   │   │   └── protocols.py       # 协议定义
│   │   ├── models/                # 数据模型
│   │   │   └── schemas.py         # Pydantic 模型
│   │   ├── rag/                   # RAG 检索
│   │   │   ├── index_store.py     # VectorStoreIndex + 混合检索
│   │   │   └── loader.py          # 文档加载与切片
│   │   └── static/                # 前端静态文件
│   ├── chroma_db_li/              # LlamaIndex 专用向量库（与 LangChain 版隔离）
│   ├── requirements.txt           # Python 依赖
│   └── run.py                     # 本地启动脚本
│
├── data/reports/                  # 示例文档（比亚迪财报等）
├── chroma_db/                     # step2 示例的向量库
├── step1_agent.py                 # LangChain Function Calling 入门示例
├── step2_rag.py                   # LangChain RAG 检索入门示例
├── run_step1.bat                  # 运行 step1 示例
├── run_step2.bat                  # 运行 step2 示例
├── start-langchain.bat            # 一键启动 LangChain 版本
├── start-llamaindex.bat           # 一键启动 LlamaIndex 版本
├── nim_config.txt                 # NVIDIA API Key 配置（已 gitignore）
└── .gitignore
```

## ⚙️ 配置说明

### 环境变量

#### 基础配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM 服务地址（云端 API 或本地服务） |
| `NIM_MODEL_NAME` | `deepseek-ai/deepseek-v4-pro` | 大语言模型名称 |
| `NIM_EMBEDDING_MODEL` | `nvidia/nv-embedqa-e5-v5` | Embedding 模型名称 |
| `NVIDIA_API_KEY` | - | NVIDIA API Key（可从环境变量或 `nim_config.txt` 读取） |
| `CHROMA_DB_DIR` | `./chroma_db` | 向量库目录（LlamaIndex 版默认 `./chroma_db_li`） |
| `KNOWLEDGE_DIR` | `./data/knowledge_base` | 知识库文件目录 |
| `CHUNK_SIZE` | `500` | 文档切片大小（字符数） |
| `CHUNK_OVERLAP` | `50` | 切片重叠大小（字符数） |
| `TOP_K` | `3` | 检索返回的文档数量 |
| `SIMILARITY_THRESHOLD` | `1.3` | 相似度阈值（L2 距离） |
| `RATE_LIMIT_RPM` | `29` | 每分钟请求限制 |
| `LLM_CACHE_TYPE` | `memory` | LLM 缓存类型（memory 或 sqlite） |

#### 多 Provider 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PROVIDER` | `nim` | LLM Provider 选择（`nim` / `xfyun` / `zhipu`） |
| `XFYUN_API_KEY` | - | 讯飞星辰 API Key |
| `XFYUN_BASE_URL` | - | 讯飞星辰 API 地址 |
| `XFYUN_MODEL_NAME` | `xopqwen36v35b` | 讯飞星辰模型名称 |
| `ZHIPU_API_KEY` | - | 智谱AI API Key |
| `ZHIPU_BASE_URL` | - | 智谱AI API 地址 |
| `ZHIPU_MODEL_NAME` | `glm-4.7-flash` | 智谱AI 模型名称 |

#### Reranker 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `RERANKER_MODEL` | `nvidia/llama-nemotron-rerank-1b-v2` | Reranker 模型名称（NVIDIA） |
| `RERANKER_TIMEOUT` | `30` | Reranker 超时时间（秒） |
| `ZHIPU_RERANKER_MODEL` | `rerank` | 智谱AI Reranker 模型名称 |

#### NVIDIA Blueprint 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `BLUEPRINT_API_URL` | - | NVIDIA RAG Blueprint API 地址 |
| `BLUEPRINT_API_KEY` | - | Blueprint API Key |
| `BLUEPRINT_LLM_MODELNAME` | - | Blueprint 使用的 LLM 模型名称 |

#### LangGraph 与 SSE 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LANGGRAPH_MAX_ITERATIONS` | `10` | LangGraph 最大迭代次数 |
| `SSE_EVENT_QUEUE_TIMEOUT` | `300` | SSE 事件队列超时（秒） |
| `MAX_TOOL_ROUNDS` | `5` | 最大工具调用轮数 |

#### LLM 参数配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_TEMPERATURE` | `0.1` | LLM 温度参数 |
| `LLM_TOP_P` | `0.9` | LLM Top-P 参数 |
| `LLM_MAX_TOKENS` | `4096` | LLM 最大生成 Token 数 |
| `LLM_REQUEST_TIMEOUT` | `120` | LLM 请求超时（秒） |

#### 会话与重试配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SESSION_MAX_MESSAGES` | `50` | 会话最大消息数 |
| `RETRY_MAX_ATTEMPTS` | `3` | 请求重试次数 |

### API Key 配置

两个版本共享同一个 `nim_config.txt` 文件（位于 `financial-agent-api/` 目录下）：

```bash
# 方式 1：配置文件（推荐）
echo "api_key=your_nvidia_api_key" > financial-agent-api/nim_config.txt

# 方式 2：环境变量
export NVIDIA_API_KEY="your_api_key"

# 方式 3：聊天界面配置
# 启动后在 http://localhost:8001/ 或 http://localhost:8002/ 的设置页面填入
```

> 💡 切换到讯飞/智谱 Provider 时，需配置对应的 API Key（`XFYUN_API_KEY` 或 `ZHIPU_API_KEY`）。

### 支持的模型

| 模型类型 | 模型名称 | 说明 |
|---------|---------|------|
| **LLM** | `deepseek-ai/deepseek-v4-pro` | DeepSeek V4 Pro（NVIDIA NIM，LangChain 版默认） |
| **LLM** | `deepseek-ai/deepseek-v4-flash` | DeepSeek V4 Flash（NVIDIA NIM，LlamaIndex 版默认） |
| **LLM** | `moonshotai/kimi-k2.6` | Moonshot Kimi K2.6（NVIDIA NIM） |
| **LLM** | `xopqwen36v35b` | 讯飞星辰模型 |
| **LLM** | `glm-4.7-flash` | 智谱AI 模型 |
| **Embedding** | `nvidia/nv-embedqa-e5-v5` | NVIDIA 办公 Embedding 模型（默认） |
| **Embedding** | `BAAI/bge-small-zh-v1.5` | HuggingFace 本地 Embedding（讯飞/智谱降级方案） |
| **Reranker** | `nvidia/llama-nemotron-rerank-1b-v2` | NVIDIA Reranker 精排模型 |
| **Reranker** | `rerank` | 智谱AI Reranker 精排模型 |

### 端口说明

| 端口 | 服务 | 说明 |
|------|------|------|
| 8001 | FastAPI（LangChain 版） | API 入口、Swagger UI、聊天界面 |
| 8002 | FastAPI（LlamaIndex 版） | API 入口、Swagger UI、聊天界面 |
| 8002 | NVIDIA NIM（Docker） | 大模型推理服务（仅 Docker 部署时占用） |
| 8003 | ChromaDB（Docker） | 向量数据库 API（仅 Docker 部署时占用） |

## 📡 API 接口

两个版本共享相同的 API 设计，仅端口号不同（LangChain: 8001，LlamaIndex: 8002）。

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

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | ✅ | 用户输入的问题（1-2000 字符） |
| `session_id` | string | ❌ | 会话 ID，为空时自动生成 |

**响应体**：
```json
{
  "answer": "根据内部文档检索结果，公司的报销流程如下：\n1. 员工在费用发生后30天内提交报销单...",
  "answer_format": "markdown",
  "tools_used": ["search_internal_documents"],
  "intermediate_steps": [
    {
      "tool_name": "search_internal_documents",
      "tool_args": {"query": "报销流程", "department": "通用"},
      "tool_result": "【来源：报销流程规范.pdf】\n员工报销需在费用发生后30天内...",
      "tool_result_type": "search_results",
      "sources": [
        {
          "source": "报销流程规范.pdf",
          "department": "财务部",
          "score": 0.87,
          "snippet": "员工报销需在费用发生后30天内提交..."
        }
      ],
      "duration_ms": 320,
      "status": "success"
    }
  ],
  "total_duration_ms": 1500,
  "session_id": "user-123"
}
```

#### POST `/api/v1/chat/stream` 🔴 LangChain 版本独有

SSE 流式对话接口，实时推送 Agent 执行状态

**请求体**：
```json
{
  "query": "公司的报销流程是怎样的？",
  "session_id": "user-123"
}
```

**响应**：`text/event-stream`（SSE 格式）

```
event: stream_start
data: {"session_id": "user-123", "timestamp": "2026-06-26T10:00:00Z"}

event: agent_start
data: {"agent": "conversational_agent", "query": "公司的报销流程是怎样的？"}

event: tool_call
data: {"tool": "search_internal_documents", "args": {"query": "报销流程"}}

event: tool_result
data: {"tool": "search_internal_documents", "status": "success", "duration_ms": 320}

event: retrieval_result
data: {"sources": [{"source": "报销流程规范.pdf", "score": 0.87, "snippet": "..."}]}

event: stream_end
data: {"answer": "根据内部文档检索结果...", "tools_used": ["search_internal_documents"], "total_duration_ms": 1500}
```

**SSE 事件类型说明**：

| 事件类型 | 说明 | 关键字段 |
|---------|------|---------|
| `stream_start` | 会话开始 | `session_id`, `timestamp` |
| `agent_start` | Agent 开始推理 | `agent`, `query` |
| `tool_call` | 工具调用开始 | `tool`, `args` |
| `tool_result` | 工具调用结果 | `tool`, `status`, `duration_ms` |
| `retrieval_result` | 检索结果 | `sources`（含来源、分数、片段） |
| `stream_end` | 流式结束 | `answer`, `tools_used`, `total_duration_ms` |

> 💡 客户端可使用 `EventSource` 或 `fetch` + `ReadableStream` 消费 SSE 流。

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

上传文档（LangChain 版：PDF/TXT/DOCX/MD；LlamaIndex 版：PDF/TXT/DOCX/MD/CSV）

**请求**：`multipart/form-data`
- `file`: 文件对象
- `department`: 所属部门（可选，默认"通用"）

**响应体**：
```json
{
  "filename": "report.pdf",
  "chunks_added": 42,
  "message": "文档 report.pdf 已成功上传并入库",
  "department": "财务部"
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

### 系统接口

#### GET `/health`

检查服务健康状态

**响应体**：
```json
{
  "status": "ok",
  "vectorstore_count": 128,
  "model_name": "deepseek-ai/deepseek-v4-pro",
  "blueprint_available": false
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 服务状态（`ok` / `error`） |
| `vectorstore_count` | int | 向量库文档数量 |
| `model_name` | string | 当前使用的 LLM 模型名称 |
| `blueprint_available` | bool | NVIDIA RAG Blueprint 是否可用（仅 LangChain 版本） |

#### GET `/api/v1/config/apikey`

查询 API Key 配置状态

**响应体**：
```json
{
  "configured": true,
  "hint": "nvap****abcd",
  "model_name": "deepseek-ai/deepseek-v4-pro",
  "provider": "nim"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `configured` | bool | 是否已配置 API Key |
| `hint` | string | API Key 脱敏提示 |
| `model_name` | string | 当前使用的模型名称 |
| `provider` | string | 当前 LLM Provider（`nim` / `xfyun` / `zhipu`） |

#### POST `/api/v1/config/apikey`

保存 API Key、模型名称和 Provider

**请求体**：
```json
{
  "api_key": "nvapi-xxxxx",
  "model_name": "deepseek-ai/deepseek-v4-pro",
  "provider": "nim"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `api_key` | string | ✅ | API Key |
| `model_name` | string | ❌ | 模型名称 |
| `provider` | string | ❌ | LLM Provider（`nim` / `xfyun` / `zhipu`） |

**响应体**：
```json
{
  "message": "配置已保存",
  "configured": true
}
```

### 完整 API 文档

启动服务后访问：
- LangChain 版 Swagger UI: http://localhost:8001/docs
- LlamaIndex 版 Swagger UI: http://localhost:8002/docs

## 🔧 核心工具列表

| 工具名称 | 功能 | LangChain | LlamaIndex |
|---------|------|:---------:|:----------:|
| `search_internal_documents` | 检索内部办公文档（混合检索 + Reranker 精排） | ✅ | ✅ |
| `get_employee_info` | 查询员工信息（模拟 OA） | ✅ | ✅ |
| `search_web` | 联网搜索（DuckDuckGo） | ✅ | ✅ |
| `send_email_notification` | 模拟发送邮件通知 | ✅ | ✅ |
| `generate_prd_document` | 生成结构化 PRD 文档 | ✅ | ✅ |
| `generate_flowchart_code` | 生成 Mermaid 流程图代码 | ✅ | ✅ |
| `generate_html_prototype` | 生成 HTML 低保真原型 | ✅ | ✅ |
| `search_csv_data` | CSV 数据统计查询 | ❌ | ✅ |

> 💡 LlamaIndex 版本的 `generate_prd_document` 集成了 NVIDIA Blueprints 优化模板，PRD 结构更完整。
> 💡 LangChain 版本的 `search_internal_documents` 支持 Reranker 精排，检索结果更精准。

## 📚 入门示例

项目根目录包含两个入门脚本，帮助你快速理解 LangChain Agent 和 RAG 的核心概念：

### step1_agent.py — Function Calling 入门

演示 LangChain Tool Calling 的基本用法，包含模拟股票查询和研报搜索工具。

```bash
run_step1.bat   # 或 python step1_agent.py
```

### step2_rag.py — RAG 检索入门

演示 LangChain RAG 检索的完整流程：文档加载 → 切片 → 向量化 → 相似度检索。

```bash
run_step2.bat   # 或 python step2_rag.py
```

> ⚠️ 运行示例前需先配置 `nim_config.txt` 中的 NVIDIA API Key。

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

### Makefile 命令速查（LangChain 版本）

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

**LangChain 版本**（`financial-agent-api/app/agent/tools.py`）：

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

然后在 `graph.py` 中注册：
```python
tools = [search_internal_documents, get_employee_info, get_meeting_room]
```

**LlamaIndex 版本**（`llama-index-app/app/agent/tools.py`）：

```python
def make_get_meeting_room_tool() -> FunctionTool:
    """创建会议室查询工具（LlamaIndex 版本）。"""

    def get_meeting_room(room_id: str) -> str:
        """查询会议室信息

        Args:
            room_id: 会议室编号

        Returns:
            会议室详细信息
        """
        # 实现逻辑
        return result

    return FunctionTool.from_defaults(fn=get_meeting_room)
```

然后在 `workflow.py` 中注册：
```python
tools = [
    make_search_internal_documents_tool(index),
    # ... 其他工具
    make_get_meeting_room_tool(),
]
```

## 🚢 部署指南

### Docker 部署（LangChain 版本，推荐生产环境）

#### 1. 准备工作

```bash
cd financial-agent-api
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

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
echo "api_key=your_api_key" > financial-agent-api/nim_config.txt

# 启动服务
python run.py
```

### GPU 支持

Docker Compose 配置已包含 GPU 支持（LangChain 版本）：

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

## 🔒 安全建议

1. **API Key 保护**：
   - 不要将 `nim_config.txt` 和 `.env` 提交到版本控制
   - 使用环境变量或密钥管理服务

2. **网络安全**：
   - 生产环境使用 HTTPS
   - 配置防火墙规则
   - 限制容器网络访问

3. **数据安全**：
   - 定期备份 `chroma_db/` 和 `data/`
   - 敏感文档考虑加密存储

4. **限流保护**：
   - 生产环境建议降低 `RATE_LIMIT_RPM`
   - 考虑添加用户认证和授权

## ❓ 常见问题

### Q1: 两个版本可以同时运行吗？

**A**: 可以。LangChain 版本运行在 8001 端口，LlamaIndex 版本运行在 8002 端口，两者共享知识库文档和 API Key 配置，互不干扰。

### Q2: 如何切换云端 API 和本地 NIM？

**A**: 修改环境变量 `NIM_BASE_URL`：

```bash
# 使用云端 API（默认）
NIM_BASE_URL=https://integrate.api.nvidia.com/v1

# 使用本地 NIM Docker
NIM_BASE_URL=http://nim:8000/v1

# 使用本机运行的 NIM
NIM_BASE_URL=http://localhost:8000/v1
```

### Q3: 如何更换大语言模型？

**A**: 修改环境变量 `NIM_MODEL_NAME`：

```bash
# 使用 DeepSeek V4 Pro
NIM_MODEL_NAME=deepseek-ai/deepseek-v4-pro

# 使用 Kimi K2.6
NIM_MODEL_NAME=moonshotai/kimi-k2.6
```

### Q4: 请求频率超限怎么办？

**A**: 系统会返回 HTTP 429 错误，响应头包含 `Retry-After`：

```bash
# 调整限速阈值
RATE_LIMIT_RPM=30  # 每分钟 30 次
```

### Q5: 如何添加新的知识库文档？

**A**: 三种方式：

1. **API 上传**：
```bash
curl -X POST http://localhost:8001/api/v1/docs/upload \
  -F "file=@report.pdf" \
  -F "department=财务部"
```

2. **文件系统**：直接放入 `financial-agent-api/data/knowledge_base/` 目录，重启服务自动加载

3. **Docker 挂载**：将宿主机目录挂载到容器：
```yaml
volumes:
  - ./my_docs:/app/data/knowledge_base
```

### Q6: 两个版本的向量库会互相影响吗？

**A**: 不会。LangChain 版本使用 `chroma_db/` 目录，LlamaIndex 版本使用 `chroma_db_li/` 目录，两者完全隔离。

### Q7: LlamaIndex 版本的 Embedding 回退机制是什么？

**A**: LlamaIndex 版本优先使用 NVIDIA 云端 Embedding（nv-embedqa-e5-v5），如果配置失败则自动回退到 HuggingFace 本地模型（BAAI/bge-small-zh-v1.5），无需 API Key 即可运行。

### Q8: 端口冲突怎么办？

**A**:

- **本地部署**：修改各版本 `app/core/config.py` 中的端口配置，或启动时指定：
```bash
uvicorn app.main:app --host 0.0.0.0 --port 9001 --reload
```

- **Docker 部署**：修改 `docker-compose.yml` 中的端口映射：
```yaml
services:
  api:
    ports:
      - "9001:8001"
```

### Q9: 如何查看容器日志？

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

### Q10: 如何完全重置系统？

**A**:
```bash
# LangChain 版本（Docker）
cd financial-agent-api
make clean
make rebuild

# LlamaIndex 版本（删除向量库目录后重启）
rm -rf llama-index-app/chroma_db_li/
```

### Q11: 如何切换 LLM Provider？

**A**: 三种方式切换：

1. **环境变量**：设置 `PROVIDER=xfyun` 或 `PROVIDER=zhipu`
2. **API 接口**：通过 `POST /api/v1/config/apikey` 传递 `provider` 字段
3. **聊天界面**：在设置页面选择 Provider

> ⚠️ 切换到讯飞/智谱 Provider 后，Embedding 会自动降级到 HuggingFace 本地模型（BAAI/bge-small-zh-v1.5），使用 hf-mirror.com 镜像下载。

### Q12: Reranker 不可用怎么办？

**A**: 系统会自动降级处理：
- NVIDIA Reranker 不可用时，尝试使用智谱AI Reranker
- 所有 Reranker 均不可用时，跳过精排步骤，直接使用 RRF 融合结果
- 降级行为会在日志中记录，不影响正常对话

### Q13: SSE 流式接口如何使用？

**A**: LangChain 版本提供 SSE 流式接口 `POST /api/v1/chat/stream`：

```javascript
// JavaScript 示例
const eventSource = new EventSource('/api/v1/chat/stream', {
  method: 'POST',
  body: JSON.stringify({ query: '你好', session_id: 'test' })
});

// 或使用 fetch
const response = await fetch('/api/v1/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: '你好', session_id: 'test' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const text = decoder.decode(value);
  // 解析 SSE 事件
  console.log(text);
}
```

```python
# Python 示例
import httpx
import json

with httpx.stream("POST", "http://localhost:8001/api/v1/chat/stream",
                   json={"query": "你好", "session_id": "test"}) as resp:
    for line in resp.iter_lines():
        if line.startswith("data:"):
            data = json.loads(line[5:])
            print(data)
```

### Q14: 什么是双 RAG 引擎路由？

**A**: LangChain 版本支持两种 RAG 方案自动路由：

- **方案 A（自建 RAG）**：使用本地 ChromaDB + BM25 + RRF 融合 + Reranker 精排，完全自主可控
- **方案 B（NVIDIA RAG Blueprint）**：调用 NVIDIA RAG Blueprint API，由 NVIDIA 托管检索和推理

路由逻辑：
1. 优先使用方案 A（自建 RAG）
2. 如果方案 A 检索结果为空或失败，自动降级到方案 B
3. 如果方案 B 也不可用，返回兜底提示
4. 可通过 `BLUEPRINT_API_URL` 和 `BLUEPRINT_API_KEY` 配置方案 B

## 📦 GitHub 上架清理清单

在将项目推送到 GitHub 之前，建议清理以下文件/文件夹：

### 缓存与编译文件（必须删除）

| 路径 | 说明 |
|------|------|
| `__pycache__/` | Python 编译缓存（根目录） |
| `.ruff_cache/` | Ruff 缓存（根目录） |
| `financial-agent-api/.mypy_cache/` | mypy 缓存 |
| `financial-agent-api/.ruff_cache/` | Ruff 缓存 |
| `financial-agent-api/app/**/__pycache__/` | 应用内 Python 缓存 |
| `llama-index-app/app/**/__pycache__/` | 应用内 Python 缓存 |

### IDE 配置文件（建议删除）

| 路径 | 说明 |
|------|------|
| `.arts/` | IDE 配置（根目录） |
| `.codeartsdoer/` | IDE 配置（根目录） |
| `financial-agent-api/.arts/` | IDE 配置 |
| `financial-agent-api/.codeartsdoer/` | IDE 配置 |

### 日志文件（建议删除）

| 路径 | 说明 |
|------|------|
| `server.log` | 空日志文件（根目录） |
| `server_error.log` | 旧日志文件（根目录） |
| `llama-index-app/server_stderr.log` | 服务日志 |
| `llama-index-app/server_stdout.log` | 服务日志 |

### 重复/测试数据（建议删除）

| 路径 | 说明 |
|------|------|
| `data/reports/` | 根目录示例文档（与 `financial-agent-api/data/reports/` 重复） |
| `chroma_db/` | 根目录 step2 示例向量库（非正式项目数据） |
| `llama-index-app/3.0` | 空文件（0 字节） |

### 建议更新 `.gitignore`

确保 `.gitignore` 包含以下条目：

```gitignore
__pycache__/
*.pyc
*.pyo
.ruff_cache/
.mypy_cache/
.arts/
.codeartsdoer/
*.log
chroma_db/
chroma_db_li/
.env
nim_config.txt
```

## 📄 License

MIT License

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/my-feature`
3. 提交更改：`git commit -m 'Add my feature'`
4. 推送分支：`git push origin feature/my-feature`
5. 提交 Pull Request

---

**Made with ❤️ by Enterprise Knowledge Agent Team**
