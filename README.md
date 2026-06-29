# 企业级 GenAI Agent 平台

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://python.langchain.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11+-purple.svg)](https://www.llamaindex.ai/)
[![Version](https://img.shields.io/badge/Version-5.1.0-blue.svg)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

企业内部办公知识库智能问答系统，提供 **LangChain** 和 **LlamaIndex** 双版本实现。LangChain 版本已升级为企业级 GenAI Agent 平台（v5.1.0），内置 API Key 认证、PII 检测与脱敏、Prompt Injection 防御、Prometheus 可观测性、K8s 云原生部署、平台 SDK，以及 v5.0 MLOps 全链路能力（实验追踪、漂移检测、RAG 评估、A/B 测试）和 v5.1 Agentic Architecture 演进（知识图谱增强检索、人机协同审批、死循环防御护栏）。两者共享知识库文档和配置文件，可同时运行在不同端口，满足不同技术栈的选型需求。

## ✨ 核心特性

### LangChain 版本（企业级平台）

- 🤖 **LangGraph 状态机**：Tool Calling + 条件路由 + 多轮对话记忆（MemorySaver）
- 📡 **SSE 流式推送**：Server-Sent Events 实时状态推送，事件类型包括 stream_start、agent_start、tool_call、tool_result、retrieval_result、stream_end 等
- 🔀 **多 Provider 支持**：NVIDIA NIM / 讯飞星辰 / 智谱AI 三大 LLM Provider，可在线切换
- 🎯 **Reranker 精排**：支持 NVIDIA Reranker（nvidia/llama-nemotron-rerank-1b-v2）和智谱AI Reranker，不可用时自动降级
- 🔀 **双 RAG 引擎路由**：方案 A（自建 RAG + Reranker）和方案 B（NVIDIA RAG Blueprint）自动路由与降级
- 🔍 **混合检索引擎**：向量 + BM25 + RRF 融合排序，提升检索精准度
- 📄 **多格式文档**：支持 PDF/TXT/DOCX/MD 文档上传、切片、向量化存储
- 👤 **员工信息查询**：演示数据，快速查询员工联系方式（未接入 OA 系统）
- 🌐 **联网搜索**：内库查无结果时自主联网搜索（DuckDuckGo）
- 📧 **数字员工动作**：模拟发送邮件通知，支持自动化任务执行（未接入 SMTP）
- 📝 **PRD 自动撰写**：基于 LLM 自动生成结构化产品需求文档，支持导出为 Word
- 📊 **流程图生成**：将业务描述转化为 Mermaid 代码，无缝对接 Visio/Draw.io
- 🖥️ **低保真原型**：快速生成基于 HTML/Tailwind 的前端原型，提升设计效率
- ⚡ **语义缓存**：高频问题直接返回缓存结果，提升响应速度
- 🔐 **API Key 认证与 RBAC**：基于 API Key 的身份认证，admin/developer/viewer 三级角色权限
- 🛡️ **PII 检测与脱敏**：自动检测邮箱/手机/身份证/银行卡/护照等 PII，输出侧自动脱敏
- 🚫 **Prompt Injection 防御**：8 种注入模式检测，HIGH/MEDIUM/LOW 三级风险评估
- ⏱️ **滑动窗口限流**：IP 限流 + 用户级限流双重保护
- 📋 **企业级审计日志**：结构化审计事件，覆盖 API/LLM/安全全链路
- 📈 **Prometheus 可观测性**：HTTP/LLM/工具/RAG 四维指标，/metrics 端点
- 📊 **Grafana Dashboard**：开箱即用的 Dashboard 配置（observability/grafana/dashboard.json）
- 🔗 **关联 ID 追踪**：X-Correlation-ID 跨模块日志关联
- 📝 **K8s JSON 日志**：结构化 JSON 日志输出，适配 Fluentd/Filebeat/Loki 采集
- ☸️ **K8s 云原生部署**：Namespace/ConfigMap/Secret/Deployment/HPA/Ingress/NetworkPolicy
- 🐳 **Docker Compose 部署**：多容器编排，一键部署
- 🧩 **平台 SDK**：Python SDK 客户端，5 分钟接入新能力
- 📦 **工具/Provider 模板**：标准模板快速开发新工具和新 Provider
- 🔍 **代码审查系统**：多 Agent 协作审查（Supervisor + Worker 模式），支持安全/架构/性能/风格四维度并行审查
- 🔧 **MCP 协议集成**：Model Context Protocol 适配层，支持 GitHub/文件系统/数据库（模拟数据）/联网搜索四大 MCP Server
- 🧪 **完整测试体系**：unit + integration + e2e 三层测试（Pytest），含安全/可观测性测试
- 🔐 **类型安全**：mypy strict 模式，完整的类型标注
- 📝 **代码质量**：ruff linter + formatter，遵循 PEP8/PEP20 规范

### v5.0 MLOps 全链路能力

- 🧪 **MLflow 实验追踪**：自动记录 RAG 链路参数（provider、model_name、chunk_size、top_k）与运行时指标（延迟、Token 用量），支持启用/禁用开关
- 📊 **查询漂移检测**：K-S 检验和 MMD 两种算法检测用户查询分布偏移，超过阈值触发 Prometheus 告警
- 📋 **RAG 四维度评估**：Faithfulness（忠实度）、Answer Relevancy（回答相关性）、Context Precision（上下文精确度）、Context Recall（上下文召回率），LLM-as-Judge 自动评估
- 🔀 **A/B 测试框架**：基于 session_id 哈希分桶的流量分发，支持动态策略更新、会话一致性、指标持久化
- 🤖 **预测性 AI 工具**：推荐相似文档工具（排除当前 RAG 上下文）+ 意图预测工具（TF-IDF + 朴素贝叶斯）
- 📈 **MLOps Dashboard**：A/B 测试配置展示、评估得分趋势图、缓存 Hit/Miss 比例、漂移检测状态
- ☁️ **Terraform IaC**：AWS ECS Fargate 云原生部署配置（VPC、Subnet、ALB、IAM）

### v5.1 Agentic Architecture 演进

- 🕸️ **知识图谱增强检索**：NetworkX 有向图存储实体关系三元组，云端 LLM 自动抽取，图谱遍历与向量检索三通道融合
- 👨‍💼 **人机协同审批（HITL）**：高风险工具（邮件发送、向量库清空等）执行前挂起状态机，SSE 推送审批弹窗，支持批准/拒绝/超时自动拒绝
- 🛡️ **死循环防御护栏（Guardrails）**：ToolRepetitionDetector 滑动窗口检测重复工具调用，超过阈值抛出 InfiniteLoopDetectedError 优雅终止
- 📊 **增强评估框架**：工具选择准确率、幻觉检测率、知识图谱检索准确率、HITL 合规性四维度评估
- 🔍 **知识图谱可视化**：Mermaid.js 力导向图渲染，实体搜索与关系展示

### LlamaIndex 版本

- 🤖 **AgentWorkflow**：LlamaIndex ReActAgent 工作流
- 🔍 **混合检索引擎**：VectorStoreIndex + BM25 + QueryFusionRetriever (RRF)
- 📈 **CSV 数据统计**：自然语言查询 CSV 数据（PandasQueryEngine）
- 📝 **NVIDIA Blueprints**：PRD 优化模板 + RRF 融合优化
- 🔄 **HuggingFace Embedding 回退**：NVIDIA Embedding 不可用时自动回退到 HuggingFace 本地模型（BAAI/bge-small-zh-v1.5）

## 📋 目录

- [双版本对比](#双版本对比)
- [架构概览](#架构概览)
- [安全与合规](#安全与合规)
- [可观测性](#可观测性)
- [K8s 云原生部署](#k8s-云原生部署)
- [平台 SDK 与模板](#平台-sdk-与模板)
- [技术栈](#技术栈)
- [快速启动](#快速启动)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [API 接口](#api-接口)
- [代码审查系统](#代码审查系统)
- [核心工具列表](#核心工具列表)
- [入门示例](#入门示例)
- [开发指南](#开发指南)
- [测试体系](#测试体系)
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
| **Embedding** | NVIDIAEmbeddings (nv-embedqa-e5-v5) | NVIDIAEmbedding (nv-embedqa-e5-v5)，回退 HuggingFace (BAAI/bge-small-zh-v1.5) |
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
| **代码审查** | ✅ 多 Agent 协作（Supervisor + 4 Worker + Summary） | ❌ |
| **MCP 协议** | ✅ GitHub/文件系统/数据库（模拟数据）/联网搜索 4 个适配器 | ❌ |
| **API Key 认证 + RBAC** | ✅ admin/developer/viewer 三级权限 | ❌ |
| **PII 检测与脱敏** | ✅ 邮箱/手机/身份证/银行卡/护照 | ❌ |
| **Prompt Injection 防御** | ✅ 8 种注入模式检测 | ❌ |
| **企业级审计日志** | ✅ 结构化审计事件 | ❌ |
| **Prometheus 指标** | ✅ /metrics 端点，四维指标 | ❌ |
| **Grafana Dashboard** | ✅ 开箱即用 | ❌ |
| **关联 ID 追踪** | ✅ X-Correlation-ID | ❌ |
| **K8s JSON 日志** | ✅ 结构化 JSON 输出 | ❌ |
| **K8s 云原生部署** | ✅ Deployment/HPA/Ingress/NetworkPolicy | ❌ |
| **平台 SDK** | ✅ Python SDK 客户端 | ❌ |
| **工具/Provider 模板** | ✅ 标准开发模板 | ❌ |
| **测试体系** | ✅ unit + integration + e2e（含安全/可观测性测试） | ❌ |
| **一键启动** | `start-langchain.bat` | `start-llamaindex.bat` |
| **MLflow 实验追踪** | ✅ RAG 链路参数与指标自动记录 | ❌ |
| **查询漂移检测** | ✅ K-S 检验 / MMD 两种算法 | ❌ |
| **RAG 四维度评估** | ✅ LLM-as-Judge 自动评估 | ❌ |
| **A/B 测试框架** | ✅ session_id 哈希分桶 + 动态策略更新 | ❌ |
| **预测性 AI 工具** | ✅ 推荐相似文档 + 意图预测 | ❌ |
| **MLOps Dashboard** | ✅ 评估趋势图 / 缓存比例 / 漂移状态 | ❌ |
| **Terraform IaC** | ✅ AWS ECS Fargate 云原生部署 | ❌ |
| **知识图谱增强检索** | ✅ NetworkX 三元组 + 三通道融合 | ❌ |
| **人机协同审批 (HITL)** | ✅ 高风险工具审批 + SSE 弹窗 | ❌ |
| **死循环防御护栏** | ✅ ToolRepetitionDetector + 优雅终止 | ❌ |
| **增强评估框架** | ✅ 工具选择 / 幻觉 / 图谱 / HITL 合规 | ❌ |

### 如何选择？

- **选 LangChain 版本**：需要企业级安全合规（API Key 认证、PII 脱敏、注入防御）、可观测性（Prometheus + Grafana）、K8s 生产部署、Docker 部署、LangGraph 状态机、SSE 流式推送、Reranker 精排、双 RAG 引擎路由、审计日志、语义缓存、代码审查系统（多 Agent 协作）、MCP 协议集成、平台 SDK、完整测试体系、MLOps 全链路能力（MLflow 实验追踪、漂移检测、RAG 评估、A/B 测试）、Agentic Architecture 演进（知识图谱增强检索、人机协同审批、死循环防御护栏）
- **选 LlamaIndex 版本**：需要 CSV 数据统计查询、NVIDIA Blueprints 优化模板、HuggingFace 本地 Embedding 回退
- **两个都跑**：同时启动，分别监听 8001 和 8002 端口，共享同一份知识库

## 🏗️ 架构概览

### 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            项目根目录 (rag/)                              │
│                                                                          │
│  ┌──────────────────────────────────────────┐  ┌──────────────────────┐ │
│  │       LangChain 版本 (:8001)              │  │  LlamaIndex 版本      │ │
│  │       企业级 GenAI Agent 平台              │  │  (:8002)              │ │
│  │                                            │  │                      │ │
│  │  ┌──────────────────────────────────────┐ │  │  ┌────────────────┐ │ │
│  │  │         API Service (FastAPI)         │ │  │  │ FastAPI 路由层 │ │ │
│  │  │  ┌──────────┐ ┌──────────┐ ┌───────┐ │ │  │  └───────┬────────┘ │ │
│  │  │  │ Security │ │Observab- │ │ Agent │ │ │  │          │          │ │
│  │  │  │ Layer    │ │ility     │ │ Core  │ │ │  │  ┌───────▼────────┐ │ │
│  │  │  │-API Key  │ │-Promethe-│ │-Lang- │ │ │  │  │ AgentWorkflow  │ │ │
│  │  │  │ Auth     │ │us        │ │Chain  │ │ │  │  │ (ReActAgent)   │ │ │
│  │  │  │-RBAC     │ │-Correla- │ │-Tools │ │ │  │  └───────┬────────┘ │ │
│  │  │  │-PII Guard│ │tion ID   │ │-RAG   │ │ │  │          │          │ │
│  │  │  │-Prompt   │ │-JSON Logs│ │-MCP   │ │ │  │  ┌───────▼────────┐ │ │
│  │  │  │ Guard    │ │-Metrics  │ │-Audit │ │ │  │  │ VectorStoreIdx │ │ │
│  │  │  │-Rate     │ │          │ │       │ │ │  │  │ + BM25 + RRF   │ │ │
│  │  │  │ Limiter  │ │          │ │       │ │ │  │  └───────┬────────┘ │ │
│  │  │  └──────────┘ └──────────┘ └───────┘ │ │  │          │          │ │
│  │  └──────────────────────────────────────┘ │  │  ┌───────▼────────┐ │ │
│  │                                            │  │  │ chroma_db_li/  │ │ │
│  │  ┌──────────────────────────────────────┐ │  │  │ (独立向量库)    │ │ │
│  │  │  LangGraph 状态机 + Agent Chain       │ │  │  └────────────────┘ │ │
│  │  │  ┌────────┬────────┐                 │ │  │                      │ │
│  │  │  │方案A   │方案B   │                 │ │  │                      │ │
│  │  │  │自建RAG │Blueprint│                 │ │  │                      │ │
│  │  │  │+Rerank │ RAG    │                 │ │  │                      │ │
│  │  │  └───┬────┴───┬────┘                 │ │  │                      │ │
│  │  └──────┼────────┼──────────────────────┘ │  │                      │ │
│  │         │        │                         │  │                      │ │
│  │  ┌──────▼───┐    │                         │  │                      │ │
│  │  │ Reranker │    │                         │  │                      │ │
│  │  │ 精排     │    │                         │  │                      │ │
│  │  └──────┬───┘    │                         │  │                      │ │
│  │         │        │                         │  │                      │ │
│  │  ┌──────▼────────▼────────────────────────┐ │  │                      │ │
│  │  │  ChromaDB + BM25 + RRF + Reranker 精排  │ │  │                      │ │
│  │  └──────────┬─────────────────────────────┘ │  │                      │ │
│  │             │                                │  │                      │ │
│  │  ┌──────────▼───────────────────────────────┐ │  │                      │ │
│  │  │  chroma_db/ (独立向量库)                   │ │  │                      │ │
│  │  └──────────────────────────────────────────┘ │  │                      │ │
│  └──────────────────────────────────────────────┘  └──────────────────────┘ │
│                 │                                   │                      │
│                 └──────────┬────────────────────────┘                      │
│                            │                                               │
│  ┌─────────────────────────▼─────────────────────────────────────────────┐ │
│  │                    Cloud API Providers                                 │ │
│  │  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────────┐    │ │
│  │  │ NVIDIA NIM        │  │ 讯飞星辰      │  │ 智谱AI               │    │ │
│  │  │ - Embedding       │  │ - LLM        │  │ - LLM + Reranker     │    │ │
│  │  │ - Reranker        │  │              │  │                      │    │ │
│  │  │ - LLM             │  │              │  │                      │    │ │
│  │  └──────────────────┘  └──────────────┘  └──────────────────────┘    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                            │                                               │
│                 ┌──────────▼──────────┐                                    │
│                 │  data/knowledge_base/ │  ← 共享知识库文档                 │
│                 └──────────┬──────────┘                                    │
│                            │                                               │
│                 ┌──────────▼──────────┐                                    │
│                 │  nim_config.txt      │  ← 共享 API Key 配置              │
│                 └─────────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 核心流程（LangChain 版本）

```
用户提问 → 认证鉴权 → FastAPI 路由 → LangGraph 状态机 → 条件路由 → 工具选择 → 混合检索 → Reranker 精排 → LLM 推理 → PII 脱敏 → 响应返回
               ↓            ↓              ↓                ↓            ↓           ↓            ↓            ↓           ↓
           API Key 校验  限流保护      会话管理/记忆     状态转移      工具调用    向量+BM25+RRF  相关性重排    审计日志    SSE 推送
                                       ↓
                           ┌────────┬──────┼──────┬────────┬────────┬────────┐
                           ↓        ↓      ↓      ↓        ↓        ↓        ↓
                       内部文档  员工信息  联网搜索  邮件通知  CSV查询  产品经理效能  知识图谱
                                                                               ↓
                                                                       ┌───────┼───────┐
                                                                       ↓       ↓       ↓
                                                                   PRD生成  流程图  HTML原型

                           ┌─────────────── MLOps 全链路 ───────────────┐
                           │  MLflow 追踪 │ 漂移检测 │ RAG 评估 │ A/B 测试 │
                           └────────────────────────────────────────────┘

                           ┌─────────── Agentic Architecture ──────────┐
                           │  HITL 审批 │ Guardrails 护栏 │ KG 增强检索 │
                           └───────────────────────────────────────────┘

代码审查请求 → 认证鉴权 → FastAPI 路由 → Supervisor Agent → 并行调度 Worker → Summary Agent → 审查报告
                   ↓            ↓                ↓                    ↓                  ↓              ↓
               API Key 校验  限流保护       功能开关校验        四维度并行审查        结果汇总汇总      SSE 推送
                                ↓                    ↓
                           MCP 工具调用      ┌───────┼───────┐───────┐
                                            ↓       ↓       ↓       ↓
                                        Security Architecture Performance Style
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
  stream_start            → 会话开始，包含 session_id
  agent_start             → Agent 开始推理
  tool_call               → 工具调用开始（含工具名、参数）
  tool_result             → 工具调用结果
  retrieval_result        → 检索结果（含来源、分数）
  hitl_approval_required  → 高风险工具审批请求（含 tool_name、tool_args）
  hitl_approved           → 审批已批准
  hitl_rejected           → 审批已拒绝
  guardrail_loop_detected → 死循环检测告警（含 tool_name、repetition_count）
  stream_end              → 流式结束，包含完整 answer
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

## 🛡️ 安全与合规

LangChain 版本内置多层安全防护体系，涵盖认证、授权、数据脱敏和输入防御。

### API Key 认证

支持两种认证方式：

```bash
# 方式 1：Bearer Token（推荐）
Authorization: Bearer <your-api-key>

# 方式 2：X-API-Key Header
X-API-Key: <your-api-key>
```

### RBAC 三级角色

| 角色 | 权限 | 说明 |
|------|------|------|
| **admin** | 全部权限 | 系统管理、配置修改、用户管理 |
| **developer** | 开发权限 | API 调用、文档管理、工具使用 |
| **viewer** | 只读权限 | 查询、检索、只读操作 |

### 按用户限流

基于 SlidingWindowRateLimiter 实现用户级限流，配合 IP 限流提供双重保护：

- IP 级限流：SlowAPI 中间件，按 IP 地址限速
- 用户级限流：按 API Key 限速，防止单用户过度消耗资源

### PII 检测与脱敏

自动检测并脱敏以下个人信息：

| PII 类型 | 检测模式 | 示例 |
|----------|---------|------|
| 邮箱 | 正则匹配 | `zhang***@example.com` |
| 手机号 | 正则匹配 | `138****5678` |
| 身份证 | 正则匹配 | `110***********1234` |
| 银行卡 | 正则匹配 | `6222****1234` |
| 护照号 | 正则匹配 | `E1****890` |

> **输入侧**：检测 PII 并记录审计日志，**不脱敏**（保留语义完整性）。
> **输出侧**：检测 PII 并**自动脱敏**，防止敏感信息泄露。

### Prompt Injection 防御

8 种注入模式检测，三级风险评估：

| 风险等级 | 说明 | 处理策略 |
|---------|------|---------|
| **HIGH** | 检测到明确注入攻击 | 拦截请求，记录审计日志 |
| **MEDIUM** | 疑似注入尝试 | 告警，允许请求但标记 |
| **LOW** | 轻微异常 | 记录日志，正常放行 |

### 企业级审计日志

AuditEvent 结构化记录，覆盖全链路：

- API 请求/响应日志
- LLM 调用日志
- 安全事件日志（认证失败、PII 检测、注入防御）
- 工具调用日志

## 📈 可观测性

LangChain 版本提供完整的可观测性支持，涵盖指标采集、结构化日志和链路追踪。

### Prometheus 指标

端点：`/metrics`

四维指标体系：

| 维度 | 指标 | 说明 |
|------|------|------|
| **HTTP 请求** | `agent_http_requests_total` / `agent_http_request_duration_seconds` | 请求计数、延迟分布 |
| **LLM 调用** | `agent_llm_calls_total` / `agent_llm_call_duration_seconds` / `agent_llm_tokens_total` | 调用计数、延迟、Token 用量 |
| **工具调用** | `agent_tool_calls_total` / `agent_tool_call_duration_seconds` | 工具调用计数、延迟 |
| **RAG 检索** | `agent_rag_retrievals_total` / `agent_rag_retrieval_duration_seconds` | 检索计数、延迟 |
| **缓存监控** | `agent_cache_hit_total` / `agent_cache_miss_total` | 缓存命中/未命中计数 |
| **漂移检测** | `agent_drift_alerts_total` / `agent_drift_score` / `agent_drift_status` | 漂移告警计数、漂移分数、检测器状态 |
| **知识图谱** | `agent_kg_triplet_count` / `agent_kg_search_latency` | 三元组计数、图谱检索延迟 |
| **HITL 审批** | `agent_hitl_approval_count` / `agent_hitl_approval_pending` | 审批计数、待审批数量 |
| **Guardrails** | `agent_guardrail_intervention_count` | 死循环防御干预计数 |

> 当 `prometheus_client` 未安装时，所有指标自动降级为空操作（no-op），不影响核心业务功能。

### Grafana Dashboard

开箱即用的 Grafana Dashboard 配置：

```
observability/grafana/dashboard.json
```

导入 Grafana 即可使用，包含：
- API 请求 QPS / 延迟 P50/P95/P99
- LLM 调用成功率 / Token 消耗趋势
- 工具调用分布 / 错误率
- RAG 检索命中率 / 延迟

### 关联 ID 中间件

所有请求自动注入 `X-Correlation-ID`，跨模块日志关联：

```bash
# 请求时自动生成或使用客户端提供的 Correlation ID
X-Correlation-ID: abc123-def456-ghi789
```

### K8s JSON 结构化日志

日志输出为 JSON 格式，适配 K8s 日志采集：

```json
{
  "timestamp": "2026-06-29T10:30:00.000Z",
  "level": "INFO",
  "correlation_id": "abc123-def456",
  "module": "routes_chat",
  "message": "Chat request processed",
  "duration_ms": 1500.0
}
```

## ☸️ K8s 云原生部署

LangChain 版本提供完整的 Kubernetes 部署清单，**纯 CPU 架构，无 GPU 依赖**，适用于标准 K8s 集群。

### 快速部署

```bash
# 1. 创建 Namespace 和 Secret
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml

# 2. 创建 ConfigMap
kubectl apply -f k8s/configmap.yaml

# 3. 部署 API 和 ChromaDB
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/chromadb-deployment.yaml
kubectl apply -f k8s/chromadb-service.yaml

# 4. 配置 HPA 自动扩缩容
kubectl apply -f k8s/api-hpa.yaml

# 5. 配置 Ingress 和 NetworkPolicy
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/networkpolicy.yaml
```

### 资源清单

| 清单文件 | 资源类型 | 说明 |
|---------|---------|------|
| `namespace.yaml` | Namespace | 命名空间隔离 |
| `configmap.yaml` | ConfigMap | 非敏感配置 |
| `secret.yaml` | Secret | 敏感配置（API Key 等） |
| `api-deployment.yaml` | Deployment | API 服务部署 |
| `api-service.yaml` | Service | API 服务暴露 |
| `api-hpa.yaml` | HorizontalPodAutoscaler | 自动扩缩容 |
| `chromadb-deployment.yaml` | StatefulSet | ChromaDB 向量数据库 |
| `chromadb-service.yaml` | Service | ChromaDB 服务暴露 |
| `ingress.yaml` | Ingress | 外部访问入口 + TLS |
| `networkpolicy.yaml` | NetworkPolicy | Egress 出站流量限制 |

### HPA 自动扩缩容

基于 CPU/内存使用率自动扩缩容：

| 参数 | 值 | 说明 |
|------|---|------|
| minReplicas | 2 | 最小副本数 |
| maxReplicas | 6 | 最大副本数 |
| CPU 阈值 | 70% | 平均 CPU 利用率超过 70% 触发扩容 |

### NetworkPolicy 网络隔离

限制 Pod 间网络访问，仅允许必要通信：

- API Pod 仅接受 Ingress 流量
- API Pod 仅访问 ChromaDB Pod
- 禁止 Pod 间非必要通信

> 详细文档请参考 `k8s/README.md`

## 🧩 平台 SDK 与模板

LangChain 版本提供 Python SDK 客户端和标准开发模板，5 分钟接入新能力。

### Python SDK 使用示例

```python
from sdk.agent_platform_client import AgentPlatformClient

# 初始化客户端
client = AgentPlatformClient(
    base_url="http://localhost:8001",
    api_key="your-api-key"
)

# 智能对话
response = client.chat(
    query="公司的报销流程是怎样的？",
    session_id="user-123"
)
print(response["answer"])

# 上传文档
result = client.upload_document(
    file_path="report.pdf",
    department="财务部"
)
print(f"上传成功，切片数：{result['chunks']}")

# 代码审查
review = client.review_code(
    code_content="def login(username, password): ...",
    review_type="full"
)
print(review["summary"])
```

### 工具模板

使用模板快速开发新工具：

```
templates/new_tool_template.py
```

### Provider 模板

使用模板快速接入新 LLM Provider：

```
templates/new_provider_template.py
```

> 详细文档请参考 `PLATFORM_GUIDE.md`

## 🛠️ 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI 0.115+ | 高性能异步框架，自动生成 OpenAPI 文档 |
| **ASGI 服务器** | Uvicorn 0.34+ | 生产级 ASGI 服务器，支持热重载 |
| **Agent 框架** | LangChain 0.3+ / LlamaIndex 0.11+ | 双版本实现 |
| **状态机** | LangGraph | 状态机拓扑 + 多轮对话记忆（MemorySaver）+ 条件路由 |
| **LLM 推理** | NVIDIA NIM / 讯飞星辰 / 智谱AI | 多 Provider 支持，可在线切换 |
| **Embedding** | nv-embedqa-e5-v5 | NVIDIA 云端 Embedding 模型 |
| **Reranker** | NVIDIA Reranker / 智谱AI Reranker | 检索结果精排，不可用时自动降级 |
| **向量数据库** | ChromaDB 0.6+ | 轻量级向量存储与相似度检索 |
| **混合检索** | BM25 + RRF | 关键字检索与向量检索融合排序 |
| **流式推送** | SSE (Server-Sent Events) | 实时状态推送，支持多种事件类型 |
| **HTTP 客户端** | httpx | NVIDIA RAG Blueprint 通信 |
| **中文分词** | jieba | BM25 中文分词支持 |
| **联网搜索** | DuckDuckGo | 外部信息查询 |
| **文档解析** | PyPDF + python-docx + unstructured | 多格式支持 |
| **语义缓存** | InMemoryCache / SQLiteCache | LLM 响应缓存 |
| **认证** | API Key + RBAC | admin/developer/viewer 三级权限 |
| **PII 检测** | 正则模式匹配 | 邮箱/手机/身份证/银行卡/护照 |
| **注入防御** | 模式匹配 + 定界符 | 8 种注入模式检测 |
| **可观测性** | Prometheus + Grafana | HTTP/LLM/工具/RAG 四维指标 |
| **容器编排** | Docker Compose / Kubernetes | 多容器编排，一键部署 / 云原生部署 |
| **限流** | SlowAPI + SlidingWindow | IP 限流 + 用户级限流 |
| **数据验证** | Pydantic 2.10+ | 数据模型验证与序列化 |
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

### 方式三：K8s 部署（LangChain 版本，推荐企业生产环境）

```bash
cd financial-agent-api
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
# 编辑 k8s/secret.yaml 填入 API Key
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/
```

### 方式四：本地开发

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
├── financial-agent-api/           # LangChain 版本（端口 8001）— 企业级 GenAI Agent 平台
│   ├── app/                       # 应用核心代码
│   │   ├── main.py                # FastAPI 入口（CORS、限速、路由挂载、安全中间件）
│   │   ├── exceptions.py          # 自定义异常体系
│   │   ├── api/                   # 路由层
│   │   │   ├── routes_chat.py     # 对话接口（含限速装饰器、SSE 流式端点、MLOps 注入）
│   │   │   ├── routes_docs.py     # 文档管理接口
│   │   │   ├── routes_review.py   # 代码审查接口
│   │   │   ├── routes_mlops.py    # MLOps API 路由（评估/A/B测试/漂移检测/健康检查）
│   │   │   └── routes_hitl.py     # HITL 审批 API 路由
│   │   ├── agent/                 # Agent 核心
│   │   │   ├── graph.py           # LangGraph 状态机核心（条件路由 + 记忆 + HITL + Guardrails）
│   │   │   ├── chain.py           # 对话链与工具调度
│   │   │   ├── hitl_manager.py    # HITL 审批管理器（创建/审批/拒绝/超时清理）
│   │   │   ├── tools.py           # @tool 工具定义
│   │   │   ├── tools/             # 工具子模块
│   │   │   │   ├── kg_tool.py     # 知识图谱检索工具
│   │   │   │   └── recommendation_tool.py  # 推荐工具 + 意图预测工具
│   │   │   └── review/            # 代码审查 Agent
│   │   │       ├── graph.py       # 审查状态图（LangGraph StateGraph）
│   │   │       ├── state.py       # 审查状态定义
│   │   │       ├── supervisor.py  # Supervisor Agent 主控
│   │   │       ├── constants.py   # 审查常量
│   │   │       ├── base_worker.py # Worker 基类
│   │   │       └── workers/       # Worker Agent 实现
│   │   │           ├── security_agent.py
│   │   │           ├── architecture_agent.py
│   │   │           ├── performance_agent.py
│   │   │           ├── style_agent.py
│   │   │           └── summary_agent.py
│   │   ├── security/              # 安全与合规模块
│   │   │   ├── auth.py            # API Key 认证与 RBAC
│   │   │   ├── rate_limiter.py    # 滑动窗口限流器
│   │   │   ├── pii_guard.py       # PII 检测与脱敏
│   │   │   ├── prompt_guard.py    # Prompt Injection 防御
│   │   │   ├── guardrails.py      # 死循环防御护栏（ToolRepetitionDetector）
│   │   │   └── audit.py           # 企业级审计日志
│   │   ├── observability/         # 可观测性模块
│   │   │   ├── metrics.py         # Prometheus 指标定义
│   │   │   ├── middleware.py      # HTTP 指标 + 关联 ID 中间件
│   │   │   └── logging_config.py  # K8s JSON 结构化日志
│   │   ├── core/                  # 核心配置
│   │   │   ├── config.py          # 环境变量配置（多 Provider 支持）
│   │   │   ├── dependencies.py    # 依赖注入
│   │   │   ├── enums.py           # 枚举定义（含审查/MCP/SSE 枚举）
│   │   │   ├── limiter.py         # 限流器实例
│   │   │   └── protocols.py       # 协议定义
│   │   ├── mcp/                   # MCP 协议模块
│   │   │   ├── client.py          # MCP 客户端
│   │   │   ├── registry.py        # MCP 工具注册表
│   │   │   ├── server.py          # MCP Server 抽象基类
│   │   │   └── adapters/          # MCP 适配器
│   │   │       ├── github_adapter.py
│   │   │       ├── filesystem_adapter.py
│   │   │       ├── database_adapter.py
│   │   │       └── websearch_adapter.py
│   │   ├── models/                # 数据模型
│   │   │   └── schemas.py         # Pydantic 模型
│   │   ├── rag/                   # RAG 检索
│   │   │   ├── loader.py          # PDF/TXT/DOCX/MD 加载与切片
│   │   │   ├── vectorstore.py     # ChromaDB 构建/检索（含 KG 三元组提取）
│   │   │   ├── engine_router.py   # 双 RAG 引擎路由（方案A/B 自动切换 + KG 融合）
│   │   │   ├── knowledge_graph.py # 知识图谱管理器（NetworkX + LLM 三元组抽取）
│   │   │   ├── reranker.py        # Reranker 精排封装（NVIDIA / 智谱AI）
│   │   │   └── nvidia_blueprint_client.py  # NVIDIA RAG Blueprint 适配层
│   │   ├── review/                # 审查配置模块
│   │   │   ├── features.py        # 功能开关管理器
│   │   │   └── settings.py        # 审查配置
│   │   ├── mlops/                 # MLOps 模块
│   │   │   ├── tracking.py        # MLflow 实验追踪器（LLMExperimentTracker）
│   │   │   ├── drift_detector.py  # 查询漂移检测器（K-S 检验 / MMD）
│   │   │   ├── evaluator.py       # RAG 四维度评估器（LLM-as-Judge）
│   │   │   └── ab_testing.py      # A/B 测试路由器（session_id 哈希分桶）
│   │   ├── eval/                  # 增强评估框架
│   │   │   └── evaluator.py       # 工具选择/幻觉/图谱/HITL 合规性评估
│   │   └── static/                # 前端静态文件
│   ├── k8s/                       # Kubernetes 部署清单
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   ├── secret.yaml
│   │   ├── api-deployment.yaml
│   │   ├── api-service.yaml
│   │   ├── api-hpa.yaml
│   │   ├── chromadb-deployment.yaml
│   │   ├── chromadb-service.yaml
│   │   ├── ingress.yaml
│   │   ├── networkpolicy.yaml
│   │   └── README.md              # K8s 详细部署文档
│   ├── sdk/                       # 平台 SDK
│   │   ├── agent_platform_client.py  # SDK 客户端
│   │   └── setup.py               # SDK 安装配置
│   ├── templates/                 # 开发者模板
│   │   ├── new_tool_template.py   # 新工具模板
│   │   └── new_provider_template.py  # 新 Provider 模板
│   ├── observability/             # 可观测性配置
│   │   └── grafana/
│   │       └── dashboard.json     # Grafana Dashboard
│   ├── data/                      # 知识库文件目录
│   │   └── knowledge_base/        # PDF/TXT/DOCX/MD 知识库文档
│   ├── chroma_db/                 # 向量数据库本地存储
│   ├── tests/                     # 测试套件
│   │   ├── unit/                  # 单元测试
│   │   │   ├── test_feature_flags.py
│   │   │   ├── test_mcp_client.py
│   │   │   ├── test_supervisor.py
│   │   │   ├── test_workers.py
│   │   │   ├── test_security.py
│   │   │   ├── test_observability.py
│   │   │   ├── test_mlops.py      # MLflow 追踪器测试
│   │   │   ├── test_drift.py      # 漂移检测器测试
│   │   │   ├── test_ab_testing.py # A/B 测试路由器测试
│   │   │   ├── test_knowledge_graph.py  # 知识图谱测试
│   │   │   ├── test_hitl_manager.py     # HITL 管理器测试
│   │   │   └── test_guardrails.py       # 死循环护栏测试
│   │   ├── integration/           # 集成测试
│   │   │   ├── test_review_flow.py
│   │   │   └── test_eval_pipeline.py    # 评估管道集成测试
│   │   ├── e2e/                   # 端到端测试
│   │   │   ├── test_review_e2e.py
│   │   │   ├── test_full_mlops_flow.py  # MLOps 全流程 E2E
│   │   │   └── test_governance_and_hitl.py  # 治理与 HITL E2E
│   │   └── eval/                  # 评估数据集
│   │       ├── eval_dataset.json  # 20 条 QA pairs 评估数据
│   │       ├── tool_selection_dataset.json  # 21 条工具选择测试
│   │       └── hallucination_dataset.json   # 15 条幻觉检测测试
│   ├── scripts/                   # 工具脚本
│   │   └── check_services.py      # 服务健康检查
│   ├── terraform/                 # AWS ECS Fargate IaC 配置
│   │   ├── main.tf                # Terraform 入口 + Provider 配置
│   │   ├── network.tf             # VPC / Subnet / SecurityGroup
│   │   ├── ecs.tf                 # ECS Cluster / Fargate Service / ALB
│   │   ├── iam.tf                 # IAM Role / Policy
│   │   ├── variables.tf           # 变量定义（敏感信息标记 sensitive）
│   │   └── outputs.tf             # 输出定义
│   ├── docker-compose.yml         # 多容器编排配置
│   ├── Dockerfile                 # 应用镜像构建
│   ├── Makefile                   # 一键操作命令
│   ├── pyproject.toml             # 项目配置
│   ├── requirements.txt           # Python 依赖
│   ├── PLATFORM_GUIDE.md          # 开发者平台指南
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
│   │   │   ├── hardware.py        # 硬件检测与模式管理
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
| `RATE_LIMIT_RPM` | `15` | 每分钟请求限制 |
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
| `RETRY_MAX_ATTEMPTS` | `5` | 请求重试次数 |

#### 安全与可观测性配置（LangChain 版本）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `ENABLE_PII_GUARD` | `true` | PII 检测开关 |
| `ENABLE_PROMPT_GUARD` | `true` | Prompt 注入防御开关 |
| `ENABLE_RATE_LIMIT` | `true` | 用户级限流开关 |
| `TEST_API_KEY` | - | 测试用 API Key（仅开发环境） |

#### MCP 与审查配置（LangChain 版本）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MULTI_AGENT_ENABLED` | `false` | 启用多Agent协作代码审查 |
| `MCP_ENABLED` | `false` | 启用 MCP 协议支持 |
| `MCP_GITHUB_ENABLED` | `false` | 启用 GitHub MCP Server |
| `MCP_FILESYSTEM_ENABLED` | `false` | 启用 Filesystem MCP Server |
| `MCP_DATABASE_ENABLED` | `false` | 启用 Database MCP Server |
| `MCP_WEBSEARCH_ENABLED` | `false` | 启用 WebSearch MCP Server |
| `WORKER_TIMEOUT_SECONDS` | `60` | Worker Agent 执行超时时间 |
| `MAX_CONCURRENT_REVIEWS` | `10` | 最大并发审查会话数 |
| `MCP_TOOL_CALL_TIMEOUT` | `30` | MCP 工具调用超时时间 |

#### MLOps 配置（LangChain 版本 v5.0+）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow Server 地址 |
| `MLFLOW_ENABLED` | `false` | MLflow 实验追踪开关 |
| `MLFLOW_EXPERIMENT_NAME` | `financial-agent-rag` | MLflow 实验名称 |
| `MLFLOW_REQUEST_TIMEOUT` | `5` | MLflow 请求超时（秒） |
| `DRIFT_ENABLED` | `true` | 查询漂移检测开关 |
| `DRIFT_DETECTION_METHOD` | `ks_test` | 漂移检测算法（`ks_test` / `mmd`） |
| `DRIFT_THRESHOLD` | `0.05` | 漂移告警阈值 |
| `DRIFT_REFERENCE_DATASET_SIZE` | `100` | 参考数据集大小 |
| `AB_TESTING_ENABLED` | `false` | A/B 测试开关 |
| `AB_BUCKET_A_RATIO` | `0.5` | Bucket A 流量比例 |
| `INTENT_PREDICTION_ENABLED` | `false` | 意图预测开关 |
| `RECOMMENDATION_ENABLED` | `false` | 推荐工具开关 |
| `RECOMMENDATION_TOP_K` | `3` | 推荐返回数量 |

#### Agentic Architecture 配置（LangChain 版本 v5.1+）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `KG_ENABLED` | `false` | 知识图谱功能开关 |
| `KG_MAX_TRIPLETS_PER_DOC` | `50` | 每篇文档最大三元组提取数 |
| `KG_SEARCH_MAX_DEPTH` | `2` | 图谱遍历最大深度 |
| `HITL_ENABLED` | `false` | 人机协同审批开关 |
| `HITL_HIGH_RISK_TOOLS` | `send_email_notification` | 高风险工具列表（逗号分隔） |
| `HITL_APPROVAL_TIMEOUT_SECONDS` | `300` | 审批超时时间（秒） |
| `GUARDRAILS_ENABLED` | `true` | 死循环防护开关 |
| `GUARDRAILS_MAX_TOOL_REPETITION` | `3` | 同一工具连续调用上限 |
| `GUARDRAILS_REPETITION_WINDOW` | `5` | 重复检测滑动窗口大小 |

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
| **Embedding** | `BAAI/bge-small-zh-v1.5` | HuggingFace 本地 Embedding（仅 LlamaIndex 版回退方案） |
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

> **认证说明**（LangChain 版本）：所有 API 接口均需携带 API Key，支持以下两种方式：
> - `Authorization: Bearer <your-api-key>`
> - `X-API-Key: <your-api-key>`

### 对话接口

#### POST `/api/v1/chat`

与 AI 助手对话（支持限流保护）

**请求头**（LangChain 版本）：
```
Authorization: Bearer <your-api-key>
```

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

**请求头**（LangChain 版本）：
```
Authorization: Bearer <your-api-key>
```

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

**请求头**（LangChain 版本）：
```
Authorization: Bearer <your-api-key>
```

**请求**：`multipart/form-data`
- `file`: 文件对象
- `department`: 所属部门（可选，默认"通用"）

**响应体**：
```json
{
  "filename": "report.pdf",
  "chunks_added": 42,
  "message": "文档 report.pdf 已成功上传并入库",
  "department": "财务部",
  "processing_time_ms": 1500.0,
  "acceleration_mode": "cloud_api"
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

### 代码审查接口 🔴 LangChain 版本独有

#### POST `/api/v1/review/code`

提交代码审查请求（支持 SSE 流式输出）

**请求头**：
```
Authorization: Bearer <your-api-key>
```

**请求体**：
```json
{
  "code_content": "def hello(): pass",
  "code_url": "https://github.com/org/repo/pull/1",
  "review_type": "full",
  "session_id": "review-123",
  "stream": false
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `code_content` | string | ❌ | 代码内容（与 code_url 二选一） |
| `code_url` | string | ❌ | 代码仓库 PR 链接（通过 GitHub MCP 获取差异） |
| `review_type` | string | ❌ | 审查类型：full/security/architecture/performance/style（默认 full） |
| `session_id` | string | ❌ | 会话 ID |
| `stream` | bool | ❌ | 是否 SSE 流式输出（默认 false） |

**SSE 事件类型**（`stream=true` 时）：

| 事件类型 | 说明 |
|---------|------|
| `review_start` | 审查开始 |
| `worker_start` | Worker Agent 开始执行 |
| `worker_result` | Worker Agent 审查结果 |
| `worker_timeout` | Worker Agent 执行超时 |
| `worker_error` | Worker Agent 执行异常 |
| `summary_start` | Summary Agent 开始汇总 |
| `review_end` | 审查结束，包含完整审查报告 |

#### GET `/api/v1/review/config`

查询当前审查配置

#### PUT `/api/v1/review/config`

更新审查配置（热更新，无需重启）

#### GET `/api/v1/review/mcp/tools`

查询 MCP 工具列表

#### POST `/api/v1/review/mcp/call`

调用 MCP 工具

### MLOps 接口 🔴 LangChain 版本 v5.0+ 独有

#### POST `/api/v1/mlops/eval`

触发 RAG 评估任务，返回四维度评分 JSON 报告

**响应体**：
```json
{
  "faithfulness_score": 0.85,
  "answer_relevancy_score": 0.92,
  "context_precision_score": 0.78,
  "context_recall_score": 0.81,
  "eval_timestamp": "2026-06-29T10:30:00Z",
  "dataset_version": "v1.0"
}
```

#### GET `/api/v1/mlops/ab-config`

获取当前 A/B 测试策略配置

#### PUT `/api/v1/mlops/ab-config`

动态更新 A/B 测试策略配置（仅限管理员，立即生效无需重启）

#### GET `/api/v1/mlops/ab-metrics`

获取各 Bucket 的性能指标（平均延迟、总请求数等）

#### GET `/api/v1/mlops/drift-status`

获取查询漂移检测状态（检测器状态、漂移分数、告警计数）

#### GET `/api/v1/mlops/health`

MLOps 模块健康检查（MLflow 连接状态、漂移检测器状态、A/B 测试配置状态）

### HITL 审批接口 🔴 LangChain 版本 v5.1+ 独有

#### GET `/api/v1/hitl/approvals`

获取所有待审批项列表

#### GET `/api/v1/hitl/approvals/{approval_id}`

获取指定审批项详情

#### POST `/api/v1/hitl/approvals/{approval_id}/approve`

批准审批（执行被挂起的高风险工具）

#### POST `/api/v1/hitl/approvals/{approval_id}/reject`

拒绝审批（Agent 生成替代话术）

#### GET `/api/v1/hitl/status`

查询 HITL 功能状态（是否启用、高风险工具列表）

### 系统接口

#### GET `/health`

检查服务健康状态

**响应体**：
```json
{
  "status": "ok",
  "acceleration_mode": "cloud_api",
  "vectorstore_count": 128,
  "model_name": "deepseek-ai/deepseek-v4-pro"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 服务状态（`ok` / `error`） |
| `acceleration_mode` | string | 加速模式（`cloud_api`） |
| `vectorstore_count` | int | 向量库文档数量 |
| `model_name` | string | 当前使用的 LLM 模型名称 |

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

### 完整 API 文档

启动服务后访问：
- LangChain 版 Swagger UI: http://localhost:8001/docs
- LlamaIndex 版 Swagger UI: http://localhost:8002/docs

## 🔍 代码审查系统 🔴 LangChain 版本独有

### 多 Agent 协作架构

代码审查系统采用 **Supervisor + Worker + Summary** 三层架构，基于 LangGraph StateGraph 实现：

```
┌──────────────────────────────────────────────────────────────────────┐
│                       代码审查系统架构                                  │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Supervisor Agent（主控）                    │    │
│  │                                                              │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  功能开关校验 → 审查类型路由 → Worker 并行调度 → 超时管理  │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                              │                                       │
│              ┌───────────────┼───────────────┐                      │
│              ↓               ↓               ↓                      │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ │
│  │ SecurityAgent │ │ Architecture  │ │ Performance   │ │  StyleAgent   │ │
│  │   安全审查     │ │   Agent      │ │   Agent       │ │   风格审查     │ │
│  │               │ │   架构审查    │ │   性能审查     │ │               │ │
│  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘ │
│          │                 │                 │                 │         │
│          └─────────────────┴─────────────────┴─────────────────┘         │
│                              │                                       │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Summary Agent（汇总）                       │    │
│  │                                                              │    │
│  │  汇总所有 Worker 结果 → 生成综合审查报告 → 严重度排序            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    MCP 协议层（工具调用）                       │    │
│  │                                                              │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │
│  │  │ GitHub   │ │ 文件系统  │ │ 数据库   │ │ 联网搜索  │       │    │
│  │  │ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │       │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### 审查类型

| 类型 | 值 | 说明 |
|------|-----|------|
| 完整审查 | `full` | 四维度并行审查 + 汇总（默认） |
| 安全审查 | `security` | 仅执行 SecurityAgent |
| 架构审查 | `architecture` | 仅执行 ArchitectureAgent |
| 性能审查 | `performance` | 仅执行 PerformanceAgent |
| 风格审查 | `style` | 仅执行 StyleAgent |

### MCP 协议集成

代码审查系统通过 MCP（Model Context Protocol）协议集成外部工具，支持 4 个适配器：

| 适配器 | 文件 | 功能 |
|--------|------|------|
| GitHub Adapter | `github_adapter.py` | 获取 GitHub PR 差异、文件内容 |
| Filesystem Adapter | `filesystem_adapter.py` | 读取本地文件系统代码文件 |
| Database Adapter | `database_adapter.py` | 查询数据库结构和数据（模拟数据） |
| Websearch Adapter | `websearch_adapter.py` | 联网搜索最佳实践和漏洞信息 |

> 💡 MCP 工具调用前会进行安全校验，确保仅允许已注册的工具被执行。

### 功能开关与热更新

审查系统支持功能开关（FeatureFlags）和配置热更新：

- **multi_agent_enabled**：是否启用多 Agent 协作模式（默认 true）
- **mcp_enabled**：是否启用 MCP 工具调用（默认 true）
- **mcp_servers**：启用的 MCP Server 列表
- **review_types**：支持的审查类型
- **worker_timeout_seconds**：Worker 执行超时时间（默认 60s）
- **max_concurrent_reviews**：最大并发审查数（默认 5）

> 💡 通过 `PUT /api/v1/review/config` 接口可热更新配置，无需重启服务。FeatureFlags 采用线程安全设计，支持配置持久化。

### 降级策略

```
Supervisor 调度成功 → 多 Agent 并行审查 → Summary 汇总
        ↓ 失败
Supervisor 调度失败 → 单 Agent 模式（仅执行指定类型的 Worker）
        ↓ 仍失败
返回错误信息，记录审计日志
```

- Worker 超时时自动跳过，其他 Worker 继续执行
- Supervisor 调度失败时降级为单 Agent 模式
- MCP 工具调用失败时不影响审查流程，仅记录警告

## 🔧 核心工具列表

| 工具名称 | 功能 | LangChain | LlamaIndex |
|---------|------|:---------:|:----------:|
| `search_internal_documents` | 检索内部办公文档（混合检索 + Reranker 精排） | ✅ | ✅ |
| `get_employee_info` | 查询员工信息（演示数据，未接入 OA） | ✅ | ✅ |
| `search_web` | 联网搜索（DuckDuckGo） | ✅ | ✅ |
| `send_email_notification` | 模拟发送邮件通知 | ✅ | ✅ |
| `generate_prd_document` | 生成结构化 PRD 文档 | ✅ | ✅ |
| `generate_flowchart_code` | 生成 Mermaid 流程图代码 | ✅ | ✅ |
| `generate_html_prototype` | 生成 HTML 低保真原型 | ✅ | ✅ |
| `search_knowledge_graph` | 知识图谱实体关系检索 | ✅ | ❌ |
| `recommend_similar_documents` | 推荐相似但未检索到的文档 | ✅ | ❌ |
| `predict_user_intent` | 意图预测（TF-IDF + 朴素贝叶斯） | ✅ | ❌ |
| `search_csv_data` | CSV 数据统计查询 | ❌ | ✅ |

> 💡 LlamaIndex 版本的 `generate_prd_document` 集成了 NVIDIA Blueprints 优化模板，PRD 结构更完整。
> 💡 LangChain 版本的 `search_internal_documents` 支持 Reranker 精排，检索结果更精准。

### 代码审查 Worker Agent 🔴 LangChain 版本独有

| Worker Agent | 功能 | 审查维度 |
|-------------|------|---------|
| `SecurityAgent` | 安全审查 | SQL 注入、XSS、硬编码密钥等 |
| `ArchitectureAgent` | 架构审查 | 设计模式、模块耦合、SOLID 原则 |
| `PerformanceAgent` | 性能审查 | N+1 查询、内存泄漏、算法复杂度 |
| `StyleAgent` | 风格审查 | 命名规范、代码格式、注释质量 |
| `SummaryAgent` | 结果汇总 | 汇总所有 Worker 结果，生成综合报告 |

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

然后在 `chain.py` 中注册：
```python
tools = [search_internal_documents, get_employee_info, get_meeting_room]
```

> 💡 LangChain 版本还需在 `app/security/auth.py` 的 `ROLE_PERMISSIONS` 中配置工具权限，详见 `PLATFORM_GUIDE.md`。

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

### 异常体系

项目采用层级异常设计，所有自定义异常均继承自 `AgentError` 基类：

```
AgentError（基类）
├── ConfigurationError           # 配置错误
├── DocumentLoadError            # 文档加载错误
├── UnsupportedFormatError       # 不支持的格式
├── VectorStoreError             # 向量库错误
├── LLMInvocationError           # LLM 调用错误
├── RateLimitExceededError       # 限流超限
├── ToolExecutionError           # 工具执行错误
├── ServiceConnectionError       # 服务连接错误
├── ValidationError              # 数据验证错误
├── InfiniteLoopDetectedError    # 死循环检测异常（含 tool_name、repetition_count）
└── ReviewError                  # 审查错误
    ├── WorkerTimeoutError           # Worker 执行超时
    ├── WorkerExecutionError         # Worker 执行异常
    ├── SupervisorDispatchError      # Supervisor 调度失败
    ├── MCPConnectionError           # MCP 连接错误
    ├── MCPToolCallError             # MCP 工具调用错误
    ├── MCPToolCallTimeoutError      # MCP 工具调用超时
    ├── MCPAuthenticationError       # MCP 认证失败
    ├── MCPToolNotAllowedError       # MCP 工具未授权
    └── ConfigPersistenceError       # 配置持久化错误
```

### 枚举体系

项目使用枚举类型确保类型安全，主要枚举定义在 `core/enums.py` 中：

| 枚举类型 | 值 | 说明 |
|---------|-----|------|
| `ModelProvider` | `nim`, `xfyun`, `zhipu`, `openai`, `deepseek` | LLM Provider 选择 |
| `ReviewType` | `full`, `security`, `architecture`, `performance`, `style` | 审查类型 |
| `ReviewStatus` | `completed`, `failed`, `timeout` | 审查状态 |
| `Severity` | `critical`, `high`, `medium`, `low`, `info` | 审查结果严重度 |
| `MCPServerName` | `github`, `filesystem`, `database`, `websearch` | MCP Server 名称 |
| `ReviewEventType` | `review_start`, `worker_start`, `worker_result`, `worker_timeout`, `worker_error`, `summary_start`, `review_end` | 审查 SSE 事件类型 |
| `SSEEventType` | `stream_start`, `agent_start`, `agent_end`, `tool_call`, `tool_result`, `tool_error`, `retrieval_result`, `retrieval_empty`, `agent_retry`, `agent_force_end`, `memory_load_failed`, `hitl_approval_required`, `hitl_approved`, `hitl_rejected`, `guardrail_loop_detected`, `stream_end`, `stream_error` | 对话 SSE 事件类型 |
| `DriftDetectionMethod` | `ks_test`, `mmd` | 漂移检测算法 |
| `DriftStatus` | `normal`, `degraded`, `unavailable` | 漂移检测器状态 |
| `ABBucket` | `bucket_a`, `bucket_b` | A/B 测试分桶 |
| `RAGStrategy` | `self_hosted_rag`, `blueprint_rag` | A/B 测试 RAG 策略 |
| `HITLStatus` | `pending`, `approved`, `rejected`, `expired` | HITL 审批状态 |
| `GuardrailAction` | `allow`, `warn`, `block` | 死循环防御动作 |

## 🧪 测试体系 🔴 LangChain 版本独有

LangChain 版本采用 **unit + integration + e2e** 三层测试结构（基于 Pytest），包含安全、可观测性、MLOps 和 Agentic Architecture 测试：

### 测试目录结构

```
financial-agent-api/
└── tests/
    ├── unit/                      # 单元测试
    │   ├── test_feature_flags.py  # 功能开关测试
    │   ├── test_mcp_client.py     # MCP 客户端测试
    │   ├── test_supervisor.py     # Supervisor Agent 测试
    │   ├── test_workers.py        # Worker Agent 测试
    │   ├── test_security.py       # 安全模块测试（PII/注入防御/限流/认证）
    │   ├── test_observability.py  # 可观测性测试（指标/日志）
    │   ├── test_mlops.py          # MLflow 追踪器测试（27 个用例）
    │   ├── test_drift.py          # 漂移检测器测试（33 个用例）
    │   ├── test_ab_testing.py     # A/B 测试路由器测试（31 个用例）
    │   ├── test_knowledge_graph.py  # 知识图谱管理器测试
    │   ├── test_hitl_manager.py     # HITL 审批管理器测试
    │   └── test_guardrails.py       # 死循环护栏测试
    ├── integration/               # 集成测试
    │   ├── test_review_flow.py    # 审查流程集成测试
    │   └── test_eval_pipeline.py  # 评估管道集成测试（15 个用例）
    ├── e2e/                       # 端到端测试
    │   ├── test_review_e2e.py     # 端到端审查测试
    │   ├── test_full_mlops_flow.py    # MLOps 全流程 E2E 测试（17 个用例）
    │   └── test_governance_and_hitl.py  # 治理与 HITL E2E 测试（11 个用例）
    └── eval/                      # 评估数据集
        ├── eval_dataset.json      # 20 条 QA pairs 评估数据
        ├── tool_selection_dataset.json  # 21 条工具选择测试
        └── hallucination_dataset.json   # 15 条幻觉检测测试
```

### 测试层级说明

| 层级 | 范围 | 说明 |
|------|------|------|
| **unit** | 单个模块/函数 | 功能开关、MCP 客户端、Supervisor 调度逻辑、Worker 审查逻辑、PII 检测、Prompt 注入防御、限流器、认证、可观测性指标/日志、MLflow 追踪器、漂移检测器、A/B 测试路由器、知识图谱管理器、HITL 审批管理器、死循环护栏 |
| **integration** | 模块间协作 | 审查流程集成测试、评估管道集成测试 |
| **e2e** | 完整系统 | 端到端审查测试、MLOps 全流程测试、治理与 HITL 测试 |

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 仅运行单元测试
pytest tests/unit/

# 仅运行集成测试
pytest tests/integration/

# 仅运行端到端测试
pytest tests/e2e/

# 运行指定测试文件
pytest tests/unit/test_security.py -v
```

## 🚢 部署指南

### K8s 部署（LangChain 版本，推荐企业生产环境）

K8s 是企业生产环境推荐的部署方式，提供自动扩缩容、网络隔离、滚动更新等企业级特性。

#### 1. 准备工作

```bash
# 确保 kubectl 已配置正确的集群
kubectl cluster-info

cd financial-agent-api
```

#### 2. 配置 Secret

```bash
# 编辑 k8s/secret.yaml，填入 API Key 等敏感信息
# 或使用 kubectl create secret 命令
kubectl create secret generic agent-platform-secrets \
  --from-literal=NVIDIA_API_KEY=your_api_key \
  -n agent-platform
```

#### 3. 部署所有资源

```bash
# 一键部署
kubectl apply -f k8s/

# 或逐个部署
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/api-hpa.yaml
kubectl apply -f k8s/chromadb-deployment.yaml
kubectl apply -f k8s/chromadb-service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/networkpolicy.yaml
```

#### 4. 验证部署

```bash
# 查看 Pod 状态
kubectl get pods -n agent-platform

# 查看服务
kubectl get svc -n agent-platform

# 查看 HPA 状态
kubectl get hpa -n agent-platform

# 查看日志
kubectl logs -f deployment/api -n agent-platform
```

> 详细文档请参考 `k8s/README.md`

### Docker Compose 部署（LangChain 版本，推荐开发环境）

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

## 🔒 安全建议

1. **API Key 认证**：
   - API Key 认证已内置（LangChain 版本），生产环境务必配置
   - 不要将 API Key 提交到版本控制
   - 使用环境变量或 K8s Secret 管理密钥

2. **PII 检测**：
   - PII 检测默认启用，可按需关闭
   - 生产环境建议保持启用，满足数据保护合规要求

3. **Prompt Injection 防御**：
   - Prompt Injection 防御默认启用
   - 建议保持启用，防止恶意 Prompt 注入攻击

4. **审计日志**：
   - 审计日志覆盖全链路，满足合规要求
   - 建议将审计日志持久化存储，定期审查

5. **限流保护**：
   - 生产环境建议降低 `RATE_LIMIT_RPM`
   - 启用用户级限流，防止单用户过度消耗资源

6. **网络安全**：
   - 使用 HTTPS
   - 配置防火墙规则
   - K8s NetworkPolicy 实现网络隔离
   - 限制容器网络访问

7. **数据安全**：
   - 定期备份 `chroma_db/` 和 `data/`
   - 敏感文档考虑加密存储

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
  -H "Authorization: Bearer <your-api-key>" \
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

### Q15: 如何配置 API Key 认证？

**A**: 设置 `TEST_API_KEY` 环境变量（开发环境）或通过 K8s Secret 注入（生产环境）：

```bash
# 开发环境
export TEST_API_KEY="my-test-api-key"

# 生产环境（K8s）
kubectl create secret generic agent-platform-secrets \
  --from-literal=API_KEYS='[{"key":"admin-key","role":"admin"},{"key":"dev-key","role":"developer"}]' \
  -n agent-platform
```

### Q16: 如何关闭 PII 检测？

**A**: 设置环境变量：

```bash
ENABLE_PII_GUARD=false
```

### Q17: 如何关闭 Prompt 注入防御？

**A**: 设置环境变量：

```bash
ENABLE_PROMPT_GUARD=false
```

### Q18: 如何部署到 K8s？

**A**: 参考 `k8s/README.md`，或使用以下快速部署命令：

```bash
kubectl apply -f k8s/
```

### Q19: 如何启用多Agent代码审查功能？

**A**: 设置环境变量或通过 API 启用：

```bash
# 方式 1：环境变量
MULTI_AGENT_ENABLED=true
MCP_ENABLED=true

# 方式 2：API 热更新
curl -X PUT http://localhost:8001/api/v1/review/config \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"multi_agent": true, "mcp": true}'
```

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

**Made with ❤️ by Enterprise GenAI Agent Platform Team**
