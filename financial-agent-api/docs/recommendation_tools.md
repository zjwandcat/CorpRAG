# v5.0 MLOps 升级 - 推荐工具和意图预测工具

## 概述

本次升级新增两个 MLOps 工具，用于提升 RAG 系统的智能化水平：

1. **推荐工具 (RecommendationTool)**：推荐相似但未在当前 RAG 上下文中的文档
2. **意图预测工具 (IntentPredictionTool)**：预测用户查询的意图类型

## 推荐工具 (RecommendationTool)

### 功能说明

推荐工具用于在用户查询后，推荐相关但尚未检索到的文档，扩展用户视野。

### 工厂函数

```python
from app.agent.tools import make_recommend_similar_documents_tool

tool = make_recommend_similar_documents_tool(
    vectorstore=vectorstore,
    embeddings=embeddings,
)
```

### 参数说明

- `vectorstore`: ChromaDB 向量存储实例
- `embeddings`: Embeddings 实例（必须使用云端 API，通过 `get_embeddings()` 获取）

### 工具参数

- `query`: 用户查询文本
- `exclude_ids`: 需要排除的文档 ID 列表（当前 RAG 上下文中的文档）
- `top_k`: 返回的推荐文档数量，默认使用配置中的 `RECOMMENDATION_TOP_K`

### 返回值

返回 `list[RecommendationItem]`，每个条目包含：
- `document_id`: 文档 ID
- `content`: 文档内容
- `source`: 来源
- `similarity_score`: 相似度得分（0-1）

### 性能约束

- 执行时间 < 3 秒
- 异常时优雅降级，返回空结果

### 配置项

```python
# app/core/config.py
RECOMMENDATION_ENABLED: bool = False  # 是否启用推荐功能
RECOMMENDATION_TOP_K: int = 3         # 推荐文档数量
```

## 意图预测工具 (IntentPredictionTool)

### 功能说明

意图预测工具使用预训练的 TF-IDF + 朴素贝叶斯模型进行意图分类，不调用 LLM API。

### 工厂函数

```python
from app.agent.tools import make_predict_user_intent_tool

tool = make_predict_user_intent_tool(model_path="./intent_model.pkl")
```

### 参数说明

- `model_path`: 预训练模型文件路径，默认使用配置中的 `INTENT_MODEL_PATH`

### 工具参数

- `query`: 用户查询文本

### 返回值

返回 `IntentPredictionResult`，包含：
- `intent_label`: 意图标签（见 `IntentLabel` 枚举）
- `confidence`: 置信度（0-1）

如果模型不可用或预测失败，返回 `None`。

### 性能约束

- 执行时间 < 500ms
- 异常时优雅降级，返回 `None`

### 配置项

```python
# app/core/config.py
INTENT_PREDICTION_ENABLED: bool = False  # 是否启用意图预测
INTENT_MODEL_PATH: str = "./intent_model.pkl"  # 模型文件路径
```

### 意图标签

```python
# app/core/enums.py
class IntentLabel(StrEnum):
    REIMBURSEMENT = "报销咨询"
    LEAVE_PROCESS = "请假流程"
    IT_SUPPORT = "IT支持"
    HR_MANAGEMENT = "人事管理"
    FINANCIAL_MANAGEMENT = "财务管理"
    OTHER = "其他"
```

### 模型训练示例

```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# 训练数据
queries = [
    "我想报销差旅费",
    "如何申请报销",
    "请假流程是什么",
    "我想请病假",
    # ... 更多数据
]
labels = [
    "报销咨询",
    "报销咨询",
    "请假流程",
    "请假流程",
    # ... 更多标签
]

# 训练模型
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(queries)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

classifier = MultinomialNB()
classifier.fit(X, y)

# 保存模型
model_data = {
    "vectorizer": vectorizer,
    "classifier": classifier,
    "label_encoder": label_encoder,
}

with open("intent_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
```

## 使用示例

### 推荐工具

```python
from langchain_chroma import Chroma
from app.core.dependencies import get_embeddings, get_vectorstore
from app.agent.tools import make_recommend_similar_documents_tool

# 获取向量存储和 Embeddings
vectorstore = get_vectorstore()
embeddings = get_embeddings()

# 创建工具
tool = make_recommend_similar_documents_tool(vectorstore, embeddings)

# 调用工具
result = tool.invoke({
    "query": "报销流程",
    "exclude_ids": ["doc1", "doc2"],  # 排除当前 RAG 上下文
    "top_k": 3,
})

for item in result:
    print(f"文档: {item.source}, 相似度: {item.similarity_score:.2f}")
```

### 意图预测工具

```python
from app.agent.tools import make_predict_user_intent_tool

# 创建工具
tool = make_predict_user_intent_tool()

# 调用工具
result = tool.invoke({"query": "我想报销差旅费"})

if result:
    print(f"意图: {result.intent_label}, 置信度: {result.confidence:.2f}")
else:
    print("意图预测不可用")
```

## 验收标准

- ✅ 推荐工具可排除当前 RAG 上下文，返回相似但未检索的文档
- ✅ 意图预测工具可返回意图标签和置信度
- ✅ 两个工具均注册为 LangChain @tool
- ✅ 推荐工具延迟 < 3 秒
- ✅ 意图预测延迟 < 500ms
- ✅ 异常时优雅降级

## 测试

```bash
# 运行单元测试
pytest tests/test_recommendation_tool.py -v

# 运行演示
python examples/demo_recommendation_tools.py
```

## 注意事项

1. **推荐工具**必须使用云端 Embedding API（通过 `get_embeddings()` 获取），禁止本地 Embedding
2. **意图预测工具**使用 TF-IDF + 朴素贝叶斯，不调用 LLM API
3. 两个工具都有性能约束，超时会记录 warning 日志
4. 异常时优雅降级，不影响主流程