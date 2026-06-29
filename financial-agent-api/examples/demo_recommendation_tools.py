"""推荐工具和意图预测工具使用示例

演示 v5.0 MLOps 升级的推荐工具和意图预测工具的基本用法。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock

from app.agent.tools import (
    make_recommend_similar_documents_tool,
    make_predict_user_intent_tool,
)
from app.core.config import settings


def demo_recommendation_tool():
    """演示推荐工具的基本用法"""
    print("=" * 60)
    print("推荐工具演示")
    print("=" * 60)

    # 创建模拟的 vectorstore 和 embeddings
    mock_vectorstore = MagicMock()
    mock_embeddings = MagicMock()

    # 创建工具
    tool = make_recommend_similar_documents_tool(
        vectorstore=mock_vectorstore,
        embeddings=mock_embeddings,
    )

    print(f"工具名称: {tool.name}")
    print(f"工具描述: {tool.description[:100]}...")
    print(f"配置 - RECOMMENDATION_ENABLED: {settings.RECOMMENDATION_ENABLED}")
    print(f"配置 - RECOMMENDATION_TOP_K: {settings.RECOMMENDATION_TOP_K}")
    print()


def demo_intent_prediction_tool():
    """演示意图预测工具的基本用法"""
    print("=" * 60)
    print("意图预测工具演示")
    print("=" * 60)

    # 创建工具
    tool = make_predict_user_intent_tool()

    print(f"工具名称: {tool.name}")
    print(f"工具描述: {tool.description[:100]}...")
    print(f"配置 - INTENT_PREDICTION_ENABLED: {settings.INTENT_PREDICTION_ENABLED}")
    print(f"配置 - INTENT_MODEL_PATH: {settings.INTENT_MODEL_PATH}")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("v5.0 MLOps 升级 - 推荐工具和意图预测工具")
    print("=" * 60)
    print()

    demo_recommendation_tool()
    demo_intent_prediction_tool()

    print("=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()