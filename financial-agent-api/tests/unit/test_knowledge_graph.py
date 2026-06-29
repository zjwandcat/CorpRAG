"""KnowledgeGraphManager 单元测试

覆盖：
- 三元组提取（extract_triplets）
- 三元组添加（add_triplets）
- 图谱检索（search）
- 遍历深度限制
- 图谱持久化与加载
- 统计信息
- LLM 调用失败降级
- 空文本处理
- JSON 解析容错
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.rag.knowledge_graph import KnowledgeGraphManager


class TestKnowledgeGraphManager:
    """KnowledgeGraphManager 核心功能测试"""

    @pytest.fixture
    def kg_manager(self, tmp_path: pytest.TempPathFactory) -> KnowledgeGraphManager:
        """创建使用临时目录的 KnowledgeGraphManager 实例"""
        with patch("app.rag.knowledge_graph.KG_TRIPLET_COUNT", new=MagicMock()), \
             patch("app.rag.knowledge_graph.KG_SEARCH_LATENCY", new=MagicMock()):
            return KnowledgeGraphManager(storage_path=str(tmp_path))

    # =====================================================================
    # extract_triplets 测试
    # =====================================================================

    def test_extract_triplets_success(self, kg_manager: KnowledgeGraphManager) -> None:
        """三元组提取正确性"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '[{"subject": "张三", "relation": "隶属", "object": "研发中心"}]'
        mock_llm.invoke.return_value = mock_response

        result = kg_manager.extract_triplets("张三隶属于研发中心", mock_llm)
        assert len(result) == 1
        assert result[0]["subject"] == "张三"
        assert result[0]["relation"] == "隶属"
        assert result[0]["object"] == "研发中心"

    def test_extract_triplets_multiple(self, kg_manager: KnowledgeGraphManager) -> None:
        """多三元组提取"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '[{"subject": "张三", "relation": "隶属", "object": "研发中心"},'
            '{"subject": "张三", "relation": "负责", "object": "项目A"}]'
        )
        mock_llm.invoke.return_value = mock_response

        result = kg_manager.extract_triplets("张三隶属于研发中心并负责项目A", mock_llm)
        assert len(result) == 2

    def test_extract_triplets_llm_failure(self, kg_manager: KnowledgeGraphManager) -> None:
        """LLM 调用失败时返回空列表"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM 不可用")
        result = kg_manager.extract_triplets("测试文本", mock_llm)
        assert result == []

    def test_extract_triplets_empty_text(self, kg_manager: KnowledgeGraphManager) -> None:
        """空文本返回空列表"""
        mock_llm = MagicMock()
        result = kg_manager.extract_triplets("", mock_llm)
        assert result == []
        mock_llm.invoke.assert_not_called()

    def test_extract_triplets_whitespace_text(self, kg_manager: KnowledgeGraphManager) -> None:
        """纯空白文本返回空列表"""
        mock_llm = MagicMock()
        result = kg_manager.extract_triplets("   \n\t  ", mock_llm)
        assert result == []
        mock_llm.invoke.assert_not_called()

    def test_extract_triplets_markdown_wrapped(self, kg_manager: KnowledgeGraphManager) -> None:
        """LLM 返回 markdown 代码块包裹的 JSON"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '```json\n'
            '[{"subject": "A", "relation": "r", "object": "B"}]\n'
            '```'
        )
        mock_llm.invoke.return_value = mock_response

        result = kg_manager.extract_triplets("测试", mock_llm)
        assert len(result) == 1
        assert result[0]["subject"] == "A"

    def test_extract_triplets_invalid_json(self, kg_manager: KnowledgeGraphManager) -> None:
        """LLM 返回无效 JSON 时返回空列表"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "这不是JSON格式"
        mock_llm.invoke.return_value = mock_response

        result = kg_manager.extract_triplets("测试", mock_llm)
        assert result == []

    def test_extract_triplets_missing_keys(self, kg_manager: KnowledgeGraphManager) -> None:
        """三元组缺少必要字段时被过滤"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '[{"subject": "A", "relation": "r"},'
            '{"subject": "B", "relation": "r2", "object": "C"}]'
        )
        mock_llm.invoke.return_value = mock_response

        result = kg_manager.extract_triplets("测试", mock_llm)
        assert len(result) == 1
        assert result[0]["subject"] == "B"

    def test_extract_triplets_max_limit(self, kg_manager: KnowledgeGraphManager) -> None:
        """三元组数量超过 max_triplets_per_doc 时截断"""
        # 创建一个 max_triplets_per_doc=2 的管理器
        with patch("app.rag.knowledge_graph.KG_TRIPLET_COUNT", new=MagicMock()), \
             patch("app.rag.knowledge_graph.KG_SEARCH_LATENCY", new=MagicMock()):
            limited_manager = KnowledgeGraphManager(
                storage_path=str(kg_manager._storage_path),
                max_triplets_per_doc=2,
            )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '[{"subject": "A", "relation": "r1", "object": "B"},'
            '{"subject": "C", "relation": "r2", "object": "D"},'
            '{"subject": "E", "relation": "r3", "object": "F"}]'
        )
        mock_llm.invoke.return_value = mock_response

        result = limited_manager.extract_triplets("测试", mock_llm)
        assert len(result) == 2

    # =====================================================================
    # add_triplets 测试
    # =====================================================================

    def test_add_triplets_success(self, kg_manager: KnowledgeGraphManager) -> None:
        """三元组添加到图谱"""
        triplets = [{"subject": "张三", "relation": "隶属", "object": "研发中心"}]
        count = kg_manager.add_triplets(triplets, source="test_doc")
        assert count == 1

    def test_add_triplets_empty_list(self, kg_manager: KnowledgeGraphManager) -> None:
        """空三元组列表返回 0"""
        count = kg_manager.add_triplets([], source="test_doc")
        assert count == 0

    def test_add_triplets_incomplete_entry(self, kg_manager: KnowledgeGraphManager) -> None:
        """不完整的三元组被跳过"""
        triplets = [
            {"subject": "", "relation": "r", "object": "B"},  # 空 subject
            {"subject": "A", "relation": "", "object": "B"},  # 空 relation
            {"subject": "A", "relation": "r", "object": ""},  # 空 object
            {"subject": "A", "relation": "r", "object": "B"},  # 有效
        ]
        count = kg_manager.add_triplets(triplets)
        assert count == 1

    def test_add_triplets_duplicate_edge(self, kg_manager: KnowledgeGraphManager) -> None:
        """重复三元组会覆盖边属性但仍计数"""
        triplets = [
            {"subject": "A", "relation": "r", "object": "B"},
            {"subject": "A", "relation": "r", "object": "B"},
        ]
        count = kg_manager.add_triplets(triplets, source="test")
        assert count == 2

    # =====================================================================
    # search 测试
    # =====================================================================

    def test_search_by_entity(self, kg_manager: KnowledgeGraphManager) -> None:
        """按实体搜索"""
        kg_manager.add_triplets([{"subject": "张三", "relation": "隶属", "object": "研发中心"}])
        results = kg_manager.search("张三")
        assert len(results) >= 1
        assert results[0].entity == "张三"

    def test_search_by_relation(self, kg_manager: KnowledgeGraphManager) -> None:
        """按关系搜索"""
        kg_manager.add_triplets([
            {"subject": "张三", "relation": "隶属", "object": "研发中心"},
            {"subject": "张三", "relation": "负责", "object": "项目A"},
        ])
        results = kg_manager.search("张三", relation="隶属")
        assert all(r.relation == "隶属" for r in results)

    def test_search_nonexistent_entity(self, kg_manager: KnowledgeGraphManager) -> None:
        """搜索不存在的实体返回空列表"""
        results = kg_manager.search("不存在的实体")
        assert results == []

    def test_search_empty_entity(self, kg_manager: KnowledgeGraphManager) -> None:
        """空实体名返回空列表"""
        results = kg_manager.search("")
        assert results == []

    def test_search_max_depth(self, kg_manager: KnowledgeGraphManager) -> None:
        """遍历深度限制"""
        kg_manager.add_triplets([
            {"subject": "A", "relation": "r1", "object": "B"},
            {"subject": "B", "relation": "r2", "object": "C"},
            {"subject": "C", "relation": "r3", "object": "D"},
        ])
        results_depth1 = kg_manager.search("A", max_depth=1)
        results_depth2 = kg_manager.search("A", max_depth=2)
        results_depth3 = kg_manager.search("A", max_depth=3)
        assert len(results_depth1) < len(results_depth2)
        assert len(results_depth2) <= len(results_depth3)

    def test_search_default_depth(self, kg_manager: KnowledgeGraphManager) -> None:
        """默认深度使用配置值"""
        kg_manager.add_triplets([
            {"subject": "A", "relation": "r1", "object": "B"},
            {"subject": "B", "relation": "r2", "object": "C"},
        ])
        # 默认 search_max_depth=2，应能检索到两层
        results = kg_manager.search("A")
        assert len(results) >= 1

    # =====================================================================
    # persist / load 测试
    # =====================================================================

    def test_persist_and_load(self, tmp_path: pytest.TempPathFactory) -> None:
        """图谱持久化和加载"""
        with patch("app.rag.knowledge_graph.KG_TRIPLET_COUNT", new=MagicMock()), \
             patch("app.rag.knowledge_graph.KG_SEARCH_LATENCY", new=MagicMock()):
            kg_manager = KnowledgeGraphManager(storage_path=str(tmp_path))
            kg_manager.add_triplets([{"subject": "X", "relation": "r", "object": "Y"}])
            kg_manager.persist()

            kg_manager2 = KnowledgeGraphManager(storage_path=str(tmp_path))
            results = kg_manager2.search("X")
            assert len(results) >= 1

    def test_persist_creates_directory(self, tmp_path: pytest.TempPathFactory) -> None:
        """持久化时自动创建目录"""
        nested_path = tmp_path / "nested" / "dir"
        with patch("app.rag.knowledge_graph.KG_TRIPLET_COUNT", new=MagicMock()), \
             patch("app.rag.knowledge_graph.KG_SEARCH_LATENCY", new=MagicMock()):
            kg_manager = KnowledgeGraphManager(storage_path=str(nested_path))
            kg_manager.add_triplets([{"subject": "A", "relation": "r", "object": "B"}])
            kg_manager.persist()

            assert (nested_path / "knowledge_graph.json").exists()

    # =====================================================================
    # get_stats 测试
    # =====================================================================

    def test_get_stats_empty(self, kg_manager: KnowledgeGraphManager) -> None:
        """空图谱统计信息"""
        stats = kg_manager.get_stats()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0

    def test_get_stats_with_data(self, kg_manager: KnowledgeGraphManager) -> None:
        """有数据时统计信息正确性"""
        kg_manager.add_triplets([{"subject": "A", "relation": "r", "object": "B"}])
        stats = kg_manager.get_stats()
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["triplet_count"] == 1

    def test_get_stats_shared_node(self, kg_manager: KnowledgeGraphManager) -> None:
        """共享节点的统计信息"""
        kg_manager.add_triplets([
            {"subject": "A", "relation": "r1", "object": "B"},
            {"subject": "A", "relation": "r2", "object": "C"},
        ])
        stats = kg_manager.get_stats()
        assert stats["node_count"] == 3  # A, B, C
        assert stats["edge_count"] == 2