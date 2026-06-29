"""Worker Agent 执行逻辑单元测试

测试 Security/Architecture/Performance/Style 四个 Worker Agent 的审查逻辑。
所有 LLM 调用均通过 Mock 模拟，不依赖真实 LLM 服务。
"""

import json
from unittest.mock import MagicMock

import pytest

from app.agent.review.workers.architecture_agent import ArchitectureAgent
from app.agent.review.workers.performance_agent import PerformanceAgent
from app.agent.review.workers.security_agent import SecurityAgent
from app.agent.review.workers.style_agent import StyleAgent
from app.core.enums import ReviewStatus, ReviewType, Severity
from app.models.schemas import WorkerResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MagicMock:
    """创建 Mock LLM 实例"""
    llm = MagicMock()
    return llm


@pytest.fixture
def sql_injection_code() -> str:
    """包含 SQL 注入漏洞的示例代码"""
    return """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
"""


@pytest.fixture
def god_class_code() -> str:
    """违反单一职责原则的代码（God Class）"""
    return """
class UserManager:
    def create_user(self, name, email):
        pass
    def delete_user(self, user_id):
        pass
    def send_email(self, to, subject, body):
        pass
    def generate_report(self, user_id):
        pass
    def authenticate(self, username, password):
        pass
"""


@pytest.fixture
def n_plus_one_query_code() -> str:
    """包含 N+1 查询问题的代码"""
    return """
def get_users_with_orders():
    users = db.query("SELECT * FROM users")
    for user in users:
        user.orders = db.query(f"SELECT * FROM orders WHERE user_id = {user.id}")
    return users
"""


@pytest.fixture
def bad_naming_code() -> str:
    """命名不规范的代码"""
    return """
def GetData(x, y):
    result = CalculateTotal(x, y)
    return result

def CalculateTotal(a, b):
    TOTAL = a + b
    return TOTAL
"""


# ---------------------------------------------------------------------------
# Security Agent 测试
# ---------------------------------------------------------------------------


class TestSecurityAgent:
    """Security Agent 安全审查测试"""

    def test_security_agent_sql_injection(
        self, mock_llm: MagicMock, sql_injection_code: str
    ) -> None:
        """Security Agent 检测 SQL 注入

        验证：Security Agent 能识别代码中的 SQL 注入漏洞并返回 critical 级别发现
        """
        # Mock LLM 返回包含 SQL 注入发现的 JSON
        llm_response = json.dumps(
            {
                "findings": [
                    {
                        "severity": "critical",
                        "description": "SQL 注入漏洞：使用字符串拼接构建 SQL 查询",
                        "location": "get_user 函数",
                        "suggestion": "使用参数化查询替代字符串拼接",
                    }
                ]
            }
        )
        response = MagicMock()
        response.content = llm_response
        mock_llm.invoke.return_value = response

        agent = SecurityAgent(llm=mock_llm)
        result = agent.execute(sql_injection_code)

        assert isinstance(result, WorkerResult)
        assert result.dimension == ReviewType.SECURITY
        assert result.status == ReviewStatus.COMPLETED
        assert len(result.findings) >= 1
        # 验证检测到 SQL 注入
        assert any("SQL" in f.description or "注入" in f.description for f in result.findings)
        # 验证严重程度为 critical
        assert any(f.severity == Severity.CRITICAL for f in result.findings)

    def test_security_agent_no_issues(self, mock_llm: MagicMock) -> None:
        """Security Agent 对安全代码返回空发现列表

        验证：安全代码审查后 findings 为空
        """
        safe_code = """
def get_user(user_id: int) -> dict:
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()
"""
        llm_response = json.dumps({"findings": []})
        response = MagicMock()
        response.content = llm_response
        mock_llm.invoke.return_value = response

        agent = SecurityAgent(llm=mock_llm)
        result = agent.execute(safe_code)

        assert result.status == ReviewStatus.COMPLETED
        assert len(result.findings) == 0

    def test_security_agent_dimension(self, mock_llm: MagicMock) -> None:
        """Security Agent 维度标识为 security"""
        agent = SecurityAgent(llm=mock_llm)
        assert agent.dimension == "security"

    def test_security_agent_parse_json_with_code_block(self, mock_llm: MagicMock) -> None:
        """Security Agent 能解析包含 ```json 代码块的 LLM 返回"""
        llm_response = """```json
{
    "findings": [
        {
            "severity": "high",
            "description": "硬编码密码",
            "location": "config.py:10",
            "suggestion": "使用环境变量"
        }
    ]
}
```"""
        response = MagicMock()
        response.content = llm_response
        mock_llm.invoke.return_value = response

        agent = SecurityAgent(llm=mock_llm)
        result = agent.execute("password = '123456'")

        assert result.status == ReviewStatus.COMPLETED
        assert len(result.findings) == 1
        assert result.findings[0].severity == Severity.HIGH


# ---------------------------------------------------------------------------
# Architecture Agent 测试
# ---------------------------------------------------------------------------


class TestArchitectureAgent:
    """Architecture Agent 架构审查测试"""

    def test_architecture_agent_pattern(self, mock_llm: MagicMock, god_class_code: str) -> None:
        """Architecture Agent 检测设计模式违规

        验证：Architecture Agent 能识别违反单一职责原则的 God Class
        """
        llm_response = json.dumps(
            {
                "findings": [
                    {
                        "severity": "high",
                        "description": "违反单一职责原则：UserManager 类承担了过多职责",
                        "location": "UserManager 类",
                        "suggestion": "将邮件发送、报告生成、认证逻辑拆分为独立的类",
                    }
                ]
            }
        )
        response = MagicMock()
        response.content = llm_response
        mock_llm.invoke.return_value = response

        agent = ArchitectureAgent(llm=mock_llm)
        result = agent.execute(god_class_code)

        assert isinstance(result, WorkerResult)
        assert result.dimension == ReviewType.ARCHITECTURE
        assert result.status == ReviewStatus.COMPLETED
        assert len(result.findings) >= 1
        # 验证检测到 SRP 违规
        assert any(
            "单一职责" in f.description or "SRP" in f.description or "职责" in f.description
            for f in result.findings
        )

    def test_architecture_agent_dimension(self, mock_llm: MagicMock) -> None:
        """Architecture Agent 维度标识为 architecture"""
        agent = ArchitectureAgent(llm=mock_llm)
        assert agent.dimension == "architecture"


# ---------------------------------------------------------------------------
# Performance Agent 测试
# ---------------------------------------------------------------------------


class TestPerformanceAgent:
    """Performance Agent 性能审查测试"""

    def test_performance_agent_bottleneck(
        self, mock_llm: MagicMock, n_plus_one_query_code: str
    ) -> None:
        """Performance Agent 检测性能瓶颈

        验证：Performance Agent 能识别 N+1 查询问题
        """
        llm_response = json.dumps(
            {
                "findings": [
                    {
                        "severity": "high",
                        "description": "N+1 查询问题：循环中执行数据库查询",
                        "location": "get_users_with_orders 函数",
                        "suggestion": "使用 JOIN 或预加载批量获取关联数据",
                    }
                ]
            }
        )
        response = MagicMock()
        response.content = llm_response
        mock_llm.invoke.return_value = response

        agent = PerformanceAgent(llm=mock_llm)
        result = agent.execute(n_plus_one_query_code)

        assert isinstance(result, WorkerResult)
        assert result.dimension == ReviewType.PERFORMANCE
        assert result.status == ReviewStatus.COMPLETED
        assert len(result.findings) >= 1
        # 验证检测到 N+1 查询
        assert any(
            "N+1" in f.description or "查询" in f.description or "循环" in f.description
            for f in result.findings
        )

    def test_performance_agent_dimension(self, mock_llm: MagicMock) -> None:
        """Performance Agent 维度标识为 performance"""
        agent = PerformanceAgent(llm=mock_llm)
        assert agent.dimension == "performance"


# ---------------------------------------------------------------------------
# Style Agent 测试
# ---------------------------------------------------------------------------


class TestStyleAgent:
    """Style Agent 风格审查测试"""

    def test_style_agent_naming(self, mock_llm: MagicMock, bad_naming_code: str) -> None:
        """Style Agent 检测命名规范

        验证：Style Agent 能识别不符合 PEP8 的命名规范（PascalCase 函数名）
        """
        llm_response = json.dumps(
            {
                "findings": [
                    {
                        "severity": "medium",
                        "description": "函数名应使用 snake_case 而非 PascalCase：GetData → get_data",
                        "location": "GetData 函数",
                        "suggestion": "将函数名改为 snake_case 格式",
                    },
                    {
                        "severity": "low",
                        "description": "常量 TOTAL 应使用 UPPER_SNAKE_CASE 格式",
                        "location": "CalculateTotal 函数内 TOTAL 变量",
                        "suggestion": "将变量名改为 upper_snake_case 或 snake_case",
                    },
                ]
            }
        )
        response = MagicMock()
        response.content = llm_response
        mock_llm.invoke.return_value = response

        agent = StyleAgent(llm=mock_llm)
        result = agent.execute(bad_naming_code)

        assert isinstance(result, WorkerResult)
        assert result.dimension == ReviewType.STYLE
        assert result.status == ReviewStatus.COMPLETED
        assert len(result.findings) >= 1
        # 验证检测到命名问题
        assert any(
            "snake_case" in f.description
            or "命名" in f.description
            or "PascalCase" in f.description
            for f in result.findings
        )

    def test_style_agent_dimension(self, mock_llm: MagicMock) -> None:
        """Style Agent 维度标识为 style"""
        agent = StyleAgent(llm=mock_llm)
        assert agent.dimension == "style"
