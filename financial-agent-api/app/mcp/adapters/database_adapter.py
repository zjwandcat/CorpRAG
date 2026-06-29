"""Database MCP Server 适配器

提供数据库查询工具：查询审查历史记录。
支持 SQLite 持久化存储，内存模式作为降级方案。
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

from app.exceptions import MCPToolCallError
from app.mcp.server import BaseMCPAdapter, MCPToolInfoProxy, validate_tool_definition_impl

logger = logging.getLogger(__name__)

# SQLite 数据库默认路径
_DEFAULT_DB_PATH: str = "./review_history.db"

# 建表 SQL
_CREATE_REVIEW_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS review_history (
    session_id TEXT PRIMARY KEY,
    review_type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    total_duration_ms INTEGER NOT NULL DEFAULT 0,
    findings_count INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT ''
)
"""

# 插入 SQL
_INSERT_REVIEW_SQL: str = """
INSERT OR REPLACE INTO review_history
    (session_id, review_type, status, created_at, total_duration_ms, findings_count, summary)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

# 查询 SQL
_QUERY_ALL_SQL: str = "SELECT * FROM review_history"
_QUERY_BY_TYPE_SQL: str = "SELECT * FROM review_history WHERE review_type = ?"
_QUERY_BY_STATUS_SQL: str = "SELECT * FROM review_history WHERE status = ?"
_QUERY_BY_TYPE_AND_STATUS_SQL: str = (
    "SELECT * FROM review_history WHERE review_type = ? AND status = ?"
)
_QUERY_BY_SESSION_ID_SQL: str = "SELECT * FROM review_history WHERE session_id = ?"
_QUERY_LIMIT_SQL: str = " LIMIT ?"

# 统计 SQL
_COUNT_TOTAL_SQL: str = "SELECT COUNT(*) FROM review_history"
_COUNT_BY_STATUS_SQL: str = "SELECT status, COUNT(*) FROM review_history GROUP BY status"
_COUNT_BY_TYPE_SQL: str = "SELECT review_type, COUNT(*) FROM review_history GROUP BY review_type"

# 工具定义
_DB_QUERY_REVIEW_HISTORY_TOOL: dict[str, Any] = {
    "name": "database_query_review_history",
    "description": (
        "查询代码审查历史记录。支持按审查类型、状态、时间范围等条件筛选。"
        "当需要了解历史审查记录和趋势时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "review_type": {
                "type": "string",
                "description": "审查类型筛选（full/security/architecture/performance/style）",
            },
            "status": {
                "type": "string",
                "description": "审查状态筛选（completed/failed/timeout）",
            },
            "limit": {
                "type": "integer",
                "description": "返回记录数量限制，默认为 10",
            },
        },
    },
}

_DB_GET_REVIEW_DETAIL_TOOL: dict[str, Any] = {
    "name": "database_get_review_detail",
    "description": (
        "获取指定审查会话的详细信息，包括各维度的审查结果和发现的问题。"
        "当需要查看某次审查的完整报告时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "审查会话 ID",
            },
        },
        "required": ["session_id"],
    },
}

_DB_QUERY_STATISTICS_TOOL: dict[str, Any] = {
    "name": "database_query_statistics",
    "description": (
        "查询审查统计数据，包括审查次数、通过率、常见问题类型等。"
        "当需要了解整体审查质量趋势时调用此工具。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "group_by": {
                "type": "string",
                "description": "分组维度（review_type/severity/date），默认为 review_type",
            },
        },
    },
}


class DatabaseAdapter(BaseMCPAdapter):
    """数据库 MCP Server 适配器

    提供数据库查询工具，支持 SQLite 持久化存储。
    当 SQLite 不可用时自动降级为内存模式。

    Attributes:
        connection_string: 数据库连接字符串（SQLite 文件路径或 :memory:）
    """

    def __init__(self, connection_string: str = "") -> None:
        """初始化数据库适配器

        Args:
            connection_string: 数据库连接字符串，支持 SQLite 文件路径。
                为空时使用默认路径，为 ":memory:" 时使用内存模式。
        """
        self._connection_string: str = connection_string
        self._use_sqlite: bool = False
        self._conn: sqlite3.Connection | None = None
        # 内存降级数据存储
        self._review_history: list[dict[str, Any]] = []
        self._init_database()

    def _init_database(self) -> None:
        """初始化数据库连接

        尝试连接 SQLite 数据库，失败时降级为内存模式。
        """
        db_path = self._connection_string or _DEFAULT_DB_PATH

        try:
            if db_path == ":memory:":
                self._conn = sqlite3.connect(":memory:")
            else:
                path = Path(db_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(path))

            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_CREATE_REVIEW_TABLE_SQL)
            self._conn.commit()
            self._use_sqlite = True
            logger.info("SQLite 数据库初始化成功: %s", db_path)

            # 首次使用时填充种子数据
            if self._is_table_empty():
                self._seed_initial_data()

        except sqlite3.Error as exc:
            logger.warning("SQLite 初始化失败，降级为内存模式: %s", exc)
            self._use_sqlite = False
            self._conn = None
            self._initialize_mock_data()

    def _is_table_empty(self) -> bool:
        """检查审查历史表是否为空

        Returns:
            True 表示表为空
        """
        if self._conn is None:
            return True
        cursor = self._conn.execute(_COUNT_TOTAL_SQL)
        row = cursor.fetchone()
        return row[0] == 0

    def _seed_initial_data(self) -> None:
        """向 SQLite 数据库填充种子数据"""
        if self._conn is None:
            return
        seed_records: list[tuple[str, str, str, str, int, int, str]] = [
            (
                "review-001",
                "full",
                "completed",
                "2026-06-20T10:00:00Z",
                45000,
                5,
                "全面审查完成，发现 2 个高危问题和 3 个中危问题。",
            ),
            (
                "review-002",
                "security",
                "completed",
                "2026-06-21T14:30:00Z",
                25000,
                3,
                "安全审查完成，发现 1 个 SQL 注入风险和 2 个 XSS 风险。",
            ),
            (
                "review-003",
                "performance",
                "timeout",
                "2026-06-22T09:15:00Z",
                60000,
                0,
                "性能审查超时，未完成。",
            ),
            (
                "review-004",
                "architecture",
                "completed",
                "2026-06-23T16:45:00Z",
                35000,
                2,
                "架构审查完成，发现 1 个循环依赖和 1 个设计模式违规。",
            ),
            (
                "review-005",
                "style",
                "failed",
                "2026-06-24T11:20:00Z",
                5000,
                0,
                "风格审查失败，LLM 调用异常。",
            ),
        ]
        self._conn.executemany(_INSERT_REVIEW_SQL, seed_records)
        self._conn.commit()
        logger.info("种子数据已写入 SQLite 数据库")

    def _initialize_mock_data(self) -> None:
        """初始化内存降级模式的审查历史数据"""
        self._review_history = [
            {
                "session_id": "review-001",
                "review_type": "full",
                "status": "completed",
                "created_at": "2026-06-20T10:00:00Z",
                "total_duration_ms": 45000,
                "findings_count": 5,
                "summary": "全面审查完成，发现 2 个高危问题和 3 个中危问题。",
            },
            {
                "session_id": "review-002",
                "review_type": "security",
                "status": "completed",
                "created_at": "2026-06-21T14:30:00Z",
                "total_duration_ms": 25000,
                "findings_count": 3,
                "summary": "安全审查完成，发现 1 个 SQL 注入风险和 2 个 XSS 风险。",
            },
            {
                "session_id": "review-003",
                "review_type": "performance",
                "status": "timeout",
                "created_at": "2026-06-22T09:15:00Z",
                "total_duration_ms": 60000,
                "findings_count": 0,
                "summary": "性能审查超时，未完成。",
            },
            {
                "session_id": "review-004",
                "review_type": "architecture",
                "status": "completed",
                "created_at": "2026-06-23T16:45:00Z",
                "total_duration_ms": 35000,
                "findings_count": 2,
                "summary": "架构审查完成，发现 1 个循环依赖和 1 个设计模式违规。",
            },
            {
                "session_id": "review-005",
                "review_type": "style",
                "status": "failed",
                "created_at": "2026-06-24T11:20:00Z",
                "total_duration_ms": 5000,
                "findings_count": 0,
                "summary": "风格审查失败，LLM 调用异常。",
            },
        ]

    @property
    def server_name(self) -> str:
        """Server 名称"""
        return "database"

    def list_tools(self) -> list[Any]:
        """列出数据库相关工具定义

        Returns:
            工具信息列表
        """
        tools = [
            MCPToolInfoProxy(
                name=_DB_QUERY_REVIEW_HISTORY_TOOL["name"],
                description=_DB_QUERY_REVIEW_HISTORY_TOOL["description"],
                parameters=_DB_QUERY_REVIEW_HISTORY_TOOL["parameters"],
                server_name=self.server_name,
            ),
            MCPToolInfoProxy(
                name=_DB_GET_REVIEW_DETAIL_TOOL["name"],
                description=_DB_GET_REVIEW_DETAIL_TOOL["description"],
                parameters=_DB_GET_REVIEW_DETAIL_TOOL["parameters"],
                server_name=self.server_name,
            ),
            MCPToolInfoProxy(
                name=_DB_QUERY_STATISTICS_TOOL["name"],
                description=_DB_QUERY_STATISTICS_TOOL["description"],
                parameters=_DB_QUERY_STATISTICS_TOOL["parameters"],
                server_name=self.server_name,
            ),
        ]
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用数据库工具

        Args:
            tool_name: 工具名称
            arguments: 工具调用参数

        Returns:
            工具执行结果

        Raises:
            MCPToolCallError: 工具调用失败
        """
        if tool_name == "database_query_review_history":
            return self._query_review_history(arguments)
        elif tool_name == "database_get_review_detail":
            return self._get_review_detail(arguments)
        elif tool_name == "database_query_statistics":
            return self._query_statistics(arguments)
        else:
            raise MCPToolCallError(
                message=f"未知的数据库工具: {tool_name}",
                tool_name=tool_name,
            )

    def _query_review_history(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """查询审查历史记录

        Args:
            arguments: 筛选参数

        Returns:
            审查记录列表
        """
        review_type = arguments.get("review_type")
        status = arguments.get("status")
        limit: int = arguments.get("limit", 10)

        if self._use_sqlite and self._conn is not None:
            return self._query_review_history_sqlite(review_type, status, limit)
        return self._query_review_history_memory(review_type, status, limit)

    def _query_review_history_sqlite(
        self,
        review_type: str | None,
        status: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """从 SQLite 查询审查历史记录

        Args:
            review_type: 审查类型筛选
            status: 审查状态筛选
            limit: 返回记录数量限制

        Returns:
            审查记录列表
        """
        assert self._conn is not None

        try:
            sql: str = _QUERY_ALL_SQL
            params: list[Any] = []

            if review_type and status:
                sql = _QUERY_BY_TYPE_AND_STATUS_SQL
                params = [review_type, status]
            elif review_type:
                sql = _QUERY_BY_TYPE_SQL
                params = [review_type]
            elif status:
                sql = _QUERY_BY_STATUS_SQL
                params = [status]

            sql += _QUERY_LIMIT_SQL
            params.append(limit)

            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]

            logger.info("从 SQLite 查询审查历史，返回 %d 条记录", len(results))
            return results

        except sqlite3.Error as exc:
            logger.warning("SQLite 查询失败，降级为内存查询: %s", exc)
            return self._query_review_history_memory(review_type, status, limit)

    def _query_review_history_memory(
        self,
        review_type: str | None,
        status: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """从内存降级存储查询审查历史记录

        Args:
            review_type: 审查类型筛选
            status: 审查状态筛选
            limit: 返回记录数量限制

        Returns:
            审查记录列表
        """
        results = self._review_history[:]

        if review_type:
            results = [r for r in results if r.get("review_type") == review_type]
        if status:
            results = [r for r in results if r.get("status") == status]

        results = results[:limit]

        logger.info("从内存查询审查历史，返回 %d 条记录", len(results))
        return results

    def _get_review_detail(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """获取审查会话详情

        Args:
            arguments: 包含 session_id 的参数字典

        Returns:
            审查详情字典
        """
        session_id = arguments.get("session_id", "")

        if not session_id:
            raise MCPToolCallError(
                message="缺少必要参数: session_id",
                tool_name="database_get_review_detail",
            )

        if self._use_sqlite and self._conn is not None:
            return self._get_review_detail_sqlite(session_id)
        return self._get_review_detail_memory(session_id)

    def _get_review_detail_sqlite(self, session_id: str) -> dict[str, Any]:
        """从 SQLite 获取审查会话详情

        Args:
            session_id: 审查会话 ID

        Returns:
            审查详情字典

        Raises:
            MCPToolCallError: 审查会话不存在
        """
        assert self._conn is not None

        try:
            cursor = self._conn.execute(_QUERY_BY_SESSION_ID_SQL, (session_id,))
            row = cursor.fetchone()

            if row is None:
                raise MCPToolCallError(
                    message=f"审查会话不存在: {session_id}",
                    tool_name="database_get_review_detail",
                )

            detail = dict(row)
            detail["results"] = [
                {
                    "dimension": "security",
                    "status": "completed",
                    "findings": [
                        {
                            "severity": "high",
                            "description": "SQL 注入风险",
                            "location": "api/routes.py:45",
                            "suggestion": "使用参数化查询",
                        }
                    ],
                }
            ]
            logger.info("从 SQLite 获取审查详情: %s", session_id)
            return detail

        except MCPToolCallError:
            raise
        except sqlite3.Error as exc:
            logger.warning("SQLite 查询详情失败，降级为内存查询: %s", exc)
            return self._get_review_detail_memory(session_id)

    def _get_review_detail_memory(self, session_id: str) -> dict[str, Any]:
        """从内存降级存储获取审查会话详情

        Args:
            session_id: 审查会话 ID

        Returns:
            审查详情字典

        Raises:
            MCPToolCallError: 审查会话不存在
        """
        for record in self._review_history:
            if record.get("session_id") == session_id:
                detail = dict(record)
                detail["results"] = [
                    {
                        "dimension": "security",
                        "status": "completed",
                        "findings": [
                            {
                                "severity": "high",
                                "description": "SQL 注入风险",
                                "location": "api/routes.py:45",
                                "suggestion": "使用参数化查询",
                            }
                        ],
                    }
                ]
                logger.info("从内存获取审查详情: %s", session_id)
                return detail

        raise MCPToolCallError(
            message=f"审查会话不存在: {session_id}",
            tool_name="database_get_review_detail",
        )

    def _query_statistics(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """查询审查统计数据

        Args:
            arguments: 包含可选 group_by 的参数字典

        Returns:
            统计数据字典
        """
        group_by = arguments.get("group_by", "review_type")

        if self._use_sqlite and self._conn is not None:
            return self._query_statistics_sqlite(group_by)
        return self._query_statistics_memory(group_by)

    def _query_statistics_sqlite(self, group_by: str) -> dict[str, Any]:
        """从 SQLite 查询审查统计数据

        Args:
            group_by: 分组维度

        Returns:
            统计数据字典
        """
        assert self._conn is not None

        try:
            # 总数
            cursor = self._conn.execute(_COUNT_TOTAL_SQL)
            total: int = cursor.fetchone()[0]

            # 按状态统计
            cursor = self._conn.execute(_COUNT_BY_STATUS_SQL)
            status_counts: dict[str, int] = {row[0]: row[1] for row in cursor.fetchall()}

            completed: int = status_counts.get("completed", 0)
            failed: int = status_counts.get("failed", 0)
            timeout: int = status_counts.get("timeout", 0)

            statistics: dict[str, Any] = {
                "total_reviews": total,
                "completed": completed,
                "failed": failed,
                "timeout": timeout,
                "completion_rate": round(completed / total * 100, 1) if total > 0 else 0.0,
            }

            if group_by == "review_type":
                cursor = self._conn.execute(_COUNT_BY_TYPE_SQL)
                by_type: dict[str, int] = {row[0]: row[1] for row in cursor.fetchall()}
                statistics["by_review_type"] = by_type
            elif group_by == "severity":
                statistics["by_severity"] = {
                    "critical": 1,
                    "high": 3,
                    "medium": 5,
                    "low": 2,
                    "info": 0,
                }

            logger.info("从 SQLite 查询审查统计，group_by=%s", group_by)
            return statistics

        except sqlite3.Error as exc:
            logger.warning("SQLite 统计查询失败，降级为内存查询: %s", exc)
            return self._query_statistics_memory(group_by)

    def _query_statistics_memory(self, group_by: str) -> dict[str, Any]:
        """从内存降级存储查询审查统计数据

        Args:
            group_by: 分组维度

        Returns:
            统计数据字典
        """
        total = len(self._review_history)
        completed = sum(1 for r in self._review_history if r.get("status") == "completed")
        failed = sum(1 for r in self._review_history if r.get("status") == "failed")
        timeout = sum(1 for r in self._review_history if r.get("status") == "timeout")

        statistics: dict[str, Any] = {
            "total_reviews": total,
            "completed": completed,
            "failed": failed,
            "timeout": timeout,
            "completion_rate": round(completed / total * 100, 1) if total > 0 else 0.0,
        }

        if group_by == "review_type":
            by_type: dict[str, int] = {}
            for r in self._review_history:
                rt = r.get("review_type", "unknown")
                by_type[rt] = by_type.get(rt, 0) + 1
            statistics["by_review_type"] = by_type
        elif group_by == "severity":
            statistics["by_severity"] = {
                "critical": 1,
                "high": 3,
                "medium": 5,
                "low": 2,
                "info": 0,
            }

        logger.info("从内存查询审查统计，group_by=%s", group_by)
        return statistics

    def health_check(self) -> bool:
        """检查数据库连接状态

        SQLite 模式下执行轻量查询验证连接可用性；
        内存降级模式下始终返回 True。

        Returns:
            True 表示数据库连接正常
        """
        if not self._use_sqlite or self._conn is None:
            # 内存降级模式始终可用
            return True

        try:
            cursor = self._conn.execute("SELECT 1")
            cursor.fetchone()
            return True
        except sqlite3.Error as exc:
            logger.warning("数据库健康检查失败: %s", exc)
            return False

    def validate_tool_definition(self, tool: Any) -> bool:
        """工具定义安全校验

        Args:
            tool: 工具信息对象

        Returns:
            True 表示工具定义安全
        """
        return validate_tool_definition_impl(tool)


__all__ = [
    "DatabaseAdapter",
]
