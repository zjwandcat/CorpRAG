"""Agent Platform Client SDK 安装配置

使用 ``pip install .`` 从 sdk/ 目录安装本 SDK，
或 ``pip install -e .`` 以可编辑模式安装用于开发。
"""

from setuptools import setup, find_packages

setup(
    name="agent-platform-client",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["httpx>=0.27"],
    python_requires=">=3.10",
    description="Python SDK for Enterprise GenAI Agent Platform",
)
