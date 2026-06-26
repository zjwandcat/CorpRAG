import os
import shutil
import sys
from pathlib import Path
from typing import Final

from app.core.enums import DocumentFormat, UserCommand
from app.exceptions import VectorStoreError
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

__all__ = [
    "CHROMA_DB_DIR",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "DEFAULT_NIM_EMBEDDING_MODEL",
    "DEFAULT_OPENAI_EMBEDDING_MODEL",
    "DEFAULT_TOP_K",
    "REPORTS_DIR",
    "SIMILARITY_THRESHOLD",
    "build_or_load_vectorstore",
    "create_embeddings",
    "load_documents",
    "main",
    "print_results",
    "search_documents",
    "split_documents",
]


REPORTS_DIR: Final[Path] = Path(__file__).resolve().parent / "data" / "reports"
CHROMA_DB_DIR: Final[Path] = Path(__file__).resolve().parent / "chroma_db"
CHUNK_SIZE: Final[int] = 500
CHUNK_OVERLAP: Final[int] = 50
DEFAULT_TOP_K: Final[int] = 3
SIMILARITY_THRESHOLD: Final[float] = 1.3
DEFAULT_NIM_EMBEDDING_MODEL: Final[str] = "nvidia/nv-embedqa-e5-v5"
DEFAULT_OPENAI_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"


def _load_env_from_config(filename: str = "nim_config.txt") -> None:
    config_path = Path(__file__).resolve().parent / filename
    if not config_path.exists():
        return

    with open(config_path, encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'`")
            if key and not os.getenv(key):
                os.environ[key] = value


def create_embeddings() -> Embeddings:
    _load_env_from_config()

    nim_api_key = os.getenv("NVIDIA_API_KEY")
    nim_base_url = os.getenv("NIM_BASE_URL")
    nim_model = os.getenv("NIM_EMBEDDING_MODEL", DEFAULT_NIM_EMBEDDING_MODEL)

    if nim_api_key and nim_base_url:
        print(f"[配置] 使用 NVIDIA NIM：{nim_base_url}，模型：{nim_model}")
        return NVIDIAEmbeddings(
            model=nim_model,
            api_key=nim_api_key,
            base_url=nim_base_url,
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        DEFAULT_OPENAI_EMBEDDING_MODEL,
    )

    if openai_api_key:
        if openai_base_url:
            print(f"[配置] 使用 OpenAI 兼容接口：{openai_base_url}")
        else:
            print("[配置] 使用 OpenAI 官方接口")
        return OpenAIEmbeddings(
            model=openai_model,
            api_key=openai_api_key,
            base_url=openai_base_url,
        )

    print("[错误] 未设置 Embedding API 凭证。")
    print(
        "[提示] 请在 nim_config.txt 中配置 NVIDIA_API_KEY + NIM_BASE_URL，"
        "或 OPENAI_API_KEY + OPENAI_BASE_URL。",
    )
    sys.exit(1)


def load_documents(reports_dir: Path) -> list[Document]:
    documents: list[Document] = []

    if not reports_dir.exists():
        print(f"[警告] 目录不存在：{reports_dir}")
        return documents

    supported_suffixes = {DocumentFormat.PDF.value, DocumentFormat.TXT.value}
    files = sorted(reports_dir.iterdir())

    for file_path in files:
        if file_path.suffix.lower() not in supported_suffixes:
            continue

        print(f"[加载] {file_path.name}")
        try:
            if file_path.suffix.lower() == DocumentFormat.PDF.value:
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding="utf-8")

            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
            documents.extend(docs)
        except Exception as exc:
            print(f"[警告] 加载文件 {file_path.name} 失败：{exc}")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[切片] 共生成 {len(chunks)} 个文本块")
    return chunks


def _vectorstore_exists(chroma_db_dir: Path) -> bool:
    return chroma_db_dir.exists() and any(chroma_db_dir.iterdir())


def _vectorstore_count(vectorstore: Chroma) -> int:
    try:
        return vectorstore._collection.count()
    except Exception as exc:
        print(f"[警告] 无法获取向量库文档数量：{exc}")
        return 0


def build_or_load_vectorstore() -> Chroma:
    embeddings = create_embeddings()

    if _vectorstore_exists(CHROMA_DB_DIR):
        print(f"[加载] 发现已有向量库：{CHROMA_DB_DIR}")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=embeddings,
        )
        count = _vectorstore_count(vectorstore)
        print(f"[诊断] 向量库中文档数量：{count}")
        if count > 0:
            return vectorstore
        print("[警告] 向量库为空，将删除后重新构建...")
        try:
            shutil.rmtree(CHROMA_DB_DIR)
        except OSError as exc:
            raise VectorStoreError(
                f"无法自动删除空向量库：{exc}。"
                "请手动删除 chroma_db 文件夹后重新运行脚本。",
            ) from exc

    print("[构建] 未找到已有向量库，开始加载文档并入库...")

    documents = load_documents(REPORTS_DIR)
    if not documents:
        print("[错误] 没有加载到任何文档，请检查 data/reports/ 目录。")
        sys.exit(1)

    chunks = split_documents(documents)
    if not chunks:
        print("[错误] 文档切片后为空，请检查文档内容。")
        sys.exit(1)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )
    count = _vectorstore_count(vectorstore)
    print(f"[完成] 向量库已保存到：{CHROMA_DB_DIR}，共 {count} 个文本块")
    return vectorstore


def search_documents(
    vectorstore: Chroma,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SIMILARITY_THRESHOLD,
) -> list[Document]:
    print(f"\n[检索] 查询：{query}")
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)

    results: list[Document] = []
    for doc, score in docs_and_scores:
        if score <= score_threshold:
            doc.metadata["score"] = round(score, 4)
            results.append(doc)

    return results


def print_results(results: list[Document]) -> None:
    print(f"[结果] 命中 {len(results)} 个文本块：")
    if not results:
        print(
            "[提示] 未命中任何文本块。可能原因：\n"
            "  1. 向量库为空或未成功入库；\n"
            "  2. 入库与检索使用的 Embedding 模型不一致；\n"
            "  3. 查询与文档内容语义差异较大。\n"
            "可尝试删除 chroma_db 目录后重新运行脚本。",
        )
        return

    for idx, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "未知来源")
        score = doc.metadata.get("score", "未知")
        print(f"\n--- 文本块 {idx} / 来源：{source} / 距离：{score} ---")
        print(doc.page_content)
        print("-" * 60)


def main() -> None:
    vectorstore = build_or_load_vectorstore()
    doc_count = _vectorstore_count(vectorstore)

    print("\n" + "=" * 60)
    print("LangChain RAG 检索交互示例")
    print(f"当前向量库中文本块数量：{doc_count}")
    print("输入查询即可（例如：比亚迪的销量情况）")
    print("输入 exit / quit / 退出 即可结束")
    print("=" * 60)

    while True:
        try:
            query = input("\n查询: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue

        if query.lower() in {
            UserCommand.EXIT.value,
            UserCommand.QUIT.value,
            UserCommand.EXIT_CN.value,
        }:
            print("再见！")
            break

        results = search_documents(vectorstore, query, DEFAULT_TOP_K)
        print_results(results)


if __name__ == "__main__":
    main()
